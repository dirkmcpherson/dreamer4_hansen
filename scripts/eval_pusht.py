
import os
import sys
import argparse
import random
import numpy as np
import torch
import gymnasium as gym
import gym_pusht
import imageio
import wandb

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dreamer4.model import Dynamics, temporal_patchify, pack_bottleneck_to_spatial, temporal_unpatchify, unpack_spatial_to_bottleneck
from dreamer4.train_dynamics import load_frozen_tokenizer_from_pt_ckpt, make_tau_schedule, sample_one_timestep_packed, decode_packed_to_frames

class MPPIPlanner:
    def __init__(self, dyn, enc, dec, args, device):
        self.dyn = dyn
        self.enc = enc
        self.dec = dec
        self.args = args
        self.device = device
        
        # Shortcuts
        self.H = args.H
        self.W = args.W
        self.patch = args.patch
        self.k_max = args.k_max
        self.packing_factor = args.packing_factor
        
        # Schedule
        self.sched = make_tau_schedule(k_max=self.k_max, schedule="finest") # or shortcut?
        
        # Action space: PushT is [-1, 1] for 2 dims? 
        # Actually gym_pusht observations "agent_pos" are in [0, 512].
        # But actions? "gym_pusht/PushT-v0" actions are usually [0, 512] target pos?
        # Let's check environment details or assume we need to normalize.
        # eval_pusht_monolithic.py used: (best_action + 1) * 256.0
        # which implies model predicts [-1, 1] and env expects [0, 512].
        self.action_dim = 2
        
    @torch.no_grad()
    def plan(self, obs, num_samples=64, horizon=16):
        # obs: (H, W, 3) uint8 or (3, H, W)? Gym returns (H, W, 3) usually.
        # Check normalization
        if obs.dtype == np.uint8:
             obs = obs.astype(np.float32) / 255.0
        
        # (H, W, 3) -> (1, 3, H, W)
        frame = torch.from_numpy(obs).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        
        # Resize if mismatch
        if frame.shape[2] != self.H or frame.shape[3] != self.W:
             import torch.nn.functional as F
             frame = F.interpolate(frame, size=(self.H, self.W), mode='bilinear', align_corners=False)
        
        # 1. Encode Obs -> Latent z_t
        # Tokenizer expects (B, T, C, H, W) -> (1, 1, 3, H, W)
        frames = frame.unsqueeze(1)
        patches = temporal_patchify(frames, self.patch)
        z_btLd, _ = self.enc(patches) # (1, 1, L, D)
        
        n_spatial = z_btLd.shape[2] // self.packing_factor
        z_curr_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=self.packing_factor) # (1, 1, Sz, Dz)
        
        # 2. Sample random actions
        # (N, H, A)
        actions = torch.rand((num_samples, horizon, self.action_dim), device=self.device) * 2 - 1
        
        # Repeat state for samples
        # z_curr_packed: (1, 1, Sz, Dz) -> (N, 1, Sz, Dz)
        curr_state = z_curr_packed.repeat(num_samples, 1, 1, 1)
        
        cumulative_rewards = torch.zeros(num_samples, device=self.device)
        
        # 3. Rollout Horizon
        # Note: This simple rollout assumes we just need reward prediction at each step.
        # Our dynamics model predicts Next State.
        # Does it predict Reward? Yes, r_hat is 3rd output.
        # dyn(..., z_tilde, ...) -> (z1_hat, h, r_hat)
        
        # We need step_idx and sigma_idx to run dyn "cleanly" (inference mode).
        # "Clean" means step_idx=emax (finest resolution), signal_idx=k_max-1 (clean signal).
        # Actually sample_one_timestep_packed runs the diffusion refinement K steps.
        # For planning speed, maybe we run fewer steps?
        # But let's follow sample_one_timestep_packed.
        
        # But sample_one_timestep_packed runs loop K times. Multiplied by Horizon. Multiplied by Num Samples.
        # That's SLOW if K is large (16). 16*16*64 = 16k inference calls?
        # We can parallelize num_samples (batch dim).
        
        # To get REWARD for (s_t, a_t):
        # We can run dyn ONCE mixed with clean signal?
        # dyn(actions=a_t, step_idx=..., z=s_t) -> r_hat
        
        # Let's define a helper for "predict next state and reward"
        
        history_state = curr_state # (N, T_accum, Sz, Dz)
        
        for t in range(horizon):
            act_step = actions[:, t:t+1] # (N, 1, A)
            
            # Predict Next State (sample_one_timestep_packed is for next z)
            # Efficient implementation: pass (N, t+1, ...) to sample_one_timestep
            # Wait, sample_one_timestep_packed takes 'past_packed'.
            # It samples ONE step.
            
            # Reward? The sample function doesn't return reward.
            # We need to compute reward explicitly.
            # Reward depends on (s_t, a_t) or (s_t+1)?
            # In dreamer4, reward is usually predicted from latent s_t (or s_t+1).
            # Let's inspect 'train_dynamics.py' loss.
            # 'r_hat_full' comes from 'dynamics(...)'.
            # It uses 'z_tilde_full' (corrupted input).
            # If we pass 'clean' input, we get 'clean' reward prediction?
            # Yes.
            
            # So, before stepping, we compute reward for (current_state, current_action)?
            # Or after stepping?
            # If r_t corresponds to transition t->t+1. 
            
            # Let's compute reward using clean pass.
            # We need dummy step_idx/signal_idx for "Clean".
            # Clean means: we provide FULL information (signal_idx = k_max?).
            # Dynamics usually expects `signal_idxs`: index of signal level in [0, k_max].
            # k_max means "clean data" (no noise)? 
            # In training: `tau, tau_idx` are sampled. 
            # `tau_idx` goes up to k_max-1.
            # `k_max` is fully clean?
            # In `make_tau_schedule`: `tau_idx` includes `k_max` as final "clean" index.
            
            # Step for reward:
            # We use `step_idxs` (resolution) and `signal_idxs` (noise level).
            # For reward, we assume resolution `emax` (finest).
            # And signal `k_max` (clean).
            
            emax = int(round(np.log2(self.k_max)))
            B_samp = num_samples
            
            # We need to construct inputs for dynamics
            # dyn(actions, step_idxs, signal_idxs, packed_seq, act_mask, ...)
            
            # We pass current history.
            step_idxs = torch.full((B_samp, history_state.shape[1]), emax, device=self.device, dtype=torch.long)
            signal_idxs = torch.full((B_samp, history_state.shape[1]), self.k_max, device=self.device, dtype=torch.long)
            
            # For reward, we might just need the last step?
            # Dynamics processes whole sequence usually, but maybe we can optimize?
            # It uses Attention. So explicit seq length matters.
            
            # Just run forward pass once on history?
            # This returns r_hat for ALL steps. We take last.
            
            # NOTE: MPPI with full attention over horizon history is heavy.
            # We limit context length?
            # Let's just pass `history_state` (which grows).
            
            # Append action to history? No, dynamics takes actions separately.
            # We have actions for [0..t].
            actions_in = actions[:, :t+1]
            act_mask_in = torch.ones((B_samp, t+1, self.action_dim), device=self.device)
            
            with torch.no_grad():
                _, _, r_hat_seq = self.dyn(
                    actions_in,
                    step_idxs,
                    signal_idxs,
                    history_state,
                    act_mask=act_mask_in
                )
            
            # r_hat_seq: (N, t+1) (or similar shape? check dyn output)
            # Dynamics returns (z1_hat, h, r_hat).
            r_curr = r_hat_seq[:, -1] # (N,)
            cumulative_rewards += r_curr
            
            # Sample next state
            # Pad actions to match sequence length (T_hist + 1 for new frame)
            # history_state is (N, t+1, ...). z_next adds 1. Total T=t+2.
            # actions_in is (N, t+1, A). Need T=t+2.
            
            # Use zeros for the "future" action slot associated with the frame we are generating?
            # or just repeat last?
            # It doesn't strictly matter for generating z_{t+1} because causal mask 
            # prevents z_{t+1} from seeing a_{t+1}.
            # But shape must match.
            
            # Pad with zeros
            pad_act = torch.zeros((B_samp, 1, self.action_dim), device=self.device)
            actions_padded = torch.cat([actions_in, pad_act], dim=1)
            
            act_mask_in = torch.ones((B_samp, t+2, self.action_dim), device=self.device)
            # Mask the padded action?
            act_mask_in[:, -1, :] = 0
            
            z_next = sample_one_timestep_packed(
                self.dyn, 
                past_packed=history_state,
                k_max=self.k_max,
                sched=self.sched,
                actions=actions_padded, 
                act_mask=act_mask_in
            )
            
            # Append next state
            history_state = torch.cat([history_state, z_next.unsqueeze(1)], dim=1)
            
        
        # 4. Select Best Action
        best_idx = torch.argmax(cumulative_rewards)
        best_action = actions[best_idx, 0] # First action
        
        # Denormalize
        # [-1, 1] -> [0, 512]
        return (best_action.cpu().numpy() + 1) * 256.0

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    print("Loading tokenizer...")
    enc, dec, tok_args = load_frozen_tokenizer_from_pt_ckpt(args.tokenizer_ckpt, device=device)
    
    # 2. Load Dynamics
    print("Loading Dynamics...")
    # Reconstruct Dynamics config
    # We load arguments from checkpoint if possible?
    dyn_ckpt = torch.load(args.dyn_ckpt, map_location="cpu")
    dyn_args = dyn_ckpt["args"]
    
    # Extract structural args
    d_bottleneck = dyn_args.get("d_bottleneck", int(tok_args.get("d_bottleneck", 32)))
    packing_factor = dyn_args.get("packing_factor", 1)
    d_spatial = dyn_args.get("d_spatial", d_bottleneck * packing_factor)
    n_spatial = dyn_args.get("n_spatial", int(tok_args.get("n_latents", 16)) // packing_factor)
    
    dyn = Dynamics(
        d_model=dyn_args["d_model"],
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=0,
        n_agent=0,
        n_heads=dyn_args["n_heads"],
        depth=dyn_args["depth"],
        k_max=dyn_args["k_max"],
        dropout=0.0,
        mlp_ratio=dyn_args.get("mlp_ratio", 4.0),
        time_every=dyn_args.get("time_every", 1),
        space_mode=dyn_args.get("space_mode", "wm_agent_isolated"),
        action_dim=2
    ).to(device)
    
    dyn.load_state_dict(dyn_ckpt["dynamics"])
    dyn.eval()
    
    # 3. Setup Planner
    # Ensure args has 'k_max' etc matching dynamics if needed, or MPPI params
    args.H = tok_args["H"]
    args.W = tok_args["W"]
    args.patch = tok_args["patch"]
    args.k_max = dyn_args["k_max"]
    args.packing_factor = dyn_args["packing_factor"]
    
    planner = MPPIPlanner(dyn, enc, dec, args, device)
    
    # 4. Run Env
    try:
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    except Exception:
        # Fallback if specific version/arg differs?
        env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
        print("Warning: Could not set obs_type='pixels'. Obs might be state.")
    
    wandb.init(
        project=args.wandb_project, 
        name=args.wandb_run_name,
        config=vars(args),
        mode="online" if not args.no_wandb else "offline"
    )
    
    obs, info = env.reset(seed=42)
    done = False
    truncated = False
    step = 0
    total_reward = 0
    frames = []
    
    print("Starting evaluation loop...")
    
    while not (done or truncated) and step < 300:
        action = planner.plan(obs)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(obs)
        step += 1
        if step % 10 == 0:
            print(f"Eval Step {step}: Reward {reward:.2f}, Total {total_reward:.2f}")
            
    print(f"Finished. Total Reward: {total_reward}")
    wandb.log({"eval/total_reward": total_reward})
    
    # Save GIF
    save_path = os.path.join(args.save_dir, "eval.gif")
    imageio.mimsave(save_path, frames, fps=10)
    print(f"Saved GIF to {save_path}")
    
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--dyn_ckpt", type=str, required=True) 
    parser.add_argument("--save_dir", type=str, default=".")
    
    # Wandb
    parser.add_argument("--wandb_project", type=str, default="dreamer4-eval")
    parser.add_argument("--wandb_run_name", type=str, default="eval")
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    evaluate(args)
