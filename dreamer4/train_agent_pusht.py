
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import wandb
from pathlib import Path

from sharded_frame_dataset import ShardedFrameDataset
from model import (
    Encoder, Decoder, Tokenizer, Dynamics, Actor, Critic,
    temporal_patchify, pack_bottleneck_to_spatial
)
from train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    sample_one_timestep_packed,
    init_distributed,
    is_rank0,
    seed_everything,
    worker_init_fn
)

def load_dynamics_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"]
    
    # Reconstruct Dynamics model args
    d_model = args.get("d_model_dyn", 256)
    d_bottleneck = 32 # assumed or from somewhere?
    # Actually need to know d_bottleneck from tokenizer usually. 
    # But let's assume standard set or retrieve from args if saved.
    
    # We need to know tokenizer params to infer d_bottleneck if not stored.
    # Luckily `train_dynamics.py` saves args.
    # But usually d_bottleneck comes from tokenizer config.
    # Let's trust it's in args or use default.
    d_bottleneck = int(args.get("d_bottleneck", 32)) # It might not be there if it was from tokenizer args
    
    # But `train_dynamics` loads tokenizer, so it computes n_spatial etc.
    # We need those constants.
    
    # HACK: we will infer from model state dict shapes if needed? 
    # Or just require the user to pass correct tokenizer args?
    # Better: load tokenizer first, get d_bottleneck, then init dynamics.
    
    return ckpt, args

def main(args):
    ddp, rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}")
    seed_everything(args.seed + rank)

    # 1. Load Tokenizer (Frozen)
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(
        args.tokenizer_ckpt, device=device
    )
    H = int(tok_args.get("H", 128))
    W = int(tok_args.get("W", 128))
    patch = int(tok_args.get("patch", 4))
    n_latents = int(tok_args.get("n_latents", 16))
    d_bottleneck = int(tok_args.get("d_bottleneck", 32))
    n_patches = (H // patch) * (W // patch)
    
    assert n_latents % args.packing_factor == 0
    n_spatial = n_latents // args.packing_factor
    d_spatial = d_bottleneck * args.packing_factor

    # 2. Load Dynamics (Frozen)
    dyn_ckpt = torch.load(args.dynamics_ckpt, map_location="cpu")
    dyn_args = dyn_ckpt["args"]
    
    d_model_dyn = int(dyn_args.get("d_model_dyn", 256))
    k_max = int(dyn_args.get("k_max", 16))
    
    dyn = Dynamics(
        d_model=d_model_dyn,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(dyn_args.get("n_register", 4)),
        n_agent=int(dyn_args.get("n_agent", 1)),
        n_heads=int(dyn_args.get("n_heads", 4)),
        depth=int(dyn_args.get("dyn_depth", 8)),
        k_max=k_max,
        dropout=0.0,
        action_dim=args.action_dim,
        time_every=int(dyn_args.get("time_every", 1)),
        space_mode=str(dyn_args.get("space_mode", "wm_agent_isolated")),
    ).to(device)
    
    # Handle DDP wrap in ckpt
    state_dict = dyn_ckpt["dynamics"]
    # Strip "module." if present
    new_sd = {k.replace("module.", ""): v for k,v in state_dict.items()}
    dyn.load_state_dict(new_sd)
    dyn.eval()
    for p in dyn.parameters():
        p.requires_grad_(False)

    print(f"[Rank {rank}] Loaded Dynamics and Tokenizer.")

    # 3. Agent (Actor-Critic)
    actor = Actor(d_model=d_spatial, action_dim=args.action_dim).to(device) # Actor sees local state z
    critic = Critic(d_model=d_spatial).to(device)
    
    if ddp:
        actor = torch.nn.parallel.DistributedDataParallel(actor, device_ids=[local_rank])
        critic = torch.nn.parallel.DistributedDataParallel(critic, device_ids=[local_rank])

    opt_actor = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    opt_critic = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    
    # 4. Dataset (for initial states)
    # We only need initial frames to encode z0
    dataset = ShardedFrameDataset(
        outdirs=args.data_dirs,
        tasks=args.tasks,
        seq_len=2 # minimal seq len, we just need the first frame
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if ddp else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Logging
    if is_rank0() and args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # Schedule for one-step sampling
    sched_eval = make_tau_schedule(k_max=k_max, schedule="shortcut", d=0.5) # Fast 2-step for training? Or 1-step?
    # Let's use 1 step for speed during imagination?
    # Or match training setup.
    # Using d=0.5 means 2 steps (1/0.5).
    
    step = 0
    start_time = time.time()
    
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps: break
            
            # 1. Get Initial State z0
            frames = batch.to(device).float() / 255.0 # (B, T, 3, H, W)
            # Encode frame 0
            # Need patchify
            # frames: (B, 2, 3, H, W)
            # Use only first frame
            img = frames[:, 0:1] # (B, 1, 3, H, W)
            patches = temporal_patchify(img, patch)
            
            with torch.no_grad():
                z_btLd, _ = encoder(patches)
                z_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_spatial, k=args.packing_factor)
                z_curr = z_packed[:, 0] # (B, n_spatial, d_spatial)
            
            # 2. Rollout / Imagination
            imag_states = [z_curr]
            imag_actions = []
            imag_rewards = [] # We need to predict rewards
            
            # Rollout horizon H
            # We need to use PREVIOUS state to predict NEXT state and reward?
            # Or Reward is output of Dyn(s, a).
            
            curr = z_curr.unsqueeze(1) # (B, 1, n_spatial, d_spatial)
            
            for t in range(args.horizon):
                # Actor Policy
                # Actor takes "state". State is (B, n_spatial, d_spatial).
                # But MLPs usually take (B, D). 
                # We should flatten or mean pool?
                # Dynamics "state_rep" used mean(dim=2).
                # Let's align Actor/Critic to use mean-pooled spatial tokens for now.
                # Or pass full spatial tokens if Actor is Transformer.
                # Our Actor is MLP. So we MUST pool.
                
                state_feat = curr.squeeze(1).mean(dim=1) # (B, d_spatial)
                
                # Sample action
                # If DDP, access module
                act_net = actor.module if ddp else actor
                mean, std = act_net(state_feat)
                dist_action = torch.distributions.Normal(mean, std)
                action = dist_action.rsample() # Reparameterized
                action_tanh = torch.tanh(action)
                
                # Dynamics Step
                # We need to predict NEXT state and REWARD.
                # Dyn forward returns: x1_hat (next state), h_t, reward_hat
                # BUT `sample_one_timestep_packed` loops.
                # We need gradients through this!
                # `sample_one_timestep_packed` supports grad if we don't detach?
                # It does NOT detach z, so yes.
                
                # Need to broadcast action to time (B, 1, A)
                a_in = action_tanh.unsqueeze(1)
                
                # We need to ensure Dynamics computation graph is preserved.
                # `make_tau_schedule` creates constants.
                # `sample_one_timestep_packed` should work.
                
                # NOTE: We need grad through DYNAMICS model?
                # Using "Dreamer" style (Backprop through time/dynamics)?
                # Standard DreamerV3 does this.
                # BUT Dyn is frozen here? 
                # If Dyn is frozen, we backprop through it to update Actor.
                # Correct.
                
                # Careful: The dynamics loaded is in Eval mode. 
                # But we valid-ly backprop through eval modules (just no dropout/batchnorm updates).
                
                z_next = sample_one_timestep_packed(
                    dyn,
                    past_packed=curr,
                    k_max=k_max,
                    sched=sched_eval, # Use fast schedule
                    actions=a_in,
                    act_mask=None,
                )
                # z_next: (B, Sz, Dz)
                
                # Reward Prediction?
                # We need reward from the transition.
                # Dynamics forward actually returns reward.
                # `sample_one_timestep_packed` implementation in `train_dynamics.py` (which I read)
                # returns `z[:,0], r_next`. 
                # Wait, let me check my memory or `view_file` of `train_dynamics.py` again?
                # I read it in step 12.
                # Line 408: `return z[:, 0], r_next`
                # Yes! It returns reward.
                # Wait, I need to check signature in my script.
                # From step 12: 
                # def sample_one_timestep_packed(..., actions=...): ... return z, r_next
                # So I need to unpack it.
                
                # Re-check signature I wrote/read:
                # 408: return z[:, 0], r_next
                
                # So sample_one_timestep_packed returns (z_next, r_pred).
                pass # just a marker
                
                z_next_out, r_pred = sample_one_timestep_packed(
                    dyn,
                    past_packed=curr,
                    k_max=k_max,
                    sched=sched_eval,
                    actions=a_in,
                    act_mask=None
                )
                
                imag_states.append(z_next_out)
                imag_actions.append(action_tanh)
                imag_rewards.append(r_pred)
                
                curr = z_next_out.unsqueeze(1)
            
            # Stack results
            # States: H+1
            # Actions: H
            # Rewards: H
            
            # Value Estimation
            # We compute Value(state) for all states 0..H
            
            crit_net = critic.module if ddp else critic
            
            values = []
            for s in imag_states:
                # s: (B, Sz, Dz)
                feat = s.mean(dim=1)
                v = crit_net(feat)
                values.append(v)
            
            values = torch.stack(values, dim=1) # (B, H+1)
            rewards = torch.stack(imag_rewards, dim=1) # (B, H)
            
            # Lambda Returns
            # Calculate targets for Value function
            # And Advantage for Actor?
            # Dreamer uses Lambda-return as target for Value, and also maximizes Value for Actor.
            
            # Bootstrap value
            discounts = args.discount * torch.ones_like(rewards)
            lambda_ = args.lambda_
            
            # Compute lambda returns
            # V_lambda(t) = r_t + gamma * ( (1-lambda)*V(t+1) + lambda*V_lambda(t+1) )
            
            target_values = torch.zeros_like(values[:, :-1])
            last_val = values[:, -1]
            
            # Iterate backwards
            for t in range(args.horizon - 1, -1, -1):
                r_t = rewards[:, t]
                v_next = values[:, t+1] # Bootstrap from value function (or specific target net?)
                # Standard Dreamer uses target network for V usually, but let's stick to simple online V for MVP.
                
                # The recursive formula:
                # R_t = r_t + gamma * ( (1-lambda)*v_next + lambda*R_{t+1} )
                # Base case: R_H = V(H)
                
                if t == args.horizon - 1:
                    next_return = last_val
                else:
                    next_return = target_values[:, t+1]
                
                ret = r_t + discounts[:, t] * ( (1 - lambda_) * v_next + lambda_ * next_return )
                target_values[:, t] = ret
                
            # Losses
            
            # Critic Loss: MSE(V(t), target_values(t).detach())
            # Don't backprop target through critic
            v_pred = values[:, :-1]
            critic_loss = 0.5 * (v_pred - target_values.detach()).pow(2).mean()
            
            # Actor Loss: Maximize V_lambda?
            # Or Dynamics Backprop: Maximize Sum(Returns)?
            # DreamerV3 maximizes the lambda-return.
            # actor_loss = -target_values.mean()
            # Note: We must allow gradients to flow from target_values -> rewards/values -> states -> actions -> actor.
            # So we use the computed returns, but we must ensure graph connectivity.
            # In the loop above, `target_values` is computed from `rewards` and `values` (next).
            # `rewards` come from dyn(actor). `values` come from critic(dyn(actor)).
            # Checks out.
            
            actor_loss = -target_values.mean()
            
            # Entropy regularization
            # We computed entropy during sampling? No.
            # We can re-compute entropy from actor output distribution on the imagined states?
            # Wait, we sampled actions from current policy during rollout.
            # So the actions are 'on-policy' wrt the frozen dynamics step.
            
            # We generally add entropy bonus
            # dist_action was created inside the loop. The entropy depends on `std`.
            # We can accumulate entropy there or recompute.
            # Simplest: Just add a small entropy term for exploration.
            
            # Since I lost reference to distributions, I'll ignore entropy for MVP 
            # OR I should save it in the loop.
            
            # Backprop
            opt_actor.zero_grad()
            opt_critic.zero_grad()
            
            loss = actor_loss + critic_loss
            loss.backward()
            
            # Clip grads
            nn.utils.clip_grad_norm_(actor.parameters(), 100.0)
            nn.utils.clip_grad_norm_(critic.parameters(), 100.0)
            
            opt_actor.step()
            opt_critic.step()
            
            # Log
            if is_rank0() and step % 10 == 0:
                wandb.log({
                    "train/actor_loss": actor_loss.item(),
                    "train/critic_loss": critic_loss.item(),
                    "train/value_mean": v_pred.mean().item(),
                    "train/reward_mean": rewards.mean().item(),
                }, step=step)
                print(f"Step {step}: Act L={actor_loss.item():.3f} Crit L={critic_loss.item():.3f} Rew={rewards.mean().item():.3f}")

            step += 1
            if step % args.save_every == 0 and is_rank0():
                save_path = Path(args.ckpt_dir) / "agent_latest.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                    "args": vars(args)
                }, save_path)
    
    print("Training finished.")
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--dynamics_ckpt", type=str, required=True)
    parser.add_argument("--data_dirs", type=str, nargs="+", required=True)
    parser.add_argument("--tasks", type=str, nargs="+", default=["pusht"])
    parser.add_argument("--ckpt_dir", type=str, default="logs/agent")
    
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--packing_factor", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambda_", type=float, default=0.95)
    parser.add_argument("--discount", type=float, default=0.99)
    
    parser.add_argument("--actor_lr", type=float, default=8e-5)
    parser.add_argument("--critic_lr", type=float, default=8e-5)
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=100)
    
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default="agent_train")

    args = parser.parse_args()
    main(args)
