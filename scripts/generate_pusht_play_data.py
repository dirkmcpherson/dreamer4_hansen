
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
import gym_pusht
from pathlib import Path

def generate_play_data(args):
    """
    Generates random play data from the PushT environment.
    """
    out_dir = Path(args.out_dir) / "pusht_play" / "pusht"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_episodes} episodes of play data to {out_dir}...")
    
    try:
        env = gym.make("gym_pusht/PushT-v0", obs_type="pixels", render_mode="rgb_array")
    except Exception as e:
        print(f"Error creating env: {e}")
        return

    for i in range(args.num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        
        frames = []
        actions = []
        rewards = []
        is_terminal = []
        is_first = []
        
        step = 0
        while not (done or truncated) and step < args.max_steps:
            # Random Action [0, 512]
            action = env.action_space.sample()
            
            # Store 'is_first' (True for step 0)
            is_first.append(step == 0)
            
            # Step
            next_obs, reward, done, truncated, _ = env.step(action)
            
            # Store Frame (obs is H,W,3 uint8?)
            # gym_pusht returns (H,W,C) uint8 for pixels
            frames.append(obs)
            actions.append(action)
            rewards.append(reward)
            is_terminal.append(done) # Truncated is not terminal for bootstrapping usually?
            
            obs = next_obs
            step += 1
            
        # Add final observation frame?
        # Dreamer dataset usually expects T frames, T actions.
        # But `generate_pusht_dataset.py` logic:
        # npz has T images, T actions.
        # Let's align with that. 
        # But we missed the final frame if we stop at done.
        # Actually `obs` becomes `next_obs` at end of loop.
        # We did not store the final `next_obs`.
        # Is that okay?
        # `generate_pusht_dataset.py` loads whatever is in npz.
        # Usually demos include the final state.
        # Let's just store what we have. T steps.
        
        # Convert to Tensor
        # Frames: (T, H, W, 3) -> (T, 3, H, W)
        frames_np = np.stack(frames) # (T, H, W, 3)
        frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0
        
        # Resize to 64x64
        import torchvision.transforms.functional as TF
        if frames_tensor.shape[-2:] != (64, 64):
             frames_tensor = TF.resize(frames_tensor, [64, 64], antialias=True)
             
        # Convert back to uint8 [0, 255] for efficiency?
        # The ShardedFrameDataset expects uint8?
        # Let's check sharded_frame_dataset.py.
        # Line 16: {"frames": (N, 3, H, W) uint8}
        # Line 121: return seq.to(torch.float32) / 255.0
        # So yes, it expects uint8.
        
        frames_tensor = (frames_tensor * 255.0).clamp(0, 255).to(torch.uint8)
        
        actions_tensor = torch.from_numpy(np.stack(actions)).float()
        rewards_tensor = torch.from_numpy(np.stack(rewards)).float()
        is_terminal_tensor = torch.from_numpy(np.stack(is_terminal))
        is_first_tensor = torch.from_numpy(np.stack(is_first))
        
        shard_path = out_dir / f"play_{i:04d}.pt"
        torch.save({
            "frames": frames_tensor,
            "actions": actions_tensor,
            "rewards": rewards_tensor,
            "is_terminal": is_terminal_tensor,
            "is_first": is_first_tensor
        }, shard_path)
        
        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{args.num_episodes}")

    print("Generation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=60)
    args = parser.parse_args()
    generate_play_data(args)
