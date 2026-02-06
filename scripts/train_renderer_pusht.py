import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import glob
import argparse
import torchvision
from pathlib import Path

# --- Import Layout ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from dreamer4.model import Tokenizer, temporal_patchify, temporal_unpatchify
    from dreamer4.train_dynamics import load_frozen_tokenizer_from_pt_ckpt
except ImportError:
    sys.path.append(os.getcwd())
    from dreamer4.model import Tokenizer, temporal_patchify, temporal_unpatchify
    from dreamer4.train_dynamics import load_frozen_tokenizer_from_pt_ckpt

# --- Simple Dataset Class ---
class PushTShardedDataset(Dataset):
    def __init__(self, outdirs, seq_len=16):
        self.seq_len = seq_len
        self.outdirs = [outdirs] if isinstance(outdirs, str) else outdirs
        self.shards = []
        for d in self.outdirs:
            search_paths = [os.path.join(d, "pusht", "*.pt"), os.path.join(d, "*.pt")]
            for p in search_paths:
                found = sorted(glob.glob(p))
                if found: self.shards.extend(found); break
        
        self.index = []
        for shard_idx, p in enumerate(self.shards):
            try:
                data = torch.load(p, map_location="cpu")
                N = data["frames"].shape[0]
                if N > seq_len:
                    for i in range(0, N - seq_len, seq_len):
                        self.index.append((shard_idx, i))
            except: pass

    def __len__(self): return len(self.index)
    
    def __getitem__(self, idx):
        shard_idx, start = self.index[idx]
        data = torch.load(self.shards[shard_idx], map_location="cpu")
        end = start + self.seq_len
        frames = data["frames"][start:end].float() / 255.0
        return {"frames": frames}

# --- Helper to force 0% Masking ---
def set_mae_p0(model):
    for m in model.modules():
        if hasattr(m, 'p_min') and hasattr(m, 'p_max'):
            m.p_min = 0.0
            m.p_max = 0.0

# --- WEIGHT MASK GENERATOR (The Fix) ---
def make_weight_mask(frames, agent_weight=100.0):
    """
    Creates a pixel-wise weight map.
    PushT Agent is Blue. We detect blue-dominant pixels.
    """
    # Frames: (B, T, 3, H, W) in [0, 1]
    r = frames[:, :, 0, :, :]
    g = frames[:, :, 1, :, :]
    b = frames[:, :, 2, :, :]
    
    # Heuristic: Blue is significantly higher than Red and Green
    # The PushT agent is a distinct blue circle.
    is_blue = (b > 0.6) & (b > r * 1.2) & (b > g * 1.2)
    
    # Base weight = 1.0
    weights = torch.ones_like(r)
    
    # Agent weight = 100.0 (or whatever you set)
    weights[is_blue] = agent_weight
    
    # Expand back to (B, T, 3, H, W) so we can multiply with image
    weights = weights.unsqueeze(2).expand_as(frames)
    
    return weights, is_blue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, default="tok_latest.pt")
    parser.add_argument("--data_dir", type=str, default="../data/pusht_train")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="finetune_weighted")
    parser.add_argument("--agent_weight", type=float, default=10.0, help="How much more important is the blue dot?")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer from {args.tokenizer_ckpt}...")
    enc, dec, tok_args = load_frozen_tokenizer_from_pt_ckpt(args.tokenizer_ckpt, device=device)
    
    patch = int(tok_args.get("patch", 4))
    H_img = int(tok_args.get("H", 96))
    W_img = int(tok_args.get("W", 96))
    
    # 2. Unfreeze Everything (End-to-End)
    print("Unfreezing Encoder & Decoder for Weighted Fine-tuning...")
    enc.train(); enc.requires_grad_(True); set_mae_p0(enc)
    dec.train(); dec.requires_grad_(True)
    
    # 3. Optimizer
    # We use a smaller LR for encoder to respect pre-training
    params = [
        {"params": enc.parameters(), "lr": 1e-6}, 
        {"params": dec.parameters(), "lr": 5e-4}, 
    ]
    opt = optim.AdamW(params)

    # 4. Data
    dataset = PushTShardedDataset(args.data_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    iter_loader = iter(loader)

    print(f"Starting weighted training (Agent Weight = {args.agent_weight}x)...")
    
    for step in range(args.steps):
        try:
            batch = next(iter_loader)
        except StopIteration:
            iter_loader = iter(loader)
            batch = next(iter_loader)
            
        frames = batch['frames'].to(device) # (B, T, C, H, W)
        
        # --- Weighted Loss Calculation ---
        
        # 1. Get Latents & Reconstruction
        patches = temporal_patchify(frames, patch)
        z_dense, _ = enc(patches) 
        pred_patches = dec(z_dense)
        recon_imgs = temporal_unpatchify(pred_patches, H_img, W_img, 3, patch)
        
        # 2. Create Weight Mask (in Pixel Space)
        weights, is_agent = make_weight_mask(frames, agent_weight=args.agent_weight)
        
        # 3. Compute Weighted MSE
        # Note: We compute loss in Pixel Space to apply the mask easily
        diff = (recon_imgs - frames) ** 2
        weighted_diff = diff * weights
        loss = weighted_diff.mean()
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 100 == 0:
            # Check how many agent pixels we found to ensure mask logic is good
            agent_px_count = is_agent.sum().item() / frames.shape[0]
            print(f"Step {step}: Loss = {loss.item():.5f} | Avg Agent Pixels/Batch: {agent_px_count:.1f}")
            
            # --- Visualization ---
            with torch.no_grad():
                gt_vis = frames[0, :8]
                recon_vis = recon_imgs[0, :8].clamp(0, 1)
                
                # Visualize the Mask too (Blue channel only for visibility)
                mask_vis = weights[0, :8] / args.agent_weight # Normalize to 0-1 for vis
                
                display_batch = torch.stack([gt_vis, recon_vis, mask_vis], dim=1).reshape(24, 3, H_img, W_img)
                path = os.path.join(args.save_dir, f"step_{step:04d}.png")
                torchvision.utils.save_image(display_batch, path, nrow=3) 
                
    save_path = os.path.join(args.save_dir, "tok_finetuned_weighted.pt")
    # Save the whole tokenizer state (enc + dec)
    state = {
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "args": tok_args
    }
    torch.save(state, save_path)
    print(f"Done! Saved weighted tokenizer to {save_path}")

if __name__ == "__main__":
    main()