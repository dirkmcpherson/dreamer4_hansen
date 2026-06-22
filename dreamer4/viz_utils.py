
import os
import torch
from torchvision.utils import save_image
from dreamer4.model import temporal_patchify, temporal_unpatchify

def save_tokenizer_viz(
    x_btchw: torch.Tensor,          # (B,T,C,H,W) float in [0,1]
    pred_btnd: torch.Tensor,        # (B,T,Np,Dp) float in [0,1]
    mae_mask_btNp1: torch.Tensor,   # (B,T,Np,1) bool True=masked
    patch: int,
    step: int,
    save_dir: str,
    max_items: int = 4,
    max_T: int = 8,
    tag: str = "viz",
):
    B, T, C, H, W = x_btchw.shape
    Tv = min(T, max_T)
    Bv = min(B, max_items)

    # patchify target
    target_btnd = temporal_patchify(x_btchw[:, :Tv], patch)  # (B,Tv,Np,Dp)

    # panels (patch space)
    masked_input_btnd = torch.where(mae_mask_btNp1[:, :Tv], torch.zeros_like(target_btnd), target_btnd)
    recon_masked_btnd = torch.where(mae_mask_btNp1[:, :Tv], pred_btnd[:, :Tv], target_btnd)
    recon_full_btnd   = pred_btnd[:, :Tv]

    # to image space (B,T,C,H,W)
    target_img = temporal_unpatchify(target_btnd,       H, W, C, patch)
    masked_img = temporal_unpatchify(masked_input_btnd, H, W, C, patch)
    rmask_img  = temporal_unpatchify(recon_masked_btnd, H, W, C, patch)
    rfull_img  = temporal_unpatchify(recon_full_btnd,   H, W, C, patch)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        # (B,T,C,H,W) -> (B,C,H,T*W)
        x = x[:, :Tv]
        return x.permute(0, 2, 3, 1, 4).contiguous().view(x.shape[0], C, H, Tv * W)

    tgt = tile_time(target_img[:Bv])
    msk = tile_time(masked_img[:Bv])
    rm  = tile_time(rmask_img[:Bv])
    rf  = tile_time(rfull_img[:Bv])

    panel = torch.cat([tgt, msk, rm, rf], dim=2)  # (Bv,C,4H,Tv*W)
    big = torch.cat([panel[i] for i in range(Bv)], dim=1)  # (C,Bv*4H,Tv*W)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{tag}_step_{step:06d}.png")
    save_image(big, save_path)
    print(f"Saved visualization to {save_path}")

def save_dynamics_eval(
    gt: torch.Tensor,          # (B,T,C,H,W) float [0,1]
    pred: torch.Tensor,        # (B,T,C,H,W) float [0,1]
    ctx_length: int,
    step: int,
    save_dir: str,
    tag: str = "eval",
    max_items: int = 4,
    gap_px: int = 16,
):
    B, T, C, H, W = gt.shape
    Bv = min(B, max_items)

    def tile_time(x: torch.Tensor) -> torch.Tensor:
        x = x[:Bv]
        B_, T_, C_, H_, W_ = x.shape
        ctx = int(max(0, min(ctx_length, T_)))

        y = x.permute(0, 2, 3, 1, 4).contiguous().view(B_, C_, H_, T_ * W_)

        if gap_px > 0 and 0 < ctx < T_:
            split = ctx * W_
            left = y[..., :split]
            right = y[..., split:]
            gap = torch.zeros((B_, C_, H_, gap_px), device=y.device, dtype=y.dtype)
            y = torch.cat([left, gap, right], dim=-1)
        return y

    gt_t = tile_time(gt)
    pr_t = tile_time(pred)
    
    # stack vertically: GT above Pred
    combined = torch.cat([gt_t, pr_t], dim=2) # (Bv, C, 2H, T*W + gap)
    
    # stack batches vertically
    big = torch.cat([combined[i] for i in range(Bv)], dim=1) # (C, Bv*2H, Width)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{tag}_step_{step:06d}.png")
    save_image(big, save_path)
    print(f"Saved visualization to {save_path}")
