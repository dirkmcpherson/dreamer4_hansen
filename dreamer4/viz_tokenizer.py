# viz_tokenizer.py
"""
Standalone tokenizer reconstruction video: side-by-side  GT | decoded.

Loads a *frozen* tokenizer checkpoint (e.g. logs/bimanual/tok/latest.pt), samples a
few clips from the frame shards, encodes->decodes them (clean, no MAE masking), and
writes a video where each clip is shown as [GT | decoded] side by side. The time axis
is the video's frame axis.

Run it like the trainers (PYTHONPATH must include the dreamer4/ dir):

    python -m dreamer4.viz_tokenizer \
        --ckpt   logs/bimanual/tok/latest.pt \
        --data_dirs data/bimanual_pusht/train --tasks pusht \
        --seq_len 16 --num_videos 4 --out tokenizer_recon.mp4

Add --wandb to also log it to a wandb run.
"""
import os
import sys
import argparse

import torch

# Make sibling modules (model.py, sharded_frame_dataset.py, train_dynamics.py) importable
# whether or not PYTHONPATH is set -- matches how the trainers import each other.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import temporal_patchify, temporal_unpatchify
from sharded_frame_dataset import ShardedFrameDataset
# Reuse the exact tokenizer loader + video-grid builder the dynamics trainer uses,
# so "what viz_tokenizer shows" == "what train_dynamics decodes".
from train_dynamics import load_frozen_tokenizer_from_pt_ckpt, pair_video_grid


def save_video_file(grid_tchw: torch.Tensor, out_path: str, fps: int) -> str:
    """grid_tchw: (T,C,H,W) in [0,1]. Writes an mp4 (falls back to gif). Returns the path written."""
    arr = (grid_tchw.clamp(0, 1).permute(0, 2, 3, 1) * 255.0).to(torch.uint8).cpu().numpy()  # (T,H,W,C)
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio  # type: ignore
    try:
        imageio.mimwrite(out_path, arr, fps=fps, macro_block_size=None)
        return out_path
    except Exception as e:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        print(f"[viz] mp4 write failed ({type(e).__name__}: {e}); falling back to {gif_path} "
              f"(for mp4: pip install imageio-ffmpeg)")
        imageio.mimwrite(gif_path, arr, duration=1.0 / max(1, fps))
        return gif_path


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="tokenizer checkpoint (e.g. logs/bimanual/tok/latest.pt)")
    p.add_argument("--data_dirs", type=str, nargs="+", required=True, help="frame-shard root(s), e.g. data/bimanual_pusht/train")
    p.add_argument("--tasks", type=str, nargs="+", default=["pusht"])
    p.add_argument("--seq_len", type=int, default=16, help="frames per clip (video length)")
    p.add_argument("--num_videos", type=int, default=4, help="number of clips shown side by side")
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", type=str, default="tokenizer_recon.mp4")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb", action="store_true", help="also log the video to a wandb run")
    p.add_argument("--wandb_project", type=str, default="dreamer4-bimanual")
    p.add_argument("--wandb_run_name", type=str, default="tokenizer-viz")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Frozen tokenizer: this path sets MAE masking off + eval mode, so it's a clean recon.
    encoder, decoder, tok_args = load_frozen_tokenizer_from_pt_ckpt(args.ckpt, device=device)
    H = int(tok_args.get("H", 128)); W = int(tok_args.get("W", 128))
    C = int(tok_args.get("C", 3));   patch = int(tok_args.get("patch", 4))

    ds = ShardedFrameDataset(outdirs=args.data_dirs, tasks=args.tasks, seq_len=args.seq_len, iid_sampling=True)
    clips = [ds[i] for i in range(args.num_videos)]   # each (T,3,H,W) float in [0,1]
    frames = torch.stack(clips, dim=0).to(device)     # (B,T,3,H,W)

    patches = temporal_patchify(frames, patch)
    z, _ = encoder(patches)
    pred = decoder(z)
    recon = temporal_unpatchify(pred, H, W, C, patch).clamp(0, 1)  # (B,T,C,H,W)

    grid = pair_video_grid(frames, [recon], max_items=args.num_videos)  # (T,C,H,W)
    out = save_video_file(grid, args.out, args.fps)

    mse = (recon.float() - frames.float()).pow(2).mean()
    psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))
    print(f"[viz] wrote {out}  (each clip: left=GT | right=decoded)")
    print(f"[viz] recon MSE={mse.item():.6f}  PSNR={psnr.item():.2f} dB  over {args.num_videos} clips x {args.seq_len} frames")

    if args.wandb:
        import wandb
        from train_dynamics import log_video_grid
        wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                   mode=os.environ.get("WANDB_MODE", "online"), config=vars(args))
        log_video_grid(grid, step=0, tag="tokenizer/recon_video", fps=args.fps, caption="left=GT | right=decoded")
        wandb.log({"tokenizer/recon_mse": float(mse.item()), "tokenizer/recon_psnr": float(psnr.item())}, step=0)
        wandb.finish()


if __name__ == "__main__":
    main()
