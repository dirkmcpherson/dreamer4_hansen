#!/usr/bin/env python3
"""Evaluate a trained bimanual-pushT world model: open-loop rollout vs ground truth.

Loads the frozen tokenizer + trained dynamics, takes a converted val episode,
gives the model `ctx_length` real frames, then rolls the WM forward `horizon`
steps under the *recorded* actions (open loop). Writes a side-by-side
[GT | WM prediction] mp4 and prints PSNR vs the static-frame floor.

    PYTHONPATH=dreamer4 python eval_bimanual_wm.py \
        --tokenizer_ckpt logs/bimanual/tok/latest.pt \
        --dynamics_ckpt  logs/bimanual/dyn/latest.pt \
        --shard data/bimanual_pusht/val/pusht/shard_0000.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio

from dreamer4.model import Dynamics, temporal_patchify, pack_bottleneck_to_spatial
from dreamer4.train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt,
    make_tau_schedule,
    sample_autoregressive_packed_sequence,
    decode_packed_to_frames,
)

ACTION_DIM_REAL = 4   # bimanual pushT (padded to the model's 16-dim encoder)


def load_dynamics(ckpt_path, *, d_bottleneck, n_latents, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    a = ckpt["args"]
    a = vars(a) if hasattr(a, "__dict__") else a
    pf = int(a.get("packing_factor", 1))
    n_spatial = n_latents // pf
    d_spatial = d_bottleneck * pf
    dyn = Dynamics(
        d_model=int(a.get("d_model_dyn", a.get("d_model", 512))),
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(a.get("n_register", 0)),
        n_agent=int(a.get("n_agent", 1)),
        n_heads=int(a.get("n_heads", 4)),
        depth=int(a.get("dyn_depth", a.get("depth", 8))),
        k_max=int(a.get("k_max", 16)),
        mlp_ratio=float(a.get("mlp_ratio", 4.0)),
        time_every=int(a.get("time_every", 4)),
        space_mode=str(a.get("space_mode", "wm_agent_isolated")),
    ).to(device)
    sd = {k.replace("module.", ""): v for k, v in ckpt["dynamics"].items()
          if not k.startswith("reward_head")}
    dyn.load_state_dict(sd, strict=True)
    dyn.eval()
    return dyn, pf, int(a.get("k_max", 16))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer_ckpt", required=True)
    ap.add_argument("--dynamics_ckpt", required=True)
    ap.add_argument("--shard", required=True, help="a converted val shard_*.pt")
    ap.add_argument("--ctx_length", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--out", default="eval_bimanual.mp4")
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder, decoder, tok = load_frozen_tokenizer_from_pt_ckpt(args.tokenizer_ckpt, device=device)
    H, W, C, patch = tok["H"], tok["W"], tok["C"], tok["patch"]
    d_bottleneck, n_latents = tok["d_bottleneck"], tok["n_latents"]
    dyn, pf, k_max = load_dynamics(args.dynamics_ckpt, d_bottleneck=d_bottleneck,
                                   n_latents=n_latents, device=device)

    sd = torch.load(args.shard, map_location="cpu", weights_only=False)
    T_need = args.ctx_length + args.horizon
    frames = sd["frames"][:T_need].float().div(255.0).unsqueeze(0).to(device)   # (1,T,C,H,W)
    act4 = sd["actions"][:T_need].to(device)                                    # (1?,T,4)
    T = frames.shape[1]
    # pad 4-dim action -> model's 16-dim encoder, with a [1,1,1,1,0..] mask
    A = dyn.action_encoder.action_dim
    actions = torch.zeros(1, T, A, device=device)
    actions[0, :, :ACTION_DIM_REAL] = act4[:T, :ACTION_DIM_REAL]
    act_mask = torch.zeros(A, device=device); act_mask[:ACTION_DIM_REAL] = 1.0

    sched = make_tau_schedule(k_max=k_max, schedule="shortcut", d=0.5)

    patches = temporal_patchify(frames, patch)
    z_btLd, _ = encoder(patches)
    z_gt_packed = pack_bottleneck_to_spatial(z_btLd, n_spatial=n_latents // pf, k=pf)

    z_pred = sample_autoregressive_packed_sequence(
        dyn, z_gt_packed=z_gt_packed, ctx_length=args.ctx_length, horizon=args.horizon,
        k_max=k_max, sched=sched, actions=actions, act_mask=act_mask,
    )
    pred = decode_packed_to_frames(decoder, z_packed=z_pred, H=H, W=W, C=C, patch=patch, packing_factor=pf)

    # PSNR over the imagined horizon vs static-frame floor
    c, h = args.ctx_length, args.horizon
    gt_h, pred_h = frames[:, c:c + h], pred[:, c:c + h]
    floor_h = frames[:, c - 1:c].expand(-1, h, -1, -1, -1)
    mse_pred = (pred_h - gt_h).pow(2).mean()
    mse_floor = (floor_h - gt_h).pow(2).mean()
    psnr = lambda m: 10.0 * torch.log10(1.0 / m.clamp_min(1e-12))
    print(f"horizon={h}  PSNR_pred={psnr(mse_pred):.2f}dB  PSNR_floor={psnr(mse_floor):.2f}dB  "
          f"mse_ratio={(mse_pred / mse_floor.clamp_min(1e-12)):.3f} (<1 = beats floor)")

    # side-by-side video [GT | WM]
    def to_u8(x):  # (C,H,W) -> (H,W,C)
        return (x.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
    wr = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for t in range(pred.shape[1]):
        tag = "ctx" if t < args.ctx_length else "imagined"
        gt, pr = to_u8(frames[0, t]), to_u8(pred[0, t])
        sep = np.full((gt.shape[0], 4, 3), 255, np.uint8)
        wr.append_data(np.concatenate([gt, sep, pr], axis=1))
    wr.close()
    print(f"wrote {args.out}  (left=GT, right=WM; first {args.ctx_length} frames are context)")


if __name__ == "__main__":
    main()
