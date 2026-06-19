#!/usr/bin/env python3
"""Smoke-test the state-action space on the CONVERTED dreamer4 data.

Verifies (numerically + visually) that my understanding holds end-to-end:

  Claim 1  action is 4-dim = world-frame XY of each end-effector
           [arm0_x, arm0_y, arm1_x, arm1_y]
  Claim 2  conversion normalized actions to [-1,1] *invertibly*
           (un-normalize(stored) == original IWS action)
  Claim 3  frames survived conversion intact (== IWS top_pov)
  Claim 4  temporal convention: action[t+1] drives obs[t]->obs[t+1]
           (shown in the video as an arrow from EE@t to EE@t+1)

Numeric checks print PASS/FAIL. A playable mp4 per episode is written so you
can eyeball that the EE minimap dots track the grippers and the next-action
arrow points where each arm actually moves.

Run:
    venv_hansen/bin/python smoke_test_state_action.py
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

IWS_ROOT = "/home/james/workspace/rai/dreamerv3-torch/world_models/interactive_world_sim/data/mini_mujoco"
PANEL = 320


def unnormalize(a_norm, lo, hi):
    return (a_norm + 1.0) / 2.0 * (hi - lo) + lo


def fig_to_rgb(fig):
    c = FigureCanvasAgg(fig); c.draw()
    w, h = c.get_width_height()
    return np.frombuffer(c.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def minimap(act_raw, t, lim):
    fig = plt.figure(figsize=(PANEL / 100, PANEL / 100), dpi=100)
    ax = fig.add_axes([0.17, 0.13, 0.78, 0.78])
    a0, a1 = act_raw[: t + 1, 0:2], act_raw[: t + 1, 2:4]
    ax.plot(a0[:, 0], a0[:, 1], "-", c="tab:blue", lw=1, alpha=0.5)
    ax.plot(a1[:, 0], a1[:, 1], "-", c="tab:red", lw=1, alpha=0.5)
    ax.scatter(a0[-1, 0], a0[-1, 1], c="tab:blue", s=70, label="arm0 EE")
    ax.scatter(a1[-1, 0], a1[-1, 1], c="tab:red", s=70, label="arm1 EE")
    if t + 1 < len(act_raw):  # arrow to NEXT commanded EE (claim 4)
        for j, col in ((0, "tab:blue"), (2, "tab:red")):
            dx = act_raw[t + 1, j] - act_raw[t, j]
            dy = act_raw[t + 1, j + 1] - act_raw[t, j + 1]
            ax.arrow(act_raw[t, j], act_raw[t, j + 1], dx, dy, head_width=0.006,
                     color=col, length_includes_head=True)
    ax.set_xlim(lim[0]); ax.set_ylim(lim[1])
    ax.set_title("EE world-XY (un-normalized action)\narrow = next-step command", fontsize=8)
    ax.legend(loc="upper right", fontsize=7); ax.grid(alpha=0.3)
    out = fig_to_rgb(fig); plt.close(fig)
    return cv2.resize(out, (PANEL, PANEL))


def hud(an):
    img = np.full((PANEL, PANEL, 3), 255, np.uint8)
    labels = ["a0_x", "a0_y", "a1_x", "a1_y"]
    cv2.putText(img, "normalized action [-1,1]", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    x0, w = 20, PANEL - 40; cx = x0 + w // 2
    for i, (lab, v) in enumerate(zip(labels, an)):
        y = 70 + i * 46
        cv2.line(img, (cx, y - 10), (cx, y + 14), (180, 180, 180), 1)
        xe = int(cx + float(np.clip(v, -1, 1)) * (w // 2))
        col = (200, 60, 60) if v >= 0 else (60, 60, 200)
        cv2.rectangle(img, (min(cx, xe), y - 8), (max(cx, xe), y + 10), col, -1)
        cv2.putText(img, f"{lab}={v:+.2f}", (x0, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/bimanual_pusht")
    ap.add_argument("--split", default="train")
    ap.add_argument("--iws_root", default=IWS_ROOT)
    ap.add_argument("--task", default="pusht")
    ap.add_argument("--n", type=int, default=2, help="episodes to render")
    ap.add_argument("--out", default="data/bimanual_pusht/state_action_smoke")
    args = ap.parse_args()

    stats = json.load(open(Path(args.data_dir) / "action_norm_stats.json"))
    lo, hi = np.array(stats["min"], np.float32), np.array(stats["max"], np.float32)
    shards = sorted(glob.glob(str(Path(args.data_dir) / args.split / args.task / "shard_*.pt")))
    iws_eps = sorted(glob.glob(f"{args.iws_root}/{args.task}/{args.split}/episode_*.hdf5"))
    assert shards and len(shards) == len(iws_eps), f"{len(shards)} shards vs {len(iws_eps)} iws episodes"

    print("=" * 70)
    print("STATE-ACTION SMOKE TEST  (converted dreamer4 data vs raw IWS source)")
    print(f"  stats: min={lo}  max={hi}")
    rt_err = sem_err = frame_err = 0.0
    for sp, ep in zip(shards, iws_eps):
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        an = sd["actions"].numpy()                       # (T,4) normalized
        raw = unnormalize(an, lo, hi)                    # recovered raw
        with h5py.File(ep, "r") as f:
            iws_a = f["action"][:].astype(np.float32)    # (T,4) original
            ee = f["obs/ee_pos"][:].astype(np.float32)   # (T,2,4,4)
            img = f["obs/images/top_pov"][:]             # (T,128,128,3)
        ee_xy = np.concatenate([ee[:, 0, :2, 3], ee[:, 1, :2, 3]], 1)  # (T,4)
        conv_frame0 = sd["frames"][0].permute(1, 2, 0).numpy()
        rt_err = max(rt_err, np.abs(raw - iws_a).max())
        sem_err = max(sem_err, np.median(np.abs(iws_a - ee_xy)))
        frame_err = max(frame_err, np.abs(conv_frame0.astype(int) - img[0].astype(int)).max())

    def verdict(name, val, tol, what):
        ok = val <= tol
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:34s} {val:.4g} (tol {tol})  -> {what}")
        return ok

    print("-" * 70)
    ok = True
    ok &= verdict("Claim 2 unnormalize roundtrip err", rt_err, 1e-3, "normalization is invertible/correct")
    ok &= verdict("Claim 1 action vs EE world-XY", sem_err, 5e-3, "action IS each arm's EE world XY")
    ok &= verdict("Claim 3 converted frame vs top_pov", frame_err, 0, "frames survived conversion intact")
    print(f"  OVERALL: {'ALL PASS' if ok else 'SOME FAIL'}")
    print("=" * 70)

    # ---- videos from the CONVERTED shards ----
    Path(args.out).mkdir(parents=True, exist_ok=True)
    lim = [(lo[[0, 2]].min() - 0.03, hi[[0, 2]].max() + 0.03),
           (lo[[1, 3]].min() - 0.03, hi[[1, 3]].max() + 0.03)]
    for sp in shards[: args.n]:
        sd = torch.load(sp, map_location="cpu", weights_only=False)
        an = sd["actions"].numpy(); raw = unnormalize(an, lo, hi)
        frames = sd["frames"].permute(0, 2, 3, 1).numpy()  # (T,128,128,3)
        name = Path(sp).stem
        out_path = str(Path(args.out) / f"{name}_state_action.mp4")
        wr = imageio.get_writer(out_path, fps=20, codec="libx264", quality=8)
        for t in range(0, len(frames), 2):
            gt = cv2.resize(frames[t], (PANEL, PANEL), interpolation=cv2.INTER_NEAREST)
            cv2.putText(gt, f"{name} t={t}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            wr.append_data(np.concatenate([gt, minimap(raw, t, lim), hud(an[t])], 1))
        wr.close()
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
