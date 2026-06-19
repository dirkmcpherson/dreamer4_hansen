#!/usr/bin/env python3
"""Convert IWS bimanual-pushT MuJoCo HDF5 episodes -> dreamer4_hansen format.

Source (per episode .hdf5):
    obs/images/top_pov : (T,128,128,3) uint8   -> frames (T,3,H,W) uint8
    action             : (T,4) float32         -> normalized to [-1,1] (global)
    (no reward in the sim pushT data)          -> zeros

Target layout (consumed by ShardedFrameDataset + WMDataset):
    <out>/<split>/<task>/shard_XXXX.pt = {frames, actions, rewards, is_first, is_terminal}
    <out>/<split>/<task>.pt            = {episode, action, reward}   (consolidated demo)

Action normalization is a single global range map (limits -> [-1,1]) fit across
ALL episodes found, matching IWS's range normalizer. Stats are written to
<out>/action_norm_stats.json so the same mapping can be reused on the full
dataset when training for real on a bigger GPU.
"""
import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F

DEF_IWS = "/home/james/workspace/rai/dreamerv3-torch/world_models/interactive_world_sim/data/mini_mujoco"


def find_episodes(iws_root: str, task: str):
    """Return {split: [paths...]}.

    Handles both layouts:
      - mini_mujoco:  <iws_root>/<task>/{train,val}/episode_*.hdf5
      - full HF data: <iws_root>/{train,val}/episode_*.hdf5
    """
    out = {}
    for split in ("train", "val"):
        for d in (Path(iws_root) / task / split, Path(iws_root) / split):
            eps = sorted(glob.glob(str(d / "episode_*.hdf5")))
            if eps:
                out[split] = eps
                break
    return out


def read_action(path: str):
    with h5py.File(path, "r") as f:
        return f["action"][:].astype(np.float32)  # (T,4); cheap, no images


def read_episode(path: str):
    with h5py.File(path, "r") as f:
        img = f["obs/images/top_pov"][:]          # (T,128,128,3) uint8
        action = f["action"][:].astype(np.float32)  # (T,4)
    return img, action


def global_action_stats(splits):
    """Single cheap pass reading ONLY the (tiny) action arrays, not images."""
    lo = hi = None
    n = 0
    for paths in splits.values():
        for p in paths:
            a = read_action(p)
            amin, amax = a.min(0), a.max(0)
            lo = amin if lo is None else np.minimum(lo, amin)
            hi = amax if hi is None else np.maximum(hi, amax)
            n += 1
            if n % 1000 == 0:
                print(f"  ...action stats: {n} episodes scanned")
    return lo, hi


def to_frames(img_thwc: np.ndarray, size: int) -> torch.Tensor:
    """(T,H,W,3) uint8 -> (T,3,size,size) uint8."""
    t = torch.from_numpy(img_thwc).permute(0, 3, 1, 2).contiguous()  # (T,3,H,W) uint8
    if t.shape[-1] != size or t.shape[-2] != size:
        t = F.interpolate(t.float(), size=(size, size), mode="bilinear", align_corners=False)
        t = t.round().clamp(0, 255).to(torch.uint8)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iws_root", default=DEF_IWS, help="dir containing <task>/{train,val}/episode_*.hdf5")
    ap.add_argument("--out_dir", default="data/bimanual_pusht")
    ap.add_argument("--task", default="pusht")
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--eps", type=float, default=1e-4, help="range floor to avoid div-by-zero")
    args = ap.parse_args()

    splits = find_episodes(args.iws_root, args.task)
    assert splits, f"no episodes found under {args.iws_root}/{args.task}/(train|val)"
    print(f"Found splits: { {k: len(v) for k, v in splits.items()} }")

    lo, hi = global_action_stats(splits)
    rng = np.maximum(hi - lo, args.eps)
    print(f"Global action min: {lo}")
    print(f"Global action max: {hi}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    with open(out_root / "action_norm_stats.json", "w") as f:
        json.dump({"min": lo.tolist(), "max": hi.tolist(), "mode": "limits[-1,1]",
                   "action_dim": int(lo.shape[0])}, f, indent=2)

    def normalize(a):  # limits -> [-1,1]
        return ((a - lo) / rng) * 2.0 - 1.0

    for split, paths in splits.items():
        task_dir = out_root / split / args.task
        task_dir.mkdir(parents=True, exist_ok=True)
        episodes, actions_all, rewards_all = [], [], []
        amin_seen, amax_seen = 1e9, -1e9
        for i, p in enumerate(paths):
            img, action = read_episode(p)
            T = action.shape[0]
            frames = to_frames(img, args.size)                       # (T,3,S,S) uint8
            actions = torch.from_numpy(normalize(action)).float()    # (T,4) in [-1,1]
            rewards = torch.zeros(T, dtype=torch.float32)
            is_first = torch.zeros(T, dtype=torch.bool); is_first[0] = True
            is_terminal = torch.zeros(T, dtype=torch.bool); is_terminal[-1] = True

            torch.save(
                {"frames": frames, "actions": actions, "rewards": rewards,
                 "is_first": is_first, "is_terminal": is_terminal},
                task_dir / f"shard_{i:04d}.pt",
            )
            episodes.append(torch.full((T,), i + 1, dtype=torch.long))
            actions_all.append(actions)
            rewards_all.append(rewards)
            amin_seen = min(amin_seen, float(actions.min()))
            amax_seen = max(amax_seen, float(actions.max()))
            if (i + 1) % 500 == 0:
                print(f"  [{split}] converted {i + 1}/{len(paths)} episodes")

        # consolidated demo file for WMDataset
        torch.save(
            {"episode": torch.cat(episodes), "action": torch.cat(actions_all),
             "reward": torch.cat(rewards_all)},
            out_root / split / f"{args.task}.pt",
        )
        nfr = sum(len(e) for e in episodes)
        print(f"[{split}] {len(paths)} episodes, {nfr} frames -> {task_dir}  "
              f"(norm action range [{amin_seen:.3f}, {amax_seen:.3f}])")

    print("Done.")


if __name__ == "__main__":
    main()
