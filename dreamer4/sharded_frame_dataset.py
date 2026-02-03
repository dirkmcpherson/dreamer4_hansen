# sharded_frame_dataset.py
import os
import bisect
from pathlib import Path
from typing import Sequence, List, Dict, Union, Optional
import numpy as np

import torch
from torch.utils.data import Dataset


class ShardedFrameDataset(Dataset):
    """
    Samples contiguous sequences from preprocessed shards across multiple roots:

      root/<task>/*.pt  with {"frames": (N, 3, H, W) uint8}

    Returns: (T, 3, H, W) float32 in [0,1], where T = seq_len.

    If iid_sampling=True, ignores idx and samples a random starting position
    uniformly over all valid sequence starts across all shards.
    """

    def __init__(
        self,
        outdirs: Union[str, Sequence[str]],
        tasks: Sequence[str] = (),
        seq_len: int = 16,
        iid_sampling: bool = True,
        cache_size: int = 128,
    ):
        super().__init__()
        assert outdirs is not None, "outdirs must be specified"

        if isinstance(outdirs, (str, Path)):
            self.outdirs = [str(outdirs)]
        else:
            self.outdirs = [str(p) for p in outdirs]

        self.tasks = list(tasks)
        self.seq_len = int(seq_len)
        self.iid_sampling = bool(iid_sampling)
        
        self._cache = {}
        self._cache_keys = []
        self._cache_size = int(cache_size)

        self.shards: List[Dict] = []
        self.cum_starts: List[int] = []
        total_starts = 0

        for root in self.outdirs:
            root = Path(root)
            for task in self.tasks:
                task_dir = root / task
                if not task_dir.exists():
                    continue

                for fname in sorted(os.listdir(task_dir)):
                    if not fname.endswith(".pt"):
                        continue
                    path = task_dir / fname

                    try:
                        td = torch.load(path, map_location="cpu")
                    except Exception as e:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (load error): {e}")
                        continue

                    frames = td.get("frames", None)
                    if not isinstance(frames, torch.Tensor):
                        print(f"[ShardedFrameDataset] Skipping shard {path} (no 'frames' tensor)")
                        continue
                    if frames.ndim != 4 or frames.shape[1] != 3:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (unexpected shape {frames.shape})")
                        continue

                    N = int(frames.shape[0])
                    if N < self.seq_len:
                        print(f"[ShardedFrameDataset] Skipping shard {path} (N={N} < seq_len={self.seq_len})")
                        continue

                    num_starts = N - self.seq_len + 1
                    self.shards.append(
                        {"path": str(path), "num_frames": N, "num_starts": num_starts}
                    )
                    self.cum_starts.append(total_starts)
                    total_starts += num_starts

        self.total_starts = total_starts
        self.seq_starts = np.array(self.cum_starts + [total_starts])
        
        # We store shard_paths for fast access:
        self.shard_paths = [s["path"] for s in self.shards]

        if self.total_starts == 0:
            print("[ShardedFrameDataset] WARNING: no usable sequences found in outdirs")
        else:
            print(
                f"[ShardedFrameDataset] roots={len(self.outdirs)}, "
                f"shards={len(self.shards):,}, seq_starts={self.total_starts:,}"
            )

    def __len__(self) -> int:
        if self.iid_sampling:
            return self.total_starts
        else:
            raise NotImplementedError("Streaming length not defined")

    def __getitem__(self, idx: int):
        # iid sampling
        shard_idx = np.searchsorted(self.seq_starts, idx, side='right') - 1
        start_seq_idx = self.seq_starts[shard_idx]
        local_seq_idx = idx - start_seq_idx

        shard_path = self.shard_paths[shard_idx]
        frames = self._load_shard(shard_path)
        
        # frames is stored as uint8, convert to float32 [0,1]
        seq = frames[local_seq_idx : local_seq_idx + self.seq_len]
        return seq.to(torch.float32) / 255.0

    def _load_shard(self, path: str) -> torch.Tensor:
        if path in self._cache:
            # Move to end (LRU)
            val = self._cache.pop(path)
            self._cache[path] = val
            self._cache_keys.remove(path)
            self._cache_keys.append(path)
            return val
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            td = torch.load(path, map_location="cpu")
        frames = td["frames"]
        
        # Cache
        if len(self._cache) >= self._cache_size:
            # Evict LRU (first item)
            oldest = self._cache_keys.pop(0)
            del self._cache[oldest]

        self._cache[path] = frames
        self._cache_keys.append(path)
        return frames
