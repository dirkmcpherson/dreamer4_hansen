#!/usr/bin/env python3
"""Download the IWS bimanual-pushT MuJoCo dataset from Hugging Face.

The dataset (~120 GB, 10,100 HDF5 episodes) is Xet-backed; fetching 10k files
trips HF's 1000-requests/5-min quota. We disable Xet (classic CDN path) and
retry with backoff on 429. snapshot_download is resumable -- completed files
are skipped each round, so re-running continues where it left off.

    pip install huggingface_hub
    python download_bimanual_data.py --local_dir /data/iws_mujoco

Then convert into dreamer4 format:
    python iws_to_dreamer4.py --iws_root /data/iws_mujoco --out_dir data/bimanual_pusht
"""
import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # avoid per-file xet-token 429s

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

REPO = "yixuan1999/interactive-world-sim-mujoco-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local_dir", required=True, help="where to download (~120 GB)")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max_rounds", type=int, default=60)
    ap.add_argument("--backoff", type=int, default=320, help="sleep on 429 (sec); > HF's 5-min window")
    args = ap.parse_args()

    for rnd in range(1, args.max_rounds + 1):
        try:
            print(f"[round {rnd}] snapshot_download -> {args.local_dir} (xet disabled)", flush=True)
            path = snapshot_download(
                repo_id=args.repo,
                repo_type="dataset",
                local_dir=args.local_dir,
                max_workers=args.workers,
            )
            print(f"\nDONE: {path}", flush=True)
            return 0
        except HfHubHTTPError as e:
            code = getattr(getattr(e, "response", None), "status_code", None)
            if code == 429:
                print(f"[round {rnd}] 429 rate-limited; sleeping {args.backoff}s then resuming", flush=True)
                time.sleep(args.backoff)
                continue
            raise
        except Exception as e:  # transient network etc. -- resume
            print(f"[round {rnd}] {e!r}; sleeping 60s then resuming", flush=True)
            time.sleep(60)
    print("Exhausted retry rounds without completing.", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
