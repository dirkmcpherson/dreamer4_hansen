#!/usr/bin/env bash
# End-to-end bimanual-pushT pipeline for dreamer4_hansen on a single big GPU.
#
# Stages (run one, several, or "all"):
#   download  -> fetch the IWS MuJoCo dataset from HuggingFace (~120 GB)
#   convert   -> HDF5 -> dreamer4 shards + demo files (~149 GB)
#   tok       -> train tokenizer
#   dyn       -> train dynamics / world model (needs tokenizer latest.pt)
#
# Usage:
#   ./run_bimanual.sh all
#   ./run_bimanual.sh convert tok
#   DATA=/scratch/iws_mujoco OUT=/scratch/bimanual ./run_bimanual.sh tok dyn
set -euo pipefail
cd "$(dirname "$0")"

# ---- config (override via env) ----
export REPO="${REPO:-$(pwd)}"
export DATA="${DATA:-$REPO/data/iws_mujoco}"          # raw HDF5 download
export OUT="${OUT:-$REPO/data/bimanual_pusht}"        # converted shards
export PYTHONPATH="$REPO/dreamer4${PYTHONPATH:+:$PYTHONPATH}"   # sibling `import task_set`
export WANDB_MODE="${WANDB_MODE:-online}"             # set to "offline" if no network
# 128^2 tokenizer uses MASKED space-attention over 1024 patches, which forces
# SDPA off the flash path and materializes the full NxN scores -> memory scales
# hard with batch size. Keep batches modest even on a 140GB H200.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PY:-python}"
TOK_DIR="${TOK_DIR:-$REPO/logs/bimanual/tok}"
DYN_DIR="${DYN_DIR:-$REPO/logs/bimanual/dyn}"

# Conservative defaults that fit ~140GB at 128^2; raise gradually watching nvidia-smi.
TOK_BS="${TOK_BS:-16}"
DYN_BS="${DYN_BS:-8}"

# Gradient accumulation for the tokenizer. At seq_len 32 the batch must shrink (16->4)
# for memory, which starved the optimizer (effective batch ~4 highly-correlated clips)
# and collapsed the tanh bottleneck (z saturated at +/-1, z_std->1, garbage recon).
# TOK_GRAD_ACCUM restores the effective batch WITHOUT extra memory: BS=4 x accum=4 == 16,
# matching the healthy seq_len=8 run. NOTE: --max_steps counts micro-batch steps, so an
# accum of 4 yields 1/4 as many optimizer updates -- bump TOK_STEPS by the same factor
# (e.g. TOK_GRAD_ACCUM=4 TOK_STEPS=400000) to match the seq8 update count.
TOK_GRAD_ACCUM="${TOK_GRAD_ACCUM:-1}"

# Tokenizer training sequence length. Upstream uses 8 and relies on the causal tokenizer
# generalizing to the longer dynamics length (32). If YOUR tokenizer doesn't extrapolate
# (reconstructions break down past frame 8), train it at the dynamics length instead:
#   TOK_SEQ=32 TOK_BS=4 TOK_DIR=$(pwd)/logs/bimanual/tok32 ./run_bimanual.sh tok
# Tokenizer memory scales with batch*seq_len, so seq_len 32 needs ~4x smaller batch (16 -> 4).
TOK_SEQ="${TOK_SEQ:-8}"

# Optional resume: RESUME=/abs/path/to/latest.pt adds --resume to whichever single stage
# you run (handy on flaky hardware). Leave empty to start fresh.
RESUME="${RESUME:-}"
RESUME_FLAG=""; [ -n "$RESUME" ] && RESUME_FLAG="--resume $RESUME"

# Tokenizer checkpoint the dynamics stage builds on. Defaults to the tok stage's latest.pt;
# override to pin a specific one, e.g. a freshly retrained seq_len=32 tokenizer:
#   TOK_CKPT=$(pwd)/logs/bimanual/tok32/latest.pt ./run_bimanual.sh dyn
TOK_CKPT="${TOK_CKPT:-$TOK_DIR/latest.pt}"

# Optional: gradient checkpointing trades ~25-33% compute/step for a large drop in
# activation memory (exact same math + dropout). Off by default; flip on to run a
# much bigger batch, e.g.  GRAD_CKPT=1 TOK_BS=64 ./run_bimanual.sh tok
GRAD_CKPT="${GRAD_CKPT:-0}"
GC_FLAG=""; [ "$GRAD_CKPT" = "1" ] && GC_FLAG="--grad_checkpoint"

# Guard the tanh bottleneck against saturation collapse (z pinned at +/-1, z_std->1) by
# RMSNorm-ing the residual stream before it. The structural fix (caps the pre-activation so
# tanh can't reach its flat tails), independent of batch size. Off by default to keep older
# (seq8) checkpoints loadable.  TOK_BOTTLENECK_NORM=1 ./run_bimanual.sh tok
TOK_BOTTLENECK_NORM="${TOK_BOTTLENECK_NORM:-0}"
BN_FLAG=""; [ "$TOK_BOTTLENECK_NORM" = "1" ] && BN_FLAG="--bottleneck_norm"
TOK_STEPS="${TOK_STEPS:-100000}"
DYN_STEPS="${DYN_STEPS:-300000}"

stages=("$@"); [ ${#stages[@]} -eq 0 ] && stages=("all")
has() { for s in "${stages[@]}"; do [ "$s" = "$1" ] || [ "$s" = "all" ] && return 0; done; return 1; }
gate() { [ -e "$1" ] || { echo "GATE FAILED: missing $1"; exit 1; }; }

echo "REPO=$REPO  DATA=$DATA  OUT=$OUT  WANDB_MODE=$WANDB_MODE"

if has download; then
  echo "=== [download] IWS MuJoCo dataset -> $DATA ==="
  $PY download_bimanual_data.py --local_dir "$DATA"
  gate "$DATA/train"; gate "$DATA/val"
fi

if has convert; then
  echo "=== [convert] $DATA -> $OUT ==="
  $PY iws_to_dreamer4.py --iws_root "$DATA" --out_dir "$OUT"
  gate "$OUT/train/pusht.pt"; gate "$OUT/action_norm_stats.json"
fi

if has tok; then
  echo "=== [tok] training tokenizer -> $TOK_DIR ==="
  gate "$OUT/train/pusht.pt"
  $PY -m dreamer4.train_tokenizer \
    --data_dirs "$OUT/train" --tasks pusht \
    --H 128 --W 128 --patch 4 --seq_len "$TOK_SEQ" \
    --batch_size "$TOK_BS" --grad_accum "$TOK_GRAD_ACCUM" --num_workers 8 $GC_FLAG $BN_FLAG \
    --max_steps "$TOK_STEPS" --save_every 5000 --log_every 100 \
    --lpips_weight 0.2 \
    --ckpt_dir "$TOK_DIR" $RESUME_FLAG \
    --wandb_project dreamer4-bimanual --wandb_run_name tokenizer
  gate "$TOK_DIR/latest.pt"
fi

if has dyn; then
  echo "=== [dyn] training dynamics -> $DYN_DIR ==="
  gate "$TOK_CKPT"
  $PY -m dreamer4.train_dynamics --use_actions \
    --data_dirs "$OUT/train" --frame_dirs "$OUT/train" \
    --tasks pusht --tasks_json "$REPO/tasks.json" \
    --tokenizer_ckpt "$TOK_CKPT" \
    --batch_size "$DYN_BS" --num_workers 8 --seq_len 32 $GC_FLAG \
    --max_steps "$DYN_STEPS" --save_every 10000 \
    --ckpt_dir "$DYN_DIR" $RESUME_FLAG \
    --wandb_project dreamer4-bimanual --wandb_run_name dynamics
  gate "$DYN_DIR/latest.pt"
fi

echo "=== done: ${stages[*]} ==="
