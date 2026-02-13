
import sys
import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from collections import Counter
from pathlib import Path

try:
    import lpips
except ImportError:
    lpips = None

# Add parent directory to path to pick up dreamer4 package
# Assuming this script is in dreamer4_hansen/scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from dreamer4.sharded_frame_dataset import ShardedFrameDataset
    from dreamer4.model import Encoder, Decoder, Tokenizer, recon_loss_from_mae, temporal_patchify, temporal_unpatchify, lpips_on_mae_recon
    from dreamer4.train_tokenizer import log_tokenizer_viz_wandb
    from dreamer4.viz_utils import save_tokenizer_viz
except ImportError:
    # Try alternate relative path if package structure is different
    sys.path.append(os.getcwd())
    from dreamer4.sharded_frame_dataset import ShardedFrameDataset
    from dreamer4.model import Encoder, Decoder, Tokenizer, recon_loss_from_mae, temporal_patchify, temporal_unpatchify, lpips_on_mae_recon
    from dreamer4.train_tokenizer import log_tokenizer_viz_wandb
    from dreamer4.viz_utils import save_tokenizer_viz


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    # Path handling to adapt to running from scripts/ or root
    script_dir = os.path.dirname(__file__)
    default_pusht_train = os.path.join(script_dir, "./data/pusht_train")
    default_pusht_play = os.path.join(script_dir, "./data/pusht_play")
    parser.add_argument("--data_dirs", type=str, nargs="+", default=[default_pusht_train, default_pusht_play])
    parser.add_argument("--max_steps", type=int, default=200001)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--ckpt_dir", type=str, default=".")
    
    # Model args
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_bottleneck", type=int, default=128)
    parser.add_argument("--patch", type=int, default=4)
    parser.add_argument("--n_latents", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--enc_depth", type=int, default=4)
    parser.add_argument("--dec_depth", type=int, default=4)
    parser.add_argument("--H", type=int, default=64)
    parser.add_argument("--W", type=int, default=64)

    
    # Training args
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--clip_grad_norm", type=float, default=100.0)
    
    # LPIPS
    parser.add_argument("--lpips_weight", type=float, default=0.2)
    parser.add_argument("--lpips_frac", type=float, default=0.5)
    parser.add_argument("--lpips_net", type=str, default="alex", choices=["alex", "vgg", "squeeze"])

    # Monitor args
    parser.add_argument("--monitor_activations", action="store_true")
    parser.add_argument("--monitor_every", type=int, default=2500)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--print_every", type=int, default=100)
    parser.add_argument("--viz_every", type=int, default=2500)
    parser.add_argument("--viz_max_items", type=int, default=4)
    parser.add_argument("--viz_max_T", type=int, default=8)

    # Wandb
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    # Create ckpt dir
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_path = os.path.join(args.ckpt_dir, "tok_latest.pt")
    if os.path.exists(save_path) and not args.force_retrain:
        print(f"Found existing checkpoint at {save_path}. Skipping training (use --force_retrain to override).")
        print("Loading weights for visualization...")
        
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        # Use training mode to match training viz exact behavior
        model.train()
        
        # Disable MAE masking to test if p=0 is the cause of quality drop
        print("Disabling MAE masking (p=0)...")
        if hasattr(model.encoder, "mae"):
            model.encoder.mae.p_min = 0.0
            model.encoder.mae.p_max = 0.0
        
        print("Generating visualization batch (model.train() + p=0)...")

    
    seed_everything(random.randint(0, 10000))

    if (args.monitor_activations or args.wandb_project) and not args.no_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project or "dreamer4-hansen-test",
            name=args.wandb_run_name or "tokenizer_overfit",
            config=vars(args),
            # mode="offline"
        )

    print(f"Initializing ShardedFrameDataset from {args.data_dirs}...")
    
    # Assuming ShardedFrameDataset takes a list of directories
    # Check if directories exist
    valid_dirs = [d for d in args.data_dirs if os.path.exists(d)]
    if not valid_dirs:
        # Fallback to try relative to current working directory
        print(f"Warning: specified data dirs {args.data_dirs} not found.")
        print(f"Current working dir: {os.getcwd()}")
        # We will try to proceed but likely fail if no data
    
    # 1. Gather all shard paths first
    all_shard_paths = []
    for d in valid_dirs if valid_dirs else args.data_dirs:
        root = Path(d)
        for task in ["pusht"]:
            task_dir = root / task
            if not task_dir.exists():
                continue
            for fname in sorted(os.listdir(task_dir)):
                if fname.endswith(".pt"):
                    all_shard_paths.append(str(task_dir / fname))
    
    all_shard_paths = sorted(all_shard_paths)
    print(f"Found {len(all_shard_paths)} total shards.")

    # 2. Split into train and val (hold back last 10)
    val_count = 10
    if len(all_shard_paths) <= val_count:
        print("Warning: not enough shards to hold back 10 for validation. Using all for training.")
        train_paths = all_shard_paths
        val_paths = []
    else:
        train_paths = all_shard_paths[:-val_count]
        val_paths = all_shard_paths[-val_count:]
    
    print(f"Train shards: {len(train_paths)}")
    print(f"Val shards:   {len(val_paths)}")

    # 3. Create datasets
    train_dataset = ShardedFrameDataset(
        outdirs=valid_dirs if valid_dirs else args.data_dirs, # ignored if shard_paths present
        tasks=["pusht"],
        seq_len=8,
        iid_sampling=True,
        shard_paths=train_paths
    )
    
    val_dataset = None
    val_dataloader = None
    if val_paths:
        val_dataset = ShardedFrameDataset(
            outdirs=valid_dirs if valid_dirs else args.data_dirs,
            tasks=["pusht"],
            seq_len=8,
            iid_sampling=True,
            shard_paths=val_paths
        )
        val_dataloader = DataLoader(
             val_dataset,
             batch_size=args.batch_size,
             shuffle=True, # Random sampling for validation too
             num_workers=args.num_workers,
             pin_memory=True,
             drop_last=True
        )

    # Log data source breakdown
    source_counts = Counter()
    total_frames = 0
    print("-" * 40)
    print("Data Source Breakdown (Train):")
    for shard in train_dataset.shards:
        path = Path(shard['path'])
        source = path.parent.parent.name
        source_counts[source] += shard['num_frames']
        total_frames += shard['num_frames']
    
    for source, count in source_counts.items():
        percent = (count / total_frames) * 100 if total_frames > 0 else 0
        print(f"  {source:<15}: {count:10,} frames ({percent:5.1f}%)")
    print(f"  {'Total':<15}: {total_frames:10,} frames")
    print("-" * 40)
    
    dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # Model Setup
    H, W, C = args.H, args.W, 3
    patch = args.patch
    n_patches = (H // patch) * (W // patch)
    d_patch = patch * patch * C
    
    print("Initializing Tokenizer...")
    enc = Encoder(
        patch_dim=d_patch,
        d_model=args.d_model,
        n_latents=args.n_latents,
        n_patches=n_patches,
        n_heads=args.n_heads,
        depth=args.enc_depth,
        d_bottleneck=args.d_bottleneck,
        dropout=0.0,
        mlp_ratio=4.0,
        time_every=1,
        latents_only_time=True,
        mae_p_min=0.0, 
        mae_p_max=0.9
    )
    
    dec = Decoder(
        d_bottleneck=args.d_bottleneck,
        d_model=args.d_model,
        n_heads=args.n_heads,
        depth=args.dec_depth,
        n_latents=args.n_latents,
        n_patches=n_patches,
        d_patch=d_patch,
        dropout=0.0,
        mlp_ratio=4.0,
        time_every=1,
        latents_only_time=True
    )
    
    model = Tokenizer(enc, dec).to(device)
    
    # ---- optim ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(device="cuda", enabled=use_amp)
    
    # ---- check existing checkpoint ----
    if os.path.exists(save_path) and not args.force_retrain:
        print(f"Found existing checkpoint at {save_path}. Skipping training (use --force_retrain to override).")
        print("Loading weights for visualization...")
        
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        # Use training mode to match training viz exact behavior
        model.train()
        
        # Disable MAE masking to test if p=0 is the cause of quality drop
        print("Disabling MAE masking (p=0.0)...")
        if hasattr(model.encoder, "mae"):
            model.encoder.mae.p_min = 0.0
            model.encoder.mae.p_max = 0.0
        
        print("Generating visualization batch (model.train() + p=0.0)...")
        os.makedirs("viz_baseline", exist_ok=True)
        
        # Get one batch
        for batch in dataloader:
            x = batch.to(device)
            if x.dtype == torch.uint8:
                x = x.float() / 255.0

            print(f"Viz input range: [{x.min():.3f}, {x.max():.3f}]")

            patches = temporal_patchify(x, patch)
            
            with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
                # Standard forward pass with masking
                pred, mae_mask, keep_prob = model(patches)
                
                # Use save_tokenizer_viz to produce the 4-row grid
                save_tokenizer_viz(
                    x_btchw=x,
                    pred_btnd=pred,
                    mae_mask_btNp1=mae_mask,
                    patch=patch,
                    step=999999, # Dummy step
                    save_dir="viz_baseline",
                    max_items=args.viz_max_items,
                    max_T=args.viz_max_T
                )
                print("Saved viz_baseline/tokenizer_viz_step_999999.png")
            break
            
        return

    # ---- lpips ----
    lpips_fn = None
    if args.lpips_weight > 0.0:
        if lpips is not None:
            print(f"Initializing LPIPS ({args.lpips_net})...")
            lpips_fn = lpips.LPIPS(net=args.lpips_net).to(device)
            lpips_fn.eval()
            lpips_fn.requires_grad_(False)
        else:
            print("Warning: lpips not installed, skipping perceptual loss.")

    print("Starting Training...")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    
    step = 0
    t0 = time.time()
    grad_accum = max(1, int(args.grad_accum))
    grad_norm = 0.0

    while step < args.max_steps:
        for batch in dataloader:
            if step >= args.max_steps:
                break
            
            # ShardedFrameDataset yields frames (B, T, C, H, W)
            x = batch.to(device, non_blocking=True) 
            
            # Normalize if uint8
            if x.dtype == torch.uint8:
                x = x.float() / 255.0

            if step % 500 == 0:
                print(f"Step {step}: Input range [{x.min().item():.3f}, {x.max().item():.3f}], mean: {x.mean().item():.3f}")

            patches = temporal_patchify(x, patch)
            
            with torch.no_grad():
                 # Debug: check z std
                 if step == 0: 
                     z, _ = model.encoder(patches) if not isinstance(model, nn.DataParallel) else model.module.encoder(patches)
                     # if step % args.log_every == 0:
                     #    wandb.log({"debug/z_std": float(z.float().std().item())}, step=step)

            with autocast(device_type="cuda", enabled=use_amp):
                pred, mae_mask, keep_prob = model(patches)
                
                # losses in fp32
                mse = recon_loss_from_mae(pred, patches, mae_mask)
                
                if lpips_fn is not None and args.lpips_weight > 0.0:
                    # LPIPS needs reconstruction in image space
                    lp = lpips_on_mae_recon(
                        lpips_fn, pred, patches, mae_mask,
                        H=H, W=W, C=C, patch=patch,
                        subsample_frac=args.lpips_frac
                    )
                    loss = mse + args.lpips_weight * lp
                else:
                    lp = torch.zeros((), device=device)
                    loss = mse

            loss_to_backprop = loss / grad_accum
            scaler.scale(loss_to_backprop).backward()
            
            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                
                # Check for spike (infinite or extremely large)
                if not torch.isfinite(grad_norm):
                    print(f"Warning: Non-finite gradient norm {grad_norm} at step {step}")
                elif grad_norm > 1e4: # Arbitrary "huge" threshold for warning
                    print(f"Warning: Large gradient spike {grad_norm:.2f} at step {step}")
                    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Logging
            if step % args.log_every == 0:
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                if (args.monitor_activations or args.wandb_project) and not args.no_wandb:
                   wandb.log(
                       {
                           "loss/total": float(loss.item()),
                           "loss/mse": float(mse.item()),
                           "loss/lpips": float(lp.item()),
                           "stats/psnr": float(psnr.item()),
                           "stats/keep_prob": float(keep_prob.mean().item()),
                           "stats/masked_frac": float(mae_mask.float().mean().item()),
                           "lr": float(optimizer.param_groups[0]["lr"]),
                           "stats/grad_norm": float(grad_norm),
                           "time/hrs": (time.time() - t0) / 3600.0,
                       },
                       step=step
                   )

            if step % args.print_every == 0:
                psnr = 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))
                print(f"Step {step}: loss={loss.item():.4f} | mse={mse.item():.4f} | lpips={lp.item():.4f} | psnr={psnr.item():.2f}")

            # Validation Loop
            if step > 0 and step % args.viz_every == 0 and val_dataloader is not None:
                print(f"Running validation at step {step}...")
                model.eval()
                val_mse_sum = 0.0
                val_psnr_sum = 0.0
                val_batches = 0
                
                # Run a few batches for validation (e.g. 10 or 20 to be quick)
                max_val_batches = 20
                
                with torch.no_grad():
                    for i, val_batch in enumerate(val_dataloader):
                        if i >= max_val_batches:
                            break
                        
                        vx = val_batch.to(device)
                        if vx.dtype == torch.uint8:
                            vx = vx.float() / 255.0
                        
                        vpatches = temporal_patchify(vx, patch)
                        
                        # Full reconstruction (no masking for validation metric?)
                        # Or should we validate reconstruction capability WITH masking?
                        # Usually we care about reconstruction capability. 
                        # Let's test reconstruction WITHOUT masking (p=0) or WITH masking?
                        # Training uses masking. Let's stick to the training task: mae reconstruction.
                        
                        with autocast(device_type="cuda", enabled=use_amp):
                             vpred, vmask, _ = model(vpatches)
                             vmse = recon_loss_from_mae(vpred, vpatches, vmask)
                             vpsnr = 10.0 * torch.log10(1.0 / vmse.clamp_min(1e-10))
                        
                        val_mse_sum += vmse.item()
                        val_psnr_sum += vpsnr.item()
                        val_batches += 1
                
                model.train()
                if val_batches > 0:
                    val_mse = val_mse_sum / val_batches
                    val_psnr = val_psnr_sum / val_batches
                    print(f"Validation: mse={val_mse:.4f} | psnr={val_psnr:.2f}")
                    
                    if (args.monitor_activations or args.wandb_project) and not args.no_wandb:
                        wandb.log({
                            "val/mse": val_mse,
                            "val/psnr": val_psnr
                        }, step=step)
                    
                    # Save validation visualization (using last batch from loop)
                    # vx is the last batch input, vpred is the last batch prediction
                    print(f"Saving validation visualization at step {step}...")
                    save_tokenizer_viz(
                        x_btchw=vx,
                        pred_btnd=vpred,
                        mae_mask_btNp1=vmask,
                        patch=patch,
                        step=step,
                        save_dir=os.path.join(args.ckpt_dir, "viz_val"),
                        max_items=args.viz_max_items,
                        max_T=args.viz_max_T
                    )
                    
                    if (args.monitor_activations or args.wandb_project) and not args.no_wandb:
                        log_tokenizer_viz_wandb(
                            x_btchw=vx,
                            pred_btnd=vpred,
                            mae_mask_btNp1=vmask,
                            patch=patch,
                            step=step,
                            max_items=args.viz_max_items,
                            max_T=args.viz_max_T,
                            caption_prefix="val"
                        )

            if step % args.viz_every == 0 and step > 0:
                 if (args.monitor_activations or args.wandb_project) and not args.no_wandb:
                    print(f"Logging visualization at step {step}...")
                    log_tokenizer_viz_wandb(
                        x_btchw=x,
                        pred_btnd=pred,
                        mae_mask_btNp1=mae_mask,
                        patch=patch,
                        step=step,
                        max_items=args.viz_max_items,
                        max_T=args.viz_max_T
                    )
                 # Always save locally if monitor_every is hit
                 save_tokenizer_viz(
                    x_btchw=x,
                    pred_btnd=pred,
                    mae_mask_btNp1=mae_mask,
                    patch=patch,
                    step=step,
                    save_dir=os.path.join(args.ckpt_dir, "viz"),
                    max_items=args.viz_max_items,
                    max_T=args.viz_max_T
                 )

            if step % args.save_every == 0 and step > 0:
                 print(f"Saving checkpoint at step {step}...")
                 state = {"model": model.state_dict(), "args": vars(args)}
                 torch.save(state, save_path) # overwrite latest
                 # Optional: keep history? 
                 # save_ckpt(os.path.join(args.ckpt_dir, f"token_step_{step}.pt"), ...)
                 if step % (100000) == 0:
                     hist_path = os.path.join(args.ckpt_dir, f"tok_step_{step}.pt")
                     torch.save(state, hist_path)
                     print(f"Saved historical checkpoint to {hist_path}")

            step += 1

    print("Saving model...")
    state = {
        "model": model.state_dict(),
        "args": vars(args) 
    }
    torch.save(state, save_path)
    print(f"Saved to {save_path}")

if __name__ == "__main__":
    main()
