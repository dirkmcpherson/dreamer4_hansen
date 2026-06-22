
import os
import math
import json
import time
import argparse
import asyncio
from typing import Optional, List, Any, Dict
import numpy as np
import torch
from aiohttp import web, WSMsgType
from PIL import Image
import io
import gymnasium as gym
import torchvision.utils as vutils

# Environment - Assuming gym_pusht is installed
try:
    import gym_pusht
except ImportError:
    pass # Assume gym.make works if registered

from dreamer4.model import (
    Encoder, Decoder, Tokenizer,
    temporal_patchify, pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck, temporal_unpatchify
)
from dreamer4.train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt
)

def frame_to_jpeg_bytes(frame_chw_01: torch.Tensor, quality: int = 85) -> bytes:
    fr_u8 = (frame_chw_01.clamp(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
    hwc = np.transpose(fr_u8, (1, 2, 0))
    im = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()

class InteractiveTokenizerServer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.infer_lock = asyncio.Lock()
        
        # Load HTML
        # Reuse the existing one
        with open(os.path.join(os.path.dirname(__file__), "interactive_pusht.html"), "r") as f:
            self.html = f.read()
            # Hack to make frame wider for side-by-side (3 panels: 512*3 = 1536)
            self.html = self.html.replace("width: 512px;", "width: 1536px;")
            self.html = self.html.replace('width="512"', 'width="1536"')
            
            # Patch JS for 3-pane interaction (only leftmost pane controls)
            js_search = "let nx = (x / w) * 2 - 1;"
            js_replace = """
    // 3 panels, control only left-most
    const panelW = w / 3;
    if (x > panelW) return; 
    let nx = (x / panelW) * 2 - 1;
            """
            self.html = self.html.replace(js_search, js_replace)

        # Load Tokenizer
        self.encoder, self.decoder, tok_info = load_frozen_tokenizer_from_pt_ckpt(
            args.tokenizer_ckpt, device=self.device
        )
        self.H = int(tok_info.get("H", 64))
        self.W = int(tok_info.get("W", 64))
        self.C = int(tok_info.get("C", 3))
        self.patch = int(tok_info.get("patch", 4))
        self.d_bottleneck = int(tok_info.get("d_bottleneck", 32))
        self.n_latents = int(tok_info.get("n_latents", 16))
        
        # Disable MAE masking for interactive visualization
        if hasattr(self.encoder, "mae"):
             self.encoder.mae.p_min = 0.0
             self.encoder.mae.p_max = 0.0
        
        # Checkpoint args
        ckpt_args = tok_info.get("args", {})
        self.seq_len = int(ckpt_args.get("seq_len", 8))
        print(f"Using sequence length: {self.seq_len}")
        
        # Packing factor? If not provided, assume 1 or standard logic.
        # But this script only uses Tokenizer, so packing factor is irrelevant for Dynamics.
        # However, Encoder outputs Z which might need packing if we were doing dynamics.
        # Here we just decode what we encode.
        # Encoder -> Z (B, T, L, D_model) -> Decoder.
        # Note: Decoder expects (B, T, L, D_bottleneck) ? No, Decoder takes z_btLd.
        # Encoder returns `z` (tanh output) which is bottlenecked?
        # Let's check model.py:
        # Encoder.forward returns z (masked/bottlenecked) of shape (B, T, L, D_bottleneck)
        # Decoder.forward takes z (B, T, L, D_bottleneck)
        # So no packing needed for direct reconstruction!
        
        if args.interactive:
            self.args.demo_playback = False

        # Environment
        if not self.args.demo_playback:
            self.env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
            self.obs, _ = self.env.reset(seed=args.seed)
        else:
            from dreamer4.sharded_frame_dataset import ShardedFrameDataset
            from dreamer4.task_set import TASK_SET
            self.dataset = ShardedFrameDataset(
                outdirs=args.dataset_path, 
                tasks=TASK_SET,
                seq_len=self.seq_len, # Use tokenizer seq len
                iid_sampling=True,
                cache_size=16
            )
            self.playback_idx = 0
            print(f"Dataset Loaded. Length: {len(self.dataset)}")
            self.env = None
            
            # Save debug batch
            self.save_debug_batch()
        
        self.latest_action = np.zeros(2, dtype=np.float32)
        self.step_count = 0
        
        # Frame History Buffer (C,H,W) tensors
        self.frames = []
        
    def save_debug_batch(self):
        print("Saving debug batch to viz_baseline/ ...")
        os.makedirs("viz_baseline", exist_ok=True)
        
        indices = np.random.randint(0, len(self.dataset), size=8)
        batch = []
        for idx in indices:
            seq = self.dataset[idx] # (T, C, H, W)
            batch.append(seq)
            
        x = torch.stack(batch).to(self.device) # (B, T, C, H, W)
        
        with torch.no_grad():
            patches = temporal_patchify(x, self.patch)
            z, _ = self.encoder(patches) 
            recon_patches = self.decoder(z)
            recon_frames = temporal_unpatchify(recon_patches, self.H, self.W, self.C, self.patch)
            
        # Compare last frames
        last_orig = x[:, -1]
        last_recon = recon_frames[:, -1]
        comp = torch.cat([last_orig, last_recon], dim=0)
        vutils.save_image(comp, "viz_baseline/demo_debug_batch.png", nrow=8, padding=2)
        print("Saved viz_baseline/demo_debug_batch.png")

    def reset_env(self):
        new_seed = int(time.time() * 1000) % (2**32)
        print(f"Resetting env with seed {new_seed}")
        if self.env:
            self.obs, _ = self.env.reset(seed=new_seed)
        if self.args.demo_playback:
            # Randomize playback start?
            self.playback_idx = np.random.randint(0, len(self.dataset))
            
        self.step_count = 0
        self.latest_action = np.zeros(2, dtype=np.float32)
        self.frames = []
        
    def _step_and_render_sync(self):
        x = None
        reward = 0.0
        
        if self.args.demo_playback:
             if self.playback_idx >= len(self.dataset):
                 self.playback_idx = 0
             
             # (T, C, H, W)
             seq = self.dataset[self.playback_idx] 
             self.playback_idx += 1
             
             x = seq.unsqueeze(0).to(self.device) # (1, T, C, H, W)
             
        else:
            # Step Env
            # Action is normalized [-1, 1]. PushT expects pixels? 
            # Wait, gym_pusht typically expecting unnormalized actions [0, 512]?
            # Or normalized?
            # Standard gym_pusht: action space is Box(low=0, high=512, shape=(2,), dtype=float32)
            # So we must denormalize.
            
            # Mapping [-1, 1] -> [0, 512]
            # -1 -> 0
            # 1 -> 512
            # (x + 1) / 2 * 512
            
            act_unorm = (self.latest_action + 1) / 2 * 512.0
            act_unorm = np.clip(act_unorm, 0, 512)
            
            self.obs, reward, terminated, truncated, _ = self.env.step(act_unorm)
            
            # Get Image
            # obs is state vector. We need to render.
            img_np = self.env.render() # (H, W, 3)
            
            # Resize to Tokenizer input (self.H, self.W)
            # Use PIL
            pil_img = Image.fromarray(img_np)
            pil_img = pil_img.resize((self.W, self.H), Image.BILINEAR)
            
            # To Tensor (C, H, W)
            frame_tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0 # (H,W,C)
            frame_tensor = frame_tensor.permute(2, 0, 1) # (C,H,W)
            
            self.frames.append(frame_tensor)
            if len(self.frames) > self.seq_len:
                self.frames.pop(0)
                
            curr_frames = torch.stack(list(self.frames), dim=0) # (T, C, H, W)
            x = curr_frames.unsqueeze(0).to(self.device) # (1, T, C, H, W)
        
        self.step_count += 1

        # Reconstruct
        with torch.no_grad():
            patches = temporal_patchify(x, self.patch)
            z, _ = self.encoder(patches) 
            # z: (1, T, L, D_bottleneck)
            
            # Recon p=0
            recon_patches_0 = self.decoder(z)
            recon_frames_0 = temporal_unpatchify(recon_patches_0, self.H, self.W, self.C, self.patch)
            recon_frame_0 = recon_frames_0[0, -1] # (C, H, W)
            
            # Recon p=0.5
            # We must re-encode with masking enabled
            if hasattr(self.encoder, "mae"):
                 old_min = self.encoder.mae.p_min
                 old_max = self.encoder.mae.p_max
                 self.encoder.mae.p_min = 0.5
                 self.encoder.mae.p_max = 0.5
            
            z_50, _ = self.encoder(patches)
            recon_patches_50 = self.decoder(z_50)
            recon_frames_50 = temporal_unpatchify(recon_patches_50, self.H, self.W, self.C, self.patch)
            recon_frame_50 = recon_frames_50[0, -1]
            
            if hasattr(self.encoder, "mae"):
                 self.encoder.mae.p_min = old_min
                 self.encoder.mae.p_max = old_max

            # Concat with original (Last frame)
            orig_frame = x[0, -1]
            
            # Debug prints
            print(f"Orig: [{orig_frame.min():.3f}, {orig_frame.max():.3f}] mean={orig_frame.mean():.3f}")
            print(f"Rec0: [{recon_frame_0.min():.3f}, {recon_frame_0.max():.3f}] mean={recon_frame_0.mean():.3f}")
            print(f"Rec50: [{recon_frame_50.min():.3f}, {recon_frame_50.max():.3f}] mean={recon_frame_50.mean():.3f}")

            # dim 2 is Width (C, H, W)
            # 3 Panels: Original | Rec 0% | Rec 50%
            combined = torch.cat([orig_frame, recon_frame_0, recon_frame_50], dim=2)
            
            # Also show original? Maybe later.
            jpeg = frame_to_jpeg_bytes(combined)
            
            status = {
                "type": "status",
                "text": f"Step: {self.step_count} | Act: {self.latest_action} | R: {reward:.2f} | T: {x.shape[1]} | Idx: {self.playback_idx if self.args.demo_playback else 'N/A'}"
            }
            return jpeg, status
            
    async def index(self, request):
        return web.Response(text=self.html, content_type="text/html")

    async def ws_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        async def recv_loop():
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        t = data.get("type")
                        if t == "mousemove":
                            x = float(data.get("x", 0))
                            y = float(data.get("y", 0))
                            self.latest_action = np.array([x, y], dtype=np.float32)
                        elif t == "reset":
                            self.reset_env()
                    except: pass
                    
        async def send_loop():
            while not ws.closed:
                start = time.monotonic()
                async with self.infer_lock:
                    jpeg, status = await asyncio.to_thread(self._step_and_render_sync)
                
                try:
                    await ws.send_str(json.dumps(status))
                    await ws.send_bytes(jpeg)
                except: break
                
                # Target FPS
                target_dt = 1.0 / self.args.fps
                elapsed = time.monotonic() - start
                delay = max(0, target_dt - elapsed)
                await asyncio.sleep(delay)

        await asyncio.gather(recv_loop(), send_loop())
        return ws

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_ckpt", type=str, required=True)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--demo_playback", action="store_true", help="Playback from dataset")
    parser.add_argument("--interactive", action="store_true", help="Enable mouse interaction (disables playback)")
    parser.add_argument("--dataset_path", type=str, default="scripts/data/pusht_train")
    args = parser.parse_args()

    server = InteractiveTokenizerServer(args)
    app = web.Application()
    app.add_routes([
        web.get('/', server.index),
        web.get('/ws', server.ws_handler),
    ])
    web.run_app(app, port=args.port)

if __name__ == "__main__":
    main()
