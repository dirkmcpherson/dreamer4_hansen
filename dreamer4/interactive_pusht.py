
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

# Environment - Assuming gym_pusht is installed
try:
    import gym_pusht
except ImportError:
    pass # Assume gym.make works if registered

from dreamer4.model import (
    Encoder, Decoder, Tokenizer, Dynamics,
    temporal_patchify, pack_bottleneck_to_spatial,
    unpack_spatial_to_bottleneck, temporal_unpatchify
)
from dreamer4.train_dynamics import (
    load_frozen_tokenizer_from_pt_ckpt, 
    make_tau_schedule, 
    sample_one_timestep_packed
)

# Reuse some helper functions from interactive.py where possible, 
# or re-implement simply since we can't easily import from script to script.

def _as_2d_packed(z: torch.Tensor) -> torch.Tensor:
    if z.dim() == 2: return z
    if z.dim() == 3 and z.shape[0] == 1: return z[0]
    return z

def frame_to_jpeg_bytes(frame_chw_01: torch.Tensor, quality: int = 85) -> bytes:
    fr_u8 = (frame_chw_01.clamp(0, 1) * 255.0).to(torch.uint8).detach().cpu().numpy()
    hwc = np.transpose(fr_u8, (1, 2, 0))
    im = Image.fromarray(hwc, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(quality), optimize=True)
    return buf.getvalue()

def load_dynamics_for_interactive(ckpt_path, device, d_bottleneck, packing_factor, n_latents_hint=None):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt["args"] # Dict
    
    # Checkpoint argument handling
    d_model = int(args.get("d_model_dyn", args.get("d_model", 256)))
    k_max = int(args.get("k_max", 16))
    
    # packing factor might be in args or passed in
    pf = int(args.get("packing_factor", packing_factor))
    
    # Check if n_latents is in args, otherwise trust packing
    nm_latents = args.get("n_latents", n_latents_hint)
    
    if nm_latents is None:
         # Assume packing factor holds relation if we can't find n_latents
         # But we really need n_latents or n_spatial.
         raise ValueError("Could not determine n_latents from checkpoint args or hint.")
         
    n_spatial = int(nm_latents) // pf
    d_spatial = d_bottleneck * pf
    
    dyn = Dynamics(
        d_model=d_model,
        d_bottleneck=d_bottleneck,
        d_spatial=d_spatial,
        n_spatial=n_spatial,
        n_register=int(args.get("n_register", 0)),
        n_agent=int(args.get("n_agent", 1)),
        n_heads=int(args.get("n_heads", 4)),
        depth=int(args.get("dyn_depth", args.get("depth", 8))),
        k_max=k_max,
        dropout=0.0,
        action_dim=int(args.get("action_dim", 2)), # Default to 2 for PushT if not in args
        mlp_ratio=float(args.get("mlp_ratio", 4.0)),
        time_every=int(args.get("time_every", 1)),
        space_mode=str(args.get("space_mode", "wm_agent_isolated")),
    ).to(device)
    
    sd = ckpt["dynamics"]
    new_sd = {k.replace("module.", ""): v for k,v in sd.items() if not k.startswith("reward_head")}
    dyn.load_state_dict(new_sd, strict=True)
    dyn.eval()
    return dyn, {"k_max": k_max, "n_spatial": n_spatial, "d_spatial": d_spatial, "packing_factor": pf, "action_dim": dyn.action_encoder.action_dim}

class InteractiveServer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.infer_lock = asyncio.Lock()
        
        # Load HTML
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
        
        # Load Dynamics
        self.dyn, dyn_info = load_dynamics_for_interactive(
            args.dynamics_ckpt, 
            self.device, 
            self.d_bottleneck, 
            args.packing_factor,
            n_latents_hint=self.n_latents
        )
        self.k_max = int(dyn_info["k_max"])
        self.n_spatial = int(dyn_info["n_spatial"])
        self.d_spatial = int(dyn_info["d_spatial"])
        self.packing_factor = int(dyn_info["packing_factor"])
        
        self.ctx_len = max(1, int(getattr(args, "ctx_len", 8)))

        self.sched = make_tau_schedule(k_max=self.k_max, schedule="shortcut", d=0.5)
        
        # Environment
        self.env = None
        if args.interactive:
             self.env = gym.make("gym_pusht/PushT-v0", render_mode="rgb_array")
             self.obs, _ = self.env.reset(seed=args.seed)
             # Pre-load initial frame to z_curr
             # We need to handle this in reset_env really, but first init here
        
        # Initial State
        self.z_curr = None # (n_spatial, d_spatial)
        self.z_imag = None # (n_spatial, d_spatial) Open loop imagination state
        self.reset_env()
        
        self.latest_action = torch.zeros(2, device=self.device)
        self.step_count = 0
        
    def reset_env(self):
        new_seed = int(time.time() * 1000) % (2**32)
        torch.manual_seed(new_seed)
        print(f"Resetting with seed: {new_seed}")

        if self.args.interactive:
             self.obs, _ = self.env.reset(seed=new_seed)
             # Resize to tokenizer H,W
             img_np = self.env.render()
             pil_img = Image.fromarray(img_np).resize((self.W, self.H), Image.BILINEAR)
             frame_tensor = torch.from_numpy(np.array(pil_img)).float() / 255.0
             frame_tensor = frame_tensor.permute(2, 0, 1).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,C,H,W)
             
             with torch.no_grad():
                 patches = temporal_patchify(frame_tensor, self.patch)
                 z, _ = self.encoder(patches) # (1,1, L, D)
                 # Pack
                 z_packed = pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)
                 self.z_curr = z_packed[0, 0]
                 self.z_imag = z_packed[0, 0].clone()
        else:
            # Initialize with a valid image (Gray background) or Noise
            dummy_img = torch.full((1, 1, self.C, self.H, self.W), 0.5, device=self.device)
            
            with torch.no_grad():
                # Patchify
                patches = temporal_patchify(dummy_img, self.patch)
                # Encode
                z, _ = self.encoder(patches)
                # Pack
                z_packed = pack_bottleneck_to_spatial(
                    z, 
                    n_spatial=self.n_spatial, 
                    k=self.packing_factor
                )
            
            self.z_curr = z_packed[0, 0]
            self.z_imag = z_packed[0, 0].clone() # Keep compatible
        
        self.step_count = 0
        self.latest_action = torch.zeros(2, device=self.device)

        # Rolling open-loop context: history of imagined latents and the
        # actions associated with arriving at each frame. hist_a is aligned
        # 1:1 with hist_z (the initial frame has a dummy zero action).
        self.hist_z = [self.z_imag.clone()]          # each (n_spatial, d_spatial)
        self.hist_a = [torch.zeros(2, device=self.device)]

    def _render_step_sync(self):
        # 1. Action
        # latest_action is in [-1, 1].
        # For env step, we need [0, 512].
        
        act_unorm = (self.latest_action.cpu().numpy() + 1) / 2 * 512.0
        act_unorm = np.clip(act_unorm, 0, 512)
        
        real_frame = None
        
        if self.args.interactive:
             self.obs, _, _, _, _ = self.env.step(act_unorm)
             img_np = self.env.render() # (H,W,3)
             pil_img = Image.fromarray(img_np).resize((self.W, self.H), Image.BILINEAR)
             real_frame = torch.from_numpy(np.array(pil_img)).float() / 255.0
             real_frame = real_frame.permute(2, 0, 1).to(self.device).unsqueeze(0).unsqueeze(0) # (1,1,C,H,W)
             
             # Encode Real for Z_real
             with torch.no_grad():
                 p = temporal_patchify(real_frame, self.patch)
                 z, _ = self.encoder(p)
                 zp = pack_bottleneck_to_spatial(z, n_spatial=self.n_spatial, k=self.packing_factor)
                 self.z_curr = zp[0, 0] # Used as ground truth for reconstruction or next step init?
                 # Actually, z_curr acts as "current state". In interactive closed loop, 
                 # we update z_curr from Real.
                 # In Imagination, we have z_imag.
        
        # 2. Dynamics (Open Loop Imagination) with rolling context.
        # Feed the last ctx_len imagined latents as context (the model was
        # trained on sequences, so >1 frame lets it infer velocity/history),
        # with actions aligned per-frame plus the new action for the frame
        # being predicted.
        t = len(self.hist_z)
        past = torch.stack(self.hist_z, dim=0).unsqueeze(0)  # (1, t, n_spatial, d_spatial)
        act_seq = self.hist_a + [self.latest_action]
        act_in = torch.stack(act_seq, dim=0).unsqueeze(0)    # (1, t+1, 2)

        with torch.no_grad():
            z_next_out = sample_one_timestep_packed(
                self.dyn,
                past_packed=past,
                k_max=self.k_max,
                sched=self.sched,
                actions=act_in,
                act_mask=None
            )
            # Append to rolling buffers and truncate to the context window.
            z_new = z_next_out[0]
            self.hist_z.append(z_new)
            self.hist_a.append(self.latest_action.clone())
            if len(self.hist_z) > self.ctx_len:
                self.hist_z.pop(0)
                self.hist_a.pop(0)
            self.z_imag = z_new
            self.step_count += 1
            
            # Decode Imagination
            z_btLd_imag = unpack_spatial_to_bottleneck(z_next_out.unsqueeze(1), k=self.packing_factor)
            patches_imag = self.decoder(z_btLd_imag)
            frames_imag = temporal_unpatchify(patches_imag, self.H, self.W, self.C, self.patch)
            frame_imag = frames_imag[0, 0]
            
            # Reconstruction (Decode current Real Z)
            # z_curr is packed. Unpack.
            z_btLd_real = unpack_spatial_to_bottleneck(self.z_curr.view(1, 1, self.n_spatial, self.d_spatial), k=self.packing_factor)
            patches_real = self.decoder(z_btLd_real)
            frames_real = temporal_unpatchify(patches_real, self.H, self.W, self.C, self.patch)
            frame_recon = frames_real[0, 0]
            
            # Real Frame
            if real_frame is not None:
                frame_real = real_frame[0, 0]
            else:
                # If not interactive, maybe just show noise/blank for real?
                frame_real = torch.zeros_like(frame_recon)
            
            # 3 Panels: Real | Recon | Imagined
            combined = torch.cat([frame_real, frame_recon, frame_imag], dim=2)
            jpeg = frame_to_jpeg_bytes(combined)
            
            z_mean = z_btLd_imag.mean().item()
            z_std = z_btLd_imag.std().item()

            status = {
                "type": "status",
                "text": f"Step: {self.step_count} | Act: [{self.latest_action[0]:.2f}, {self.latest_action[1]:.2f}] | Z_imag: {z_mean:.2f}+/-{z_std:.2f}"
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
                            self.latest_action = torch.tensor([x, y], device=self.device)
                        elif t == "reset":
                            self.reset_env()
                    except: pass
                    
        async def send_loop():
            while not ws.closed:
                start = time.monotonic()
                async with self.infer_lock:
                    jpeg, status = await asyncio.to_thread(self._render_step_sync)
                
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
    parser.add_argument("--dynamics_ckpt", type=str, required=True)
    parser.add_argument("--packing_factor", type=int, default=2)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--interactive", action="store_true", help="Enable mouse interaction (Real | Recon | Imagined)")
    parser.add_argument("--ctx_len", type=int, default=8, help="Rolling context window (# past imagined frames fed to the world model)")
    parser.add_argument("--dataset_path", type=str, default="scripts/data/pusht_train")
    args = parser.parse_args()

    server = InteractiveServer(args)
    app = web.Application()
    app.add_routes([
        web.get('/', server.index),
        web.get('/ws', server.ws_handler),
    ])
    web.run_app(app, port=args.port)

if __name__ == "__main__":
    main()
