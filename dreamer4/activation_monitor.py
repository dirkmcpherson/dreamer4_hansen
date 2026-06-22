
import torch
import torch.nn as nn
import wandb

class ActivationMonitor:
    def __init__(self, model: nn.Module, log_every: int = 1000, active: bool = False):
        self.model = model
        self.log_every = log_every
        self.active = active  # Master switch
        self.recording = False # Per-step switch
        
        self.hooks = []
        self.stats = {}
        
        if self.active:
            self.register_hooks(model)
            
    def register_hooks(self, module: nn.Module, prefix=""):
        # Recursive registration
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Hook worthy layers
            if isinstance(child, (nn.Linear, nn.Conv2d, nn.LayerNorm, nn.MultiheadAttention)):
                self._register_hook(child, full_name)
            
            # Recurse
            self.register_hooks(child, full_name)
            
    def _register_hook(self, layer: nn.Module, name: str):
        def hook_fn(module, input, output):
            if not self.recording:
                return
            
            # Handle tuple outputs (e.g. RNNs, Transformers)
            if isinstance(output, tuple):
                out_tensor = output[0]
            else:
                out_tensor = output
            
            if not isinstance(out_tensor, torch.Tensor):
                return
                
            # Log stats
            # We store: mean, std, and a histogram
            # To save memory, we detach and maybe move to CPU or just keeping stats?
            # Storing full tensor for histogram might be heavy if many layers.
            # We'll clone and detach.
            with torch.no_grad():
                # self.stats[name] = out_tensor.detach().clone() 
                # Better: computed stats immediately? 
                # But histograms need data.
                # Wandb histogram can take tensor.
                # We'll store a subsample to save memory?
                
                # simple subsample
                flat = out_tensor.detach().flatten()
                if flat.numel() > 10000:
                    indices = torch.randint(0, flat.numel(), (10000,), device=flat.device)
                    flat = flat[indices]
                
                self.stats[name] = flat.cpu()

        handle = layer.register_forward_hook(hook_fn)
        self.hooks.append(handle)
        
    def step(self, global_step: int):
        if not self.active:
            return
            
        # Decision: Record this step?
        if global_step % self.log_every == 0:
            self.recording = True
        else:
            self.recording = False
            
    def log_and_clear(self, global_step: int):
        if not self.active or not self.stats:
            self.recording = False # Ensure off
            return
            
        # Log to wandb
        log_dict = {}
        for name, tensor in self.stats.items():
            # Create a histogram
            # Grouping meaningful names?
            # just log 'activations/{name}'
            log_dict[f"activations/{name}"] = wandb.Histogram(tensor)
            
            # Optional: Mean/Std tracking
            log_dict[f"act_stats/{name}_mean"] = tensor.mean().item()
            log_dict[f"act_stats/{name}_std"] = tensor.std().item()
            # Pct Dead (for relus, though we monitor layers before relu usually if inside block? 
            # Hooks on Linear are pre-activation usually if separated, or post if strictly Linear.
            # Most layers here are Linear/Conv.
            log_dict[f"act_stats/{name}_dead%"] = (tensor <= 0.0).float().mean().item()
            
        wandb.log(log_dict, step=global_step)
        
        # Cleanup
        self.stats = {}
        self.recording = False
        
    def close(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
