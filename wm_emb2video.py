import os
import sys
import gc
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, LCMScheduler

# =====================================================================
# 1. GLOBAL CONFIGURATION & DEVICE RESOLUTION
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SD_CKPT = Path(r"D:\AI\sd.webui\webui\models\Stable-diffusion\realisticVisionV60B1_v51HyperVAE.safetensors")
ADAPTER_CKPT = Path("vjepa_t2i_adapter_sd15_step4550.pt")
OUTPUT_DIR = Path("experiments/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

_model_cache = {
    "sd_components": None,
    "adapter": None,
    "ckpt_hash": None
}

# =====================================================================
# 2. NEURAL NETWORK ARCHITECTURE (T2I-ADAPTER)
# =====================================================================
class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels, eps=1e-6), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): 
        return x + self.block(x)

class AdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.resnets = nn.Sequential(*[ResnetBlock(out_channels) for _ in range(num_res_blocks)])
    def forward(self, x): 
        return self.resnets(self.in_conv(x))

class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): 
        return self.conv(x)

class JEPAProjectionHead(nn.Module):
    def __init__(self, token_dim=1024, out_channels=320):
        super().__init__()
        self.reduce = nn.Conv2d(token_dim, 512, 1)
        self.refine = nn.Sequential(
            nn.GroupNorm(32, 512, eps=1e-6), nn.SiLU(),
            nn.Conv2d(512, out_channels, 3, padding=1),
            nn.GroupNorm(32, out_channels, eps=1e-6), nn.SiLU(),
        )
    def forward(self, x):
        B = x.shape[0]
        x = x.float().reshape(B, 24, 24, 1024).permute(0, 3, 1, 2)
        x = self.reduce(x)
        x = F.interpolate(x, size=(48, 48), mode='bilinear', align_corners=False)
        return self.refine(x)

class JEPAAdapter(nn.Module):
    def __init__(self, num_res_blocks=2):
        super().__init__()
        self.projection = JEPAProjectionHead(1024, 320)
        self.block0 = AdapterBlock(320, 320, num_res_blocks)
        self.block1 = AdapterBlock(320, 640, num_res_blocks)
        self.block2 = AdapterBlock(640, 1280, num_res_blocks)
        self.block3 = AdapterBlock(1280, 1280, num_res_blocks)
        self.down0 = DownsampleBlock(320)
        self.down1 = DownsampleBlock(640)
        self.down2 = DownsampleBlock(1280)

    def forward(self, jepa_emb):
        x = self.projection(jepa_emb)
        f0 = self.block0(x); x = self.down0(f0)
        f1 = self.block1(x); x = self.down1(f1)
        f2 = self.block2(x); x = self.down2(f2)
        f3 = self.block3(x)
        return [f0, f1, f2, f3]

# =====================================================================
# 3. SPATIAL UPSAMPLING LAYER
# =====================================================================
def upsample_pooled_embeddings(embeddings_tensor):
    B, L, C = embeddings_tensor.shape
    x = embeddings_tensor.view(B, 12, 12, C).permute(0, 3, 1, 2)
    x_up = F.interpolate(x, size=(24, 24), mode='bicubic', align_corners=False)
    return x_up.permute(0, 2, 3, 1).view(B, 576, C)

# =====================================================================
# 4. OPENCV VIDEO & IMAGE HELPER UTILITIES
# =====================================================================
def extract_video_frames(video_path, target_fps=0):
    cap = cv2.VideoCapture(video_path)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 24
        
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames, target_fps if target_fps > 0 else source_fps

def create_side_by_side(img1, img2):
    if img1.size[1] != img2.size[1]:
        scale = img1.size[1] / img2.size[1]
        new_w = int(img2.size[0] * scale)
        img2 = img2.resize((new_w, img1.size[1]), Image.Resampling.BILINEAR)
    
    dst = Image.new('RGB', (img1.size[0] + img2.size[0], img1.size[1]))
    dst.paste(img1, (0, 0))
    dst.paste(img2, (img1.size[0], 0))
    return dst

def frames_to_video(pil_frames, out_path, fps):
    if not pil_frames:
        return
    w, h = pil_frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
    
    for frame in pil_frames:
        frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    video_writer.release()

def auto_open_file(file_path):
    path_str = str(file_path)
    if sys.platform == "win32":
        os.startfile(path_str)
    elif sys.platform == "darwin":
        subprocess.run(["open", path_str])
    else:
        subprocess.run(["xdg-open", path_str])

# =====================================================================
# 5. MODEL LOADERS & INFERENCE PIPELINE
# =====================================================================
def load_sd():
    if _model_cache["sd_components"] and _model_cache["ckpt_hash"] == str(SD_CKPT):
        return _model_cache["sd_components"]

    print(f"Loading SD Base Checkpoint: {SD_CKPT.name}")
    pipe = StableDiffusionPipeline.from_single_file(
        str(SD_CKPT), torch_dtype=torch.float16, safety_checker=None
    ).to(DEVICE)

    scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    comps = {
        "vae": pipe.vae.eval(), "unet": pipe.unet.eval(),
        "tokenizer": pipe.tokenizer, "text_enc": pipe.text_encoder.eval(),
        "scheduler": scheduler
    }
    for m in [comps["vae"], comps["unet"], comps["text_enc"]]:
        for p in m.parameters(): 
            p.requires_grad_(False)

    del pipe; gc.collect(); torch.cuda.empty_cache()
    _model_cache["sd_components"] = comps
    _model_cache["ckpt_hash"] = str(SD_CKPT)
    return comps

def load_adapter():
    if _model_cache["adapter"] is not None:
        return _model_cache["adapter"]
    print(f"Loading Target Adapter weights: {ADAPTER_CKPT}")
    adapter = JEPAAdapter(num_res_blocks=2).to(DEVICE)
    ckpt = torch.load(str(ADAPTER_CKPT), map_location=DEVICE, weights_only=True)
    adapter.load_state_dict(ckpt.get("adapter", ckpt))
    adapter.eval()
    _model_cache["adapter"] = adapter
    return adapter

@torch.no_grad()
def get_empty_text_emb(tokenizer, text_enc):
    tokens = tokenizer([""], padding="max_length", max_length=tokenizer.model_max_length,
                       truncation=True, return_tensors="pt").input_ids.to(DEVICE)
    return text_enc(tokens).last_hidden_state

@torch.no_grad()
def generate_image(jepa_emb, adapter, vae, unet, scheduler, text_emb, num_steps=4, seed=42, cond_scale=1.0):
    scheduler.set_timesteps(num_steps)
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    latent = torch.randn(1, 4, 48, 48, dtype=torch.float16, device=DEVICE, generator=gen)
            
    jepa_gpu = jepa_emb.unsqueeze(0).to(DEVICE)

    for t in scheduler.timesteps:
        feats = adapter(jepa_gpu)
        residuals = [f.to(torch.float16) * cond_scale for f in feats]
        noise_pred = unet(latent, t, text_emb, down_intrablock_additional_residuals=residuals).sample
        latent = scheduler.step(noise_pred, t, latent).prev_sample

    latent = latent / vae.config.scaling_factor
    img = vae.decode(latent).sample
    img = ((img.float().clamp(-1, 1) + 1) / 2 * 255).byte()
    return Image.fromarray(img[0].permute(1, 2, 0).cpu().numpy())

# =====================================================================
# 6. PIPELINE EXECUTION WRAPPER WITH AUTOMATIC STRIDE RATIO RESOLUTION
# =====================================================================
def run_pooled_pipeline(npz_path, video_path=None, steps=4, seed=42, cond_scale=1.0, video_fps=0):
    npz_path = Path(npz_path)
    
    print(f"Extracting pooled arrays: {npz_path.name}")
    npz_data = np.load(npz_path)
    array_key = 'embeddings' if 'embeddings' in npz_data else npz_data.files[0]
    pooled_arr = npz_data[array_key]
    
    pooled_tensor = torch.from_numpy(pooled_arr).float()
    if pooled_tensor.ndim == 2:
        pooled_tensor = pooled_tensor.unsqueeze(0)
        
    print("Mapping spatial scaling operations via cell-centered alignment layers...")
    upsampled_embeddings = upsample_pooled_embeddings(pooled_tensor)
    num_embs = upsampled_embeddings.shape[0]
    
    sd = load_sd()
    adapter = load_adapter()
    text_emb = get_empty_text_emb(sd["tokenizer"], sd["text_enc"])
    
    frames = []
    final_fps = 24.0 # Default base playback fallback speed
    
    if video_path:
        video_path = Path(video_path)
        if video_path.exists():
            print(f"Extracting verification references from target: {video_path.name}")
            orig_frames, source_fps = extract_video_frames(str(video_path), video_fps)
            
            # --- AUTOMATIC STRIDE RATIO CALCULATION BLOCK ---
            stride = max(1, round(len(orig_frames) / num_embs))
            print(f"-> Detected Embedding Subsampling Stride: {stride}x")
            print(f"-> Base video frames: {len(orig_frames)} | Target embedding slices: {num_embs}")
            
            # Step the original video frames by the stride sequence so timestamps align
            frames = [orig_frames[i * stride] for i in range(num_embs) if i * stride < len(orig_frames)]
            
            # Lock sequence bounds cleanly
            min_len = min(len(frames), num_embs)
            frames = frames[:min_len]
            upsampled_embeddings = upsampled_embeddings[:min_len]
            
            # Mathematically adjust frame pacing down based on dropped elements
            final_fps = source_fps / stride
            print(f"-> Calibrated Output Video Playback Frame Rate: {final_fps:.2f} FPS (Original: {source_fps} FPS)")
    else:
        # If no video passed, fallback to user argument or an estimated 2 FPS (10 FPS / 5 stride)
        final_fps = video_fps if video_fps > 0 else 2.0
            
    generated = []
    for i in tqdm(range(upsampled_embeddings.shape[0]), desc="Diffusing Array State"):
        emb = upsampled_embeddings[i]
        img = generate_image(emb, adapter, sd["vae"], sd["unet"], sd["scheduler"],
                             text_emb, steps, seed, cond_scale)
        generated.append(img)
        
    timestamp = datetime.now().strftime("%H%M%S")
    is_video_out = len(generated) > 1 or video_path is not None
    
    if is_video_out:
        if frames:
            render_frames = [create_side_by_side(o, g) for o, g in zip(frames, generated)]
            out_path = OUTPUT_DIR / f"{npz_path.stem}_sbs_{timestamp}.mp4"
        else:
            render_frames = generated
            out_path = OUTPUT_DIR / f"{npz_path.stem}_gen_{timestamp}.mp4"
            
        frames_to_video(render_frames, str(out_path), final_fps)
    else:
        if frames:
            sbs_img = create_side_by_side(frames[0], generated[0])
            out_path = OUTPUT_DIR / f"{npz_path.stem}_sbs_{timestamp}.png"
        else:
            sbs_img = generated[0]
            out_path = OUTPUT_DIR / f"{npz_path.stem}_gen_{timestamp}.png"
        sbs_img.save(out_path)
        
    print(f"✓ Pipeline execution complete: {out_path}")
    auto_open_file(out_path)
    return out_path

# =====================================================================
# 7. CLI / GUI ENTRY EXECUTION LOGIC
# =====================================================================
def launch_gui_fallback():
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    
    print("No system parameters detected. Launching modular file system pickers...")
    
    npz_path = filedialog.askopenfilename(
        title="Select Pooled Embeddings Archive (.npz)",
        filetypes=[("NumPy Archive", "*.npz")]
    )
    if not npz_path:
        print("Pipeline aborted: Missing required .npz input path constraint.")
        return None
        
    video_path = filedialog.askopenfilename(
        title="Select Optional Reference Video (Bypass to generate solo result)",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.webm")]
    )
    video_path = video_path if video_path else None
    
    return argparse.Namespace(
        npz_path=npz_path,
        video_path=video_path,
        steps=4,
        seed=42,
        cond_scale=1.0,
        video_fps=0
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V-JEPA 2.1 Spatial Core Upsampling and Stable Diffusion Generation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-i", "--npz_path", type=str, help="Path to your input pooled .npz workspace file.")
    parser.add_argument("-v", "--video_path", type=str, default=None, help="Path to original source video tracking file for comparison layouts.")
    parser.add_argument("--steps", type=int, default=4, help="Inference denoising steps configuration parameter.")
    parser.add_argument("--seed", type=int, default=42, help="Hardware pseudorandomization tracking sequence.")
    parser.add_argument("--cond_scale", type=float, default=1.0, help="Latent injector gain tuning value.")
    parser.add_argument("--video_fps", type=int, default=0, help="Output sequence metadata reproduction rate.")

    if len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = launch_gui_fallback()

    if args and args.npz_path:
        run_pooled_pipeline(
            npz_path=args.npz_path,
            video_path=args.video_path,
            steps=args.steps,
            seed=args.seed,
            cond_scale=args.cond_scale,
            video_fps=args.video_fps
        )