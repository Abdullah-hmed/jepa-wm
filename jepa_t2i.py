"""
JEPA T2I-Adapter Inference Script - Enhanced Edition
----------------------------------------------------
Image → V-JEPA 2.1 encode → JEPAAdapter → SD1.5 generation
Video → per-frame encode → generate → ffmpeg reassemble

Features:
  • Side-by-side comparison outputs (original | generated)
  • Configurable video frame sampling FPS
  • Persistent SD 1.5 model caching in VRAM
  • Modern Tkinter UI with progress bars & live logging
  • GPU-optimized for 4GB VRAM (1050 Ti compatible)
"""

import os
import sys
import gc
import json
import shutil
import tempfile
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION – Edit these paths to match your setup
# ─────────────────────────────────────────────────────────────────
SD_CKPT = Path(
    r"D:\AI\sd.webui\webui\models\Stable-diffusion"
    r"\realisticVisionV60B1_v51HyperVAE.safetensors"
)
ADAPTER_CKPT = Path("vjepa_t2i_adapter_sd15_step4550.pt")
OUTPUT_DIR = Path("experiments/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 384
MAX_VRAM_GB = 4.0  # Target VRAM usage for memory management

# ─────────────────────────────────────────────────────────────────
# GLOBAL MODEL CACHE (persistent across generations)
# ─────────────────────────────────────────────────────────────────
_model_cache = {
    "vjepa_encoder": None,
    "adapter": None,
    "sd_components": None,  # {vae, unet, tokenizer, text_enc, scheduler}
    "ckpt_hash": None,
}


def clear_cache(component: str = None):
    """Clear specific or all cached models to free VRAM."""
    global _model_cache
    if component is None or component == "vjepa":
        if _model_cache["vjepa_encoder"] is not None:
            _model_cache["vjepa_encoder"].cpu()
            del _model_cache["vjepa_encoder"]
            _model_cache["vjepa_encoder"] = None
    if component is None or component == "adapter":
        if _model_cache["adapter"] is not None:
            del _model_cache["adapter"]
            _model_cache["adapter"] = None
    if component is None or component == "sd":
        if _model_cache["sd_components"] is not None:
            for key in ["vae", "unet", "text_enc"]:
                if _model_cache["sd_components"].get(key):
                    _model_cache["sd_components"][key].cpu()
            del _model_cache["sd_components"]
            _model_cache["sd_components"] = None
    gc.collect()
    torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────
# VJEPA TRANSFORM & UTILS
# ─────────────────────────────────────────────────────────────────
# Crop mode is set at runtime via the UI; default is center-crop
# "center_crop" or "letterbox"
_current_crop_mode = "center_crop"

def _make_vjepa_transform(crop_mode: str = "center_crop"):
    """Return a torchvision transform for the chosen crop/pad strategy."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if crop_mode == "letterbox":
        def letterbox_transform(img: Image.Image) -> torch.Tensor:
            # Pad to square with black bars, then resize
            w, h = img.size
            max_side = max(w, h)
            padded = Image.new("RGB", (max_side, max_side), (0, 0, 0))
            padded.paste(img, ((max_side - w) // 2, (max_side - h) // 2))
            padded = padded.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            t = T.ToTensor()(padded)
            return normalize(t)
        return letterbox_transform
    else:
        transform = T.Compose([
            T.Resize(IMG_SIZE, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(IMG_SIZE),
            T.ToTensor(),
            normalize,
        ])
        return transform

vjepa_transform = _make_vjepa_transform("center_crop")

# ─────────────────────────────────────────────────────────────────
# ADAPTER ARCHITECTURE (must match training exactly)
# ─────────────────────────────────────────────────────────────────
class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, channels, eps=1e-6), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels, eps=1e-6), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
    def forward(self, x): return x + self.block(x)


class AdapterBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_res_blocks=2):
        super().__init__()
        self.in_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.resnets = nn.Sequential(*[ResnetBlock(out_channels) for _ in range(num_res_blocks)])
    def forward(self, x): return self.resnets(self.in_conv(x))


class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


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
        self.block0 = AdapterBlock(320,  320,  num_res_blocks)
        self.block1 = AdapterBlock(320,  640,  num_res_blocks)
        self.block2 = AdapterBlock(640,  1280, num_res_blocks)
        self.block3 = AdapterBlock(1280, 1280, num_res_blocks)
        self.down0  = DownsampleBlock(320)
        self.down1  = DownsampleBlock(640)
        self.down2  = DownsampleBlock(1280)

    def forward(self, jepa_emb):
        x  = self.projection(jepa_emb)
        f0 = self.block0(x);        x = self.down0(f0)
        f1 = self.block1(x);        x = self.down1(f1)
        f2 = self.block2(x);        x = self.down2(f2)
        f3 = self.block3(x)
        return [f0, f1, f2, f3]


# ─────────────────────────────────────────────────────────────────
# MODEL LOADING WITH CACHING
# ─────────────────────────────────────────────────────────────────
# Add to _model_cache initialization
_model_cache = {
    "vjepa_encoder": None,
    "vjepa_dtype": None,  # Track encoder's parameter dtype
    "adapter": None,
    "sd_components": None,
    "ckpt_hash": None,
}

# Update load_vjepa_cached to store dtype
def load_vjepa_cached():
    if _model_cache["vjepa_encoder"] is not None:
        # Move back to GPU if needed
        _model_cache["vjepa_encoder"].to(DEVICE)
        return _model_cache["vjepa_encoder"]
    
    print("Loading V-JEPA 2.1 encoder...")
    encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
    encoder.eval().to(DEVICE)
    
    # Store dtype for later consistency checks
    _model_cache["vjepa_dtype"] = next(encoder.parameters()).dtype
    
    for p in encoder.parameters():
        p.requires_grad_(False)
    _model_cache["vjepa_encoder"] = encoder
    return encoder


@torch.no_grad()
def encode_frame(pil_image: Image.Image) -> torch.Tensor:
    """Returns (574, 1024) float32 on CPU."""
    encoder = load_vjepa_cached()
    
    # Ensure encoder is on GPU and in correct dtype for inference
    encoder.eval().to(DEVICE)
    
    x = vjepa_transform(pil_image)  # (3, 384, 384)  — crop mode applied via global
    x = x.unsqueeze(0).unsqueeze(2).to(DEVICE)  # (1, 3, 1, 384, 384)
    
    # Auto-cast for efficiency, but ensure input dtype matches encoder
    input_dtype = next(encoder.parameters()).dtype
    x = x.to(input_dtype)
    
    with torch.autocast(DEVICE, enabled=input_dtype == torch.float16):
        emb = encoder(x)  # (1, 574, 1024)
    
    return emb.squeeze(0).float().cpu()

def encode_all_frames(frames: list[Image.Image]) -> list[torch.Tensor]:
    """Encode frames, returns list of (574,1024) CPU tensors."""
    encoder = load_vjepa_cached()
    encoder.eval().to(DEVICE)  # Keep on GPU during encoding
    
    embeddings = []
    print(f"Encoding {len(frames)} frame(s) with V-JEPA...")
    for frame in tqdm(frames, desc="VJEPA encode", unit="frame"):
        embeddings.append(encode_frame(frame))
    
    # Only offload after ALL frames are encoded
    encoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


SCHEDULER_OPTIONS = ["LCM+LoRA", "DDIM", "Euler A", "PNDM", "DPM++ 2M"]

def build_scheduler(name: str, config):
    """Instantiate the requested scheduler from a diffusers config dict."""
    from diffusers import (DDIMScheduler, LCMScheduler,
                           EulerAncestralDiscreteScheduler,
                           PNDMScheduler, DPMSolverMultistepScheduler)
    mapping = {
        "DDIM":       lambda c: DDIMScheduler.from_config(c),
        "LCM+LoRA":   lambda c: LCMScheduler.from_config(c),
        "Euler A":    lambda c: EulerAncestralDiscreteScheduler.from_config(c),
        "PNDM":       lambda c: PNDMScheduler.from_config(c),
        "DPM++ 2M":   lambda c: DPMSolverMultistepScheduler.from_config(c),
    }
    return mapping.get(name, mapping["DDIM"])(config)


def load_sd_components_cached(sd_ckpt_path: Path, scheduler_name: str = "DDIM"):
    """Load SD components with persistent caching. Scheduler is rebuilt on name change."""
    cached = _model_cache["sd_components"]
    ckpt_match = _model_cache["ckpt_hash"] == str(sd_ckpt_path)

    scheduler_match = _model_cache.get("scheduler_name") == scheduler_name

    if cached is not None and ckpt_match and scheduler_match:
        # Same checkpoint + same scheduler: just restore to GPU and reuse
        for key in ["vae", "unet", "text_enc"]:
            if cached[key]:
                cached[key].to(DEVICE)
        return cached

    if cached is not None and not scheduler_match:
        # Scheduler changed: full reload so LCM LoRA fusing etc. is always clean
        print(f"Scheduler changed ({_model_cache.get('scheduler_name')} -> {scheduler_name}), reloading SD weights...")
        clear_cache("sd")

    from diffusers import StableDiffusionPipeline

    print(f"Loading SD checkpoint: {sd_ckpt_path.name}...")
    pipe = StableDiffusionPipeline.from_single_file(
        str(sd_ckpt_path),
        torch_dtype=torch.float16,
        safety_checker=None,
        load_safety_checker=False,
    ).to(DEVICE)

    # For LCM+LoRA: fuse LoRA while we still have the full pipe object
    if scheduler_name == "LCM+LoRA":
        print("Loading LCM LoRA weights and fusing…")
        try:
            pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            pipe.fuse_lora()
            print("✓ LCM LoRA fused.")
        except Exception as e:
            print(f"⚠  LCM LoRA loading failed ({e}); continuing without LoRA.")

    scheduler = build_scheduler(scheduler_name, pipe.scheduler.config)
    vae, unet, tokenizer, text_enc = pipe.vae.eval(), pipe.unet.eval(), pipe.tokenizer, pipe.text_encoder.eval()

    for model in [vae, unet, text_enc]:
        for p in model.parameters():
            p.requires_grad_(False)

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    _model_cache["sd_components"] = {
        "vae": vae, "unet": unet, "tokenizer": tokenizer,
        "text_enc": text_enc, "scheduler": scheduler,
    }
    _model_cache["ckpt_hash"] = str(sd_ckpt_path)
    _model_cache["scheduler_name"] = scheduler_name
    print(f"✓ SD components loaded and cached. Scheduler: {scheduler_name}")
    return _model_cache["sd_components"]


def load_adapter_cached(adapter_ckpt_path: Path):
    """Load adapter weights with caching."""
    if _model_cache["adapter"] is not None:
        return _model_cache["adapter"]
    
    print(f"Loading adapter from {adapter_ckpt_path}...")
    adapter = JEPAAdapter(num_res_blocks=2).to(DEVICE)
    ckpt = torch.load(str(adapter_ckpt_path), map_location=DEVICE, weights_only=True)
    state = ckpt.get("adapter", ckpt)
    adapter.load_state_dict(state)
    adapter.eval()
    _model_cache["adapter"] = adapter
    print("✓ Adapter loaded and cached.")
    return adapter


# ─────────────────────────────────────────────────────────────────
# TEXT EMBEDDING
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_text_embedding(tokenizer, text_enc, prompt: str, device=DEVICE):
    tokens = tokenizer(
        [prompt], padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(device)
    return text_enc(tokens).last_hidden_state


# ─────────────────────────────────────────────────────────────────
# GENERATION
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_image(
    jepa_emb: torch.Tensor,
    adapter: JEPAAdapter,
    vae, unet, scheduler,
    text_emb: torch.Tensor,
    cfg_scale: float = 1.0,
    num_steps: int = 25,
    seed: int = 42,
    conditioning_scale: float = 1.0,
) -> Image.Image:
    scheduler.set_timesteps(num_steps)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latent = torch.randn(1, 4, 48, 48, dtype=torch.float16, device=DEVICE, generator=generator)  # → 384×384
    jepa_gpu = jepa_emb.unsqueeze(0).to(DEVICE)
    use_cfg = cfg_scale > 1.0
    
    if use_cfg:
        null_emb = torch.zeros_like(text_emb)

    for t in scheduler.timesteps:
        features = adapter(jepa_gpu)
        residuals = [f.to(dtype=torch.float16) * conditioning_scale for f in features]

        if use_cfg:
            noise_uncond = unet(latent, t.unsqueeze(0).to(DEVICE), null_emb,
                              down_intrablock_additional_residuals=residuals).sample
            noise_cond = unet(latent, t.unsqueeze(0).to(DEVICE), text_emb,
                            down_intrablock_additional_residuals=residuals).sample
            noise_pred = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(latent, t.unsqueeze(0).to(DEVICE), text_emb,
                            down_intrablock_additional_residuals=residuals).sample

        latent = scheduler.step(noise_pred, t, latent).prev_sample

    # Decode
    latent = latent / vae.config.scaling_factor
    img_tensor = vae.decode(latent).sample
    img_tensor = (img_tensor.float().clamp(-1, 1) + 1) / 2
    img_np = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ─────────────────────────────────────────────────────────────────
# VIDEO HELPERS WITH FPS SAMPLING
# ─────────────────────────────────────────────────────────────────
def extract_video_frames_sampled(video_path: str, target_fps: float) -> tuple[list[Image.Image], float, float]:
    """
    Extract frames from video with optional FPS downsampling.
    Returns: (frames, original_fps, effective_fps)
    """
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine sampling interval
    if target_fps <= 0 or target_fps >= orig_fps:
        sample_interval = 1
        effective_fps = orig_fps
    else:
        sample_interval = max(1, int(orig_fps / target_fps))
        effective_fps = orig_fps / sample_interval
    
    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += 1
    cap.release()
    
    print(f"  Extracted {len(frames)} frames (sampled at {effective_fps:.2f} fps from {orig_fps:.2f} fps)")
    return frames, orig_fps, effective_fps

def create_side_by_side_video_frames(orig_frames: list[Image.Image], 
                                     gen_frames: list[Image.Image]) -> list[Image.Image]:
    """Create side-by-side frames for video output."""
    result = []
    for orig, gen in zip(orig_frames, gen_frames):
        side_by_side = create_side_by_side(orig, gen)
        result.append(side_by_side)
    return result

def create_side_by_side(original: Image.Image, generated: Image.Image, 
                        labels: tuple = ("Original", "Generated"),
                        gap: int = 4, label_height: int = 30) -> Image.Image:
    """Create horizontal side-by-side comparison with guaranteed even dimensions."""
    # Match heights to original/generated max
    h = max(original.height, generated.height)
    
    # Resize proportionally using round() to avoid truncation artifacts
    orig_resized = original.resize((round(original.width * h / original.height), h))
    gen_resized  = generated.resize((round(generated.width * h / generated.height), h))
    
    total_width  = orig_resized.width + gen_resized.width + gap
    total_height = h + label_height
    
    # 🔑 H.264/libx264 requires dimensions divisible by 2
    if total_width % 2 != 0:
        total_width += 1
    if total_height % 2 != 0:
        total_height += 1
        
    # Create dark canvas
    result = Image.new("RGB", (total_width, total_height), color=(30, 30, 30))
    
    # Draw labels
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    draw.text((10, h + 5), labels[0], fill=(200, 200, 200), font=font)
    draw.text((orig_resized.width + gap + 10, h + 5), labels[1], fill=(200, 200, 200), font=font)
    
    # Paste images (extra 1px padding just shows as dark background)
    result.paste(orig_resized, (0, 0))
    result.paste(gen_resized, (orig_resized.width + gap, 0))
    
    return result


def frames_to_video(frames: list[Image.Image], output_path: str, fps: float, 
                    progress_callback=None):
    """Write PIL frames to mp4 using ffmpeg with robust error handling."""
    tmp = tempfile.mkdtemp()
    try:
        # Save frames
        for i, frame in enumerate(tqdm(frames, desc="Saving frames", unit="frame")):
            frame.save(os.path.join(tmp, f"frame_{i:06d}.png"))
            if progress_callback and i % 10 == 0:
                progress_callback(f"Saved {i+1}/{len(frames)} frames")
        
        # Round fps to 3 decimals to avoid ffmpeg parsing issues
        fps_rounded = round(fps, 3)
        
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",  # Only show errors
            "-framerate", str(fps_rounded),
            "-i", os.path.join(tmp, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", 
            "-preset", "medium", "-crf", "23",
            output_path,
        ]
        
        if progress_callback:
            progress_callback(f"Encoding video @ {fps_rounded} fps...")
            
        # Run with full error capture
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            check=False  # Don't auto-raise, we'll handle errors
        )
        
        if result.returncode != 0:
            error_msg = result.stderr.strip() or "Unknown ffmpeg error"
            # Common fixes to suggest
            if "Invalid framerate" in error_msg:
                error_msg += "\n💡 Try setting Video Sample FPS to a whole number (1, 2, 5, etc.)"
            elif "libx264" in error_msg:
                error_msg += "\n💡 Ensure ffmpeg was compiled with libx264 support"
            raise RuntimeError(f"ffmpeg failed: {error_msg}")
            
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_pipeline(
    input_path: str,
    prompt: str = "",
    cfg_scale: float = 1.0,
    num_steps: int = 25,
    seed: int = 42,
    conditioning_scale: float = 1.0,
    video_sample_fps: float = 0,  # 0 = use original fps
    crop_mode: str = "center_crop",   # "center_crop" | "letterbox"
    scheduler_name: str = "DDIM",
    log_fn=print,
    progress_fn=None,
):
    input_path = Path(input_path)
    is_video = input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # ── 0. Apply crop mode globally ──────────────────────────────
    global vjepa_transform, _current_crop_mode
    if crop_mode != _current_crop_mode:
        log_fn(f"🔄 Crop mode → {crop_mode}")
        vjepa_transform = _make_vjepa_transform(crop_mode)
        _current_crop_mode = crop_mode
    log_fn(f"🖼  Crop mode: {crop_mode} | Scheduler: {scheduler_name}")

    # ── 1. Load/extract frames ───────────────────────────────────
    if is_video:
        log_fn("📹 Extracting video frames...")
        frames, orig_fps, eff_fps = extract_video_frames_sampled(str(input_path), video_sample_fps)
        log_fn(f"  ✓ {len(frames)} frames at {eff_fps:.2f} fps (original: {orig_fps:.2f} fps)")
    else:
        frames = [Image.open(input_path).convert("RGB")]
        orig_fps = eff_fps = None
        log_fn(f"  ✓ Loaded image: {input_path.name}")

    if progress_fn:
        progress_fn(0.1, "Encoding with V-JEPA...")

    # ── 2. Encode all frames ─────────────────────────────────────
    embeddings = encode_all_frames(frames)
    
    if progress_fn:
        progress_fn(0.3, "Loading models...")

    # ── 3. Load cached models ────────────────────────────────────
    sd = load_sd_components_cached(SD_CKPT, scheduler_name)
    adapter = load_adapter_cached(ADAPTER_CKPT)
    vae, unet, tokenizer, text_enc, scheduler = [sd[k] for k in ["vae", "unet", "tokenizer", "text_enc", "scheduler"]]

    # ── 4. Text embedding ────────────────────────────────────────
    use_prompt = prompt.strip() != ""
    effective_prompt = prompt.strip() if use_prompt else ""
    log_fn(f"📝 Prompt: «{effective_prompt}»" if use_prompt else "📝 Prompt: (null conditioning)")
    log_fn(f"⚙️  CFG: {cfg_scale} | Steps: {num_steps} | Seed: {seed}")
    
    text_emb = get_text_embedding(tokenizer, text_enc, effective_prompt)

    # ── 5. Generate frames ───────────────────────────────────────
    generated = []
    log_fn(f"🎨 Generating {len(embeddings)} frame(s)...")
    
    for i, emb in enumerate(tqdm(embeddings, desc="Generating", unit="frame")):
        img = generate_image(
            jepa_emb=emb, adapter=adapter, vae=vae, unet=unet, scheduler=scheduler,
            text_emb=text_emb, cfg_scale=cfg_scale, num_steps=num_steps,
            seed=seed, conditioning_scale=conditioning_scale,
        )
        generated.append(img)
        
        if progress_fn:
            progress = 0.3 + (0.6 * (i + 1) / len(embeddings))
            progress_fn(progress, f"Generated {i+1}/{len(embeddings)} frames")
        
        if (i + 1) % 10 == 0:
            log_fn(f"  → {i+1}/{len(embeddings)} frames done")

    if progress_fn:
        progress_fn(0.95, "Creating side-by-side output...")

    # ── 6. Create side-by-side outputs ───────────────────────────
    if is_video:
        side_by_side_frames = create_side_by_side_video_frames(frames, generated)
        stem = input_path.stem
        out_path = str(OUTPUT_DIR / f"{stem}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        log_fn(f"🎬 Assembling comparison video → {out_path}")
        frames_to_video(side_by_side_frames, out_path, eff_fps, progress_callback=log_fn)
    else:
        side_by_side = create_side_by_side(frames[0], generated[0])
        stem = input_path.stem
        out_path = str(OUTPUT_DIR / f"{stem}_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        side_by_side.save(out_path)
        log_fn(f"💾 Saved comparison image → {out_path}")

    if progress_fn:
        progress_fn(1.0, "Done!")
    
    log_fn(f"✅ Complete! Output: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# MODERN TKINTER UI
# ─────────────────────────────────────────────────────────────────
class ModernApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎨 JEPA Adapter Studio")
        self.geometry("720x880")
        self.minsize(720, 880)
        self.configure(bg="#2b2b2b")
        self._style_widgets()
        self._build_ui()
        self._is_generating = False

    def _style_widgets(self):
        """Apply modern ttk styling."""
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure colors
        style.configure("TFrame", background="#2b2b2b")
        style.configure("TLabel", background="#2b2b2b", foreground="#e0e0e0", font=("Segoe UI", 9))
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#7ec8e3")
        style.configure("Status.TLabel", font=("Consolas", 8), foreground="#888888")
        
        # Progress bar
        style.configure("TProgressbar", thickness=20, troughcolor="#404040", background="#5aa9e6")

    def _build_ui(self):
        """Build the modern UI layout."""
        main_frame = ttk.Frame(self, padding="12")
        main_frame.pack(fill="both", expand=True)
        
        # ── Header ──────────────────────────────────────────────
        header = ttk.Label(main_frame, text="🎨 JEPA T2I-Adapter Studio", style="Header.TLabel")
        header.pack(pady=(0, 12))
        
        # ── File Selection Card ─────────────────────────────────
        file_card = ttk.LabelFrame(main_frame, text="📁 Input Source", padding=10)
        file_card.pack(fill="x", pady=(0, 8))
        
        file_frame = ttk.Frame(file_card)
        file_frame.pack(fill="x")
        
        self.file_var = tk.StringVar(value="No file selected")
        file_entry = ttk.Entry(file_frame, textvariable=self.file_var, state="readonly", width=50)
        file_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        
        browse_btn = ttk.Button(file_frame, text="Browse…", command=self._browse, width=12)
        browse_btn.pack(side="right")
        
        # File info label
        self.file_info = ttk.Label(file_card, text="", style="Status.TLabel", wraplength=650)
        self.file_info.pack(fill="x", pady=(4, 0))

        # ── Settings Card ───────────────────────────────────────
        settings_card = ttk.LabelFrame(main_frame, text="⚙️ Generation Settings", padding=10)
        settings_card.pack(fill="x", pady=(0, 8))
        
        # Grid layout for settings
        settings_grid = ttk.Frame(settings_card)
        settings_grid.pack(fill="x")
        
        # Row 0: CFG & Steps
        ttk.Label(settings_grid, text="CFG Scale:").grid(row=0, column=0, sticky="w", pady=2)
        self.cfg_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(settings_grid, from_=1.0, to=15.0, increment=0.5,
                   textvariable=self.cfg_var, width=8).grid(row=0, column=1, sticky="w", padx=(0, 20))
        
        ttk.Label(settings_grid, text="Steps:").grid(row=0, column=2, sticky="w")
        self.steps_var = tk.IntVar(value=4)
        ttk.Spinbox(settings_grid, from_=1, to=100, increment=1,
                   textvariable=self.steps_var, width=6).grid(row=0, column=3, sticky="w", padx=(0, 20))
        
        # Row 1: Seed & Adapter Scale
        ttk.Label(settings_grid, text="Seed:").grid(row=1, column=0, sticky="w", pady=2)
        self.seed_var = tk.IntVar(value=42)
        ttk.Spinbox(settings_grid, from_=0, to=2**31-1, increment=1,
                   textvariable=self.seed_var, width=12).grid(row=1, column=1, sticky="w", padx=(0, 20))
        
        ttk.Label(settings_grid, text="Adapter Scale:").grid(row=1, column=2, sticky="w")
        self.cond_scale_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(settings_grid, from_=0.1, to=2.0, increment=0.1,
                   textvariable=self.cond_scale_var, width=6).grid(row=1, column=3, sticky="w")
        
        # Row 2: Video FPS Sampling (only relevant for videos)
        ttk.Label(settings_grid, text="Video Sample FPS:").grid(row=2, column=0, sticky="w", pady=(8, 2))
        self.fps_var = tk.DoubleVar(value=5.0)
        # Fixed: removed invalid 'tooltip' parameter
        fps_spin = ttk.Spinbox(settings_grid, from_=0, to=60, increment=1,
                            textvariable=self.fps_var, width=8)
        fps_spin.grid(row=2, column=1, sticky="w", padx=(0, 20))
        # Added helper label instead of tooltip
        ttk.Label(settings_grid, text="(0=original, 1+=downsample)", 
                style="Status.TLabel", foreground="#aaaaaa").grid(row=2, column=2, columnspan=2, sticky="w")

        # Row 3: Crop Mode toggle
        ttk.Label(settings_grid, text="Crop Mode:").grid(row=3, column=0, sticky="w", pady=(8, 2))
        self.crop_mode_var = tk.StringVar(value="center_crop")
        crop_frame = ttk.Frame(settings_grid)
        crop_frame.grid(row=3, column=1, columnspan=3, sticky="w", pady=(8, 2))
        self._crop_btn_center = ttk.Button(
            crop_frame, text="⬛ Center Crop", width=14,
            command=lambda: self._set_crop_mode("center_crop"))
        self._crop_btn_center.pack(side="left", padx=(0, 4))
        self._crop_btn_letter = ttk.Button(
            crop_frame, text="⬜ Letterbox", width=14,
            command=lambda: self._set_crop_mode("letterbox"))
        self._crop_btn_letter.pack(side="left")
        self._crop_mode_label = ttk.Label(crop_frame, text="● Center Crop active",
                                          style="Status.TLabel", foreground="#5aa9e6")
        self._crop_mode_label.pack(side="left", padx=(10, 0))

        # Row 4: Scheduler selector
        ttk.Label(settings_grid, text="Scheduler:").grid(row=4, column=0, sticky="w", pady=(6, 2))
        self.scheduler_var = tk.StringVar(value="LCM+LoRA")
        scheduler_combo = ttk.Combobox(
            settings_grid, textvariable=self.scheduler_var,
            values=SCHEDULER_OPTIONS, state="readonly", width=16)
        scheduler_combo.grid(row=4, column=1, sticky="w", pady=(6, 2))
        ttk.Label(settings_grid, text="⚠ Cache cleared on scheduler switch to LCM+LoRA",
                  style="Status.TLabel", foreground="#aaaaaa").grid(
                  row=4, column=2, columnspan=2, sticky="w")

        # ── Progress Section ────────────────────────────────────
        progress_card = ttk.Frame(main_frame)
        progress_card.pack(fill="x", pady=(8, 4))
        
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(progress_card, variable=self.progress_var, 
                                           maximum=100.0, mode="determinate")
        self.progress_bar.pack(fill="x")
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_card, textvariable=self.status_var, style="Status.TLabel")
        status_label.pack(pady=(4, 0))

        # ── Action Buttons ──────────────────────────────────────
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", pady=(12, 8))
        
        self.run_btn = ttk.Button(btn_frame, text="▶ Generate", command=self._start_generation, 
                                 style="Accent.TButton" if hasattr(ttk.Style(), "Accent.TButton") else "TButton")
        self.run_btn.pack(side="left", padx=(0, 8))
        
        ttk.Button(btn_frame, text="🗑 Clear Cache", command=self._clear_cache).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="📂 Open Output", command=self._open_output).pack(side="left")

        # ── Log Panel ───────────────────────────────────────────
        log_card = ttk.LabelFrame(main_frame, text="📋 Log", padding=8)
        log_card.pack(fill="both", expand=True, pady=(4, 0))
        
        self.log_text = tk.Text(log_card, height=10, state="disabled", wrap="word",
                               bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9),
                               insertbackground="#d4d4d4")
        log_scroll = ttk.Scrollbar(log_card, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        log_scroll.pack(side="right", fill="y")
        self.log_text.pack(fill="both", expand=True)
        
        # Bind Ctrl+A for select all in log
        self.log_text.bind("<Control-a>", lambda e: self.log_text.tag_add("sel", "1.0", "end"))

    def _set_crop_mode(self, mode: str):
        """Toggle crop mode and update button labels."""
        self.crop_mode_var.set(mode)
        if mode == "center_crop":
            self._crop_mode_label.config(text="● Center Crop active", foreground="#5aa9e6")
        else:
            self._crop_mode_label.config(text="● Letterbox active", foreground="#e6a85a")

    def _browse(self):
        """Open file browser."""
        path = filedialog.askopenfilename(
            title="Select image or video",
            filetypes=[
                ("Images & Videos", "*.png *.jpg *.jpeg *.bmp *.webp *.mp4 *.avi *.mov *.mkv *.webm"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.file_var.set(path)
            # Update file info
            p = Path(path)
            size_mb = p.stat().st_size / (1024 * 1024)
            info = f"📄 {p.name} • {size_mb:.1f} MB"
            if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                cap = cv2.VideoCapture(path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                cap.release()
                info += f" • 🎬 {frames} frames @ {fps:.1f}fps ({duration:.1f}s)"
            self.file_info.config(text=info)

    def _log(self, msg: str):
        """Add message to log with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"[{timestamp}] {msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        self.update_idletasks()

    def _update_progress(self, progress: float, status: str = None):
        """Update progress bar and status (0.0 to 1.0)."""
        self.progress_var.set(progress * 100)
        if status:
            self.status_var.set(status)
        self.update_idletasks()

    def _clear_cache(self):
        """Clear all cached models."""
        if messagebox.askyesno("Clear Cache", "Clear all cached models from VRAM?"):
            clear_cache()
            self._log("🗑 Model cache cleared")
            self.status_var.set("Cache cleared")

    def _open_output(self):
        """Open output directory in file explorer."""
        if OUTPUT_DIR.exists():
            if sys.platform == "win32":
                os.startfile(str(OUTPUT_DIR))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(OUTPUT_DIR)])
            else:
                subprocess.run(["xdg-open", str(OUTPUT_DIR)])

    def _start_generation(self):
        """Start generation in background thread."""
        if self._is_generating:
            return
            
        path = self.file_var.get()
        if path == "No file selected" or not os.path.exists(path):
            messagebox.showerror("Error", "Please select a valid input file first.")
            return

        self._is_generating = True
        self.run_btn.configure(state="disabled", text="⏳ Generating…")
        self._log("=" * 60)
        self._update_progress(0, "Initializing…")

        def worker():
            try:
                out = run_pipeline(
                    input_path=path,
                    prompt="",
                    cfg_scale=self.cfg_var.get(),
                    num_steps=self.steps_var.get(),
                    seed=self.seed_var.get(),
                    conditioning_scale=self.cond_scale_var.get(),
                    video_sample_fps=self.fps_var.get(),
                    crop_mode=self.crop_mode_var.get(),
                    scheduler_name=self.scheduler_var.get(),
                    log_fn=self._log,
                    progress_fn=self._update_progress,
                )
                self._log(f"\n✅ Success! Output: {out}")

                # Automatically open output
                if sys.platform == "win32":
                    os.startfile(out)
                elif sys.platform == "darwin":
                    subprocess.run(["open", out])
                else:
                    subprocess.run(["xdg-open", out])

            except Exception as e:
                import traceback
                self._log(f"\n❌ Error: {e}")
                self._log(traceback.format_exc())
                messagebox.showerror("Error", str(e))
            finally:
                self._is_generating = False
                self.run_btn.configure(state="normal", text="▶ Generate")
                self._update_progress(0, "Ready")

        threading.Thread(target=worker, daemon=True).start()


# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ModernApp()
    app.mainloop()