"""
JEPA T2I-Adapter Inference Script - Gradio Edition
---------------------------------------------------
Image → V-JEPA 2.1 encode → JEPAAdapter → SD1.5 generation
Video → per-frame encode → generate → ffmpeg reassemble

Features:
  • Side-by-side comparison outputs (original | generated)
  • Configurable video frame sampling FPS
  • Persistent SD 1.5 model caching in VRAM
  • Gradio web UI with progress tracking & live logging
  • GPU-optimized for 4GB VRAM (1050 Ti compatible)
"""

import os
import gc
import shutil
import tempfile
import subprocess
import threading
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
import gradio as gr

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION – Edit these paths to match your setup
# ─────────────────────────────────────────────────────────────────
SD_CKPT = Path(
    r"D:\AI\sd.webui\webui\models\Stable-diffusion"
    r"\realisticVisionV60B1_v51HyperVAE.safetensors"
)
ADAPTER_CKPT = Path("vjepa_t2i_adapter_sd15_step4550.pt")
OUTPUT_DIR   = Path("experiments/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE    = 384
MAX_VRAM_GB = 4.0

# ─────────────────────────────────────────────────────────────────
# GLOBAL MODEL CACHE (persistent across generations)
# ─────────────────────────────────────────────────────────────────
_model_cache = {
    "vjepa_encoder": None,
    "vjepa_dtype":   None,
    "adapter":       None,
    "sd_components": None,
    "ckpt_hash":     None,
}


def clear_cache(component: str = None):
    """Clear specific or all cached models to free VRAM."""
    global _model_cache
    if component in (None, "vjepa"):
        if _model_cache["vjepa_encoder"] is not None:
            _model_cache["vjepa_encoder"].cpu()
            del _model_cache["vjepa_encoder"]
            _model_cache["vjepa_encoder"] = None
            _model_cache["vjepa_dtype"]   = None
    if component in (None, "adapter"):
        if _model_cache["adapter"] is not None:
            del _model_cache["adapter"]
            _model_cache["adapter"] = None
    if component in (None, "sd"):
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
def letterbox_image(img, size, fill=(0, 0, 0)):
    """Resize while preserving aspect ratio, pad to square."""
    if not isinstance(img, Image.Image):
        img = T.ToPILImage()(img)
    w, h   = img.size
    scale  = min(size / w, size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    canvas  = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


vjepa_transform = T.Compose([
    T.Lambda(lambda img: letterbox_image(img, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


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
def load_vjepa_cached():
    if _model_cache["vjepa_encoder"] is not None:
        _model_cache["vjepa_encoder"].to(DEVICE)
        return _model_cache["vjepa_encoder"]

    print("Loading V-JEPA 2.1 encoder...")
    encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
    encoder.eval().to(DEVICE)
    _model_cache["vjepa_dtype"] = next(encoder.parameters()).dtype
    for p in encoder.parameters():
        p.requires_grad_(False)
    _model_cache["vjepa_encoder"] = encoder
    return encoder


@torch.no_grad()
def encode_frame(pil_image: Image.Image) -> torch.Tensor:
    """Returns (574, 1024) float32 on CPU."""
    encoder     = load_vjepa_cached()
    encoder.eval().to(DEVICE)
    x           = vjepa_transform(pil_image)
    x           = x.unsqueeze(0).unsqueeze(2).to(DEVICE)
    input_dtype = next(encoder.parameters()).dtype
    x           = x.to(input_dtype)
    with torch.autocast(DEVICE, enabled=input_dtype == torch.float16):
        emb = encoder(x)
    return emb.squeeze(0).float().cpu()


def encode_all_frames(frames: list, log_fn=print) -> list:
    """Encode frames; returns list of (574,1024) CPU tensors."""
    encoder = load_vjepa_cached()
    encoder.eval().to(DEVICE)
    embeddings = []
    log_fn(f"Encoding {len(frames)} frame(s) with V-JEPA...")
    for frame in tqdm(frames, desc="VJEPA encode", unit="frame"):
        embeddings.append(encode_frame(frame))
    # Offload only after all frames are done
    encoder.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    return embeddings


def load_sd_components_cached(sd_ckpt_path: Path, log_fn=print):
    if (_model_cache["sd_components"] is not None and
            _model_cache["ckpt_hash"] == str(sd_ckpt_path)):
        for key in ["vae", "unet", "text_enc"]:
            if _model_cache["sd_components"][key]:
                _model_cache["sd_components"][key].to(DEVICE)
        return _model_cache["sd_components"]

    from diffusers import StableDiffusionPipeline, LCMScheduler

    log_fn(f"Loading SD checkpoint: {sd_ckpt_path.name}...")
    pipe = StableDiffusionPipeline.from_single_file(
        str(sd_ckpt_path),
        torch_dtype=torch.float16,
        safety_checker=None,
        load_safety_checker=False,
    ).to(DEVICE)

    scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    vae, unet, tokenizer, text_enc = (
        pipe.vae.eval(), pipe.unet.eval(), pipe.tokenizer, pipe.text_encoder.eval()
    )
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
    log_fn("✓ SD components loaded and cached.")
    return _model_cache["sd_components"]


def load_adapter_cached(adapter_ckpt_path: Path, log_fn=print):
    if _model_cache["adapter"] is not None:
        return _model_cache["adapter"]

    log_fn(f"Loading adapter from {adapter_ckpt_path}...")
    adapter = JEPAAdapter(num_res_blocks=2).to(DEVICE)
    ckpt    = torch.load(str(adapter_ckpt_path), map_location=DEVICE, weights_only=True)
    state   = ckpt.get("adapter", ckpt)
    adapter.load_state_dict(state)
    adapter.eval()
    _model_cache["adapter"] = adapter
    log_fn("✓ Adapter loaded and cached.")
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
    num_steps: int = 4,
    seed: int = 42,
    conditioning_scale: float = 1.0,
) -> Image.Image:
    scheduler.set_timesteps(num_steps)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    latent    = torch.randn(1, 4, 48, 48, dtype=torch.float16, device=DEVICE, generator=generator)
    jepa_gpu  = jepa_emb.unsqueeze(0).to(DEVICE)
    use_cfg   = cfg_scale > 1.0

    if use_cfg:
        null_emb = torch.zeros_like(text_emb)

    for t in scheduler.timesteps:
        features  = adapter(jepa_gpu)
        residuals = [f.to(dtype=torch.float16) * conditioning_scale for f in features]

        if use_cfg:
            noise_uncond = unet(latent, t.unsqueeze(0).to(DEVICE), null_emb,
                                down_intrablock_additional_residuals=residuals).sample
            noise_cond   = unet(latent, t.unsqueeze(0).to(DEVICE), text_emb,
                                down_intrablock_additional_residuals=residuals).sample
            noise_pred   = noise_uncond + cfg_scale * (noise_cond - noise_uncond)
        else:
            noise_pred = unet(latent, t.unsqueeze(0).to(DEVICE), text_emb,
                              down_intrablock_additional_residuals=residuals).sample

        latent = scheduler.step(noise_pred, t, latent).prev_sample

    latent     = latent / vae.config.scaling_factor
    img_tensor = vae.decode(latent).sample
    img_tensor = (img_tensor.float().clamp(-1, 1) + 1) / 2
    img_np     = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_np)


# ─────────────────────────────────────────────────────────────────
# VIDEO HELPERS
# ─────────────────────────────────────────────────────────────────
def extract_video_frames_sampled(video_path: str, target_fps: float):
    """Returns (frames, orig_fps, effective_fps)."""
    cap         = cv2.VideoCapture(video_path)
    orig_fps    = cap.get(cv2.CAP_PROP_FPS) or 24.0
    sample_interval = max(1, int(orig_fps / target_fps)) if 0 < target_fps < orig_fps else 1
    effective_fps   = orig_fps / sample_interval
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_interval == 0:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        idx += 1
    cap.release()
    return frames, orig_fps, effective_fps


def create_side_by_side(original: Image.Image, generated: Image.Image,
                        labels=("Original", "Generated"),
                        gap: int = 4, label_height: int = 30) -> Image.Image:
    """Horizontal comparison image with even dimensions for H.264."""
    h           = max(original.height, generated.height)
    orig_r      = original.resize((round(original.width * h / original.height), h))
    gen_r       = generated.resize((round(generated.width * h / generated.height), h))
    total_w     = orig_r.width + gen_r.width + gap
    total_h     = h + label_height
    if total_w % 2: total_w += 1
    if total_h % 2: total_h += 1

    result = Image.new("RGB", (total_w, total_h), color=(30, 30, 30))
    draw   = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    draw.text((10, h + 5), labels[0], fill=(200, 200, 200), font=font)
    draw.text((orig_r.width + gap + 10, h + 5), labels[1], fill=(200, 200, 200), font=font)
    result.paste(orig_r, (0, 0))
    result.paste(gen_r,  (orig_r.width + gap, 0))
    return result


def frames_to_video(frames: list, output_path: str, fps: float, log_fn=print):
    tmp = tempfile.mkdtemp()
    try:
        for i, frame in enumerate(frames):
            frame.save(os.path.join(tmp, f"frame_{i:06d}.png"))
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-framerate", str(round(fps, 3)),
            "-i", os.path.join(tmp, "frame_%06d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-preset", "medium", "-crf", "23",
            output_path,
        ]
        log_fn(f"Encoding video @ {round(fps, 3)} fps...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()}")
    finally:
        shutil.rmtree(tmp)


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────
def run_pipeline(
    input_path: str,
    prompt: str              = "",
    cfg_scale: float         = 1.0,
    num_steps: int           = 4,
    seed: int                = 42,
    conditioning_scale: float = 1.0,
    video_sample_fps: float  = 0,
    log_fn                   = print,
    progress_fn              = None,   # gr.Progress() instance or None
):
    input_path = Path(input_path)
    is_video   = input_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    # ── 1. Load / extract frames ─────────────────────────────────
    if is_video:
        log_fn("📹 Extracting video frames...")
        frames, orig_fps, eff_fps = extract_video_frames_sampled(str(input_path), video_sample_fps)
        log_fn(f"  ✓ {len(frames)} frames at {eff_fps:.2f} fps (original: {orig_fps:.2f} fps)")
    else:
        frames   = [Image.open(input_path).convert("RGB")]
        orig_fps = eff_fps = None
        log_fn(f"  ✓ Loaded image: {input_path.name}")

    if progress_fn: progress_fn(0.05, desc="Encoding with V-JEPA...")

    # ── 2. Encode all frames ─────────────────────────────────────
    embeddings = encode_all_frames(frames, log_fn=log_fn)

    if progress_fn: progress_fn(0.25, desc="Loading models...")

    # ── 3. Load cached models ────────────────────────────────────
    sd      = load_sd_components_cached(SD_CKPT, log_fn=log_fn)
    adapter = load_adapter_cached(ADAPTER_CKPT, log_fn=log_fn)
    vae, unet, tokenizer, text_enc, scheduler = [
        sd[k] for k in ["vae", "unet", "tokenizer", "text_enc", "scheduler"]
    ]

    # ── 4. Text embedding ────────────────────────────────────────
    effective_prompt = prompt.strip()
    log_fn(f"📝 Prompt: «{effective_prompt}»" if effective_prompt else "📝 Prompt: (null conditioning)")
    log_fn(f"⚙️  CFG: {cfg_scale} | Steps: {num_steps} | Seed: {seed} | Adapter scale: {conditioning_scale}")
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
            progress_fn(0.25 + 0.65 * (i + 1) / len(embeddings),
                        desc=f"Generated {i+1}/{len(embeddings)} frames")
        if (i + 1) % 10 == 0:
            log_fn(f"  → {i+1}/{len(embeddings)} frames done")

    if progress_fn: progress_fn(0.92, desc="Creating side-by-side output...")

    # ── 6. Save side-by-side output ──────────────────────────────
    stem      = input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if is_video:
        sbs_frames = [create_side_by_side(o, g) for o, g in zip(frames, generated)]
        out_path   = str(OUTPUT_DIR / f"{stem}_comparison_{timestamp}.mp4")
        log_fn(f"🎬 Assembling comparison video → {out_path}")
        frames_to_video(sbs_frames, out_path, eff_fps, log_fn=log_fn)
    else:
        sbs      = create_side_by_side(frames[0], generated[0])
        out_path = str(OUTPUT_DIR / f"{stem}_comparison_{timestamp}.png")
        sbs.save(out_path)
        log_fn(f"💾 Saved comparison image → {out_path}")

    if progress_fn: progress_fn(1.0, desc="Done!")
    log_fn(f"✅ Complete! Output: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────────────────────────
def gradio_run(
    mode,
    input_image,
    input_video,
    prompt,
    cfg_scale,
    num_steps,
    seed,
    cond_scale,
    video_fps,
    progress=gr.Progress(track_tqdm=True),
):
    input_file = input_image if mode == "Image" else input_video
    
    if not input_file:
        raise gr.Error("Please upload an image or video first.")

    log_lines = []
    def capture_log(msg): log_lines.append(msg)
    def progress_fn(val, desc=""): progress(val, desc=desc)

    try:
        out_path = run_pipeline(
            input_path=input_file,
            prompt=prompt,
            cfg_scale=cfg_scale,
            num_steps=int(num_steps),
            seed=int(seed),
            conditioning_scale=cond_scale,
            video_sample_fps=video_fps,
            log_fn=capture_log,
            progress_fn=progress_fn,
        )
    except Exception as e:
        import traceback
        log_lines.append(f"\n❌ Error: {e}")
        log_lines.append(traceback.format_exc())
        return None, None, "\n".join(log_lines)

    is_video = Path(out_path).suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"}
    if is_video:
        return None, out_path, "\n".join(log_lines)
    else:
        return out_path, None, "\n".join(log_lines)


def gradio_clear_cache():
    clear_cache()
    return "🗑 All models cleared from VRAM."


def build_ui():
    with gr.Blocks(title="JEPA Adapter Studio", theme=gr.themes.Default()) as demo:
        gr.Markdown("# 🎨 JEPA T2I-Adapter Studio")
        gr.Markdown("Image/video → V-JEPA 2.1 encode → JEPAAdapter → SD 1.5")

        with gr.Row():
            # ── Left column ──────────────────────────────
            with gr.Column(scale=1):
                mode = gr.Radio(
                    ["Image", "Video"], 
                    value="Image", 
                    label="Input Type",
                    info="Switches preview widgets"
                )
                
                input_image = gr.Image(
                    type="filepath",
                    label="Input image",
                    visible=True
                )
                input_video = gr.Video(
                    label="Input video",
                    visible=False
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Leave empty for null conditioning",
                    lines=2,
                )

                with gr.Group():
                    gr.Markdown("### Generation settings")
                    with gr.Row():
                        cfg_scale = gr.Slider(1.0, 15.0, value=1.0, step=0.5, label="CFG Scale")
                        num_steps = gr.Slider(1, 50, value=4, step=1, label="Steps")
                    with gr.Row():
                        seed = gr.Number(value=42, label="Seed", precision=0)
                        cond_scale = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Adapter Scale")
                    video_fps = gr.Slider(0, 60, value=0, step=1, label="Video Sample FPS", info="0 = original")

                with gr.Row():
                    run_btn = gr.Button("▶  Generate", variant="primary", scale=3)
                    clear_btn = gr.Button("🗑  Clear VRAM", scale=1)

            # ── Right column ─────────────────────────────
            with gr.Column(scale=1):
                output_image = gr.Image(label="Output comparison (image)", interactive=False)
                output_video = gr.Video(label="Output comparison (video)", interactive=False, visible=False)
                
                log_box = gr.Textbox(label="Log", lines=20, max_lines=40, interactive=False)
                cache_status = gr.Textbox(label="Cache status", interactive=False, lines=1)

        # toggle visibility
        def toggle_mode(m):
            is_img = m == "Image"
            return (
                gr.update(visible=is_img),      # input_image
                gr.update(visible=not is_img),  # input_video
                gr.update(visible=is_img),      # output_image
                gr.update(visible=not is_img),  # output_video
            )
        
        mode.change(toggle_mode, inputs=mode, outputs=[input_image, input_video, output_image, output_video])

        # wiring
        run_btn.click(
            fn=gradio_run,
            inputs=[mode, input_image, input_video, prompt, cfg_scale, num_steps, seed, cond_scale, video_fps],
            outputs=[output_image, output_video, log_box],
        )
        clear_btn.click(fn=gradio_clear_cache, inputs=[], outputs=[cache_status])

    return demo

# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,       # auto-opens browser tab
        share=False,          # set True to get a public gradio.live URL
    )