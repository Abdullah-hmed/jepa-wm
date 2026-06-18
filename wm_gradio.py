import os
import time
import shutil
import tempfile
import subprocess

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# =====================================================================
# CONFIG
# =====================================================================

ENC_DIM    = 1024
N_PATCHES  = 144
PATCH_H    = 12
PATCH_W    = 12
PRED_DIM   = 1024
PRED_DEPTH = 20
PRED_HEADS = 16
ACTION_DIM = 2
MAX_SEQ    = 48

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Model source: local files now, HF Hub later ----
# Flip this once predictor_vjepa21.pt / decoder_vjepa21.pt are uploaded
# to your repos, and fill in the repo IDs. resolve_checkpoint() below
# handles the rest.
USE_HF_HUB = False

HF_PREDICTOR_REPO = "abdullah-hmed/streetwm-vjepa21"
HF_DECODER_REPO   = "abdullah-hmed/streetwm-vjepa21"
PREDICTOR_FILENAME = "checkpoints/best.pt"
DECODER_FILENAME   = "decoder_vjepa21.pt"

LOCAL_PREDICTOR_CKPT = "predictor_vjepa21.pt"
LOCAL_DECODER_CKPT   = "decoder_vjepa21.pt"


def resolve_checkpoint(local_path, repo_id, filename):
    """Returns a local filesystem path to the checkpoint, either from
    disk or freshly downloaded from the Hub, depending on USE_HF_HUB."""
    if USE_HF_HUB:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename)
    return local_path


# =====================================================================
# PREDICTOR  (unchanged from the Tkinter version — no GUI dependency)
# =====================================================================

def get_1d_sincos(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (embed_dim / 2)))
    pos   = pos.reshape(-1)
    out   = torch.einsum('m,d->md', pos, omega)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def build_2d_sincos_pe(embed_dim, h, w):
    assert embed_dim % 4 == 0
    half  = embed_dim // 2
    emb_y = get_1d_sincos(half, torch.arange(h, dtype=torch.float32))
    emb_x = get_1d_sincos(half, torch.arange(w, dtype=torch.float32))
    emb_y = emb_y.unsqueeze(1).expand(-1, w, -1)
    emb_x = emb_x.unsqueeze(0).expand(h, -1, -1)
    return torch.cat([emb_y, emb_x], dim=-1).reshape(h * w, embed_dim)


class FiLMBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.cond_mapper = nn.Linear(dim, dim * 2)

    def forward(self, x, action_emb):
        B, TN, D = x.shape
        _, T_, _ = action_emb.shape
        N = TN // T_
        stats = self.cond_mapper(action_emb).unsqueeze(2)
        gamma, beta = torch.chunk(stats, 2, dim=-1)
        x = x.view(B, T_, N, D)
        x = x * (1 + gamma) + beta
        return x.view(B, TN, D)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.qkv  = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask is not None else None,
            dropout_p=0.0,
        )
        return self.proj(x.transpose(1, 2).reshape(B, N, D))


class TransformerFiLMBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadSelfAttention(dim, num_heads)
        self.film1 = FiLMBlock(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.film2 = FiLMBlock(dim)

    def forward(self, x, action_emb, attn_mask=None):
        x = x + self.film1(self.attn(self.norm1(x), attn_mask), action_emb)
        x = x + self.film2(self.mlp(self.norm2(x)), action_emb)
        return x


class ACPredictor(nn.Module):
    def __init__(self, enc_dim=ENC_DIM, pred_dim=PRED_DIM, depth=PRED_DEPTH,
                 num_heads=PRED_HEADS, action_dim=2,
                 n_patches=N_PATCHES, patch_h=PATCH_H, patch_w=PATCH_W, max_seq=MAX_SEQ):
        super().__init__()
        self.pred_dim  = pred_dim
        self.n_patches = n_patches
        self.max_seq   = max_seq

        self.patch_proj     = nn.Linear(enc_dim, pred_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 256), nn.GELU(),
            nn.Linear(256, pred_dim), nn.LayerNorm(pred_dim)
        )

        spatial_pe = build_2d_sincos_pe(pred_dim, patch_h, patch_w)
        self.register_buffer('spatial_pe', spatial_pe.unsqueeze(0))
        self.temporal_emb = nn.Embedding(max_seq, pred_dim)

        self.blocks = nn.ModuleList([
            TransformerFiLMBlock(pred_dim, num_heads) for _ in range(depth)
        ])
        self.norm     = nn.LayerNorm(pred_dim)
        self.out_proj = nn.Linear(pred_dim, enc_dim)

        base_mask = self._build_causal_mask(max_seq, n_patches)
        self.register_buffer('causal_mask', base_mask)
        self._init_weights()

    def _build_causal_mask(self, T_, n_patches):
        total = T_ * n_patches
        mask = torch.full((total, total), float('-inf'))
        for t in range(T_):
            mask[t * n_patches:(t + 1) * n_patches, :(t + 1) * n_patches] = 0.0
        return mask

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z, a):
        B, T_, N, _ = z.shape
        z_proj = self.patch_proj(z) + self.spatial_pe
        t_ids  = torch.arange(T_, device=z.device)
        z_proj = z_proj + self.temporal_emb(t_ids).unsqueeze(0).unsqueeze(2)
        a_emb  = self.action_encoder(a)
        seq    = z_proj.reshape(B, T_ * N, self.pred_dim)
        mask   = self.causal_mask[:T_ * N, :T_ * N]
        for block in self.blocks:
            seq = block(seq, a_emb, attn_mask=mask)
        return self.out_proj(self.norm(seq).view(B, T_, N, self.pred_dim))


# =====================================================================
# DECODER  (your fast CNN decoder — unchanged)
# =====================================================================

class UpsamplingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        g = lambda ch: 32 if ch >= 32 else ch
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g(out_ch), out_ch), nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(g(out_ch), out_ch), nn.GELU(),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        return self.net(x)


class FrameDecoder(nn.Module):
    GRID_H = GRID_W = 12

    def __init__(self, emb_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(emb_dim, 512, kernel_size=1), nn.GroupNorm(32, 512), nn.GELU()
        )
        self.up = nn.ModuleList([
            UpsamplingBlock(512, 512),   # 12 -> 24
            UpsamplingBlock(512, 256),   # 24 -> 48
            UpsamplingBlock(256, 128),   # 48 -> 96
            UpsamplingBlock(128,  64),   # 96 -> 192
            UpsamplingBlock(64,   32),   # 192 -> 384
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1).view(B, 1024, self.GRID_H, self.GRID_W)
        x = self.proj(x)
        for block in self.up:
            x = block(x)
        return self.to_rgb(x)


# =====================================================================
# ENCODER HELPERS  (unchanged)
# =====================================================================

def load_vjepa21_encoder():
    # The official V-JEPA 2.1 codebase doesn't provide a clean way to load
    # just the encoder, so download weights from:
    # https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
    # and place in "~/.cache/torch/hub/checkpoints/"
    encoder, _ = torch.hub.load(
        'facebookresearch/vjepa2',
        'vjepa2_1_vit_large_384',
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


def encode_and_pool(encoder, pil_img, device):
    """Encode a PIL image with V-JEPA 2.1 and avg-pool 576->144 tokens."""
    tf = T.Compose([
        T.CenterCrop(360),
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_img).unsqueeze(0).unsqueeze(2).float()  # [1, 3, 1, 384, 384]
    tensor = tensor.to(device)

    with torch.no_grad():
        feats = encoder(tensor)  # [1, 576, 1024]

    feats = feats.permute(0, 2, 1)                        # [1, 1024, 576]
    feats = feats.view(1, 1024, 24, 24)
    feats = F.avg_pool2d(feats, kernel_size=2, stride=2)   # [1, 1024, 12, 12]
    feats = feats.view(1, 1024, 144)
    feats = feats.permute(0, 2, 1)                         # [1, 144, 1024]
    return feats.half()


def tensor_to_pil(recon):
    img_np = recon.squeeze(0).detach().cpu().clamp(0, 1).permute(1, 2, 0).float().numpy()
    return Image.fromarray((img_np * 255).astype(np.uint8))


# =====================================================================
# MODEL LOADING  — runs once, at process startup, NOT inside a
# Gradio event handler. See the module docstring for why.
# =====================================================================

ENCODER   = None
PREDICTOR = None
DECODER   = None
MODELS_READY = False


def _load_state(model, ckpt_path, name):
    if ckpt_path and os.path.exists(ckpt_path):
        data = torch.load(ckpt_path, map_location=DEVICE)
        state = data.get('model', data) if isinstance(data, dict) else data
        model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
        print(f"  [ok] {name} weights loaded from {ckpt_path}")
    else:
        print(f"  [warn] {name} weights not found at {ckpt_path!r} — using random init")


def load_models():
    """Blocking, one-shot model load. Returns a status string for the UI."""
    global ENCODER, PREDICTOR, DECODER, MODELS_READY
    try:
        print("[1/3] Loading V-JEPA 2.1 encoder (kept on CPU until needed)...")
        ENCODER = load_vjepa21_encoder()

        print("[2/3] Loading predictor...")
        PREDICTOR = ACPredictor().to(DEVICE).half()
        _load_state(PREDICTOR, resolve_checkpoint(LOCAL_PREDICTOR_CKPT, HF_PREDICTOR_REPO, PREDICTOR_FILENAME), "Predictor")

        print("[3/3] Loading decoder...")
        DECODER = FrameDecoder().to(DEVICE).half()
        _load_state(DECODER, resolve_checkpoint(LOCAL_DECODER_CKPT, HF_DECODER_REPO, DECODER_FILENAME), "Decoder")

        PREDICTOR.eval()
        DECODER.eval()
        MODELS_READY = True
        return "Ready — upload a starting image to begin."
    except Exception as e:
        MODELS_READY = False
        msg = f"Model loading failed: {e}"
        print(f"[error] {msg}")
        return msg


# =====================================================================
# SESSION-LEVEL LOGIC
# Everything below operates on a per-user `session` dict carried in a
# gr.State, so concurrent visitors don't share or clobber each other's
# rollout state.
# =====================================================================

# NOTE: in the original code, 'w' maps to (0, 0) — i.e. a held W key
# with no other key produces a zero action vector. That's preserved
# here verbatim, but it's worth double-checking against your training
# data's action convention — it looks like it might have been meant
# to be (0, 1).
ACTIONS = {'w': (0, 0), 's': (0, -1), 'a': (-1, 0), 'd': (1, 0)}

DIRECTIONS = {
    "Forward (W)": ACTIONS['w'],
    "Left (A)":    ACTIONS['a'],
    "Right (D)":   ACTIONS['d'],
}


def new_session():
    return {
        'latent_history':  None,
        'is_recording':     False,
        'record_dir':       None,
        'frame_count':      0,
        'emb_buffer':       [],
        'emb_frame_count':  0,
        'step_count':       0,
    }


def load_start_image(pil_img, session):
    if pil_img is None:
        return session, gr.skip(), gr.skip(), gr.skip(), gr.skip()
    if not MODELS_READY:
        return session, gr.skip(), "0", "—", "Models aren't ready yet — wait a moment and re-upload."

    global ENCODER
    ENCODER.to(DEVICE)
    with torch.no_grad():
        seed_emb = encode_and_pool(ENCODER, pil_img, DEVICE)  # [1, 144, 1024]
    ENCODER.to('cpu')
    if DEVICE == 'cuda':
        torch.cuda.empty_cache()

    session = new_session()
    session['latent_history'] = seed_emb.unsqueeze(1)  # [1, 1, 144, 1024]
    return session, pil_img, "0", "—", "Go! Use the D-pad for single steps, or Auto-walk to explore continuously."


def run_step(session, dx, dy, intensity, save_emb, emb_stride):
    if not MODELS_READY or session is None or session.get('latent_history') is None:
        return session, gr.skip(), gr.skip(), gr.skip()

    t0 = time.time()
    action = torch.tensor(
        [[dx * intensity, dy * intensity]], dtype=torch.float16, device=DEVICE
    ).unsqueeze(1)  # [1, 1, 2]

    with torch.no_grad():
        nxt = PREDICTOR(session['latent_history'][:, -1:], action)  # [1, 1, 144, 1024]
        session['latent_history'] = torch.cat(
            [session['latent_history'], nxt], dim=1
        )[:, -MAX_SEQ:]
        recon = DECODER(nxt.squeeze(1))  # [1, 3, 384, 384]

    fps = 1.0 / max(time.time() - t0, 1e-6)
    pil = tensor_to_pil(recon)

    if session['is_recording']:
        frame_path = os.path.join(session['record_dir'], f"frame_{session['frame_count']:05d}.png")
        pil.save(frame_path)
        session['frame_count'] += 1
        if save_emb:
            stride = max(1, int(emb_stride))
            if session['emb_frame_count'] % stride == 0:
                session['emb_buffer'].append(nxt.squeeze(1).detach().cpu().float().numpy())
            session['emb_frame_count'] += 1

    session['step_count'] += 1
    return session, pil, str(session['step_count']), f"{fps:.1f}"


def auto_step(session, direction_label, intensity, save_emb, emb_stride):
    dx, dy = DIRECTIONS[direction_label]
    return run_step(session, dx, dy, intensity, save_emb, emb_stride)


def toggle_recording(session, save_emb, emb_stride):
    if session is None or session.get('latent_history') is None:
        yield session, gr.Button(value="⏺ Start Recording"), None, None, "Load a starting image first."
        return

    if not session['is_recording']:
        session['is_recording']    = True
        session['record_dir']      = tempfile.mkdtemp(prefix="wm_rec_")
        session['frame_count']     = 0
        session['emb_buffer']      = []
        session['emb_frame_count'] = 0
        yield session, gr.Button(value="⏹ Stop && Save"), None, None, "Recording..."
        return

    # stopping: finalize in the background-ish (still synchronous, but
    # we yield an intermediate state first so the button updates instantly)
    session['is_recording'] = False
    yield session, gr.Button(value="Processing...", interactive=False), gr.skip(), gr.skip(), "Encoding video..."

    record_dir = session['record_dir']
    timestamp  = time.strftime("%Y%m%d-%H%M%S")
    out_dir    = tempfile.mkdtemp(prefix="wm_out_")
    video_path = os.path.join(out_dir, f"world_model_{timestamp}.mp4")

    cmd = [
        'ffmpeg', '-y', '-framerate', '10',
        '-i', os.path.join(record_dir, 'frame_%05d.png'),
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', video_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception as e:
        print(f"ffmpeg error: {e}")
        video_path = None

    emb_path = None
    if save_emb and session['emb_buffer']:
        emb_path = os.path.join(out_dir, f"embeddings_{timestamp}.npz")
        stacked = np.concatenate(session['emb_buffer'], axis=0)
        np.savez_compressed(emb_path, embeddings=stacked, stride=np.array(max(1, int(emb_stride))))

    shutil.rmtree(record_dir, ignore_errors=True)
    session['record_dir'] = None
    session['emb_buffer'] = []

    yield session, gr.Button(value="⏺ Start Recording", interactive=True), video_path, emb_path, "Saved."


def reset_session():
    return (
        None,                  # session_state
        None,                  # image_input
        None,                  # viewport
        "0",                   # steps_box
        "—",                   # fps_box
        False,                 # auto_walk_active
        gr.Timer(active=False),  # timer
        "Session reset — upload an image to begin.",
    )


# =====================================================================
# UI
# =====================================================================

CUSTOM_CSS = """
.gradio-container {
    background-color: #121212 !important;
    color: #e0e0e0 !important;
}
.gradio-markdown h1, .gradio-markdown h2, .gradio-markdown h3 {
    color: #ffffff !important;
    border-bottom: 1px solid #333;
    padding-bottom: 8px;
    margin-bottom: 15px !important;
}
#viewport {
    background-color: #000 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 8px 20px rgba(0,0,0,0.8) !important;
    border: 1px solid #333 !important;
}
#viewport img {
    border-radius: 8px !important;
    object-fit: contain !important;
    width: 100% !important;
    height: auto !important;
    max-height: 500px !important;
    background-color: #000 !important;
}
.dpad-container {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    margin: 15px 0 !important;
}
.dpad-row {
    display: flex !important;
    justify-content: center !important;
    gap: 10px !important;
    margin-bottom: 5px !important;
}
.dpad-btn {
    width: 65px;
    height: 65px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 24px;
    font-weight: bold;

    background: #333;
    color: white;

    border: 1px solid #222;
    border-radius: 4px;

    cursor: pointer;
}

.dpad-btn:active {
    background: #222;
}
.gradio-group {
    background-color: #1e1e1e !important;
    border: 1px solid #333 !important;
    border-radius: 12px !important;
    padding: 15px !important;
    margin-bottom: 15px !important;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
}
.gradio-button {
    border-radius: 8px !important;
    font-weight: 500 !important;
}
"""

INITIAL_STATUS = load_models()

with gr.Blocks(title="World Model Explorer", css=CUSTOM_CSS, theme=gr.themes.Base(primary_hue="blue")) as demo:
    session_state    = gr.State(None)
    auto_walk_active = gr.State(False)

    gr.Markdown("# Street-JEPA")
    gr.Markdown("""
    Trained on a 1 hour long street walking video, this model learns to predict the next frame of a video given the current frame and 
    a 2D action vector (dx, dy) derived from the video's optical flow. The predictor is an action conditioned transformer that takes in the latent 
    representation of the current frame and the action vector, and outputs the latent representation of the next frame. The decoder then reconstructs 
    the predicted frame from its latent representation.
    
    The training video is: https://youtu.be/V-q8Hn_6vHY?si=9j6DA1KMfvxV4khO. The model should perform adequately on any frame from this video, but may not 
    work too well with anything else.
    """)
    
    status_md = gr.Markdown(f"**Status:** {INITIAL_STATUS}")

    with gr.Row():
        # ---- LEFT COLUMN: Screen + Recording ----
        with gr.Column(scale=2):
            viewport = gr.Image(
                label="World view",
                type="pil",
                interactive=False,
                elem_id="viewport",
                show_label=False,
            )

            with gr.Group():
                gr.Markdown("### 🎥 Recording")
                with gr.Row():
                    rec_btn = gr.Button("⏺ Start Recording", variant="primary", scale=1)
                with gr.Row():
                    save_emb_cb = gr.Checkbox(label="Save embeddings", value=False, scale=1)
                    emb_stride_num = gr.Slider(1, 64, value=1, step=1, label="Embedding stride", scale=2)
                with gr.Row():
                    video_out = gr.Video(label="Recorded clip", scale=1)
                    emb_out = gr.File(label="Embeddings (.npz)", scale=1)

        # ---- RIGHT COLUMN: Upload + Movement + Auto-walk + Stats ----
        with gr.Column(scale=1):
            image_input = gr.Image(
                label="Starting frame (Upload to begin)",
                type="pil",
                sources=["upload"],
                height=120,
                show_label=True,
            )

            with gr.Group():
                gr.Markdown("### 🎮 Movement")
                with gr.Column(elem_classes="dpad-container"):
                    with gr.Row(elem_classes="dpad-row"):
                        up_btn = gr.Button("▲", elem_classes="dpad-btn", scale=0)
                    with gr.Row(elem_classes="dpad-row"):
                        left_btn = gr.Button("◀", elem_classes="dpad-btn", scale=0)
                        right_btn = gr.Button("▶", elem_classes="dpad-btn", scale=0)
                        
                
                intensity_slider = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="Movement intensity")

            with gr.Group():
                gr.Markdown("### 🚶 Auto-walk")
                direction_radio = gr.Radio(
                    choices=list(DIRECTIONS.keys()), 
                    value="Forward (W)", 
                    label="Direction"
                )
                tick_slider = gr.Slider(50, 500, value=50, step=50, label="Step interval (ms)")
                with gr.Row():
                    start_walk_btn = gr.Button("▶ Start", variant="primary", scale=1)
                    stop_walk_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

            with gr.Group():
                gr.Markdown("### 📊 Stats")
                with gr.Row():
                    steps_box = gr.Textbox(value="0", label="Steps", interactive=False, scale=1)
                    fps_box = gr.Textbox(value="—", label="FPS", interactive=False, scale=1)
                reset_btn = gr.Button("🔄 Reset Session", variant="secondary")

    # ---- Timer for auto-walk ----
    timer = gr.Timer(value=tick_slider.value / 1000.0, active=False)

    # ---- Wiring ----

    image_input.upload(
        fn=load_start_image,
        inputs=[image_input, session_state],
        outputs=[session_state, viewport, steps_box, fps_box, status_md],
    )

    # D-pad clicks
    up_btn.click(
        fn=lambda session, intensity, save_emb, stride: run_step(session, *DIRECTIONS["Forward (W)"], intensity, save_emb, stride),
        inputs=[session_state, intensity_slider, save_emb_cb, emb_stride_num],
        outputs=[session_state, viewport, steps_box, fps_box],
    )
    left_btn.click(
        fn=lambda session, intensity, save_emb, stride: run_step(session, *DIRECTIONS["Left (A)"], intensity, save_emb, stride),
        inputs=[session_state, intensity_slider, save_emb_cb, emb_stride_num],
        outputs=[session_state, viewport, steps_box, fps_box],
    )
    right_btn.click(
        fn=lambda session, intensity, save_emb, stride: run_step(session, *DIRECTIONS["Right (D)"], intensity, save_emb, stride),
        inputs=[session_state, intensity_slider, save_emb_cb, emb_stride_num],
        outputs=[session_state, viewport, steps_box, fps_box],
    )

    # Auto-walk timer tick
    timer.tick(
        fn=auto_step,
        inputs=[session_state, direction_radio, intensity_slider, save_emb_cb, emb_stride_num],
        outputs=[session_state, viewport, steps_box, fps_box],
    )

    # Auto-walk start/stop
    start_walk_btn.click(
        fn=lambda: (True, gr.Timer(active=True)),
        inputs=None,
        outputs=[auto_walk_active, timer],
    )
    stop_walk_btn.click(
        fn=lambda: (False, gr.Timer(active=False)),
        inputs=None,
        outputs=[auto_walk_active, timer],
    )
    tick_slider.change(
        fn=lambda ms, active: gr.Timer(value=ms / 1000.0, active=active),
        inputs=[tick_slider, auto_walk_active],
        outputs=timer,
    )

    # Recording toggle
    rec_btn.click(
        fn=toggle_recording,
        inputs=[session_state, save_emb_cb, emb_stride_num],
        outputs=[session_state, rec_btn, video_out, emb_out, status_md],
    )

    # Reset
    reset_btn.click(
        fn=reset_session,
        inputs=None,
        outputs=[session_state, image_input, viewport, steps_box, fps_box, auto_walk_active, timer, status_md],
    )


if __name__ == "__main__":
    demo.queue().launch()