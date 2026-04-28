import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import os
import time
import threading
import subprocess
import shutil

ENC_DIM     = 1024
N_PATCHES   = 144
PATCH_H     = 12
PATCH_W     = 12
PRED_DIM    = 1024
PRED_DEPTH  = 20
PRED_HEADS  = 16
ACTION_DIM  = 2
MAX_SEQ     = 48

# =====================================================================
# PREDICTOR
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
        _, T, _  = action_emb.shape
        N = TN // T
        stats = self.cond_mapper(action_emb).unsqueeze(2)
        gamma, beta = torch.chunk(stats, 2, dim=-1)
        x = x.view(B, T, N, D)
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

    def _build_causal_mask(self, T, n_patches):
        total = T * n_patches
        mask = torch.full((total, total), float('-inf'))
        for t in range(T):
            mask[t*n_patches:(t+1)*n_patches, :(t+1)*n_patches] = 0.0
        return mask

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, z, a):
        B, T, N, _ = z.shape
        z_proj = self.patch_proj(z) + self.spatial_pe
        t_ids  = torch.arange(T, device=z.device)
        z_proj = z_proj + self.temporal_emb(t_ids).unsqueeze(0).unsqueeze(2)
        a_emb  = self.action_encoder(a)
        seq    = z_proj.reshape(B, T * N, self.pred_dim)
        mask   = self.causal_mask[:T*N, :T*N]
        for block in self.blocks:
            seq = block(seq, a_emb, attn_mask=mask)
        return self.out_proj(self.norm(seq).view(B, T, N, self.pred_dim))


# =====================================================================
# DECODER
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
    def forward(self, x): return self.net(x)

class FrameDecoder(nn.Module):
    GRID_H = GRID_W = 12
    def __init__(self, emb_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(emb_dim, 512, kernel_size=1), nn.GroupNorm(32, 512), nn.GELU()
        )
        self.up = nn.ModuleList([
            UpsamplingBlock(512, 512),   # 12 → 24
            UpsamplingBlock(512, 256),   # 24 → 48
            UpsamplingBlock(256, 128),   # 48 → 96
            UpsamplingBlock(128,  64),   # 96 → 192
            UpsamplingBlock(64,   32),   # 192 → 384
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.shape[0]
        x = x.permute(0, 2, 1).view(B, 1024, self.GRID_H, self.GRID_W)
        x = self.proj(x)
        for block in self.up: x = block(x)
        return self.to_rgb(x)


# =====================================================================
# ENCODER HELPER
# =====================================================================


def load_vjepa21_encoder():

    # The official V-JEPA 2.1 codebase doesn't provide a clean way to load just the encoder,
    # so download weights: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitl_dist_vitG_384.pt
    # and place in "C:\Users\<username>\.cache\torch\hub\checkpoints" or "/root/.cache/torch/hub/checkpoints/"

    encoder, _ = torch.hub.load(
        'facebookresearch/vjepa2',
        'vjepa2_1_vit_large_384',
    )
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder

def encode_and_pool(encoder, pil_img, device):
    """
    Encode a PIL image with V-JEPA 2.1 and avg-pool 576→144 tokens.
    Matches exactly the preprocessing used during shard generation.
    Returns: [1, 144, 1024] float16 on device
    """
    tf = T.Compose([
        T.CenterCrop(360),
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_img).unsqueeze(0).unsqueeze(2).float()  # [1, 3, 1, 384, 384]
    tensor = tensor.to(device)

    with torch.no_grad():
        feats = encoder(tensor)   # [1, 576, 1024]

    # avg pool 24x24 → 12x12
    feats = feats.permute(0, 2, 1)                        # [1, 1024, 576]
    feats = feats.view(1, 1024, 24, 24)                   # [1, 1024, 24, 24]
    feats = F.avg_pool2d(feats, kernel_size=2, stride=2)  # [1, 1024, 12, 12]
    feats = feats.view(1, 1024, 144)                      # [1, 1024, 144]
    feats = feats.permute(0, 2, 1)                        # [1, 144, 1024]

    return feats.half()


# =====================================================================
# GAME
# =====================================================================

ACTIONS = {'w': (0, 0), 's': (0, -1), 'a': (-1, 0), 'd': (1, 0)}

BG     = '#111111'
PANEL  = '#1c1c1c'
CARD   = '#272727'
ACCENT = '#4e7fff'
DIM    = '#666666'
BRIGHT = '#dddddd'
DANGER = '#cc4444'


class WorldModelGame:
    def __init__(self, root):
        self.root = root
        self.root.title("World Model Explorer")
        self.root.configure(bg=BG)
        self.root.geometry("960x640")
        self.root.minsize(800, 600)
        self.root.resizable(True, True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.is_recording   = False
        self.record_dir     = "temp_recording"
        self.frame_count    = 0
        self.intensity      = 1.0
        self.latent_history = None
        self.is_running     = False
        self.keys_held      = set()
        self.tick_interval  = 50
        self.last_step_time = 0.0
        self.step_count     = 0
        self.fps            = 0.0
        self.inferring      = False

        self._build_ui()
        self._load_models()
        self._game_loop()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        SIDEBAR_W = 220
        sidebar = tk.Frame(self.root, bg=PANEL, width=SIDEBAR_W)
        sidebar.pack(side='right', fill='y')
        sidebar.pack_propagate(False)

        self.main_area = tk.Frame(self.root, bg='#0a0a0a')
        self.main_area.pack(side='left', fill='both', expand=True)

        self.viewport = tk.Label(self.main_area, bg='#0a0a0a', borderwidth=0)
        self.viewport.place(relx=0.5, rely=0.5, anchor='center')

        P = dict(padx=12)
        tk.Label(sidebar, text="World Model", bg=PANEL, fg=BRIGHT,
                font=('Arial', 11, 'bold')).pack(anchor='w', pady=(14, 0), **P)

        self._sep(sidebar)
        self.load_btn = self._btn(sidebar, "Load Image", self._load_start_image)
        self.load_btn.configure(state='disabled')
        self.load_btn.pack(fill='x', pady=10, **P)

        self.rec_btn = self._btn(sidebar, "⏺ Start Recording", self._toggle_recording, color=ACCENT)
        self.rec_btn.pack(fill='x', pady=5, **P)

        self._sep(sidebar)
        tk.Label(sidebar, text="Movement intensity",
                bg=PANEL, fg=DIM, font=('Arial', 9)).pack(anchor='w', pady=(8, 0), **P)
        self.intensity_var = tk.DoubleVar(value=1.0)
        tk.Scale(sidebar, from_=0.1, to=5.0, resolution=0.1,
                orient='horizontal', variable=self.intensity_var,
                command=lambda v: setattr(self, 'intensity', float(v)),
                bg=PANEL, fg=BRIGHT, troughcolor=CARD,
                highlightthickness=0, activebackground=ACCENT,
                bd=0, length=SIDEBAR_W - 24).pack(**P)

        tk.Label(sidebar, text="Step interval (ms)",
                bg=PANEL, fg=DIM, font=('Arial', 9)).pack(anchor='w', pady=(6, 0), **P)
        self.tick_var = tk.IntVar(value=self.tick_interval)
        tk.Scale(sidebar, from_=50, to=1000, resolution=50,
                orient='horizontal', variable=self.tick_var,
                command=lambda v: setattr(self, 'tick_interval', int(float(v))),
                bg=PANEL, fg=BRIGHT, troughcolor=CARD,
                highlightthickness=0, activebackground=ACCENT,
                bd=0, length=SIDEBAR_W - 24).pack(**P)

        self._sep(sidebar)
        tk.Label(sidebar, text="D-pad", bg=PANEL, fg=DIM,
                font=('Arial', 9)).pack(anchor='w', pady=(8, 6), **P)
        self._build_dpad(sidebar)

        self._sep(sidebar)
        tk.Label(sidebar, text="Stats", bg=PANEL, fg=DIM,
                font=('Arial', 9)).pack(anchor='w', pady=(8, 4), **P)
        self.stat_steps = self._stat(sidebar, "Steps")
        self.stat_fps   = self._stat(sidebar, "FPS")

        self._sep(sidebar)
        self._btn(sidebar, "Reset Session", self._reset_session, color=DANGER).pack(fill='x', pady=8, **P)

        self.root.bind('<KeyPress>',   lambda e: self.keys_held.add(e.keysym.lower()) if e.keysym.lower() in ACTIONS else None)
        self.root.bind('<KeyRelease>', lambda e: self.keys_held.discard(e.keysym.lower()))
        self.root.bind('<Escape>',     lambda e: self.root.destroy())

    def _sep(self, p):
        tk.Frame(p, bg=CARD, height=1).pack(fill='x', pady=2)

    def _btn(self, parent, text, cmd, color=ACCENT):
        b = tk.Button(parent, text=text, command=cmd,
                      bg=CARD, fg=color, activebackground='#333333',
                      activeforeground=color, relief='flat', bd=0,
                      font=('Arial', 10), pady=6, cursor='hand2')
        b.bind('<Enter>', lambda e: b.configure(bg='#333333'))
        b.bind('<Leave>', lambda e: b.configure(bg=CARD))
        return b

    def _stat(self, parent, label):
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill='x', padx=12, pady=1)
        tk.Label(row, text=label, bg=PANEL, fg=DIM,
                 font=('Arial', 9), width=9, anchor='w').pack(side='left')
        v = tk.Label(row, text="—", bg=PANEL, fg=BRIGHT,
                     font=('Courier', 10), anchor='e')
        v.pack(side='right')
        return v

    def _build_dpad(self, parent):
        f = tk.Frame(parent, bg=PANEL)
        f.pack()
        bs = dict(bg=CARD, fg=BRIGHT, relief='flat', bd=0,
                  font=('Arial', 13), width=3, height=1, cursor='hand2')
        def mb(sym, dx, dy, r, c):
            b = tk.Button(f, text=sym, **bs,
                          command=lambda: self._one_shot(float(dx), float(dy)))
            b.grid(row=r, column=c, padx=3, pady=3, ipadx=6, ipady=4)
            b.bind('<Enter>',         lambda e: b.configure(bg='#383838'))
            b.bind('<Leave>',         lambda e: b.configure(bg=CARD))
            b.bind('<ButtonPress>',   lambda e: b.configure(bg='#1e3a8a'))
            b.bind('<ButtonRelease>', lambda e: b.configure(bg='#383838'))
        mb('▲',  0,  1, 0, 1)
        mb('◀', -1,  0, 1, 0)
        tk.Label(f, bg=CARD, width=3, height=1).grid(
            row=1, column=1, padx=3, pady=3, ipadx=6, ipady=4)
        mb('▶',  1,  0, 1, 2)
        mb('▼',  0, -1, 2, 1)

    # ------------------------------------------------------------------
    # RECORDING
    # ------------------------------------------------------------------

    def _toggle_recording(self):
        if not self.is_recording:
            if os.path.exists(self.record_dir):
                shutil.rmtree(self.record_dir)
            os.makedirs(self.record_dir)
            self.is_recording = True
            self.frame_count  = 0
            self.rec_btn.configure(text="⏹ Stop & Save", fg=DANGER)
        else:
            self.is_recording = False
            self.rec_btn.configure(text="... Processing ...", state='disabled')
            threading.Thread(target=self._finalize_video, daemon=True).start()

    def _finalize_video(self):
        timestamp   = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs('output', exist_ok=True)
        output_file = f"output/world_model_{timestamp}.mp4"
        cmd = [
            'ffmpeg', '-y', '-framerate', '10',
            '-i', f'{self.record_dir}/frame_%05d.png',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', output_file
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Video saved: {output_file}")
        except Exception as e:
            print(f"FFmpeg error: {e}")
        shutil.rmtree(self.record_dir)
        self.root.after(0, lambda: self.rec_btn.configure(
            text="⏺ Start Recording", fg=ACCENT, state='normal'))

    # ------------------------------------------------------------------
    # GAME LOOP
    # ------------------------------------------------------------------

    def _game_loop(self):
        if self.keys_held and self.is_running and not self.inferring:
            now = time.time()
            if (now - self.last_step_time) * 1000 >= self.tick_interval:
                dx = dy = 0.0
                for k in self.keys_held:
                    kx, ky = ACTIONS[k]
                    dx += kx; dy += ky
                mag = (dx**2 + dy**2) ** 0.5
                if mag > 0:
                    dx /= mag; dy /= mag
                self._step(dx, dy)
                self.last_step_time = now
        self.root.after(16, self._game_loop)

    # ------------------------------------------------------------------
    # INFERENCE
    # ------------------------------------------------------------------

    def _one_shot(self, dx, dy):
        if self.is_running and not self.inferring:
            self._step(dx, dy)

    def _step(self, dx, dy):
        if self.latent_history is None or self.inferring:
            return
        self.inferring = True
        t0 = time.time()

        action = torch.tensor(
            [[dx * self.intensity, dy * self.intensity]],
            dtype=torch.float16, device=self.device
        ).unsqueeze(1)  # [1, 1, 2]

        with torch.no_grad():
            nxt   = self.predictor(self.latent_history[:, -1:], action)  # [1, 1, 144, 1024]
            self.latent_history = torch.cat([self.latent_history, nxt], dim=1)[:, -MAX_SEQ:]
            recon = self.decoder(nxt.squeeze(1))  # [1, 3, 384, 384]

        self.fps = 1.0 / max(time.time() - t0, 1e-6)
        img_np   = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).float().numpy()
        pil      = Image.fromarray((img_np * 255).astype(np.uint8))

        if self.is_recording:
            pil.save(os.path.join(self.record_dir, f"frame_{self.frame_count:05d}.png"))
            self.frame_count += 1

        vw, vh = self.main_area.winfo_width(), self.main_area.winfo_height()
        scale  = min(vw / 384, vh / 384) * 0.9
        target = (int(384 * scale), int(384 * scale))
        if target[0] > 10:
            self._show(pil.resize(target, Image.Resampling.BILINEAR))

        self.step_count += 1
        self._refresh_stats()
        self.inferring = False

    # ------------------------------------------------------------------
    # MODEL LOADING
    # ------------------------------------------------------------------

    def _load_models(self):
        def _load():
            self._status("Loading V-JEPA 2.1 encoder (CPU)…")
            self.encoder = load_vjepa21_encoder()
            # encoder stays on CPU until needed, then swapped to VRAM

            self._status("Loading predictor…")
            self.predictor = ACPredictor().to(self.device).half()

            self._status("Loading decoder…")
            self.decoder = FrameDecoder().to(self.device).half()

            for name, model, ckpt in [
                ('Predictor', self.predictor, 'predictor_vjepa21.pt'),
                ('Decoder',   self.decoder,   'decoder_vjepa21.pt'),
            ]:
                if os.path.exists(ckpt):
                    data  = torch.load(ckpt, map_location=self.device)
                    state = data.get('model', data)
                    model.load_state_dict(
                        {k.replace('module.', ''): v for k, v in state.items()})
                    self._status(f"✓ {name} loaded")
                else:
                    self._status(f"⚠ {name} weights not found — random init")

            self.predictor.eval()
            self.decoder.eval()
            self._status("Ready — load a starting image")
            self.root.after(0, lambda: self.load_btn.configure(state='normal'))

        threading.Thread(target=_load, daemon=True).start()

    # ------------------------------------------------------------------
    # IMAGE LOADING
    # ------------------------------------------------------------------

    def _load_start_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")])
        if not path: return

        img = Image.open(path).convert('RGB')
        self._status("Encoding initial frame…")

        # Move encoder to VRAM, encode, pool, move back
        self.encoder.to(self.device)
        with torch.no_grad():
            seed_emb = encode_and_pool(self.encoder, img, self.device)  # [1, 144, 1024]
        self.encoder.to('cpu')
        torch.cuda.empty_cache()

        self.latent_history = seed_emb.unsqueeze(1)  # [1, 1, 144, 1024]
        self.step_count     = 0
        self.is_running     = True
        self._status("Go! Hold W/A/S/D to explore")

        # Show preview
        self.root.update_idletasks()
        vw = max(self.viewport.winfo_width(),  400)
        vh = max(self.viewport.winfo_height(), 400)
        preview = img.copy()
        preview.thumbnail((vw, vh), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (vw, vh), (10, 10, 10))
        canvas.paste(preview, ((vw - preview.width) // 2, (vh - preview.height) // 2))
        self._show(canvas)
        self._refresh_stats()

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _reset_session(self):
        self.latent_history = None
        self.is_running     = False
        self.step_count     = 0
        self.keys_held.clear()
        self.viewport.configure(image='', text="Load an image to begin", fg=DIM)
        self.viewport.image = None
        self._refresh_stats()
        self._status("Session reset")

    def _show(self, pil_img):
        tk_img = ImageTk.PhotoImage(pil_img)
        self.viewport.configure(image=tk_img)
        self.viewport.image = tk_img

    def _status(self, msg):
        print(f"[Status] {msg}")

    def _refresh_stats(self):
        self.stat_steps.configure(text=str(self.step_count))
        self.stat_fps.configure(text=f"{self.fps:.1f}" if self.fps > 0 else "—")


if __name__ == "__main__":
    root = tk.Tk()
    WorldModelGame(root)
    root.mainloop()