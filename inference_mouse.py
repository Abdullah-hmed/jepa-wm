import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import os
import time
import threading
from tkinter import Tk, filedialog
from PIL import Image

# ── Constants ──────────────────────────────────────────────
ENC_DIM    = 1024
N_PATCHES  = 144
PATCH_H    = 12
PATCH_W    = 12
PRED_DIM   = 1024
PRED_DEPTH = 20
PRED_HEADS = 16
ACTION_DIM = 2
MAX_SEQ    = 48

DX_SCALE   = 111.11
DY_SCALE   = 333.33

WIN_W      = 900
WIN_H      = 600
VIEW_SIZE  = 512   # square viewport

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
        mask  = torch.full((total, total), float('-inf'))
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
            UpsamplingBlock(512, 512),
            UpsamplingBlock(512, 256),
            UpsamplingBlock(256, 128),
            UpsamplingBlock(128,  64),
            UpsamplingBlock(64,   32),
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
# ENCODER
# =====================================================================

def load_vjepa21_encoder():
    encoder, _ = torch.hub.load('facebookresearch/vjepa2', 'vjepa2_1_vit_large_384')
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder

def encode_and_pool(encoder, pil_img, device):
    tf = T.Compose([
        T.CenterCrop(360),
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_img).unsqueeze(0).unsqueeze(2).float().to(device)
    with torch.no_grad():
        feats = encoder(tensor)
    feats = feats.permute(0, 2, 1).view(1, 1024, 24, 24)
    feats = F.avg_pool2d(feats, kernel_size=2, stride=2)
    feats = feats.view(1, 1024, 144).permute(0, 2, 1)
    return feats.half()


# =====================================================================
# PYGAME APP
# =====================================================================

def pick_image():
    """Open file dialog on a hidden tkinter root."""
    root = Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All", "*.*")]
    )
    root.destroy()
    return path


def pil_to_pygame(pil_img, size):
    pil_img = pil_img.resize(size, Image.Resampling.BILINEAR)
    return pygame.image.fromstring(pil_img.tobytes(), pil_img.size, 'RGB')


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ── Load models ────────────────────────────────────────
    print("[Status] Loading V-JEPA 2.1 encoder…")
    encoder = load_vjepa21_encoder()

    print("[Status] Loading predictor…")
    predictor = ACPredictor().to(device).half()

    print("[Status] Loading decoder…")
    decoder = FrameDecoder().to(device).half()

    for name, model, ckpt in [
        ('Predictor', predictor, 'predictor_vjepa21.pt'),
        ('Decoder',   decoder,   'decoder_vjepa21.pt'),
    ]:
        if os.path.exists(ckpt):
            data  = torch.load(ckpt, map_location=device)
            state = data.get('model', data)
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in state.items()})
            print(f"[Status] ✓ {name} loaded")
        else:
            print(f"[Status] ⚠ {name} not found — random init")

    predictor.eval()
    decoder.eval()
    print("[Status] Ready")

    # ── Pygame setup ───────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("World Model Explorer")
    clock  = pygame.time.Clock()
    font_l = pygame.font.SysFont('Arial', 15)
    font_s = pygame.font.SysFont('Arial', 12)

    # Layout
    VIEW_X = 20
    VIEW_Y = (WIN_H - VIEW_SIZE) // 2
    PANEL_X = VIEW_X + VIEW_SIZE + 20
    PANEL_W = WIN_W - PANEL_X - 20

    # Colors
    C_BG     = (17,  17,  17)
    C_PANEL  = (28,  28,  28)
    C_CARD   = (39,  39,  39)
    C_ACCENT = (78, 127, 255)
    C_DIM    = (100, 100, 100)
    C_BRIGHT = (220, 220, 220)
    C_DANGER = (200,  68,  68)
    C_GREEN  = ( 68, 200, 136)

    # State
    latent_history  = None
    current_surface = None
    mouse_captured  = False
    sensitivity     = 1.0
    step_count      = 0
    fps_inf         = 0.0
    status_msg      = "Press L to load image"
    inferring       = False
    pending_dx      = 0.0
    pending_dy      = 0.0

    def do_step(dx, dy):
        nonlocal latent_history, current_surface, step_count, fps_inf, inferring

        if latent_history is None or inferring:
            return
        inferring = True
        t0 = time.time()

        action = torch.tensor(
            [[dx * DX_SCALE, dy * DY_SCALE]],
            dtype=torch.float16, device=device
        ).unsqueeze(1)

        with torch.no_grad():
            nxt = predictor(latent_history[:, -1:], action)
            latent_history = torch.cat([latent_history, nxt], dim=1)[:, -MAX_SEQ:]
            recon = decoder(nxt.squeeze(1))

        fps_inf = 1.0 / max(time.time() - t0, 1e-6)
        img_np  = recon.squeeze(0).cpu().clamp(0, 1).permute(1, 2, 0).float().numpy()
        pil     = Image.fromarray((img_np * 255).astype(np.uint8))
        current_surface = pil_to_pygame(pil, (VIEW_SIZE, VIEW_SIZE))
        step_count += 1
        inferring = False

    def load_image():
        nonlocal latent_history, current_surface, step_count, status_msg
        path = pick_image()
        if not path:
            return
        status_msg = "Encoding…"
        img = Image.open(path).convert('RGB')
        encoder.to(device)
        seed_emb = encode_and_pool(encoder, img, device)
        encoder.to('cpu')
        torch.cuda.empty_cache()
        latent_history  = seed_emb.unsqueeze(1)
        step_count      = 0
        preview_pil     = img.copy()
        preview_pil.thumbnail((VIEW_SIZE, VIEW_SIZE), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', (VIEW_SIZE, VIEW_SIZE), (10, 10, 10))
        canvas.paste(preview_pil, (
            (VIEW_SIZE - preview_pil.width)  // 2,
            (VIEW_SIZE - preview_pil.height) // 2,
        ))
        current_surface = pil_to_pygame(canvas, (VIEW_SIZE, VIEW_SIZE))
        status_msg = "Click viewport to capture mouse"

    def draw_text(surf, text, x, y, color=None, font=None):
        color = color or C_BRIGHT
        font  = font  or font_l
        surf.blit(font.render(text, True, color), (x, y))

    def draw_button(surf, rect, label, color=C_ACCENT):
        pygame.draw.rect(surf, C_CARD, rect, border_radius=6)
        tw, th = font_l.size(label)
        draw_text(surf, label,
                  rect[0] + (rect[2] - tw) // 2,
                  rect[1] + (rect[3] - th) // 2,
                  color=color)
        return pygame.Rect(rect)

    # Button rects
    btn_load  = (PANEL_X, 60,  PANEL_W, 34)
    btn_reset = (PANEL_X, 102, PANEL_W, 34)

    running = True
    while running:
        dt = clock.tick(30)

        # ── Events ─────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if mouse_captured:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        mouse_captured = False
                        pending_dx = pending_dy = 0.0
                        status_msg = "Mouse released"

                elif event.key == pygame.K_l:
                    threading.Thread(target=load_image, daemon=True).start()

                elif event.key == pygame.K_r:
                    latent_history  = None
                    current_surface = None
                    step_count      = 0
                    if mouse_captured:
                        pygame.mouse.set_visible(True)
                        pygame.event.set_grab(False)
                        mouse_captured = False
                    status_msg = "Session reset"

                elif event.key == pygame.K_w:
                    if latent_history is not None:
                        threading.Thread(
                            target=do_step, args=(0.0, 0.0), daemon=True
                        ).start()

                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    sensitivity = min(5.0, sensitivity + 0.1)
                elif event.key == pygame.K_MINUS:
                    sensitivity = max(0.1, sensitivity - 0.1)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    vr = pygame.Rect(VIEW_X, VIEW_Y, VIEW_SIZE, VIEW_SIZE)
                    br_load  = pygame.Rect(btn_load)
                    br_reset = pygame.Rect(btn_reset)

                    if vr.collidepoint(mx, my) and latent_history is not None:
                        pygame.mouse.set_visible(False)
                        pygame.event.set_grab(True)
                        mouse_captured = True
                        status_msg = "Mouse captured — ESC to release"

                    elif br_load.collidepoint(mx, my):
                        threading.Thread(target=load_image, daemon=True).start()

                    elif br_reset.collidepoint(mx, my):
                        latent_history  = None
                        current_surface = None
                        step_count      = 0
                        if mouse_captured:
                            pygame.mouse.set_visible(True)
                            pygame.event.set_grab(False)
                            mouse_captured = False
                        status_msg = "Session reset"

            elif event.type == pygame.MOUSEMOTION:
                if mouse_captured:
                    # pygame relative mode gives exact per-frame deltas
                    rel_x, rel_y = event.rel
                    pending_dx += rel_x * sensitivity * 0.01
                    pending_dy += rel_y * sensitivity * 0.01
                    pending_dx  = max(-1.0, min(1.0, pending_dx))
                    pending_dy  = max(-1.0, min(1.0, pending_dy))

        # ── Dispatch accumulated mouse delta ───────────────
        if (mouse_captured and latent_history is not None and not inferring and
                (abs(pending_dx) > 0.001 or abs(pending_dy) > 0.001)):
            dx, dy     = pending_dx, pending_dy
            pending_dx = 0.0
            pending_dy = 0.0
            threading.Thread(target=do_step, args=(dx, dy), daemon=True).start()

        # ── Draw ───────────────────────────────────────────
        screen.fill(C_BG)

        # Viewport
        view_rect = pygame.Rect(VIEW_X, VIEW_Y, VIEW_SIZE, VIEW_SIZE)
        pygame.draw.rect(screen, C_PANEL, view_rect)
        if current_surface:
            screen.blit(current_surface, (VIEW_X, VIEW_Y))
        else:
            msg = font_l.render("Press L to load image", True, C_DIM)
            screen.blit(msg, (
                VIEW_X + (VIEW_SIZE - msg.get_width())  // 2,
                VIEW_Y + (VIEW_SIZE - msg.get_height()) // 2,
            ))
        pygame.draw.rect(screen, C_ACCENT if mouse_captured else C_DIM,
                         view_rect, 2, border_radius=4)

        # Capture indicator
        if mouse_captured:
            ind = font_s.render("● CAPTURED  |  ESC to release", True, C_ACCENT)
            screen.blit(ind, (VIEW_X + 8, VIEW_Y + 8))

        # Panel
        pygame.draw.rect(screen, C_PANEL,
                         (PANEL_X - 10, 0, WIN_W - PANEL_X + 10, WIN_H))

        draw_text(screen, "World Model", PANEL_X, 20, C_BRIGHT)
        draw_text(screen, "Mouse-driven inference", PANEL_X, 40, C_DIM, font_s)

        draw_button(screen, btn_load,  "📂  Load Image  (L)")
        draw_button(screen, btn_reset, "↺  Reset  (R)", color=C_DANGER)

        # Sensitivity
        draw_text(screen, f"Sensitivity: {sensitivity:.1f}  (+/- to adjust)",
                  PANEL_X, 155, C_DIM, font_s)
        bar_w = int(PANEL_W * (sensitivity / 5.0))
        pygame.draw.rect(screen, C_CARD,   (PANEL_X, 172, PANEL_W, 8), border_radius=4)
        pygame.draw.rect(screen, C_ACCENT, (PANEL_X, 172, bar_w,   8), border_radius=4)

        # Controls
        draw_text(screen, "Controls", PANEL_X, 200, C_DIM)
        for i, line in enumerate([
            "Click viewport  → capture mouse",
            "ESC             → release mouse",
            "W               → step forward",
            "Mouse move      → look / steer",
            "L               → load image",
            "R               → reset session",
            "+/-             → sensitivity",
        ]):
            draw_text(screen, line, PANEL_X, 222 + i * 18, C_DIM, font_s)

        # Stats
        draw_text(screen, "Stats", PANEL_X, 360, C_DIM)
        for i, (label, val) in enumerate([
            ("Steps",  str(step_count)),
            ("Inf FPS", f"{fps_inf:.1f}" if fps_inf > 0 else "—"),
            ("dx",     f"{pending_dx:+.3f}"),
            ("dy",     f"{pending_dy:+.3f}"),
            ("Device", device.upper()),
        ]):
            draw_text(screen, f"{label:<10}{val}", PANEL_X, 382 + i * 18, C_BRIGHT, font_s)

        # Status bar
        pygame.draw.rect(screen, C_CARD, (0, WIN_H - 28, WIN_W, 28))
        draw_text(screen, status_msg, 12, WIN_H - 20, C_DIM, font_s)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()