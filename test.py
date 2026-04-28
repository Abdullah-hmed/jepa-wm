import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# CONFIGURATION (12x12 Pooled Variant)
# =====================================================================
PRED_DEPTH = 20
SEQ_LEN    = 24
MAX_SEQ    = 48

ENC_DIM     = 1024
N_PATCHES   = 144   # 12 * 12
PATCH_H     = 12
PATCH_W     = 12
PRED_DIM    = 1024
PRED_HEADS  = 16
ACTION_DIM  = 2

print(f"Config: {PATCH_H}x{PATCH_W} grid | {N_PATCHES} patches | Depth {PRED_DEPTH}")

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

        # Correctly builds PE for the 12x12 grid
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
    # Updated for the 12x12 grid
    GRID_H = GRID_W = 12  
    def __init__(self, emb_dim=1024):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(emb_dim, 512, kernel_size=1), nn.GroupNorm(32, 512), nn.GELU()
        )
        # We need an extra upsampling stage or larger scale factor to get 
        # back to original resolution if starting from 12x12.
        # This keeps the blocks but adds one more to reach standard sizes.
        self.up = nn.ModuleList([
            UpsamplingBlock(512, 512), # 12 -> 24
            UpsamplingBlock(512, 256), # 24 -> 48
            UpsamplingBlock(256, 128), # 48 -> 96
            UpsamplingBlock(128,  64), # 96 -> 192
            UpsamplingBlock(64,   32), # 192 -> 384 (optional, depending on target)
        ])
        self.to_rgb = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        B = x.shape[0]
        # x expected as [B, 144, 1024]
        x = x.permute(0, 2, 1).view(B, 1024, self.GRID_H, self.GRID_W)
        x = self.proj(x)
        for block in self.up: x = block(x)
        return self.to_rgb(x)

# =====================================================================
# EXECUTION
# =====================================================================

model = ACPredictor(depth=PRED_DEPTH, max_seq=MAX_SEQ).cuda().half()
decoder = FrameDecoder().cuda().half()
model.eval()
decoder.eval()

# Dummy input representing pooled embeddings
# z = torch.randn(1, 32, 144, 1024, dtype=torch.float16, device='cuda')
# a = torch.randn(1, 32, 2, dtype=torch.float16, device='cuda')

z = torch.randn(1, 48, 144, 1024, dtype=torch.float16, device='cuda')
a = torch.randn(1, 48, 2, dtype=torch.float16, device='cuda')

with torch.no_grad():
    out = model(z, a)
    # Decode the last predicted frame
    frame = decoder(out[:, -1])

peak = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak VRAM with 12x12 setup: {peak:.0f}MB")
print(f"Output Frame Shape: {frame.shape}")