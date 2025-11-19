import torch
import torch.nn as nn
import math
from thop import profile

class SSCT(nn.Module):
    #spectral–spatial collaborative tokenize
    def __init__(self, c_in, tf_dim):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(c_in, c_in, 3, 1, 1, groups=c_in, bias=False),  # 空域局部特征
            nn.BatchNorm2d(c_in),
            nn.ReLU6(inplace=True)
        )
        self.spectral_attn = nn.Sequential(
            nn.Conv2d(c_in, c_in // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_in // 4, c_in, 1),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(c_in, tf_dim, 1)

    def forward(self, x):
        xs = self.spatial(x)
        attn = self.spectral_attn(x)
        x = xs * attn + x
        x = self.proj(x)
        return x

class  DBLA(nn.Module):
    #dynamic-basis low-rank aligne
    def __init__(self, in_ch, tf_dim, n_bases=2, rank=1):
        super().__init__()
        self.in_ch = in_ch
        self.tf_dim = tf_dim
        self.n_bases = n_bases
        self.rank = rank

        # Basis adapters: parameters A (tf_dim x rank) and B (rank x in_ch) per basis
        self.A = nn.Parameter(torch.randn(n_bases, tf_dim, rank) * 0.02)
        self.B = nn.Parameter(torch.randn(n_bases, rank, in_ch) * 0.02)

        # small MLP fallback for coefficient generation if KAN not available
        self.coeff_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, max(8, in_ch)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, in_ch), n_bases),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: B, in_ch, H, W
        B, C, H, W = x.shape
        # per-sample descriptor
        coeffs = self.coeff_mlp(x)  # B, n_bases

        # build per-batch low-rank adapter W_delta = sum_m coeffs[m] * (A_m @ B_m) ; shapes handled
        # Precompute basis W_m: tf_dim x in_ch
        W_bases = torch.einsum('mfr,mrc->mfc', self.A, self.B)  # m, tf_dim, in_ch
        # Expand to batch: B, m, tf_dim, in_ch
        W_bases = W_bases.unsqueeze(0).expand(B, -1, -1, -1)
        coeffs = coeffs.view(B, self.n_bases, 1, 1)
        W_delta = (coeffs * W_bases).sum(dim=1)  # B, tf_dim, in_ch

        # apply adapter: compute tokens dynamically via batched einsum
        L = H * W
        x_flat = x.view(B, C, L).permute(0, 2, 1)  # B, L, C
        tokens_delta = torch.einsum('blc,bfc->blf', x_flat, W_delta)  # B, L, tf_dim

        # base tokens via existing tokenizer are expected; here produce output to be fused
        return tokens_delta  # B, L, tf_dim
# ---------- 与上一版完全相同的模块 ----------
class CRB(nn.Module):
    #Channel Recalibration block
    def __init__(self, c_in, c_out, stride=1, expand=2):
        super().__init__()
        hidden = int(c_in * expand)
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, c_out, 1, bias=False),
            nn.BatchNorm2d(c_out),
        )

        # 轻量通道重标定
        self.recalib = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // 4, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out // 4, c_out, 1, bias=True),
            nn.Sigmoid()
        )

        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, 1, stride, bias=False),
                nn.BatchNorm2d(c_out)
            )

    def forward(self, x):
        out = self.conv(x)
        scale = self.recalib(out)
        out = out * scale + self.shortcut(x)
        return out


class TokenMix(nn.Module):
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mixer = nn.Conv2d(dim, dim, kernel, padding=kernel//2, groups=dim, bias=False)
        self.drop_path = nn.Dropout(0.05)

    def forward(self, x):
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.mixer(feat)
        feat = feat.flatten(2).transpose(1, 2)
        return x + self.drop_path(feat)


class LTFU(nn.Module):
    #lite token fusion unit
    def __init__(self, dim, mlp_ratio=1.5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = TokenMix(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(0.05)
        )
        self.res_scale = 0.5

    def forward(self, x):
        x = x + self.res_scale * self.token_mix(self.norm1(x))
        x = x + self.res_scale * self.mlp(self.norm2(x))
        return x


class ARSNet(nn.Module):
    #Adaptive Rank Spectral Network
    def __init__(self, num_classes, num_bands, patch_size,
                 stem_ch=4, stage=12, tf_dim=12):   # 6→4 / 18→12
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(num_bands, stem_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU6(inplace=True),
            nn.Conv2d(stem_ch, stem_ch, 3, 2, 1, bias=False),
            nn.BatchNorm2d(stem_ch),
            nn.ReLU6(inplace=True)
        )

        self.stage = CRB(stem_ch, stage, stride=2, expand=2)

        self.tokenizer = SSCT(stage, tf_dim)
        self.tf = LTFU(tf_dim)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Conv2d(tf_dim, num_classes, 1)
        
        self.lra = DBLA(in_ch=stage, tf_dim=tf_dim, n_bases=2, rank=1)

    def forward(self, x):
        x = self.stem(x)          # B,4 ,H/2,W/2
        x = self.stage(x)         # B,12,H/4,W/4
        b, c, h, w = x.shape
        tokens_orig = self.tokenizer(x).flatten(2).transpose(1,2)   # B,L,tf_dim
        tokens_lra = self.lra(x)                                # B,L,tf_dim (delta)
        tokens = tokens_orig + 0.4 * tokens_lra                      # small residual adapter
        tokens = self.tf(tokens)
        x = tokens.transpose(1,2).view(b, c, h, w)
        x = self.pool(x)
        logits = self.head(x).flatten(1)
        return logits, x.flatten(1)


# ---------- 自检 ----------
if __name__ == "__main__":
    net = ARSNet(num_classes=7, num_bands=3, patch_size=12)
    x = torch.randn(2, 3, 12, 12)
    pred, feat = net(x)
    print("pred:", pred.shape, "feat:", feat.shape)

    flops, params = profile(net, inputs=(x,))
    print(f"FLOPs:  {flops / 1e6:.2f} MFLOPs")
    print(f"Params: {params / 1e3:.2f} K")
    
    
