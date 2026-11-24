"""Stage 2 — the three false-positive-reduction architectures (identical trunks, identical inputs).

Input to every model: a (B, 3, 48, 48, 48) tensor — channel s is the candidate-centred crop at physical extent
24 / 48 / 96 mm resampled to 48^3 (voxel 0.5 / 1.0 / 2.0 mm), values in [0,1]. Produced on GPU by patches.make_scales.

  A  ResNet18_3D        single-scale: channel 1 (48 mm) -> 3D ResNet-18 trunk -> GAP -> FC.                 "the floor"
  B  MultiScaleCNN      three trunks (same design, separate weights) -> GAP each -> concat(3x512) -> MLP.   "the standard approach"
  C  CNNTransformer     same three trunks -> final 3x3x3x512 maps tokenised (27 tokens/scale) -> proj 256 +
                        learned scale embedding + learned position embedding + CLS -> Transformer encoder
                        (4-6 layers, 4 heads, d=256, pre-norm) -> CLS -> FC.                               "the contribution"
     Ablations (config): fusion='attention'|'concat' (concat = per-scale mean-pooled tokens concatenated, no
     attention across scales), use_scale_embedding, depth.
All models return a logit of shape (B,). Trunk = MONAI 3D ResNet-18 (BasicBlock [2,2,2,2], conv1 7^3 stride 1,
maxpool /2, stages /2 /2 /2 -> 3^3 x 512 for a 48^3 input).
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from monai.networks.nets import resnet


class Trunk(nn.Module):
    """3D ResNet-18 up to layer4; returns the (B,512,3,3,3) feature map for a 48^3 input."""
    def __init__(self, width: float = 1.0):
        super().__init__()
        r = resnet.ResNet(block=resnet.ResNetBlock, layers=[2, 2, 2, 2], block_inplanes=resnet.get_inplanes(), spatial_dims=3,
                          n_input_channels=1, conv1_t_size=7, conv1_t_stride=1, num_classes=1, widen_factor=width, feed_forward=False)
        self.conv1, self.bn1, self.act, self.maxpool = r.conv1, r.bn1, r.act, r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.out_channels = int(512 * width)

    def forward(self, x):
        x = self.maxpool(self.act(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x


class ResNet18_3D(nn.Module):
    """Model A."""
    def __init__(self, scale_channel: int = 1, dropout: float = 0.3, **kw):
        super().__init__(); self.c = scale_channel; self.trunk = Trunk()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.trunk.out_channels, 1))

    def forward(self, x):
        f = self.trunk(x[:, self.c:self.c + 1]).mean(dim=(2, 3, 4))
        return self.head(f).squeeze(1)


class MultiScaleCNN(nn.Module):
    """Model B."""
    def __init__(self, n_scales: int = 3, dropout: float = 0.3, hidden: int = 256, **kw):
        super().__init__(); self.trunks = nn.ModuleList([Trunk() for _ in range(n_scales)])
        d = self.trunks[0].out_channels * n_scales
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        f = torch.cat([t(x[:, i:i + 1]).mean(dim=(2, 3, 4)) for i, t in enumerate(self.trunks)], 1)
        return self.head(f).squeeze(1)

    def embed(self, x):
        return torch.cat([t(x[:, i:i + 1]).mean(dim=(2, 3, 4)) for i, t in enumerate(self.trunks)], 1)


class CNNTransformer(nn.Module):
    """Model C."""
    def __init__(self, n_scales: int = 3, d_model: int = 256, depth: int = 4, heads: int = 4, ff: int = 512, dropout: float = 0.1,
                 fusion: str = "attention", use_scale_embedding: bool = True, use_pos_embedding: bool = True, head_dropout: float = 0.3, **kw):
        super().__init__()
        self.n_scales, self.fusion, self.use_se, self.use_pe = n_scales, fusion, use_scale_embedding, use_pos_embedding
        self.trunks = nn.ModuleList([Trunk() for _ in range(n_scales)])
        c = self.trunks[0].out_channels
        self.proj = nn.Linear(c, d_model)
        self.scale_emb = nn.Parameter(torch.zeros(n_scales, 1, d_model)); nn.init.trunc_normal_(self.scale_emb, std=0.02)
        self.pos_emb = nn.Parameter(torch.zeros(1, 27, d_model)); nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model)); nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d_model)
        head_in = d_model if fusion == "attention" else d_model * n_scales
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(head_in, 1))

    def tokens(self, x):
        toks = []
        for i, t in enumerate(self.trunks):
            f = t(x[:, i:i + 1])                                   # (B, C, 3, 3, 3)
            f = f.flatten(2).transpose(1, 2)                       # (B, 27, C)
            f = self.proj(f)
            if self.use_pe: f = f + self.pos_emb[:, : f.shape[1]]
            if self.use_se: f = f + self.scale_emb[i]
            toks.append(f)
        return toks

    def embed(self, x):
        toks = self.tokens(x)
        if self.fusion == "attention":
            z = torch.cat([self.cls.expand(x.shape[0], -1, -1)] + toks, 1)
            z = self.encoder(z)
            return self.norm(z[:, 0])
        return torch.cat([self.norm(self.encoder(t)).mean(1) for t in toks], 1)   # 'concat' ablation: no cross-scale attention

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)


MODELS = {"A": ResNet18_3D, "B": MultiScaleCNN, "C": CNNTransformer}


def build_model(name: str, **kw) -> nn.Module:
    return MODELS[name](**kw)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__(); self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, y):
        y = y.float(); ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        p = torch.sigmoid(logits); pt = p * y + (1 - p) * (1 - y); at = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (at * (1 - pt) ** self.gamma * ce).mean()
