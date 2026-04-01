import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- UNet-based Flow Predictor ---
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if up:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = None

    def forward(self, x1, x2=None):
        if self.up:
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x2, x1], dim=1)
            return self.conv(x)
        return self.conv(x1)

class LatentFlowAndResidualPredictor(nn.Module):
    """A UNet that predicts both a 2D flow field and a latent residual map."""
    def __init__(self, in_channels, latent_channels, base_features=64):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.down1 = UNetBlock(in_channels, base_features)
        self.down2 = UNetBlock(base_features, base_features * 2)
        self.down3 = UNetBlock(base_features * 2, base_features * 4)
        self.bottleneck = UNetBlock(base_features * 4, base_features * 8)
        self.up1 = UNetBlock(base_features * 8, base_features * 4, up=True)
        self.up2 = UNetBlock(base_features * 4, base_features * 2, up=True)
        self.up3 = UNetBlock(base_features * 2, base_features, up=True)
        self.flow_head = nn.Conv2d(base_features, 2, kernel_size=1)
        self.residual_head = nn.Conv2d(base_features, latent_channels, kernel_size=1)
        torch.nn.init.zeros_(self.residual_head.weight)
        if self.residual_head.bias is not None:
            torch.nn.init.zeros_(self.residual_head.bias)

    def forward(self, z_ref, z_dri):
        x = torch.cat([z_ref, z_dri], dim=1)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        b = self.bottleneck(self.pool(d3))
        u1 = self.up1(b, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        flow = self.flow_head(u3)
        latent_residual = self.residual_head(u3)
        return flow, latent_residual


# --- Time Conditioning Module ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings