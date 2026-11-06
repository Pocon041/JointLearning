"""Spatial blocks, explicit local relations, and pixel-wise frequency fusion."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

BAND_NAMES = ("low", "mid", "high")


def norm2d(channels: int) -> nn.GroupNorm:
    if channels < 1:
        raise ValueError("channels must be positive")
    groups = min(16, max(1, channels // 4))
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class LayerNorm2d(nn.LayerNorm):
    """ConvNeXt channel LayerNorm for NCHW tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__(channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            norm2d(out_channels),
            nn.GELU(),
        )


class DropPath(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("drop path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self.probability:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class ConvNeXtBlock(nn.Module):
    """7x7 depthwise conv, channel LN, 4x MLP, layer scale, and residual."""

    def __init__(self, channels: int, drop_path: float = 0.0, layer_scale: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, channels * 4)
        self.activation = nn.GELU()
        self.reduce = nn.Linear(channels * 4, channels)
        self.gamma = nn.Parameter(torch.full((channels,), layer_scale))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(x).permute(0, 2, 3, 1)
        residual = self.reduce(self.activation(self.expand(self.norm(residual))))
        residual = (residual * self.gamma).permute(0, 3, 1, 2).contiguous()
        return x + self.drop_path(residual)


class FrequencyBlock(nn.Module):
    """Lightweight block applied separately, with tied weights, to each band."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels, 1),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class LocalFrequencyRelation(nn.Module):
    """Center minus an attention-weighted neighborhood, excluding the center.

    No masks or task IDs enter the operator. Similarity is calculated between
    learned queries and keys at the same spatial scale. Padding positions do
    not participate in the softmax.
    """

    def __init__(self, channels: int, kernel_size: int = 3, query_channels: int | None = None):
        super().__init__()
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("relation kernel must be odd and at least 3")
        self.kernel_size = kernel_size
        self.query_channels = query_channels or min(32, max(4, channels // 4))
        self.input_norm = LayerNorm2d(channels)
        self.query = nn.Conv2d(channels, self.query_channels, 1, bias=False)
        self.key = nn.Conv2d(channels, self.query_channels, 1, bias=False)
        self.project = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            LayerNorm2d(channels),
            nn.GELU(),
        )

    def difference(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        k = self.kernel_size
        neighbors = k * k
        normalized = self.input_norm(x)
        query = self.query(normalized).flatten(2).unsqueeze(2)
        key = F.unfold(self.key(normalized), k, padding=k // 2)
        key = key.reshape(b, self.query_channels, neighbors, h * w)
        logits = (query.float() * key.float()).sum(1) / math.sqrt(self.query_channels)
        valid = F.unfold(x.new_ones((1, 1, h, w)), k, padding=k // 2).bool()
        valid[:, neighbors // 2] = False
        logits = logits.masked_fill(~valid, -torch.finfo(logits.dtype).max)
        attention = logits.softmax(dim=1) * valid
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        values = F.unfold(x, k, padding=k // 2).reshape(b, c, neighbors, h * w)
        context = (values * attention.to(x.dtype).unsqueeze(1)).sum(2).reshape(b, c, h, w)
        has_context = valid.any(1).reshape(1, 1, h, w)
        return (x - context) * has_context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.difference(x))


class SpatialFrequencyFusion(nn.Module):
    """Keep three band streams, use per-pixel selection and two-way feedback."""

    def __init__(self, channels: int, relation_kernel: int = 3):
        super().__init__()
        self.frequency_relation = LocalFrequencyRelation(channels, relation_kernel)
        hidden = max(16, channels // 4)
        self.band_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 3, 1)
        )
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 3, 1), nn.Sigmoid()
        )
        self.frequency_to_spatial = nn.Conv2d(channels, channels, 1, bias=False)
        self.spatial_to_frequency = nn.Conv2d(channels, channels, 1, bias=False)
        self.output = nn.Sequential(nn.Conv2d(channels * 2, channels, 1, bias=False), LayerNorm2d(channels))
        self.spatial_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.frequency_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))
        self.output_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, spatial: torch.Tensor, bands):
        if len(bands) != 3 or any(band.shape != spatial.shape for band in bands):
            raise ValueError("fusion expects three band maps with the spatial feature shape")
        b = spatial.shape[0]
        relations = self.frequency_relation(torch.cat(tuple(bands), dim=0)).split(b, dim=0)
        descriptor = torch.cat((spatial, *relations), dim=1)
        weights = self.band_gate(descriptor).float().softmax(dim=1).to(spatial.dtype)
        feedback = self.feedback_gate(descriptor)
        selected = sum(weights[:, i:i + 1] * relation for i, relation in enumerate(relations))
        spatial_out = spatial + self.spatial_scale * self.frequency_to_spatial(selected)
        projected_spatial = self.spatial_to_frequency(spatial)
        bands_out = tuple(
            band + self.frequency_scale * (relation + feedback[:, i:i + 1] * projected_spatial)
            for i, (band, relation) in enumerate(zip(bands, relations))
        )
        selected_bands = sum(weights[:, i:i + 1] * band for i, band in enumerate(bands_out))
        fused = spatial_out + self.output_scale * self.output(torch.cat((spatial_out, selected_bands), dim=1))
        return spatial_out, bands_out, fused, weights

