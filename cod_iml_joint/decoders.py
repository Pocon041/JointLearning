"""Private COD/IML readers and decoders for the shared dual-domain pyramid.

Every decoder sees exactly the same shared ``fused`` and low/mid/high ``bands``
features. Frequency preferences are learned per pixel inside each decoder;
there is no task-specific encoder, external adapter or cross-task mask exchange.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import ConvNormAct, norm2d


class TaskBandReader(nn.Module):
    """Task-private, spatially varying low/mid/high feature selection."""

    def __init__(self, channels: int, width: int):
        super().__init__()
        self.spatial_projection = nn.Conv2d(channels, width, 1, bias=False)
        self.band_projections = nn.ModuleList(
            nn.Conv2d(channels, width, 1, bias=False) for _ in range(3)
        )
        # Keep H and W: global pooling would erase the desired per-pixel gate.
        self.gate = nn.Conv2d(4 * width, 3, 1)
        self.combine = ConvNormAct(2 * width, width, 1, 1)

    def forward(self, fused: torch.Tensor, bands: Sequence[torch.Tensor]):
        if len(bands) != 3 or any(band.shape != fused.shape for band in bands):
            raise ValueError("each scale requires three bands matching its fused feature")
        spatial = self.spatial_projection(fused)
        projected = [layer(band) for layer, band in zip(self.band_projections, bands)]
        weights = self.gate(torch.cat((spatial, *projected), dim=1)).softmax(dim=1)
        selected = sum(weights[:, index:index + 1] * band for index, band in enumerate(projected))
        return self.combine(torch.cat((spatial, selected), dim=1))


class FPNDecoderBase(nn.Module):
    """Top-down pyramid: each refinement contributes to the final prediction."""

    def __init__(self, in_channels: Sequence[int], width: int = 96):
        super().__init__()
        if len(in_channels) != 4 or width <= 0:
            raise ValueError("a decoder requires four channel widths and positive width")
        self.in_channels = tuple(in_channels)
        self.width = width
        self.readers = nn.ModuleList(TaskBandReader(channels, width) for channels in in_channels)
        self.refine = nn.ModuleList(ConvNormAct(width, width, 3, 1) for _ in in_channels)

    def pyramid(self, features: Mapping[str, Sequence]):
        if not isinstance(features, Mapping) or "fused" not in features or "bands" not in features:
            raise ValueError("decoder input must contain fused and bands pyramids")
        fused, bands = features["fused"], features["bands"]
        if len(fused) != 4 or len(bands) != 4:
            raise ValueError("decoder input must contain exactly four scales")
        projected = [reader(feature, scale_bands)
                     for reader, feature, scale_bands in zip(self.readers, fused, bands)]
        current = self.refine[-1](projected[-1])
        outputs = [None] * 4
        outputs[-1] = current
        for index in range(2, -1, -1):
            current = self.refine[index](
                projected[index] + F.interpolate(current, size=projected[index].shape[-2:],
                                                  mode="bilinear", align_corners=False)
            )
            outputs[index] = current
        return tuple(outputs)


class CODDecoder(FPNDecoderBase):
    """Region/contour refinement with optional explicit boundary supervision."""

    def __init__(self, in_channels: Sequence[int], width: int = 96):
        super().__init__(in_channels, width)
        self.boundary = nn.Sequential(ConvNormAct(width, width, 3, 1), nn.Conv2d(width, 1, 1))
        self.local_structure = ConvNormAct(width, width, 3, 1)
        self.object_gate = nn.Sequential(nn.Conv2d(width * 2, width, 1), nn.Sigmoid())
        self.context_projection = nn.Conv2d(width, width, 1, bias=False)
        self.output = nn.Sequential(ConvNormAct(width, width, 3, 1), nn.Conv2d(width, 1, 1))

    def forward(self, features, output_size, return_aux: bool = False):
        high_resolution = self.pyramid(features)[0]
        boundary = self.boundary(high_resolution)
        context = F.avg_pool2d(high_resolution, 7, 1, 3, count_include_pad=False)
        structure = self.local_structure(high_resolution)
        object_gate = self.object_gate(torch.cat((high_resolution, context), dim=1))
        refined = (high_resolution + boundary.sigmoid() * structure
                   + object_gate * self.context_projection(context))
        logits = F.interpolate(self.output(refined), size=output_size,
                               mode="bilinear", align_corners=False)
        if return_aux:
            return {"logits": logits,
                    "boundary": F.interpolate(boundary, size=output_size,
                                               mode="bilinear", align_corners=False)}
        return logits


class IMLDecoder(FPNDecoderBase):
    """Fine-resolution center/context differences augment the region pyramid."""

    def __init__(self, in_channels: Sequence[int], width: int = 96):
        super().__init__(in_channels, width)
        self.local_residual = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            nn.Conv2d(width, width, 1, bias=False), norm2d(width), nn.GELU(),
        )
        self.output = nn.Sequential(ConvNormAct(width * 2, width, 3, 1), nn.Conv2d(width, 1, 1))

    def forward(self, features, output_size, return_aux: bool = False):
        high_resolution = self.pyramid(features)[0]
        context = F.avg_pool2d(high_resolution, 3, 1, 1, count_include_pad=False)
        residual = self.local_residual(high_resolution - context)
        logits = self.output(torch.cat((high_resolution, residual), dim=1))
        return F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
