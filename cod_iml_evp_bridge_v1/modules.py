"""Signed spectral inputs, private evidence, and shared task-view interaction.

All task views in a forward pass describe the SAME input image. No masks,
cross-image correspondences, pseudo-labels, or persistent sample memories enter
the bridge. Private source features are detached only at the bridge write edge.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


TASKS = ("cod", "iml")


class SignedBandBank(nn.Module):
    """Three smooth, conjugate-symmetric FFT bands whose responses sum to x.

    Radius uses cycles/pixel divided by axis Nyquist (0.5 cycles/pixel).
    These are bands of the resized model input, not JPEG coefficients or native
    camera fingerprints. The complete bank preserves phase and response sign.
    """

    def __init__(self, radii=(0.25, 0.65), transition=0.08):
        super().__init__()
        if len(radii) != 2 or not 0 < radii[0] < radii[1] < math.sqrt(2):
            raise ValueError("Expected two increasing radial cutoffs below sqrt(2)")
        if not math.isfinite(transition) or not 0 < transition < radii[0]:
            raise ValueError("Invalid spectral transition width")
        self.radii = tuple(float(v) for v in radii)
        self.transition = float(transition)

    def masks(self, height, width, device):
        fy = torch.fft.fftfreq(height, device=device)[:, None]
        fx = torch.fft.fftfreq(width, device=device)[None, :]
        radius = 2 * (fy.square() + fx.square()).sqrt()

        def lowpass(cutoff):
            position = ((radius - cutoff + self.transition) / (2 * self.transition)).clamp(0, 1)
            return 0.5 * (1 + torch.cos(math.pi * position))

        low, upper = (lowpass(c) for c in self.radii)
        return torch.stack((low, upper - low, 1 - upper))

    def forward(self, image):
        if image.ndim != 4 or image.shape[1] != 3 or not image.is_floating_point():
            raise ValueError("Band bank expects floating point BCHW RGB")
        with torch.autocast(device_type=image.device.type, enabled=False):
            spectrum = torch.fft.fft2(image.float(), norm="ortho")
            masks = self.masks(*image.shape[-2:], image.device)
            bands = torch.fft.ifft2(spectrum[:, None] * masks[None, :, None], norm="ortho").real
        return bands  # B, three bands, RGB, H, W


class BandPyramid(nn.Module):
    """Strided convolutions retain signed input and avoid shared BN statistics."""

    def __init__(self, in_channels, widths):
        super().__init__()
        self.layers = nn.ModuleList()
        for i, width in enumerate(widths):
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, width, 7 if i == 0 else 3,
                          stride=4 if i == 0 else 2, padding=3 if i == 0 else 1),
                nn.GELU(),
            ))
            in_channels = width

    def forward(self, image):
        result = []
        for layer in self.layers:
            image = layer(image)
            result.append(image)
        return tuple(result)


class LocalBandRelations(nn.Module):
    """Shared operator for band value, local mean, contrast, and RGB context."""

    def __init__(self, width):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(4 * width, width, 1), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.Conv2d(width, width, 1),
        )

    def forward(self, bands, content):
        b, k, c, h, w = bands.shape
        x = bands.reshape(b * k, c, h, w)
        mean = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        context = content[:, None].expand(-1, k, -1, -1, -1).reshape_as(x)
        return self.project(torch.cat((x, mean, x - mean, context), dim=1)).reshape(b, k, c, h, w)


class CrossAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, source):
        q = self.q(self.query_norm(query))
        source = self.source_norm(source)
        k, v = self.k(source), self.v(source)
        result = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None], dropout_p=0.)
        return self.output(result[:, 0])


class SharedRelationBridge(nn.Module):
    """Image-local hub: three band groups plus two private task-view sources.

    Source order is explicit. Optional self routing replaces only the foreign
    task view, preserving the token count and shared bridge parameterization.
    """

    def __init__(self, width, latents=16, pool_size=8):
        super().__init__()
        if latents < 1 or pool_size < 1:
            raise ValueError("Positive bridge sizes required")
        self.pool_size = int(pool_size)
        self.seeds = nn.Parameter(torch.randn(1, latents, width) * .02)
        self.source_embedding = nn.Parameter(torch.randn(1, 5, 1, width) * .02)
        self.attention = CrossAttention(width)
        self.refine = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * width),
                                    nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, relations, private_views, task, *, cross_enabled=True):
        b, k, c, h, w = relations.shape
        if k != 3 or task not in TASKS:
            raise ValueError("Bridge expects three frequency bands and a known task")
        views = dict(private_views)
        other = "iml" if task == "cod" else "cod"
        if not cross_enabled:
            views[other] = views[task]
        # Only these write edges are detached. The bridge and relation operator
        # remain trainable under BOTH losses; no donor private parameter is updated.
        sources = torch.cat((relations, views["cod"].detach()[:, None],
                             views["iml"].detach()[:, None]), dim=1)
        size = (min(h, self.pool_size), min(w, self.pool_size))
        tokens = F.adaptive_avg_pool2d(sources.reshape(b * 5, c, h, w), size)
        tokens = tokens.flatten(2).transpose(1, 2).reshape(b, 5, -1, c)
        tokens = (tokens + self.source_embedding).flatten(1, 2)
        query = self.seeds.expand(b, -1, -1) + tokens[:, :3 * size[0] * size[1]].mean(1, keepdim=True)
        hub = query + self.attention(query, tokens)
        return hub + self.refine(hub)


class TaskReader(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.attention = CrossAttention(width)
        self.gate = nn.Conv2d(3 * width, width, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, private_view, content, hub):
        b, c, h, w = private_view.shape
        query = (private_view + content).flatten(2).transpose(1, 2)
        message = self.attention(query, hub).transpose(1, 2).reshape(b, c, h, w)
        gate = self.gate(torch.cat((private_view, content, message), dim=1)).sigmoid()
        return gate * message, gate
