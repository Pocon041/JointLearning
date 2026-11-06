"""Energy-preserving descriptors and spatially anchored local frequency reads."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from cod_iml_evp_bridge_v1.modules import SignedBandBank


class ChannelNorm(nn.Module):
    """Normalize channels at a pixel, with FP32 statistics and no running state."""
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float().permute(0, 2, 3, 1)).permute(0, 3, 1, 2).to(dtype)


class SpatialResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
                                    nn.SiLU(), nn.Conv2d(channels, channels, 1, bias=False))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), .1))

    def forward(self, x):
        return x + self.scale * self.layers(self.norm(x))


class EnergyBandPyramid(nn.Module):
    """Independent scale descriptors avoid sequential strided signed cancellation.

    Every scale pools positive responses, negative responses, and squared
    responses BEFORE their signs can cancel. Difference(pos,neg) equals the
    pooled signed signal; RMS remains available separately. This is not a
    lossless transform after spatial reduction. The bounded gain avoids making
    vanishing bands unit-energy; zero input maps exactly to zero.
    """
    def __init__(self, widths, energy_floor=.1):
        super().__init__()
        if not math.isfinite(energy_floor) or energy_floor <= 0:
            raise ValueError("energy_floor must be positive")
        self.energy_floor = energy_floor
        self.stems = nn.ModuleList(nn.Conv2d(9, c, 1, bias=False) for c in widths)
        self.norms = nn.ModuleList(ChannelNorm(c) for c in widths)
        self.refine = nn.ModuleList(SpatialResidual(c) for c in widths)
        self.energy_skip = nn.ModuleList(nn.Conv2d(3, c, 1, bias=False) for c in widths)

    @staticmethod
    def descriptors(bands, size):
        b, k, c, h, w = bands.shape
        with torch.autocast(device_type=bands.device.type, enabled=False):
            x = bands.float().reshape(b * k, c, h, w)
            positive = F.adaptive_avg_pool2d(x.clamp_min(0), size)
            negative = F.adaptive_avg_pool2d((-x).clamp_min(0), size)
            # sqrt(eps) subtraction keeps exact silence at zero, with bounded derivative.
            rms = (F.adaptive_avg_pool2d(x.square(), size) + 1e-8).sqrt() - 1e-4
        return torch.cat((positive, negative, rms), 1), rms

    def forward(self, bands):
        b, k, _, h, w = bands.shape
        output = []
        for i, (stem, norm, block, skip) in enumerate(zip(self.stems, self.norms, self.refine, self.energy_skip)):
            divisor = 4 * 2 ** i
            size = (math.ceil(h / divisor), math.ceil(w / divisor))
            descriptor, energy = self.descriptors(bands, size)
            gain = energy.mean(1, keepdim=True) / (energy.mean(1, keepdim=True) + self.energy_floor)
            x = block(norm(stem(descriptor))) * gain + skip(energy)
            output.append(x.reshape(b, k, -1, *size))
        return tuple(output)


class LocalRelationLibrary(nn.Module):
    """Shared band relation operators; receive supervision from either task."""
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.content_norm = ChannelNorm(channels)
        self.project = nn.Conv2d(4 * channels, channels, 1, bias=False)
        self.refine = SpatialResidual(channels)

    def forward(self, bands, content):
        b, k, c, h, w = bands.shape
        raw = bands.reshape(b * k, c, h, w)
        x = self.norm(raw)
        mean = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        context = self.content_norm(content)[:, None].expand(-1, k, -1, -1, -1).reshape_as(x)
        # Local residual retains an explicit spatial participant.
        return self.refine(raw + self.project(torch.cat((x, mean, x - mean, context), 1))).reshape_as(bands)


class LocalFrequencyBridge(nn.Module):
    """Same-image band-by-neighbour attention, anchored at every target pixel.

    Keys/values are SHARED, not foreign private task features. Queries/gates
    are task-private. Fixed relative offsets give the 3x3 neighbourhood spatial
    meaning; invalid padded neighbours receive zero attention probability.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = ChannelNorm(channels)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.local = nn.Conv2d(channels, channels, 1, bias=False)
        yy, xx = torch.meshgrid(torch.arange(-1, 2), torch.arange(-1, 2), indexing="ij")
        self.register_buffer("offset_prior", -(xx.square() + yy.square()).reshape(1, 1, 9, 1).float() / 2)
        self.band_bias = nn.Parameter(torch.zeros(1, 3, 1, 1))
        self.output_norm = ChannelNorm(channels)

    def forward(self, relation, query, *, return_aux=False):
        b, k, c, h, w = relation.shape
        source = relation.reshape(b * k, c, h, w)
        normalized = self.norm(source)
        keys = F.unfold(self.key(normalized), 3, padding=1).reshape(b, k, c, 9, h * w)
        values = F.unfold(self.value(source), 3, padding=1).reshape(b, k, c, 9, h * w)
        with torch.autocast(device_type=query.device.type, enabled=False):
            logits = (query.float().flatten(2)[:, None, :, None] * keys.float()).sum(2) / math.sqrt(c)
            logits = logits + self.offset_prior + self.band_bias.float()
            valid = F.unfold(torch.ones(1, 1, h, w, device=query.device), 3, padding=1) > 0
            logits = logits.masked_fill(~valid[:, None], float("-inf"))
            attention = logits.flatten(1, 2).softmax(1).reshape(b, k, 9, h * w)
            message = (attention[:, :, None] * values.float()).sum((1, 3)).reshape(b, c, h, w)
        message = self.output_norm(message.to(query.dtype) + self.local(relation.mean(1)))
        if not return_aux:
            return message
        probability = attention.detach().float().flatten(1, 2)
        entropy = -(probability * probability.clamp_min(1e-12).log()).sum(1).mean() / math.log(27)
        centered = message.detach().float() - message.detach().float().mean((-2, -1), keepdim=True)
        return message, {"attention_entropy": entropy,
                         "message_spatial_fraction": centered.square().mean().sqrt() /
                             message.detach().float().square().mean().sqrt().clamp_min(1e-12),
                         "invalid_attention_max": attention.detach().masked_select(
                             (~valid[:, None]).expand_as(attention)).abs().max()
                             if (~valid).any() else query.new_zeros(())}


class PrivateFrequencyReader(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.evidence = nn.Sequential(nn.Conv2d(3 * channels, channels, 1, bias=False), SpatialResidual(channels))
        self.norm = ChannelNorm(channels)
        self.query = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def prepare(self, bands, content):
        evidence = self.evidence(bands.flatten(1, 2))
        query = self.query(torch.cat((self.norm(evidence), self.norm(content)), 1))
        return evidence, query

    def finish(self, evidence, content, message):
        gate = self.gate(torch.cat((self.norm(evidence), self.norm(content), message), 1)).sigmoid()
        return torch.cat((evidence, gate * message), 1), gate
