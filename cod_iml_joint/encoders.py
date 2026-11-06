"""Task-shared ConvNeXt spatial stages and lightweight band-preserving stages."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re

import torch
from torch import nn

from .blocks import ConvNeXtBlock, FrequencyBlock, LayerNorm2d, SpatialFrequencyFusion


class ConvNeXtSpatialEncoder(nn.Module):
    """ConvNeXt-Tiny layout by default; no implicit checkpoint downloads."""

    def __init__(self, channels=(96, 192, 384, 768), depths=(3, 3, 9, 3), drop_path_rate=0.1):
        super().__init__()
        self.channels, self.depths = tuple(channels), tuple(depths)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, channels[0], 4, stride=4), LayerNorm2d(channels[0]))
        ] + [
            nn.Sequential(LayerNorm2d(channels[i - 1]), nn.Conv2d(channels[i - 1], channels[i], 2, stride=2))
            for i in range(1, 4)
        ])
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        start = 0
        for width, depth in zip(channels, depths):
            self.stages.append(nn.Sequential(*[
                ConvNeXtBlock(width, rates[start + i]) for i in range(depth)
            ]))
            start += depth

    def forward(self, x):
        features = []
        for downsample, stage in zip(self.downsample_layers, self.stages):
            x = stage(downsample(x))
            features.append(x)
        return tuple(features)

    def load_pretrained(self, path: str | Path, source: str = "auto"):
        """Load a local native/torchvision/Facebook ConvNeXt state strictly.

        Only the classifier/final classification norm may be discarded.
        All spatial encoder keys and shapes must match before any mutation.
        Initialization provenance should be recorded by the experiment caller.
        """
        state = torch.load(Path(path), map_location="cpu", weights_only=True)
        for key in ("model", "state_dict"):
            if isinstance(state, Mapping) and key in state:
                state = state[key]
        if not isinstance(state, Mapping):
            raise ValueError("spatial checkpoint must contain a state dictionary")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        if source == "auto":
            source = "torchvision" if any(k.startswith("features.") for k in state) else "native"
            if any(".dwconv." in k for k in state):
                source = "facebook"
        if source not in ("native", "torchvision", "facebook"):
            raise ValueError("source must be auto, native, torchvision, or facebook")
        converted, ignored = {}, []
        for key, value in state.items():
            name = key
            if source == "torchvision":
                match = re.fullmatch(r"features\.(\d+)\.(.*)", key)
                if match:
                    index, rest = int(match[1]), match[2]
                    if index % 2 == 0:
                        name = f"downsample_layers.{index // 2}.{rest}"
                    else:
                        name = f"stages.{index // 2}.{rest}"
                        name = name.replace(".layer_scale", ".gamma")
                        for old, new in ((".block.0.", ".depthwise."), (".block.2.", ".norm."),
                                         (".block.3.", ".expand."), (".block.5.", ".reduce.")):
                            name = name.replace(old, new)
                        if name.endswith(".gamma"):
                            value = value.flatten()
                elif key.startswith("classifier."):
                    ignored.append(key)
                    continue
            elif source == "facebook":
                for old, new in ((".dwconv.", ".depthwise."), (".pwconv1.", ".expand."),
                                 (".pwconv2.", ".reduce.")):
                    name = name.replace(old, new)
                if key.startswith(("head.", "norm.")):
                    ignored.append(key)
                    continue
            if name in converted:
                raise ValueError(f"duplicate converted checkpoint key: {name}")
            converted[name] = value
        expected = self.state_dict()
        missing, extra = set(expected) - set(converted), set(converted) - set(expected)
        wrong_shape = [key for key in set(expected) & set(converted)
                       if not isinstance(converted[key], torch.Tensor) or converted[key].shape != expected[key].shape]
        if missing or extra or wrong_shape:
            raise ValueError(f"spatial checkpoint mismatch: missing={sorted(missing)}, "
                             f"extra={sorted(extra)}, shape={wrong_shape}")
        self.load_state_dict(converted, strict=True)
        return {"source": source, "path": str(Path(path).resolve()), "loaded_keys": len(converted),
                "ignored_keys": ignored}


class BandFrequencyEncoder(nn.Module):
    """A lightweight pyramid whose weights are tied across three band inputs.

    A band has its own activation tensor throughout the encoder. Batch-wise
    concatenation reuses convolution weights; it never averages the bands.
    The 4x4/stride4 stem and 2x2/stride2 downsamples match the spatial grid.
    """

    def __init__(self, channels: Sequence[int]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], 4, stride=4), LayerNorm2d(channels[0]), nn.GELU()
        )
        self.downsample_layers = nn.ModuleList([
            nn.Identity()
        ] + [
            nn.Sequential(LayerNorm2d(channels[i - 1]), nn.Conv2d(channels[i - 1], channels[i], 2, stride=2))
            for i in range(1, 4)
        ])
        self.stages = nn.ModuleList(FrequencyBlock(channels[i]) for i in range(4))

    @staticmethod
    def apply_to_bands(module, bands):
        batch = bands[0].shape[0]
        return tuple(module(torch.cat(tuple(bands), dim=0)).split(batch, dim=0))


class SharedDualDomainEncoder(nn.Module):
    """Both tasks use exactly these modules, inputs, and feature pyramids."""

    def __init__(self, channels=(96, 192, 384, 768), depths=(3, 3, 9, 3),
                 drop_path_rate=0.1, relation_kernel=3):
        super().__init__()
        if len(channels) != 4 or len(depths) != 4 or any(c < 4 for c in channels) or any(d < 1 for d in depths):
            raise ValueError("encoder requires four positive depths and four channel widths >= 4")
        if not 0 <= drop_path_rate < 1:
            raise ValueError("drop_path_rate must be in [0,1)")
        self.channels, self.depths = tuple(channels), tuple(depths)
        self.spatial = ConvNeXtSpatialEncoder(channels, depths, drop_path_rate)
        self.frequency = BandFrequencyEncoder(channels)
        self.fusions = nn.ModuleList(SpatialFrequencyFusion(c, relation_kernel) for c in channels)
        self.register_buffer("rgb_mean", torch.tensor((.485, .456, .406))[None, :, None, None])
        self.register_buffer("rgb_std", torch.tensor((.229, .224, .225))[None, :, None, None])

    def forward(self, x, low, mid, high):
        # Input x is raw RGB/255. DCT bands retain signed responses and their
        # original scale; RGB normalization happens only on the spatial path.
        spatial = (x - self.rgb_mean) / self.rgb_std
        bands = self.frequency.apply_to_bands(self.frequency.stem, (low, mid, high))
        fused_pyramid, band_pyramid, band_weights = [], [], []
        for index in range(4):
            spatial = self.spatial.stages[index](self.spatial.downsample_layers[index](spatial))
            if index:
                bands = self.frequency.apply_to_bands(self.frequency.downsample_layers[index], bands)
            bands = self.frequency.apply_to_bands(self.frequency.stages[index], bands)
            spatial, bands, fused, weights = self.fusions[index](spatial, bands)
            fused_pyramid.append(fused)
            band_pyramid.append(bands)
            band_weights.append(weights)
        return {"fused": tuple(fused_pyramid), "bands": tuple(band_pyramid),
                "band_weights": tuple(band_weights)}
