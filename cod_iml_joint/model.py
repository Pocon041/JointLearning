"""Symmetric COD + IML model: one RGB/frequency backbone, two private decoders."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .decoders import CODDecoder, IMLDecoder
from .encoders import SharedDualDomainEncoder
from .frequency import LocalDCTBank


class JointCODIMLModel(nn.Module):
    """Inputs are float RGB in [0,1]; outputs are raw mask logits.

    Both tasks call encode() unchanged. task selects only a decoder. Padding
    extends the same image for both modalities; predictions are decoded on
    that padded grid and cropped back, rather than stretched to the input.
    """

    def __init__(self, channels=(96, 192, 384, 768), depths=(3, 3, 9, 3),
                 decoder_width=96, dct_stride=4, drop_path_rate=0.1, relation_kernel=3):
        super().__init__()
        self.config = {
            "channels": tuple(channels), "depths": tuple(depths),
            "decoder_width": decoder_width, "dct_stride": dct_stride,
            "drop_path_rate": drop_path_rate, "relation_kernel": relation_kernel,
        }
        self.frequency_bank = LocalDCTBank(patch_size=8, stride=dct_stride)
        self.encoder = SharedDualDomainEncoder(channels, depths, drop_path_rate, relation_kernel)
        self.cod_decoder = CODDecoder(channels, decoder_width)
        self.iml_decoder = IMLDecoder(channels, decoder_width)

    def encode(self, x: torch.Tensor):
        if (x.ndim != 4 or x.shape[1] != 3 or x.shape[0] < 1
                or min(x.shape[-2:]) < 8 or not x.is_floating_point()):
            raise ValueError("input must be a nonempty floating [B,3,H,W] RGB tensor, H,W >= 8")
        h, w = x.shape[-2:]
        pad_h, pad_w = (-h) % 32, (-w) % 32
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate") if pad_h or pad_w else x
        low, mid, high = self.frequency_bank(padded)
        features = self.encoder(padded, low, mid, high)
        features.update(input_size=(h, w), padded_size=padded.shape[-2:])
        return features

    def forward(self, x: torch.Tensor, task: str | None = None, *, return_aux=False):
        if task not in (None, "cod", "iml"):
            raise ValueError("task must be None, cod, or iml")
        features = self.encode(x)
        h, w = features["input_size"]
        outputs = {}
        if task in (None, "cod"):
            prediction = self.cod_decoder(features, features["padded_size"], return_aux=return_aux)
            if return_aux:
                outputs["cod"] = prediction["logits"][..., :h, :w]
                outputs["cod_boundary"] = prediction["boundary"][..., :h, :w]
            else:
                outputs["cod"] = prediction[..., :h, :w]
        if task in (None, "iml"):
            outputs["iml"] = self.iml_decoder(features, features["padded_size"])[..., :h, :w]
        return outputs

    def parameter_groups(self):
        """Disjoint shared, COD-private and IML-private groups."""
        return {
            "shared": list(self.encoder.parameters()),
            "cod_private": list(self.cod_decoder.parameters()),
            "iml_private": list(self.iml_decoder.parameters()),
        }

    def load_spatial_pretrained(self, path, source="auto"):
        return self.encoder.spatial.load_pretrained(path, source)

