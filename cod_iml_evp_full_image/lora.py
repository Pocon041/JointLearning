"""Shared attention updates that preserve the generic pretrained key names."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from cod_iml_evp_official_aligned.model import convert_mmseg_mit


class SharedLowRankLinear(nn.Module):
    """Frozen original linear map plus a shared, initially zero low-rank update.

    Original `weight`/`bias` names and Parameter identities are preserved. There
    is no train/eval weight merging, so validation cannot mutate the base model.
    """
    def __init__(self, base, rank, alpha):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("a plain Linear projection is required")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("rank must be a positive integer within the projection dimensions")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if base.weight.requires_grad or (base.bias is not None and base.bias.requires_grad):
            raise ValueError("freeze the original projection before adding shared updates")
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight, self.bias = base.weight, base.bias
        self.rank, self.alpha = rank, float(alpha)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return F.linear(x, self.weight, self.bias) + (self.alpha / self.rank) * delta


def inject_shared_qkv(backbone, rank, alpha):
    """Both tasks use these same Q and fused-KV Parameters in all four stages."""
    names = []
    for stage in range(1, 5):
        for block_index, block in enumerate(getattr(backbone, f"block{stage}")):
            for part in ("q", "kv"):
                layer = getattr(block.attn, part)
                setattr(block.attn, part, SharedLowRankLinear(layer, rank, alpha))
                names.append(f"block{stage}.{block_index}.attn.{part}")
    return names


def load_generic_backbone_with_lora(backbone, path):
    """Allow only newly initialized LoRA tensors to be absent from ImageNet init.

    An adapted checkpoint cannot masquerade as generic initialization: any
    supplied low-rank tensor is unexpected here. Learned weights are restored
    later through the model's separate sparse-checkpoint method.
    """
    saved = torch.load(path, map_location="cpu", weights_only=False)
    saved = saved.get("state_dict", saved)
    clean = {}
    for key, value in saved.items():
        for prefix in ("module.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in clean:
            raise ValueError("duplicate generic pretrained key after prefix removal: " + key)
        clean[key] = value
    if clean and all(key.startswith("layers.") for key in clean):
        clean = convert_mmseg_mit(clean)
    expected = backbone.state_dict()
    low_rank = {key for key in expected if key.endswith((".lora_A", ".lora_B"))}
    generic = set(expected) - low_rank
    missing = generic - set(clean)
    ignored = sorted(set(clean) - generic)
    if missing or any(not key.startswith("head.") for key in ignored):
        raise ValueError("generic checkpoint keys differ; missing=" + str(sorted(missing)[:5]) +
                         ", unexpected=" + str([k for k in ignored if not k.startswith('head.')][:5]))
    for key in generic:
        value = clean[key]
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise ValueError("generic checkpoint tensor shape differs: " + key)
    result = backbone.load_state_dict({key: clean[key] for key in generic}, strict=False)
    if set(result.missing_keys) != low_rank or result.unexpected_keys:
        raise ValueError("generic loading omitted tensors other than the declared shared updates")
    return {"loaded_keys": len(generic), "ignored_keys": ignored,
            "new_low_rank_keys": len(low_rank), "source": "generic ImageNet1K MiT-B4"}
