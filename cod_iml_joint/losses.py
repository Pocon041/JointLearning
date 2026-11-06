"""Symmetric task segmentation objectives; absent labels are never fabricated."""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F


def validate_weight(weight: float, name: str) -> float:
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return weight


def validate_mask(logits: torch.Tensor, target: torch.Tensor) -> None:
    if logits.ndim != 4 or logits.shape[1] != 1 or target.shape != logits.shape:
        raise ValueError("logits and target must have the same nonempty [B,1,H,W] shape")
    if any(size == 0 for size in logits.shape):
        raise ValueError("logits and mask must be nonempty")
    if logits.device != target.device:
        raise ValueError("logits and target must be on the same device")
    if not logits.is_floating_point():
        raise TypeError("logits must be floating point")
    if not torch.isfinite(logits).all() or not torch.isfinite(target).all():
        raise FloatingPointError("logits and target must be finite")
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("mask values must be within [0,1]")


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    validate_mask(logits, target)
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    probability = logits.float().sigmoid()
    target = target.float()
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dims)
    denominator = probability.square().sum(dims) + target.square().sum(dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """BCE plus squared Dice, reduced per sample before the batch mean."""
    validate_mask(logits, target)
    return (F.binary_cross_entropy_with_logits(logits.float(), target.float())
            + soft_dice_loss(logits, target))


def mask_boundary(target: torch.Tensor) -> torch.Tensor:
    """3x3 morphological boundary from this task's supplied mask only."""
    target = target.float()
    dilated = F.max_pool2d(target, 3, 1, 1)
    eroded = -F.max_pool2d(-target, 3, 1, 1)
    return (dilated - eroded).clamp(0, 1)


def joint_loss(
    outputs: Mapping[str, torch.Tensor], cod_mask=None, iml_mask=None,
    cod_weight: float = 1.0, iml_weight: float = 1.0,
    boundary_weight: float = 0.0,
):
    """Loss for requested labels, supporting either or both task predictions.

    COD and IML tensors may come from different forward calls and need not have
    equal batch/image sizes. ``boundary_weight`` is opt-in and is scaled by
    ``cod_weight`` alongside the COD segmentation loss.
    """
    cod_weight = validate_weight(cod_weight, "cod_weight")
    iml_weight = validate_weight(iml_weight, "iml_weight")
    boundary_weight = validate_weight(boundary_weight, "boundary_weight")
    values, total = {}, None
    for name, target, weight in (("cod", cod_mask, cod_weight), ("iml", iml_mask, iml_weight)):
        if target is None or weight == 0:
            continue
        if name not in outputs:
            raise ValueError(f"missing {name} prediction for its supplied mask")
        values[name] = segmentation_loss(outputs[name], target)
        weighted = weight * values[name]
        total = weighted if total is None else total + weighted
    if cod_mask is not None and cod_weight > 0 and boundary_weight > 0:
        if "cod_boundary" not in outputs:
            raise ValueError("boundary_weight requires cod_boundary logits")
        values["cod_boundary"] = segmentation_loss(outputs["cod_boundary"], mask_boundary(cod_mask))
        total = total + cod_weight * boundary_weight * values["cod_boundary"]
    if total is None:
        raise ValueError("at least one task requires a supplied mask and positive weight")
    if not torch.isfinite(total):
        raise FloatingPointError("nonfinite joint loss")
    values["total"] = total
    return total, values
