"""Original task objectives restricted to real pixels, averaged per image."""
import math

import torch
from torch.nn import functional as F


def task_loss(task, logits, target, *, valid_mask=None, iml_dice_weight=0.):
    if task not in ("cod", "iml"):
        raise ValueError("unknown task")
    if not math.isfinite(iml_dice_weight) or iml_dice_weight < 0:
        raise ValueError("invalid IML Dice weight")
    if logits.ndim != 4 or logits.shape[1] != 1 or target.shape != logits.shape:
        raise ValueError("matching B1HW prediction and target required")
    if valid_mask is None:
        valid_mask = torch.ones_like(target, dtype=torch.bool)
    if valid_mask.shape != target.shape or valid_mask.dtype != torch.bool or valid_mask.device != logits.device:
        raise ValueError("valid_mask must be boolean B1HW on the logits device")
    count = valid_mask.sum((1, 2, 3))
    if bool((count == 0).any()):
        raise ValueError("every image must contain valid pixels")
    z = torch.where(valid_mask, logits.float(), 0.)
    y = torch.where(valid_mask, target.float(), 0.)
    if not torch.isfinite(z).all() or not torch.isfinite(y).all() or bool(((y < 0) | (y > 1)).any()):
        raise ValueError("valid pixels must have finite logits and targets within [0,1]")
    bce = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * valid_mask).sum((1, 2, 3)) / count
    probability = z.sigmoid() * valid_mask
    intersection = (probability * y).sum((1, 2, 3))
    p_sum, y_sum = probability.sum((1, 2, 3)), y.sum((1, 2, 3))
    if task == "cod":
        return (bce + 1 - intersection / (p_sum + y_sum - intersection).clamp_min(1e-6)).mean()
    if iml_dice_weight:
        bce = bce + iml_dice_weight * (1 - (2 * intersection + 1.) / (p_sum + y_sum + 1.))
    return bce.mean()
