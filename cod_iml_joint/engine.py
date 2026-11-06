"""Dataset-independent symmetric optimization for unpaired COD/IML batches."""
from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Mapping, Optional

import torch

from .losses import joint_loss, validate_weight


def _validate_batch(batch: Mapping[str, torch.Tensor], task: str) -> None:
    if not isinstance(batch, Mapping) or "image" not in batch or "mask" not in batch:
        raise ValueError(f"{task} batch must supply image and mask tensors")
    image, mask = batch["image"], batch["mask"]
    if not isinstance(image, torch.Tensor) or not isinstance(mask, torch.Tensor):
        raise TypeError(f"{task} image and mask must be tensors")
    if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] == 0:
        raise ValueError(f"{task} image must have nonempty [B,3,H,W] shape")
    if mask.shape != (image.shape[0], 1, *image.shape[-2:]):
        raise ValueError(f"{task} mask must have [B,1,H,W] matching its image")
    if not image.is_floating_point():
        raise TypeError(f"{task} image must be floating point")
    if image.device != mask.device:
        raise ValueError(f"{task} image and mask must be on the same device")
    if not torch.isfinite(image).all() or not torch.isfinite(mask).all():
        raise FloatingPointError(f"{task} image and mask must be finite")
    if bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError(f"{task} mask values must be within [0,1]")


def joint_train_step(
    model, optimizer,
    cod_batch: Optional[Mapping[str, torch.Tensor]] = None,
    iml_batch: Optional[Mapping[str, torch.Tensor]] = None,
    cod_weight: float = 1.0, iml_weight: float = 1.0,
    *, boundary_weight: float = 0.0, amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16, max_grad_norm: float | None = None,
):
    """Accumulate two task gradients and take exactly one optimizer update.

    Unpaired task batches can differ in batch size and image resolution. Only
    each batch's labeled decoder runs. Gradients are accumulated sequentially
    to release each forward graph before constructing the next. Equal task
    weights are defaults; neither task acts as the primary/anchor task.

    BF16 autocast is optional; FP16 scaling is deliberately outside this API.
    Optimizer updates are skipped on nonfinite loss/gradient and all partially
    accumulated gradients are cleared before re-raising.
    """
    cod_weight = validate_weight(cod_weight, "cod_weight")
    iml_weight = validate_weight(iml_weight, "iml_weight")
    boundary_weight = validate_weight(boundary_weight, "boundary_weight")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("max_grad_norm must be finite and positive")
    if amp and amp_dtype != torch.bfloat16:
        raise ValueError("joint_train_step supports BF16 autocast only")
    active = [("cod", cod_batch, cod_weight), ("iml", iml_batch, iml_weight)]
    active = [(name, batch, weight) for name, batch, weight in active if batch is not None and weight > 0]
    if not active:
        raise ValueError("at least one task batch with positive weight is required")
    for name, batch, _ in active:
        _validate_batch(batch, name)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    values, total_value = {}, None
    try:
        for task, batch, weight in active:
            context = (torch.autocast(device_type=batch["image"].device.type, dtype=amp_dtype)
                       if amp else nullcontext())
            with context:
                predictions = model(batch["image"], task=task,
                                    return_aux=(task == "cod" and boundary_weight > 0))
                masks = {f"{task}_mask": batch["mask"]}
                weighted_loss, task_values = joint_loss(
                    predictions, **masks,
                    cod_weight=weight if task == "cod" else 0.0,
                    iml_weight=weight if task == "iml" else 0.0,
                    boundary_weight=boundary_weight if task == "cod" else 0.0,
                )
            if not torch.isfinite(weighted_loss):
                raise FloatingPointError(f"nonfinite {task} loss")
            weighted_loss.backward()
            values.update({name: value.detach() for name, value in task_values.items() if name != "total"})
            detached = weighted_loss.detach()
            total_value = detached if total_value is None else total_value + detached
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]
                      if parameter.grad is not None]
        if not parameters or any(not torch.isfinite(parameter.grad).all() for parameter in parameters):
            raise FloatingPointError("missing or nonfinite gradients; optimizer update skipped")
        if max_grad_norm is not None:
            values["grad_norm"] = torch.nn.utils.clip_grad_norm_(
                parameters, max_grad_norm, error_if_nonfinite=True).detach()
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    values["total"] = total_value
    return values
