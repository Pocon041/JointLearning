"""Sequential unpaired graphs, simultaneous task-wise optimizer transaction."""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from cod_iml_evp_full_image.losses import task_loss
from cod_iml_evp_full_image.engine import joint_train_step as original_step
from .optimizer import TaskAdamW


def joint_train_step(model, optimizer, cod_batch=None, iml_batch=None, cod_weight=1., iml_weight=1.,
                     *, boundary_weight=0., amp=False, max_grad_norm=5., iml_dice_weight=0., clip_mode="global"):
    if not isinstance(optimizer, TaskAdamW):
        return original_step(model, optimizer, cod_batch=cod_batch, iml_batch=iml_batch,
            cod_weight=cod_weight, iml_weight=iml_weight, boundary_weight=boundary_weight,
            amp=amp, max_grad_norm=max_grad_norm, iml_dice_weight=iml_dice_weight, clip_mode=clip_mode)
    from cod_iml_joint.engine import _validate_batch
    if boundary_weight != 0 or clip_mode != "global":
        raise ValueError("task-wise comparison retains original global clipping and no boundary loss")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("invalid gradient clipping limit")
    if any(not math.isfinite(w) or w < 0 for w in (cod_weight, iml_weight)):
        raise ValueError("invalid task weight")
    active = [(task, batch, weight) for task, batch, weight in
              (("cod", cod_batch, cod_weight), ("iml", iml_batch, iml_weight)) if batch is not None and weight > 0]
    if not active:
        raise ValueError("no supervised task")
    for task, batch, _ in active:
        _validate_batch(batch, task)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    shared = optimizer.shared_parameters()
    gradients, logs, total = {}, {}, None
    try:
        for task, batch, weight in active:
            for p in shared:
                p.grad = None
            kwargs = {"processed_sizes": batch["processed_size"]} if "processed_size" in batch else {}
            with torch.autocast(batch["image"].device.type, dtype=torch.bfloat16) if amp else nullcontext():
                logits = model(batch["image"], task=task, **kwargs)[task]
                loss = task_loss(task, logits, batch["mask"], valid_mask=batch.get("valid_mask"), iml_dice_weight=iml_dice_weight)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite task loss")
            (weight * loss).backward()
            gradients[task] = [p.grad.detach().clone() if p.grad is not None else None for p in shared]
            logs[task] = loss.detach()
            total = weight * loss.detach() if total is None else total + weight * loss.detach()
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                for p, g in zip(shared, gradients[task])])
            logs["shared_grad_" + task] = vector.float().norm()
        if len(active) == 2:
            vectors = [torch.cat([(g if g is not None else torch.zeros_like(p)).flatten()
                                 for p, g in zip(shared, gradients[t])]).float() for t, _, _ in active]
            logs["shared_grad_cosine"] = F.cosine_similarity(vectors[0][None], vectors[1][None], eps=1e-12)[0]
        # Reconstruct the original aggregate gradient only for MATCHED clipping.
        # Adam histories receive the separate gradients scaled by this same factor.
        for i, p in enumerate(shared):
            parts = [gradients[t][i] for t, _, _ in active if gradients[t][i] is not None]
            p.grad = torch.stack(parts).sum(0) if parts else None
        params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
        if not params or any(not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError("missing or nonfinite gradients")
        for role in ("shared", "cod_private", "iml_private"):
            values = [p.grad.detach().float().square().sum() for group in optimizer.param_groups
                      if group["role"] == role for p in group["params"] if p.grad is not None]
            logs["gradient_norm_" + role] = torch.stack(values).sum().sqrt() if values else total.new_zeros(())
        if max_grad_norm is not None:
            norm = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
            coefficient = (max_grad_norm / (norm + 1e-6)).clamp(max=1.)
            for values in gradients.values():
                for grad in values:
                    if grad is not None:
                        grad.mul_(coefficient.to(grad.device))
            logs["grad_norm"] = norm
        optimizer.step(task_gradients=gradients)
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    logs["total"] = total
    return logs
