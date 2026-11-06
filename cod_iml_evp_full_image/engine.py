"""Optional isolated objective and gradient-clipping interventions.

Defaults reproduce the existing losses and global clipping. Loss/clip changes
must be explicitly selected in a recorded experiment config, not silently
combined with the local-bridge architecture change.
"""
from contextlib import nullcontext
import math

import torch
from torch.nn import functional as F

from .losses import task_loss


def joint_train_step(model, optimizer, cod_batch=None, iml_batch=None, cod_weight=1., iml_weight=1.,
                     *, boundary_weight=0., amp=False, max_grad_norm=5.,
                     iml_dice_weight=0., clip_mode="global"):
    from cod_iml_joint.engine import _validate_batch
    if boundary_weight != 0 or clip_mode not in ["global", "roles"]:
        raise ValueError("Unsupported boundary loss or clipping mode")
    if max_grad_norm is not None and (not math.isfinite(max_grad_norm) or max_grad_norm <= 0):
        raise ValueError("Invalid maximum gradient norm")
    if any(not math.isfinite(w) or w < 0 for w in [cod_weight, iml_weight]):
        raise ValueError("Invalid task weight")
    batches = [(t, b, w) for t, b, w in [("cod", cod_batch, cod_weight), ("iml", iml_batch, iml_weight)]
               if b is not None and w > 0]
    if not batches:
        raise ValueError("No active task")
    for t, b, _ in batches:
        _validate_batch(b, t)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    logs, previous, total = {}, None, None
    shared = [p for g in optimizer.param_groups if g["name"].startswith("shared") for p in g["params"]]
    try:
        for task, batch, weight in batches:
            context = torch.autocast(batch["image"].device.type, dtype=torch.bfloat16) if amp else nullcontext()
            with context:
                kwargs = {"processed_sizes": batch["processed_size"]} if "processed_size" in batch else {}
                logits = model(batch["image"], task=task, **kwargs)[task]
                loss = task_loss(task, logits, batch["mask"], iml_dice_weight=iml_dice_weight, valid_mask=batch.get("valid_mask"))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            (weight * loss).backward()
            logs[task] = loss.detach()
            total = weight * loss.detach() if total is None else total + weight * loss.detach()
            if len(batches) == 2:
                current = torch.cat([(p.grad.detach().float() if p.grad is not None else torch.zeros_like(p)).flatten()
                                     for p in shared])
                gradient = current if previous is None else current - previous
                logs["shared_grad_" + task] = gradient.norm()
                if previous is not None:
                    logs["shared_grad_cosine"] = F.cosine_similarity(previous[None], gradient[None], eps=1e-12)[0]
                previous = current
        role_params = {}
        for group in optimizer.param_groups:
            role = "shared" if group["name"].startswith("shared") else (
                "cod_private" if group["name"].startswith("cod_private") else "iml_private")
            role_params.setdefault(role, []).extend(p for p in group["params"] if p.grad is not None)
        params = [p for values in role_params.values() for p in values]
        if not params or any(not torch.isfinite(p.grad).all() for p in params):
            raise FloatingPointError("Missing or nonfinite gradients")
        for role, values in role_params.items():
            logs["gradient_norm_" + role] = torch.stack([p.grad.detach().float().square().sum() for p in values]).sum().sqrt() if values else total.new_zeros(())
        if max_grad_norm is not None:
            if clip_mode == "global":
                logs["grad_norm"] = torch.nn.utils.clip_grad_norm_(params, max_grad_norm, error_if_nonfinite=True)
            else:
                for values in role_params.values():
                    if values:
                        torch.nn.utils.clip_grad_norm_(values, max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    logs["total"] = total
    return logs
