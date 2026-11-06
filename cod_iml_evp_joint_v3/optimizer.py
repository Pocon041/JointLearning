"""Task-wise Adam histories on shared parameters, normal AdamW on private ones.

AdaTask's core is separate task histories and a SUM of adaptive proposals. That
is the default here. Mean aggregation is explicitly a separate adaptation.
AdamW decay, role learning rates and missing-gradient handling are engineering
extensions to the author's Adam reference, not a claim of full reproduction.
"""
import math

import torch
from torch.optim import Optimizer


class TaskAdamW(Optimizer):
    def __init__(self, groups, *, tasks=("cod", "iml"), betas=(.9, .999), eps=1e-8, aggregation="sum"):
        if aggregation not in ("sum", "mean") or not tasks or len(set(tasks)) != len(tasks):
            raise ValueError("explicit unique tasks and sum/mean aggregation required")
        if len(betas) != 2 or any(not 0 <= b < 1 for b in betas) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("invalid Adam hyperparameters")
        self.tasks = tuple(tasks)
        self.aggregation = aggregation
        super().__init__(groups, dict(betas=tuple(betas), eps=float(eps)))
        seen = set()
        for group in self.param_groups:
            if group.get("role") not in ("shared", "cod_private", "iml_private"):
                raise ValueError("every group requires an explicit parameter role")
            if not math.isfinite(group["lr"]) or group["lr"] < 0 or not math.isfinite(group.get("weight_decay", 0.)) or group.get("weight_decay", 0.) < 0:
                raise ValueError("invalid learning rate or weight decay")
            for p in group["params"]:
                if id(p) in seen:
                    raise ValueError("overlapping optimizer parameters")
                if not p.is_floating_point() or p.dtype not in (torch.float32, torch.float64):
                    raise ValueError("Adam states require FP32/FP64 master parameters")
                seen.add(id(p))

    def shared_parameters(self):
        return [p for group in self.param_groups if group["role"] == "shared" for p in group["params"]]

    def state_dict(self):
        result = super().state_dict()
        result["task_adamw"] = dict(tasks=list(self.tasks), aggregation=self.aggregation, version=1)
        return result

    def load_state_dict(self, saved):
        expected = dict(tasks=list(self.tasks), aggregation=self.aggregation, version=1)
        if saved.get("task_adamw") != expected:
            raise ValueError("task optimizer schema or aggregation differs")
        super().load_state_dict({k: v for k, v in saved.items() if k != "task_adamw"})

    @torch.no_grad()
    def step(self, closure=None, *, task_gradients=None):
        if closure is not None or not isinstance(task_gradients, dict) or not task_gradients:
            raise ValueError("supply explicit per-task shared gradients; closure is unsupported")
        if set(task_gradients) - set(self.tasks):
            raise ValueError("unknown task in shared gradients")
        shared = self.shared_parameters()
        # Validate the complete transaction before any parameter/moment mutation.
        for task, gradients in task_gradients.items():
            if len(gradients) != len(shared):
                raise ValueError("shared gradient count differs")
            for p, grad in zip(shared, gradients):
                if grad is not None:
                    self._validate_gradient(p, grad)
        for group in self.param_groups:
            if group["role"] != "shared":
                for p in group["params"]:
                    if p.grad is not None:
                        self._validate_gradient(p, p.grad)
        offset = 0
        active_tasks = [task for task in self.tasks if task in task_gradients]
        coefficient = 1. / len(active_tasks) if self.aggregation == "mean" else 1.
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, wd, eps = group["lr"], group.get("weight_decay", 0.), group["eps"]
            for p in group["params"]:
                if group["role"] == "shared":
                    gradients = [(task, task_gradients[task][offset]) for task in active_tasks]
                    offset += 1
                    gradients = [(task, grad) for task, grad in gradients if grad is not None]
                    if not gradients:
                        continue
                    slots = self.state[p].setdefault("task_moments", {})
                    if wd:
                        p.mul_(1. - lr * wd)  # Exactly once per shared optimizer update.
                    for task, grad in gradients:
                        state = slots.setdefault(task, {})
                        self._proposal_step(p, grad, state, beta1, beta2, eps, lr * coefficient)
                else:
                    if p.grad is None:
                        continue  # Missing foreign-task supervision: no moments/decay.
                    if wd:
                        p.mul_(1. - lr * wd)
                    self._proposal_step(p, p.grad, self.state[p], beta1, beta2, eps, lr)
        return None

    @staticmethod
    def _validate_gradient(p, grad):
        if (grad.is_sparse or grad.shape != p.shape or grad.dtype != p.dtype or grad.device != p.device):
            raise ValueError("gradient shape, dtype, device or layout differs")
        if not torch.isfinite(grad).all():
            raise FloatingPointError("nonfinite task gradient; update refused")

    @staticmethod
    def _proposal_step(p, grad, state, beta1, beta2, eps, lr):
        if not state:
            state.update(step=0, exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
        state["step"] += 1
        step = state["step"]
        state["exp_avg"].lerp_(grad, 1. - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1. - beta2)
        denom = state["exp_avg_sq"].sqrt().div_(math.sqrt(1. - beta2 ** step)).add_(eps)
        p.addcdiv_(state["exp_avg"], denom, value=-lr / (1. - beta1 ** step))


def build_optimizer(model, config):
    from cod_iml_evp_training_v2.strategy import build_optimizer as original
    base, report = original(model, config)
    mode = config.get("shared_optimizer", "adamw")
    if mode == "adamw":
        return base, report
    if mode not in ("task_adamw_sum", "task_adamw_mean"):
        raise ValueError("unknown shared optimizer")
    groups = []
    for group in base.param_groups:
        group = dict(group)
        name = group["name"]
        group["role"] = ("shared" if name.startswith("shared") else
                         "cod_private" if name.startswith("cod_private") else "iml_private")
        groups.append(group)
    return TaskAdamW(groups, betas=tuple(config.get("betas", (.9, .999))),
                     aggregation=mode.rsplit("_", 1)[-1]), report
