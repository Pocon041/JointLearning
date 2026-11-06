# Snapshot of training_v2 runner; sampling, losses, evaluator and scheduler retained.
# Source SHA256: dce6c4e3b5db51731c9cb2b87496791eb4ed44e91006d2066ab8c66b3e12d629
# Isolated EVP runner: shared trainable prompts, frozen generic backbone, matched task sampling.
# Reference train.py SHA256: 44fefb2a47b4aebf4d173f63157439b4c9f108757201e285fc894733a9191ad8
"""Auditable COD-only, IML-only and symmetric joint-training experiments.

Each arm uses the same per-task sampling, augmentation and stochastic-forward
seeds. Checkpoints resume only at completed epoch boundaries. Independent test
data are evaluated after training and never participate in model selection.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Sampler

from cod_iml_evp_bridge_v2.engine import joint_train_step, task_loss
from cod_iml_evp_training_v2.strategy import build_optimizer, make_scheduler
from sklearn.metrics import precision_recall_curve
from tools.casia_train import add_tampered_metrics
from cod_iml_joint.losses import segmentation_loss
from cod_iml_joint.metrics import SegmentationMetrics, StandardCODMetrics
from .model import ReferenceEVPModel as JointCODIMLModel


TASKS = ("cod", "iml")
ARM_TASKS = {"cod_only": ("cod",), "iml_only": ("iml",), "joint": TASKS}


def derived_seed(seed: int, *parts) -> int:
    content = ":".join(map(str, (seed, *parts))).encode()
    return int.from_bytes(hashlib.sha256(content).digest()[:8], "little") % (2**63 - 1)


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)


class EpochTaskSampler(Sampler):
    """Fixed sample count, repeated shuffled passes, task-local epoch seeds.

    A draw is ``(dataset_index, occurrence)``. The dataset uses occurrence to
    give repeated samples different reproducible geometric augmentations.
    """

    def __init__(self, length: int, draws: int, *, seed: int, task: str, epoch: int):
        if length < 1 or draws < 1:
            raise ValueError("sampler length and draws must be positive")
        self.length, self.draws = length, draws
        self.seed, self.task, self.epoch = seed, task, epoch

    def __len__(self):
        return self.draws

    def __iter__(self):
        drawn, cycle = 0, 0
        while drawn < self.draws:
            generator = torch.Generator().manual_seed(derived_seed(self.seed, self.task, self.epoch, cycle))
            for index in torch.randperm(self.length, generator=generator).tolist():
                if drawn == self.draws:
                    return
                yield index, drawn
                drawn += 1
            cycle += 1


class TaskSeededForward(nn.Module):
    """Keep DropPath and other stochastic forwards matched between arms."""

    def __init__(self, model: nn.Module, seed: int, device: torch.device):
        super().__init__()
        self.base_model, self.seed, self.device = model, seed, device
        self.epoch = self.step = 0

    def forward(self, image, task=None, **kwargs):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(derived_seed(self.seed, "forward", task, self.epoch, self.step))
            return self.base_model(image, task=task, **kwargs)


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_copy(source: Path, target: Path) -> None:
    # Source checkpoints are immutable inodes; later last.pt saves use replace.
    temporary = target.with_name(target.name + ".link.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state: dict) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        tensor = tensor.detach().cpu().contiguous()
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_rng() -> dict:
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        if not torch.cuda.is_available():
            raise ValueError("CUDA checkpoint cannot resume an exact run on CPU")
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("arm") not in ARM_TASKS:
        raise ValueError("arm must be cod_only, iml_only or joint")
    for key in ("output_dir", "spatial_pretrained", "initial_checkpoint"):
        if config.get(key):
            value = Path(config[key])
            config[key] = str(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
    if not config.get("output_dir"):
        raise ValueError("output_dir is required")
    def resolve_spec(spec):
        if isinstance(spec, str):
            value = Path(spec)
            return str(value.resolve() if value.is_absolute() else (path.parent / value).resolve())
        if isinstance(spec, dict):
            if "manifest" in spec:
                return {key: resolve_spec(value) if key in ("manifest", "root") and value is not None else value
                        for key, value in spec.items()}
            return {key: resolve_spec(value) for key, value in spec.items()}
        raise TypeError("data split must be a manifest path or mapping")
    config["data"] = {task: {split: resolve_spec(spec) for split, spec in splits.items()}
                      for task, splits in config["data"].items()}
    config.setdefault("seed", 2026)
    config.setdefault("epochs", 50)
    config.setdefault("size", [512, 512])
    if isinstance(config["size"], int):
        config["size"] = [config["size"], config["size"]]
    config.setdefault("batch_size", 4)
    config.setdefault("workers", 4)
    config.setdefault("amp", True)
    config.setdefault("evaluate_test", True)
    config.setdefault("cod_metrics", True)
    config.setdefault("warmup_epochs", 3)
    config.setdefault("min_lr_ratio", 0.01)
    config.setdefault("selection_metric", {"cod": "smeasure", "iml": "f1"})
    sizes = config["size"].values() if isinstance(config["size"], dict) else [config["size"]]
    if config["epochs"] < 1 or any(len(size)!=2 or min(size)<32 for size in sizes):
        raise ValueError("positive epochs and task image sizes >=32 required")
    if int(config["workers"]) < 0 or int(config.get("log_interval", 25)) < 1:
        raise ValueError("workers must be nonnegative and log_interval positive")
    return config


def manifest_spec(spec) -> dict:
    return {"manifest": spec} if isinstance(spec, str) else dict(spec)


def make_dataset(config, task: str, split: str, *, spec=None):
    from cod_iml_joint.data import ManifestDataset
    arguments = manifest_spec(spec if spec is not None else config["data"][task][split])
    return ManifestDataset(**arguments, size=tuple(config["size"][task] if isinstance(config["size"], dict) else config["size"]), training=split == "train",
                           task=task, seed=derived_seed(config["seed"], "data", task))


def task_batch_size(config, task):
    size = config["batch_size"]
    result = int(size[task] if isinstance(size, dict) else size)
    if result < 1:
        raise ValueError("batch sizes must be positive")
    return result


def steps_for_protocol(config: dict, lengths: dict) -> int:
    if set(lengths) != set(TASKS):
        raise ValueError("both task train manifests are required to match the optimizer/exposure budget")
    default = max(math.ceil(lengths[task] / task_batch_size(config, task)) for task in TASKS)
    steps = int(config.get("steps_per_epoch", default))
    if steps < 1:
        raise ValueError("steps_per_epoch must be positive")
    return steps


def train_loader(dataset, config, task: str, epoch: int, steps: int):
    from cod_iml_joint.data import seed_worker
    dataset.set_epoch(epoch)
    batch_size = task_batch_size(config, task)
    sampler = EpochTaskSampler(len(dataset), steps * batch_size, seed=config["seed"], task=task, epoch=epoch)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True,
                      num_workers=int(config["workers"]), pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(derived_seed(config["seed"], "worker", task, epoch)))


def eval_loader(dataset, config):
    from cod_iml_joint.data import seed_worker
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=int(config["workers"]),
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(derived_seed(config["seed"], "evaluation")))


def evaluate(model, loader, task: str, device, *, amp: bool = False, auroc_bins=4096,
             rows_path: Path | None = None, standard_cod: bool = True, iml_dice_weight: float = 0.) -> dict:
    was_training = model.training
    model.eval()
    accumulator = SegmentationMetrics(auroc_bins=auroc_bins)
    cod_accumulator = StandardCODMetrics() if task == "cod" and standard_cod else None
    family_accumulators = {}
    oracle_values = []
    loss_sum, rows = 0.0, []
    start = time.monotonic()
    try:
        with torch.inference_mode():
            for batch in loader:
                image = batch["image"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16) if amp else nullcontext():
                    logits = model(image, task=task)[task]
                loss_sum += float(task_loss(task, logits, mask, iml_dice_weight=iml_dice_weight))
                if "original_mask" not in batch:
                    raise ValueError("evaluation dataset must retain original_mask at native GT resolution")
                original = batch["original_mask"]
                if isinstance(original, (list, tuple)):
                    original = original[0]
                original = original.squeeze().cpu().numpy()
                if original.ndim != 2:
                    raise ValueError("original_mask must represent a two-dimensional GT")
                restored = F.interpolate(logits.float(), size=original.shape, mode="bilinear", align_corners=False)
                probability = restored.sigmoid()[0, 0].cpu().numpy()
                result = accumulator.update(probability, original)
                if task == "iml" and original.any():
                    precision, recall, _ = precision_recall_curve(original.reshape(-1).astype(np.int64), probability.reshape(-1))
                    oracle_values.append(float(np.max(2*precision*recall/(precision+recall+1e-10))))
                if cod_accumulator is not None:
                    cod_accumulator.update(probability, original)
                identifier = batch.get("id", [str(accumulator.count - 1)])[0]
                source = batch.get("source_id", [identifier])[0]
                row = {"id": str(identifier), "source_id": str(source), **result}
                if task == "iml" and "family" in batch:
                    family = batch["family"]
                    family = str(family[0] if isinstance(family, (list, tuple)) else family)
                    if family not in family_accumulators:
                        family_accumulators[family] = SegmentationMetrics(auroc_bins=auroc_bins)
                    family_accumulators[family].update(probability, original)
                    row["family"] = family
                rows.append(row)
        summary = accumulator.compute()
        if oracle_values:
            summary["oracle_f1_macro"] = float(np.mean(oracle_values))
            summary["oracle_f1_defined_images"] = len(oracle_values)
            summary["oracle_f1_scope"] = "Per-image GT-informed PR maximum; native GT; diagnostic only, no selection change"
        if cod_accumulator is not None:
            summary.update(cod_accumulator.compute())
        if family_accumulators:
            summary["by_family"] = {family: metric.compute()
                                    for family, metric in sorted(family_accumulators.items())}
        summary.update(loss=loss_sum / accumulator.count, elapsed_seconds=time.monotonic() - start,
                       prediction_resize="bilinear_logits_to_native_GT_then_sigmoid")
        if rows_path is not None:
            rows_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = rows_path.with_name(rows_path.name + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            os.replace(temporary, rows_path)
        return add_tampered_metrics(summary)
    finally:
        model.train(was_training)


def protocol_signature(config: dict) -> str:
    stable = {key: value for key, value in config.items() if key not in ("output_dir",)}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def append_metrics(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def recover_epoch_log(path: Path, completed: int, last_record: dict) -> None:
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if int(row["epoch"]) <= completed:
                rows.append(row)
        canonical = {int(row["epoch"]): row for row in rows}
        if completed:
            canonical[completed] = last_record
        rows = [canonical[epoch] for epoch in sorted(canonical)]
        expected = [json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows]
        if lines == expected:
            return
        backup = path.with_name(path.name + f".recovery-{time.time_ns()}.bak")
        shutil.copyfile(path, backup)
    elif completed:
        rows = [last_record]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def test_specs(config, task):
    spec = config["data"][task].get("test")
    if spec is None:
        return {}
    return {"test": spec} if isinstance(spec, str) or "manifest" in spec else spec


def final_evaluation(model, config, output, tasks, device):
    results = {}
    cache = {}
    for task in tasks:
        results[task] = {}
        for checkpoint_name, path in (("final", output / "last.pt"), ("best_val", output / f"best_{task}.pt")):
            saved = torch.load(path, map_location="cpu", weights_only=False)
            model.restore_training_state(saved["model"], saved["immutable_pretrained_sha256"])
            model.to(device)
            checkpoint_result = {"checkpoint": str(path), "epoch": saved["completed_epoch"],
                                 "model_sha256": saved["model_sha256"], "datasets": {}}
            for name, spec in test_specs(config, task).items():
                manifest = manifest_spec(spec)["manifest"]
                cache_key = (task, saved["model_sha256"], file_sha256(manifest),
                             json.dumps(manifest_spec(spec), sort_keys=True))
                if cache_key in cache:
                    metrics = dict(cache[cache_key], reused_identical_checkpoint_evaluation=True)
                else:
                    dataset = make_dataset(config, task, "test", spec=spec)
                    metrics = evaluate(model, eval_loader(dataset, config), task, device,
                                       amp=bool(config.get("eval_amp",False)), auroc_bins=int(config.get("auroc_bins", 4096)),
                                       standard_cod=bool(config["cod_metrics"]), iml_dice_weight=float(config.get("iml_dice_weight",0.)),
                                       rows_path=output / "evaluation" / f"{task}_{name}_{checkpoint_name}_images.jsonl")
                    cache[cache_key] = metrics
                checkpoint_result["datasets"][name] = metrics
            results[task][checkpoint_name] = checkpoint_result
            atomic_json(output / "test_results.json", results)
    return results


def validate_reference_source(config):
    if config.get("model_variant") != "evp_reference":
        raise ValueError("This isolated runner requires original EVP with explicit stage sharing")
    import importlib
    for package, expected in config["frozen_package_sources"].items():
        directory = Path(importlib.import_module(package).__file__).parent
        actual = {p.name: file_sha256(p) for p in sorted(directory.glob("*.py"))}
        if actual != expected:
            raise ValueError("Frozen package source differs: " + package)
    if file_sha256(Path(sys.modules[add_tampered_metrics.__module__].__file__)) != config["casia_evaluation_runner_sha256"]:
        raise ValueError("Frozen CASIA evaluation helper differs")


def run_experiment(config: dict, *, resume: str | Path | None = None):
    validate_reference_source(config)
    tasks = ARM_TASKS[config["arm"]]
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if resume and Path(resume).resolve() != (output / "last.pt").resolve():
        raise ValueError("exact resume must use this output directory's last.pt; use initial_checkpoint for new initialization")
    if not resume and any((output / name).exists() for name in ("status.json", "last.pt", "metrics.jsonl")):
        raise FileExistsError("output already contains an experiment; use an explicit epoch-boundary resume")
    status = {"arm": config["arm"], "state": "initializing", "completed_epochs": 0,
              "resume_policy": "completed_epoch_boundaries_only", "pid": os.getpid(), "time": time.time()}
    atomic_json(output / "status.json", status)
    try:
        device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; refusing an implicit CPU fallback")
        if config["amp"] and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but unsupported; change the recorded protocol explicitly")
        if config.get("deterministic", False):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        seed_everything(int(config["seed"]), deterministic=bool(config.get("deterministic", False)))
        model = JointCODIMLModel(**config.get("model", {}))
        if model.encoder.freeze_backbone and not config.get("spatial_pretrained"):
            raise ValueError("Frozen EVP backbone requires verified generic pretrained weights")
        initialization = {"kind": "random", "spatial_pretrained": None}
        if config.get("spatial_pretrained"):
            initialization["kind"] = "generic_ImageNet_MiT_B4_original_EVP_shared_stage_prompt_basis"
            initialization["spatial_pretrained"] = model.load_spatial_pretrained(config["spatial_pretrained"])
            initialization["pretrained_sha256"] = file_sha256(config["spatial_pretrained"])
        if config.get("initial_checkpoint"):
            saved_initial = torch.load(config["initial_checkpoint"], map_location="cpu", weights_only=False)
            model.load_state_dict(saved_initial.get("model", saved_initial), strict=True)
            initialization.update(kind="explicit_full_initial_checkpoint",
                                  initial_checkpoint_sha256=file_sha256(config["initial_checkpoint"]))
        initialization["full_model_sha256"] = state_sha256(model.state_dict())
        model.to(device)
        optimizer, optimizer_report = build_optimizer(model, config)
        datasets = {task: make_dataset(config, task, "train") for task in TASKS}
        lengths = {task: len(dataset) for task, dataset in datasets.items()}
        steps = steps_for_protocol(config, lengths)
        total_steps = config["epochs"] * steps
        warmup_steps = int(config["warmup_epochs"] * steps)
        if warmup_steps >= total_steps:
            raise ValueError("warmup must be shorter than the full experiment")
        scheduler = make_scheduler(optimizer, total_steps, warmup_steps,
                                   min_lr=float(config.get("min_lr",1e-6)),
                                   start_factor=float(config.get("warmup_start_factor",.1)))
        val_datasets = {task: make_dataset(config, task, "val") for task in tasks}
        manifest_hashes = {}
        for task, splits in config["data"].items():
            for split, spec in splits.items():
                specs = test_specs(config, task) if split == "test" else {split: spec}
                for name, item in specs.items():
                    manifest = manifest_spec(item)["manifest"]
                    manifest_hashes[f"{task}/{split}/{name}"] = {"path": manifest, "sha256": file_sha256(manifest)}
        signature = protocol_signature(config)
        protocol = {"config": config, "signature": signature, "initialization": initialization,
                    "scheduler": {"type":"LinearLR+CosineAnnealingLR" if warmup_steps else "CosineAnnealingLR", "step_unit":"optimizer_update", "total_steps":total_steps, "warmup_steps":warmup_steps, "eta_min":float(config.get("min_lr",1e-6))},
                    "train_lengths": lengths, "steps_per_epoch": steps,
                    "samples_per_task_per_epoch": {task: steps * task_batch_size(config, task) for task in TASKS},
                    "active_tasks": tasks, "optimizer_groups": optimizer_report, "manifests": manifest_hashes,
                    "selection": config["selection_metric"],
                    "independent_test_access": "only_after_training" if config["evaluate_test"] else "disabled_for_this_stage",
                    "sampling": "task_seed_epoch_cycle_permutation_with_occurrence_augmentation",
                    "forward_randomness": "task_seed_epoch_step_forked_torch_RNG",
                    "normalization": "raw_sigmoid_no_per_image_minmax",
                    "source_sha256": {str(path.relative_to(Path(__file__).parent)): file_sha256(path)
                                      for path in sorted(Path(__file__).parent.glob("*.py"))},
                    "environment": {"python": sys.version, "torch": torch.__version__, "device": str(device),
                                    "cuda": torch.version.cuda,
                                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}}
        start_epoch, best = 0, {task: {"score": None, "epoch": None} for task in tasks}
        if resume:
            saved = torch.load(resume, map_location="cpu", weights_only=False)
            if saved["phase"] != "epoch_complete" or saved["protocol_signature"] != signature:
                raise ValueError("resume requires an epoch-complete checkpoint with the identical protocol")
            if saved["manifest_hashes"] != manifest_hashes:
                raise ValueError("manifest contents changed since the checkpoint")
            if saved["initialization"] != initialization:
                raise ValueError("initialization provenance changed since the checkpoint")
            if saved["source_sha256"] != protocol["source_sha256"]:
                raise ValueError("training source code changed since the checkpoint")
            initialization = saved["initialization"]
            model.restore_training_state(saved["model"], saved["immutable_pretrained_sha256"])
            optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"])
            start_epoch, best = saved["completed_epoch"], saved["best"]
            restore_rng(saved["rng"])
            recover_epoch_log(output / "metrics.jsonl", start_epoch, saved["epoch_record"])
            for task in tasks:
                if best[task]["epoch"] == start_epoch:
                    atomic_copy(Path(resume), output / f"best_{task}.pt")
            if Path(resume).resolve() != (output / "last.pt").resolve():
                atomic_copy(Path(resume), output / "last.pt")
        else:
            atomic_json(output / "protocol.json", protocol)
            atomic_json(output / "config_resolved.json", config)
        wrapper = TaskSeededForward(model, int(config["seed"]), device)
        training_start = time.monotonic()
        for epoch in range(start_epoch, config["epochs"]):
            epoch_start = time.monotonic()
            wrapper.epoch = epoch
            loaders = {task: train_loader(datasets[task], config, task, epoch, steps) for task in tasks}
            iterators = {task: iter(loader) for task, loader in loaders.items()}
            exposures = {task: hashlib.sha256() for task in tasks}
            counts = {task: 0 for task in tasks}
            sums = {}
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(steps):
                wrapper.step = step
                batches = {}
                for task in tasks:
                    batch = next(iterators[task])
                    for identifier in batch["id"]:
                        exposures[task].update(str(identifier).encode() + b"\n")
                    counts[task] += len(batch["image"])
                    batches[f"{task}_batch"] = {key: batch[key].to(device, non_blocking=True)
                                                for key in ("image", "mask")}
                values = joint_train_step(wrapper, optimizer, **batches,
                                          cod_weight=float(config.get("cod_weight", 1.0)),
                                          iml_weight=float(config.get("iml_weight", 1.0)),
                                          boundary_weight=float(config.get("boundary_weight", 0.0)),
                                          amp=bool(config["amp"]), max_grad_norm=config.get("max_grad_norm", 1.0),
                                          iml_dice_weight=float(config.get("iml_dice_weight",0.)),
                                          clip_mode=config.get("clip_mode","global"))
                scheduler.step()
                for key, value in values.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                interval = int(config.get("log_interval", 25))
                if step == 0 or (step + 1) % interval == 0 or step + 1 == steps:
                    status.update(state="training", epoch=epoch + 1, step_in_epoch=step + 1,
                                  steps_per_epoch=steps, completed_epochs=epoch, time=time.time(),
                                  losses={key: value / (step + 1) for key, value in sums.items()},
                                  lr={group["name"]: group["lr"] for group in optimizer.param_groups})
                    atomic_json(output / "status.json", status)
                    print(json.dumps(status, ensure_ascii=False), flush=True)
            validation = {}
            for task in tasks:
                status.update(state="validating", task=task, time=time.time())
                atomic_json(output / "status.json", status)
                validation[task] = evaluate(model, eval_loader(val_datasets[task], config), task, device,
                                            amp=bool(config.get("eval_amp",False)), auroc_bins=int(config.get("auroc_bins", 4096)),
                                            standard_cod=bool(config["cod_metrics"]), iml_dice_weight=float(config.get("iml_dice_weight",0.)))
            improved = []
            for task in tasks:
                metric = config["selection_metric"][task]
                score = validation[task].get(metric)
                if score is None or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise ValueError(f"invalid validation selection metric {task}/{metric}")
                sign = -1 if metric in ("mae", "loss", "fpr") else 1
                if best[task]["score"] is None or sign * score > sign * best[task]["score"]:
                    best[task] = {"score": score, "epoch": epoch + 1, "metric": metric}
                    improved.append(task)
            row = {"epoch": epoch + 1, "arm": config["arm"], "global_step": (epoch + 1) * steps,
                   "train": {key: value / steps for key, value in sums.items()}, "validation": validation,
                   "samples": counts,
                   "sample_id_order_sha256": {task: value.hexdigest() for task, value in exposures.items()},
                   "best": {task: dict(value) for task, value in best.items()},
                   "epoch_seconds": time.monotonic() - epoch_start,
                   "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                   "lr": {group["name"]: group["lr"] for group in optimizer.param_groups}}
            model_state = model.training_state()
            payload = {"phase": "epoch_complete", "completed_epoch": epoch + 1,
                       "protocol_signature": signature, "manifest_hashes": manifest_hashes,
                       "model_config": model.config, "model": model_state, "model_sha256": state_sha256(model_state),
                       "immutable_pretrained_sha256": model._pretrained_sha256, "checkpoint_scope": "trainable_parameters_and_all_mutable_buffers_plus_hash_verified_immutable_backbone",
                       "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                       "rng": capture_rng(), "best": best, "epoch_record": row,
                       "initialization": initialization, "source_sha256": protocol["source_sha256"], "config": config}
            atomic_checkpoint(output / "last.pt", payload)
            for task in improved:
                atomic_copy(output / "last.pt", output / f"best_{task}.pt")
            append_metrics(output / "metrics.jsonl", row)
            status.update(state="epoch_complete", completed_epochs=epoch + 1, time=time.time(), best=best,
                          latest_validation=validation)
            atomic_json(output / "status.json", status)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        results = {}
        if config["evaluate_test"]:
            status.update(state="final_testing", completed_epochs=config["epochs"], time=time.time())
            atomic_json(output / "status.json", status)
            results = final_evaluation(model, config, output, tasks, device)
        status.update(state="complete", time=time.time(), training_elapsed_seconds=time.monotonic() - training_start,
                      completed_epochs=config["epochs"],
                      test_results=str(output / "test_results.json") if config["evaluate_test"] else None,
                      independent_test_access="completed" if config["evaluate_test"] else "disabled_for_this_stage", best=best)
        atomic_json(output / "status.json", status)
        return results
    except BaseException as error:
        status.update(state="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc(), time=time.time())
        atomic_json(output / "status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", help="Explicit epoch-complete checkpoint; partial epochs restart")
    arguments = parser.parse_args()
    run_experiment(resolve_config(arguments.config), resume=arguments.resume)


if __name__ == "__main__":
    main()
