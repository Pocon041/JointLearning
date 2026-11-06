"""Manifest-driven paired segmentation data with matched deterministic geometry.

JSONL records contain ``id``, ``image``, ``mask``, ``task`` and ``source_id``.
Relative paths resolve against explicit ``root``, a record ``base``/``root``,
or the manifest directory, in that order. This module never invents splits or
infers common source images from filenames. Split/source audits belong to the
manifest construction stage.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate


def seed_worker(worker_id: int) -> None:
    """Seed incidental worker-side Python/NumPy randomness from DataLoader."""
    del worker_id
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _sample_seed(seed: int, task: str, epoch: int, index: int, occurrence: int) -> int:
    payload = json.dumps([seed, task, epoch, index, occurrence], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _read_manifest(path: Path) -> tuple[list[dict], str | None]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
        if isinstance(value, dict):
            rows, base = value.get("records", value.get("samples")), value.get("base", value.get("root"))
        else:
            rows, base = value, None
        if not isinstance(rows, list):
            raise ValueError(f"JSON manifest must be a list or contain records: {path}")
    else:
        rows, base = [], None
        for line_number, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest must contain nonempty object records: {path}")
    return rows, base


def _authentic(row: Mapping) -> bool:
    return (row.get("is_authentic") is True or row.get("authentic") is True
            or any(row.get(key) == "authentic" for key in ("kind", "type", "label_type")))


class ManifestDataset(Dataset):
    """Paired RGB/mask samples from an audited manifest.

    ``size`` is output (height, width). Training applies paired flips and 90
    degree rotations before resize, using a seed determined by
    (seed, task, epoch, index, occurrence). An index may be ``(index,
    occurrence)`` so repeated draws get distinct, reproducible geometry.

    Images use bilinear resize; masks use nearest-neighbor resize. Mask
    foreground is nonzero by default (works for both 0/1 and 0/255 masks).
    Optional ``mask_threshold``/``mask_invert`` can be set per record after a
    dataset audit. Missing masks are permitted only for explicitly authentic
    records. No photographic, JPEG, or color augmentations are applied.

    Evaluation samples retain ``original_mask`` [1,H,W] and
    ``original_size`` (H,W). Use batch_size=1 with default collate, or
    ``eval_collate`` for batches containing different native GT sizes.
    """

    def __init__(self, manifest: str | Path, size: int | Sequence[int] = (512, 512),
                 training: bool = True, seed: int = 0, root: str | Path | None = None,
                 task: str | None = None, *, hflip: float = 0.5, vflip: float = 0.5,
                 rotate90: bool = True, mask_threshold: float = 0.0,
                 validate_paths: bool = True):
        self.manifest = Path(manifest).resolve()
        rows, document_base = _read_manifest(self.manifest)
        self.size = (size, size) if isinstance(size, int) else tuple(size)
        if (len(self.size) != 2 or any(not isinstance(x, int) or isinstance(x, bool)
                                     or x < 1 for x in self.size)):
            raise ValueError("size must contain positive integer (height, width)")
        if task is not None and task not in ("cod", "iml"):
            raise ValueError("task must be cod, iml, or None")
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in (hflip, vflip)):
            raise ValueError("flip probabilities must be finite and in [0,1]")
        if not math.isfinite(mask_threshold):
            raise ValueError("mask_threshold must be finite")
        self.training, self.seed, self.epoch = bool(training), int(seed), 0
        self.task, self.hflip, self.vflip = task, float(hflip), float(vflip)
        self.rotate90, self.mask_threshold = bool(rotate90), float(mask_threshold)
        self.records = []
        identifiers = set()
        for position, original in enumerate(rows):
            row = dict(original)
            actual_task = row.get("task", task)
            if actual_task not in ("cod", "iml") or (task is not None and actual_task != task):
                raise ValueError(f"manifest task mismatch at row {position}: {actual_task!r}")
            row["task"] = actual_task
            image_value = row.get("image")
            if not isinstance(image_value, str) or not image_value:
                raise ValueError(f"row {position} requires an image path")
            identifier = str(row.get("id", image_value))
            if not identifier or identifier in identifiers:
                raise ValueError(f"empty or duplicate manifest id: {identifier!r}")
            identifiers.add(identifier)
            row["id"] = identifier
            row["source_id"] = str(row.get("source_id", identifier))
            base = root if root is not None else row.get("base", row.get("root", document_base))
            base_path = self.manifest.parent if base is None else Path(base)
            if not base_path.is_absolute():
                base_path = self.manifest.parent / base_path
            image_path = Path(image_value)
            row["image"] = image_path if image_path.is_absolute() else base_path / image_path
            mask_value = row.get("mask")
            if mask_value in (None, ""):
                if not _authentic(row):
                    raise ValueError(f"missing mask requires explicit authentic type: {identifier}")
                row["mask"] = None
            elif isinstance(mask_value, str):
                mask_path = Path(mask_value)
                row["mask"] = mask_path if mask_path.is_absolute() else base_path / mask_path
            else:
                raise ValueError(f"invalid mask path: {identifier}")
            threshold = float(row.get("mask_threshold", self.mask_threshold))
            if not math.isfinite(threshold):
                raise ValueError(f"mask threshold must be finite: {identifier}")
            row["mask_threshold"] = threshold
            if "mask_invert" in row and not isinstance(row["mask_invert"], bool):
                raise ValueError(f"mask_invert must be boolean: {identifier}")
            if validate_paths:
                for key in ("image", "mask"):
                    if row[key] is not None and not row[key].is_file():
                        raise FileNotFoundError(f"{key} missing for {identifier}: {row[key]}")
            self.records.append(row)

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, draw):
        if isinstance(draw, tuple):
            if len(draw) != 2:
                raise ValueError("draw tuple must be (index, occurrence)")
            index, occurrence = draw
        else:
            index, occurrence = draw, 0
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self)
                or not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0):
            raise IndexError("index and occurrence must be nonnegative integers in range")
        row = self.records[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        if row["mask"] is None:
            original = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            with Image.open(row["mask"]) as source:
                if source.size != image.size:
                    raise ValueError(f"image/mask native sizes differ: {row['id']}")
                original = (np.asarray(source.convert("L")) > row["mask_threshold"]).astype(np.uint8)
            if row.get("mask_invert", False):
                original = 1 - original
        original_size = (image.height, image.width)
        mask = Image.fromarray(original * 255, mode="L")
        if self.training:
            rng = random.Random(_sample_seed(self.seed, row["task"], self.epoch, index, occurrence))
            transforms = []
            if rng.random() < self.hflip:
                transforms.append(Image.Transpose.FLIP_LEFT_RIGHT)
            if rng.random() < self.vflip:
                transforms.append(Image.Transpose.FLIP_TOP_BOTTOM)
            if self.rotate90:
                rotation = rng.randrange(4)
                if rotation:
                    transforms.append((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                       Image.Transpose.ROTATE_270)[rotation - 1])
            for transform in transforms:
                image, mask = image.transpose(transform), mask.transpose(transform)
        pil_size = (self.size[1], self.size[0])
        image = image.resize(pil_size, Image.Resampling.BILINEAR)
        mask = mask.resize(pil_size, Image.Resampling.NEAREST)
        image_array = np.array(image, dtype=np.float32, copy=True).transpose(2, 0, 1) / 255.0
        mask_array = np.array(mask, dtype=np.float32, copy=True)[None] / 255.0
        result = {"image": torch.from_numpy(image_array), "mask": torch.from_numpy(mask_array),
                  "id": row["id"], "source_id": row["source_id"], "task": row["task"]}
        if "family" in row:
            result["family"] = str(row["family"])
        if not self.training:
            result.update(original_mask=torch.from_numpy(original.copy()[None].astype(np.float32)),
                          original_size=original_size)
        return result


def eval_collate(samples: list[dict]) -> dict:
    """Stack model-sized tensors while retaining variable-sized GT as a list."""
    if not samples:
        raise ValueError("cannot collate an empty evaluation batch")
    result = default_collate([{k: v for k, v in sample.items()
                              if k not in ("original_mask", "original_size")} for sample in samples])
    result["original_mask"] = [sample["original_mask"] for sample in samples]
    result["original_size"] = [sample["original_size"] for sample in samples]
    return result


def infinite_batches(loader: Iterable):
    """Repeat an iterable without caching its batches or mutating its sampler."""
    while True:
        found = False
        for batch in loader:
            found = True
            yield batch
        if not found:
            raise ValueError("cannot repeat an empty loader")


def balanced_batches(loaders: Mapping[str, Iterable], steps: int):
    """Yield one batch per task per step; loader sampling budgets stay explicit."""
    if not loaders or not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
        raise ValueError("loaders must be nonempty and steps must be a positive integer")
    streams = {task: infinite_batches(loader) for task, loader in loaders.items()}
    for _ in range(steps):
        yield {task: next(stream) for task, stream in streams.items()}
