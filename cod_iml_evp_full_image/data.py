"""Whole-image geometry and batch padding, retaining every original region.

This module is isolated from historical experiments. Native mode never upsamples;
fit mode preserves aspect ratio. Padding is a batching operation, not background
supervision. ImageNet-mean RGB becomes zero after the model's normalization.
"""
from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Sampler

from cod_iml_joint.data import ManifestDataset, _sample_seed

IMAGENET_MEAN = (.485, .456, .406)


class FullImageDataset(ManifestDataset):
    def __init__(self, *args, max_side=None, resize_mode="native", **kwargs):
        super().__init__(*args, **kwargs)
        if resize_mode not in ("native", "fit"):
            raise ValueError("resize_mode must be native or fit")
        if max_side is not None and (isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 32):
            raise ValueError("max_side must be an integer >=32 or None")
        if resize_mode == "fit" and max_side is None:
            raise ValueError("fit requires an explicit maximum side")
        self.max_side, self.resize_mode = max_side, resize_mode
        self._native_sizes = {}

    def _draw(self, draw):
        if isinstance(draw, tuple):
            if len(draw) != 2:
                raise ValueError("draw must be (index, occurrence)")
            index, occurrence = draw
        else:
            index, occurrence = draw, 0
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(self)
                or not isinstance(occurrence, int) or isinstance(occurrence, bool) or occurrence < 0):
            raise IndexError("invalid sample draw")
        return index, occurrence

    def _geometry(self, index, occurrence):
        if not self.training:
            return []
        rng = random.Random(_sample_seed(self.seed, self.records[index]["task"], self.epoch, index, occurrence))
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
        return transforms

    def _resize_shape(self, h, w):
        scale = self.max_side / max(h, w) if self.max_side else 1.
        if self.resize_mode == "native":
            scale = min(1., scale)
        return max(1, round(h * scale)), max(1, round(w * scale))

    def output_size(self, draw):
        """Header-only shape query for deterministic shape-aware batching."""
        index, occurrence = self._draw(draw)
        if index not in self._native_sizes:
            with Image.open(self.records[index]["image"]) as source:
                self._native_sizes[index] = (source.height, source.width)
        h, w = self._native_sizes[index]
        for transform in self._geometry(index, occurrence):
            if transform in (Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_270):
                h, w = w, h
        return self._resize_shape(h, w)

    def __getitem__(self, draw):
        index, occurrence = self._draw(draw)
        row = self.records[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        if row["mask"] is None:
            original = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            with Image.open(row["mask"]) as source:
                if source.size != image.size:
                    raise ValueError("image/mask native sizes differ: " + row["id"])
                original = (np.asarray(source.convert("L")) > row["mask_threshold"]).astype(np.uint8)
            if row.get("mask_invert", False):
                original = 1 - original
        original_size = (image.height, image.width)
        mask = Image.fromarray(original * 255)
        for transform in self._geometry(index, occurrence):
            image, mask = image.transpose(transform), mask.transpose(transform)
        h, w = self._resize_shape(image.height, image.width)
        if image.size != (w, h):
            image = image.resize((w, h), Image.Resampling.BILINEAR)
            mask = mask.resize((w, h), Image.Resampling.NEAREST)
        result = {"image": torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.),
                  "mask": torch.from_numpy(np.array(mask, dtype=np.float32)[None] / 255.),
                  "id": row["id"], "source_id": row["source_id"], "task": row["task"],
                  "processed_size": (h, w)}
        if "family" in row:
            result["family"] = str(row["family"])
        if not self.training:
            result.update(original_mask=torch.from_numpy(original[None].astype(np.float32)),
                          original_size=original_size)
        return result


def pad_collate(samples, *, multiple=32):
    if not samples or isinstance(multiple, bool) or not isinstance(multiple, int) or multiple < 1:
        raise ValueError("nonempty samples and a positive integer multiple are required")
    sizes = [tuple(sample["image"].shape[-2:]) for sample in samples]
    h = math.ceil(max(s[0] for s in sizes) / multiple) * multiple
    w = math.ceil(max(s[1] for s in sizes) / multiple) * multiple
    image = samples[0]["image"].new_tensor(IMAGENET_MEAN)[None, :, None, None].expand(len(samples), 3, h, w).clone()
    mask = image.new_zeros((len(samples), 1, h, w))
    valid = torch.zeros_like(mask, dtype=torch.bool)
    for i, (sample, (sh, sw)) in enumerate(zip(samples, sizes)):
        if sample["mask"].shape != (1, sh, sw) or tuple(sample["processed_size"]) != (sh, sw):
            raise ValueError("image, mask and processed_size must agree")
        image[i, :, :sh, :sw] = sample["image"]
        mask[i, :, :sh, :sw] = sample["mask"]
        valid[i, :, :sh, :sw] = True
    result = {"image": image, "mask": mask, "valid_mask": valid,
              "processed_size": torch.tensor(sizes, dtype=torch.long)}
    for key in ("id", "source_id", "task", "family", "original_mask", "original_size"):
        present = [key in sample for sample in samples]
        if any(present):
            if not all(present):
                raise ValueError("inconsistent sample metadata: " + key)
            result[key] = [sample[key] for sample in samples]
    return result


class ShapeBatchSampler(Sampler):
    """Regroup a fixed draw stream without adding, dropping or repeating draws.

    Full buckets are emitted as they fill; the small tail is batched in original
    stream order. Thus shape grouping does not sort the epoch into a curriculum.
    It changes batch composition; matching single/joint arms use the same rule.
    """
    def __init__(self, sampler, dataset, batch_size, bucket_step=64):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (batch_size, bucket_step)):
            raise ValueError("batch_size and bucket_step must be positive integers")
        self.sampler, self.dataset = sampler, dataset
        self.batch_size, self.bucket_step = batch_size, bucket_step

    def __len__(self):
        return math.ceil(len(self.sampler) / self.batch_size)

    def __iter__(self):
        buckets = {}
        for position, draw in enumerate(self.sampler):
            h, w = self.dataset.output_size(draw)
            key = (math.ceil(h / self.bucket_step), math.ceil(w / self.bucket_step))
            bucket = buckets.setdefault(key, [])
            bucket.append((position, draw))
            if len(bucket) == self.batch_size:
                yield [item[1] for item in bucket]
                bucket.clear()
        tail = sorted(item for bucket in buckets.values() for item in bucket)
        for start in range(0, len(tail), self.batch_size):
            yield [item[1] for item in tail[start:start + self.batch_size]]
