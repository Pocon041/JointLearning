"""Image-macro segmentation metrics on raw sigmoid probabilities.

No per-image prediction normalization or test-time threshold search is used.
AUROC is explicitly a fixed-bin approximation; constant-label images have no
defined image AUROC and are counted separately rather than assigned a score.
"""
from __future__ import annotations

import inspect
from importlib.metadata import version

import numpy as np


def histogram_auroc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with ties inside each probability histogram bin."""
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 1:
        raise ValueError("AUROC histograms must be matching one-dimensional arrays")
    if (not np.isfinite(positive).all() or not np.isfinite(negative).all()
            or (positive < 0).any() or (negative < 0).any()):
        raise ValueError("AUROC histogram counts must be finite and nonnegative")
    p, n = positive.sum(), negative.sum()
    if p == 0 or n == 0:
        return None
    below = np.cumsum(negative) - negative
    return float(np.dot(positive, below + 0.5 * negative) / (p * n))


class SegmentationMetrics:
    """Accumulate per-image F1/IoU/MAE and explicitly named histogram AUROC.

    F1/IoU use a fixed >=0.5 threshold. An empty GT and empty prediction score
    one for F1/IoU. Image FPR excludes full-foreground images. AUROC uses raw
    probabilities and >=0.5 binary GT. A pooled pixel AUROC is also reported;
    it is distinct from the mean of defined per-image AUROCs.
    """

    def __init__(self, *, threshold: float = 0.5, auroc_bins: int = 4096):
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0,1]")
        if not isinstance(auroc_bins, int) or auroc_bins < 2:
            raise ValueError("auroc_bins must be an integer >=2")
        self.threshold, self.auroc_bins = float(threshold), auroc_bins
        self.count = 0
        self.sums = dict.fromkeys(("f1", "iou", "mae", "precision", "recall"), 0.0)
        self.fpr_sum = self.auroc_sum = 0.0
        self.fpr_count = self.auroc_count = self.empty_gt = self.full_gt = 0
        self.positive = np.zeros(auroc_bins, dtype=np.int64)
        self.negative = np.zeros(auroc_bins, dtype=np.int64)
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, probability, target) -> dict:
        probability = np.asarray(probability, dtype=np.float64).squeeze()
        target = np.asarray(target, dtype=np.float64).squeeze()
        if probability.shape != target.shape or probability.size == 0:
            raise ValueError("prediction and target must have matching nonempty shapes")
        if not np.isfinite(probability).all() or not np.isfinite(target).all():
            raise ValueError("prediction and target must be finite")
        if probability.min() < -1e-7 or probability.max() > 1 + 1e-7:
            raise ValueError("prediction must contain sigmoid probabilities in [0,1]")
        if target.min() < 0 or target.max() > 1:
            raise ValueError("target must be in [0,1]")
        probability = probability.clip(0, 1)
        truth, predicted = target >= 0.5, probability >= self.threshold
        tp = int(np.count_nonzero(predicted & truth))
        fp = int(np.count_nonzero(predicted & ~truth))
        fn = int(np.count_nonzero(~predicted & truth))
        tn = int(np.count_nonzero(~predicted & ~truth))
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        result = {
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "mae": float(np.abs(probability - target).mean()),
            "precision": tp / (tp + fp) if tp + fp else 1.0,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "fpr": fp / (fp + tn) if fp + tn else None,
        }
        for key in self.sums:
            self.sums[key] += result[key]
        if result["fpr"] is not None:
            self.fpr_sum += result["fpr"]
            self.fpr_count += 1
        self.empty_gt += int(not truth.any())
        self.full_gt += int(truth.all())
        bins = np.minimum((probability * self.auroc_bins).astype(np.int64), self.auroc_bins - 1)
        positive = np.bincount(bins[truth], minlength=self.auroc_bins)
        negative = np.bincount(bins[~truth], minlength=self.auroc_bins)
        self.positive += positive
        self.negative += negative
        auc = histogram_auroc(positive, negative)
        result["pixel_auroc_hist"] = auc
        if auc is not None:
            self.auroc_sum += auc
            self.auroc_count += 1
        self.count += 1
        return result

    def compute(self) -> dict:
        if not self.count:
            raise ValueError("cannot compute metrics before any image is evaluated")
        return {
            "images": self.count,
            "threshold": self.threshold,
            **{key: value / self.count for key, value in self.sums.items()},
            "fpr": self.fpr_sum / self.fpr_count if self.fpr_count else None,
            "pixel_auroc_macro_hist": self.auroc_sum / self.auroc_count if self.auroc_count else None,
            "pixel_auroc_pooled_hist": histogram_auroc(self.positive, self.negative),
            "auroc_bins": self.auroc_bins,
            "auroc_defined_images": self.auroc_count,
            "empty_gt_images": self.empty_gt,
            "full_gt_images": self.full_gt,
            "pooled_confusion": {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn},
            "prediction_normalization": "sigmoid_only_no_per_image_minmax",
            "aggregation": "image_macro_except_explicitly_pooled_metrics",
        }


class StandardCODMetrics:
    """Published S/wF/E implementations with normalization explicitly disabled.

    Requires a PySODMetrics release whose public step API accepts normalize.
    Passing float probabilities and boolean GT avoids the library's default
    uint8 preparation and per-image min-max normalization.
    """

    def __init__(self):
        try:
            from py_sod_metrics import Emeasure, MAE, Smeasure, WeightedFmeasure
        except ImportError as error:
            raise ImportError("standard COD metrics require pysodmetrics>=1.6.2") from error
        self.measures = {"smeasure": Smeasure(), "weighted_fmeasure": WeightedFmeasure(),
                         "emeasure": Emeasure(), "mae": MAE()}
        for name, measure in self.measures.items():
            if "normalize" not in inspect.signature(measure.step).parameters:
                raise RuntimeError(f"{name} lacks explicit normalize=False support; update pysodmetrics")
        self.package_version = version("pysodmetrics")
        self.count = 0

    def update(self, probability, target):
        probability = np.asarray(probability, dtype=np.float64).squeeze()
        target = np.asarray(target).squeeze() >= 0.5
        if probability.ndim != 2 or probability.shape != target.shape:
            raise ValueError("standard COD metrics require matching 2D prediction and GT")
        if not np.isfinite(probability).all():
            raise ValueError("COD probabilities must be finite")
        for measure in self.measures.values():
            measure.step(pred=probability, gt=target, normalize=False)
        self.count += 1

    def compute(self):
        if not self.count:
            raise ValueError("cannot compute COD metrics without images")
        em = self.measures["emeasure"].get_results()["em"]
        return {
            "smeasure": float(self.measures["smeasure"].get_results()["sm"]),
            "weighted_fmeasure": float(self.measures["weighted_fmeasure"].get_results()["wfm"]),
            "mean_emeasure": float(np.mean(em["curve"])),
            "max_emeasure": float(np.max(em["curve"])),
            "adaptive_emeasure": float(em["adp"]),
            "sod_mae": float(self.measures["mae"].get_results()["mae"]),
            "cod_metric_backend": f"pysodmetrics=={self.package_version}",
            "cod_metric_normalize": False,
        }
