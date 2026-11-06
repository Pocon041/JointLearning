"""Overlapping local DCT with signed, spatially aligned band reconstructions.

The three masks partition every DCT coefficient, so their overlap-normalized
inverse transforms sum to the input (up to floating-point roundoff). This is
an RGB-domain local transform; it does not read JPEG bitstream coefficients.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


def _orthonormal_dct(n: int, *, dtype=torch.float64, device=None) -> torch.Tensor:
    index = torch.arange(n, dtype=dtype, device=device)
    basis = torch.cos(torch.pi * index[:, None] * (2 * index[None, :] + 1) / (2 * n))
    scales = torch.full((n,), (2.0 / n) ** 0.5, dtype=dtype, device=device)
    scales[0] = (1.0 / n) ** 0.5
    return basis * scales[:, None]


class LocalDCTBank(nn.Module):
    """8x8 DCT/IDCT bands, default overlap stride 4, with right/bottom padding.

    Frequency indices are zero-based. Low is ``u+v <= 2``, high is
    ``u+v >= 7``, and mid contains the remaining coefficients. These cutoffs
    are explicit implementation choices, not claims about task-specific
    useful frequencies. No magnitude, clipping, or min/max normalization is
    applied: negative detail responses and absolute amplitudes are retained.

    Float64 input keeps float64 arithmetic. All other floating input uses
    float32 DCT/IDCT, including under autocast; the returned bands are float32.
    """

    def __init__(self, patch_size: int = 8, stride: int = 4):
        super().__init__()
        if patch_size != 8 or not isinstance(patch_size, int):
            raise ValueError("patch_size must be 8")
        if not isinstance(stride, int) or isinstance(stride, bool) or not 1 <= stride <= patch_size:
            raise ValueError("stride must be an integer in [1, patch_size]")
        self.patch_size = patch_size
        self.stride = stride
        self.register_buffer("basis", _orthonormal_dct(patch_size), persistent=False)
        u = torch.arange(patch_size)[:, None]
        v = torch.arange(patch_size)[None, :]
        diagonal = u + v
        low = diagonal <= 2
        high = diagonal >= 7
        self.register_buffer("low_mask", low, persistent=False)
        self.register_buffer("mid_mask", ~(low | high), persistent=False)
        self.register_buffer("high_mask", high, persistent=False)

    def _work_basis(self, x: torch.Tensor) -> torch.Tensor:
        # Regenerate if the whole model was cast to a low-precision dtype;
        # upcasting an already rounded fp16/bfloat16 basis loses accuracy.
        if self.basis.dtype in (torch.float32, torch.float64) and (
            self.basis.dtype == x.dtype or self.basis.dtype == torch.float64
        ):
            return self.basis.to(device=x.device, dtype=x.dtype)
        return _orthonormal_dct(self.patch_size, dtype=x.dtype, device=x.device)

    def _inverse_band(self, coefficients, mask, basis, h, w, divisor):
        # [B,C,L,P,P] -> inverse DCT -> [B,C,P,P,L] -> fold columns.
        masked = coefficients * mask.to(device=coefficients.device)[None, None, None]
        patches = basis.t() @ masked @ basis
        b, c, count, p1, p2 = patches.shape
        columns = patches.permute(0, 1, 3, 4, 2).reshape(b, c * p1 * p2, count)
        return F.fold(columns, (h, w), self.patch_size, stride=self.stride) / divisor

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("x must have shape [B,C,H,W]")
        if not x.is_floating_point():
            raise TypeError("x must be a real floating-point tensor")
        b, c, h, w = x.shape
        if b < 1 or c < 1 or h < self.patch_size or w < self.patch_size:
            raise ValueError("batch/channels must be nonempty and height/width must be at least 8")
        work_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        # The transform must not silently run matrix multiplies in AMP dtype.
        with torch.autocast(device_type=x.device.type, enabled=False):
            work = x.to(dtype=work_dtype)
            bottom = (-(h - self.patch_size)) % self.stride
            right = (-(w - self.patch_size)) % self.stride
            work = F.pad(work, (0, right, 0, bottom), mode="replicate")
            hp, wp = work.shape[-2:]
            columns = F.unfold(work, self.patch_size, stride=self.stride)
            count = columns.shape[-1]
            patches = columns.reshape(b, c, self.patch_size, self.patch_size, count)
            patches = patches.permute(0, 1, 4, 2, 3)
            basis = self._work_basis(work)
            coefficients = basis @ patches @ basis.t()
            # Every padded pixel is covered; one shared divisor broadcasts
            # over batch/channels, avoiding repeated full-sized ones tensors.
            coverage_columns = work.new_ones((1, self.patch_size ** 2, count))
            divisor = F.fold(coverage_columns, (hp, wp), self.patch_size, stride=self.stride)
            bands = tuple(
                self._inverse_band(coefficients, mask, basis, hp, wp, divisor)[..., :h, :w]
                for mask in (self.low_mask, self.mid_mask, self.high_mask)
            )
        return bands
