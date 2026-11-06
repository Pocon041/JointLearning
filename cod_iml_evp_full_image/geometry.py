"""Frequency extraction before batching padding, and padding-aware restoration."""
import torch
from torch.nn import functional as F


def checked_sizes(image, sizes):
    if isinstance(sizes, torch.Tensor):
        if sizes.dtype not in (torch.int32, torch.int64):
            raise ValueError("processed sizes must be integers")
        sizes = sizes.detach().cpu().tolist()
    if len(sizes) != image.shape[0]:
        raise ValueError("one valid shape is required per image")
    out = []
    for shape in sizes:
        if len(shape) != 2 or any(isinstance(x, bool) or not isinstance(x, int) for x in shape):
            raise ValueError("processed sizes must be integer pairs")
        h, w = shape
        if not 0 < h <= image.shape[-2] or not 0 < w <= image.shape[-1]:
            raise ValueError("processed size exceeds the padded image")
        out.append((h, w))
    return out


def frequency_before_padding(image, sizes, transform):
    """Apply an existing frequency transform to valid rectangles, then zero-pad.

    Equal shapes are grouped for batched FFTs. Leading transform dimensions may
    include a band axis. This does not mask attention, feature pooling or BN;
    downstream padding influence must not be described as completely removed.
    """
    sizes = checked_sizes(image, sizes)
    groups = {}
    for i, shape in enumerate(sizes):
        groups.setdefault(shape, []).append(i)
    output = [None] * image.shape[0]
    for (h, w), indices in groups.items():
        value = transform(image[indices, :, :h, :w])
        if value.shape[0] != len(indices) or value.shape[-2:] != (h, w):
            raise ValueError("frequency transform must preserve batch and spatial dimensions")
        value = F.pad(value, (0, image.shape[-1] - w, 0, image.shape[-2] - h))
        for offset, index in enumerate(indices):
            output[index] = value[offset]
    return torch.stack(output)


def handcrafted_features_before_padding(generator, image, sizes):
    """Original EVP FFT and original trainable embedding pyramid, reordered.

    Only FFT runs on each valid rectangle. All trainable handcrafted generators
    still run on the batched canvas. No frequency filter shape is changed here.
    """
    if generator.input_type != "fft" or generator.tuning_stage != "1234":
        raise ValueError("This path requires the original four-stage FFT EVP")
    x = frequency_before_padding(image, sizes,
        lambda patch: generator.fft(patch, generator.freq_nums, generator.prompt_type))
    output = []
    for stage in range(1, 5):
        tokens, h, w = getattr(generator, f"handcrafted_generator{stage}")(x)
        output.append(tokens)
        x = tokens.transpose(1, 2).reshape(image.shape[0], -1, h, w).contiguous()
    return tuple(output)


def restore_logits(logits, processed_sizes, original_masks):
    """Remove padding BEFORE interpolating to native GT; return a variable list."""
    sizes = checked_sizes(logits, processed_sizes)
    if len(original_masks) != logits.shape[0]:
        raise ValueError("one native GT is required per prediction")
    return [F.interpolate(logits[i:i + 1, :, :h, :w].float(), size=gt.shape[-2:],
                          mode="bilinear", align_corners=False)
            for i, ((h, w), gt) in enumerate(zip(sizes, original_masks))]
