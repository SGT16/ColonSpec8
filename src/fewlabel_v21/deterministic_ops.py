from __future__ import annotations

from collections.abc import Sequence
import os

import torch
from torch import Tensor


def spatial_cross_entropy(
    logits: Tensor,
    target: Tensor,
    *,
    reduction: str = "mean",
    ignore_index: int = -100,
) -> Tensor:


    if logits.ndim < 2 or logits.shape[1] < 1:
        raise ValueError("CE expects logits [N, C, ...] with at least one class")
    if not logits.is_floating_point():
        raise TypeError("CE logits must have a floating-point dtype")
    expected_shape = (logits.shape[0], *logits.shape[2:])
    if tuple(target.shape) != expected_shape:
        raise ValueError(f"Expected target shape {expected_shape}, got {tuple(target.shape)}")
    if target.dtype != torch.long:
        raise TypeError("CE target must have torch.int64 dtype")
    if target.device != logits.device:
        raise ValueError("CE logits and target must be on the same device")
    if reduction not in {"none", "sum", "mean"}:
        raise ValueError(f"Unsupported CE reduction: {reduction!r}")
    if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
        raise TypeError("ignore_index must be an integer")
    valid = target != ignore_index
    safe_target = target.masked_fill(~valid, 0)


    negative_log_likelihood = -torch.log_softmax(logits, dim=1).gather(
        1, safe_target.unsqueeze(1)
    ).squeeze(1)
    loss = negative_log_likelihood.masked_fill(~valid, 0)
    if reduction == "none":
        return loss
    total = loss.sum()
    if reduction == "sum":
        return total
    count = valid.sum()


    return torch.where(count > 0, total / count.clamp_min(1), total.new_tensor(float("nan")))


def _positive_size(size: Sequence[int], length: int, name: str) -> tuple[int, ...]:
    if not isinstance(size, Sequence) or len(size) != length:
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in size):
        raise ValueError(f"{name} must contain exactly {length} positive integers")
    return tuple(size)


def _interpolation_matrix(input_size: int, output_size: int, like: Tensor) -> Tensor:


    coordinate = (torch.arange(output_size, dtype=like.dtype, device=like.device) + 0.5)
    coordinate = (coordinate * (input_size / output_size) - 0.5).clamp_min(0)
    lower = coordinate.floor().to(torch.long)
    upper = (lower + 1).clamp_max(input_size - 1)
    fraction = coordinate - lower
    columns = torch.arange(input_size, device=like.device).unsqueeze(0)
    return ((columns == lower.unsqueeze(1)).to(like.dtype) * (1 - fraction).unsqueeze(1)
            + (columns == upper.unsqueeze(1)).to(like.dtype) * fraction.unsqueeze(1))


def resize_bilinear(x: Tensor, size: Sequence[int]) -> Tensor:


    if x.ndim != 4 or min(x.shape[-2:]) < 1:
        raise ValueError("Bilinear resize expects NCHW input with nonempty spatial axes")
    if x.dtype not in {torch.float32, torch.float64}:
        raise TypeError("Deterministic bilinear resize supports float32 and float64")
    output_height, output_width = _positive_size(size, 2, "size")
    if x.is_cuda and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    if tuple(x.shape[-2:]) == (output_height, output_width):
        return x
    height_matrix = _interpolation_matrix(x.shape[-2], output_height, x)
    width_matrix = _interpolation_matrix(x.shape[-1], output_width, x)
    horizontal = torch.matmul(x, width_matrix.transpose(0, 1))
    return torch.matmul(height_matrix, horizontal)


def spatial_mean_pool3d(x: Tensor, output_size: Sequence[int]) -> Tensor:


    if x.ndim != 5 or min(x.shape[2:]) < 1:
        raise ValueError("Spatial block mean expects NCDHW input with nonempty D, H, W")
    if not x.is_floating_point():
        raise TypeError("Spatial block mean requires floating-point input")
    output_depth, output_height, output_width = _positive_size(output_size, 3, "output_size")
    batch, channels, depth, height, width = x.shape
    if output_depth != depth:
        raise ValueError("Spatial block mean must preserve the spectral/depth axis")
    if height % output_height or width % output_width:
        raise ValueError("Input spatial dimensions must be divisible by output dimensions")
    blocks = x.reshape(batch, channels, depth, output_height, height // output_height,
                       output_width, width // output_width)
    return blocks.mean(dim=(4, 6))
