from __future__ import annotations
import torch
from torch import Tensor
from torch.nn import functional as F
from fewlabel.objectives import dice_from_statistics


def cross_entropy(logits: Tensor, labels: Tensor, *, ignore_index: int = -100,
                  reduction: str = 'mean') -> Tensor:


    if reduction not in ('sum', 'mean', 'none'):
        raise ValueError('Unsupported cross entropy reduction')
    losses = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction='none')
    if reduction == 'none':
        return losses
    total = losses.sum()
    if reduction == 'sum':
        return total
    count = labels.ne(ignore_index).sum()
    return torch.where(count > 0, total / count.clamp_min(1), total.new_tensor(float('nan')))


def masked_supervised(logits: Tensor, labels: Tensor) -> dict[str, Tensor]:

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits must have shape [B,2,H,W]")
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("labels must have shape [B,H,W]")
    labeled = labels.ne(-1)
    count = labeled.sum()

    logits32 = logits.float()
    ce_sum = cross_entropy(logits32, labels.long(), ignore_index=-1, reduction="sum")
    probabilities = logits32.softmax(dim=1) * labeled.unsqueeze(1)
    targets = F.one_hot(labels.clamp_min(0).long(), num_classes=2).permute(0, 3, 1, 2)
    targets = targets.to(probabilities.dtype) * labeled.unsqueeze(1)
    axes = (0, 2, 3)
    intersection = (probabilities * targets).sum(dim=axes)
    prediction_sum = probabilities.sum(dim=axes)
    target_sum = targets.sum(dim=axes)
    return {
        "ce": ce_sum / count.clamp_min(1),
        "dice": dice_from_statistics(intersection, prediction_sum, target_sum),
        "ce_sum": ce_sum,
        "labeled_count": count,
        "dice_intersection": intersection,
        "dice_prediction_sum": prediction_sum,
        "dice_target_sum": target_sum,
    }
