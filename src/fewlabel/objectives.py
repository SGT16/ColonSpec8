from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def dice_from_statistics(intersection: Tensor, prediction_sum: Tensor,
                         target_sum: Tensor, epsilon: float = 1e-6) -> Tensor:
    return 1.0 - ((2.0 * intersection + epsilon)
                  / (prediction_sum + target_sum + epsilon)).mean()


def masked_supervised(logits: Tensor, labels: Tensor) -> dict[str, Tensor]:

    if logits.ndim != 4 or logits.shape[1] != 2:
        raise ValueError("logits must have shape [B,2,H,W]")
    if labels.shape != (logits.shape[0], *logits.shape[2:]):
        raise ValueError("labels must have shape [B,H,W]")
    labeled = labels.ne(-1)
    count = labeled.sum()

    logits32 = logits.float()
    ce_sum = F.cross_entropy(logits32, labels.long(), ignore_index=-1, reduction="sum")
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


def consistency_loss(student_logits: Tensor, teacher_logits: Tensor,
                     labels: Tensor, tau: float) -> tuple[Tensor, Tensor, Tensor]:

    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be in [0,1]")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("teacher and student logits must be geometrically aligned")
    with torch.no_grad():
        confidence, pseudo = teacher_logits.detach().float().softmax(1).max(1)
        mask = labels.eq(-1) & confidence.ge(tau)
    pixel_loss = F.cross_entropy(student_logits.float(), pseudo, reduction="none")
    loss = (pixel_loss * mask).sum() / mask.sum().clamp_min(1)
    return loss, mask, pseudo


def prototype_loss(features: Tensor, prototypes: Tensor, valid: Tensor,
                   mask: Tensor, pseudo: Tensor, temperature: float) -> Tensor:


    if temperature <= 0:
        raise ValueError("prototype temperature must be positive")
    if prototypes.shape != (2, features.shape[1]) or valid.shape != (2,):
        raise ValueError("Expected two class prototypes and two validity flags")
    if not bool(valid.all()):
        return features.sum() * 0.0
    normalized = F.normalize(features.float(), dim=1)
    detached_prototypes = F.normalize(prototypes.detach().float(), dim=1)
    scores = torch.einsum("bchw,kc->bkhw", normalized, detached_prototypes) / temperature
    pixel_loss = F.cross_entropy(scores, pseudo.detach().long(), reduction="none")
    return (pixel_loss * mask).sum() / mask.sum().clamp_min(1)


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:

    if not 0.0 <= momentum <= 1.0:
        raise ValueError("EMA momentum must be in [0,1]")
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    if teacher_state.keys() != student_state.keys():
        raise ValueError("EMA teacher and student architectures differ")
    for name, value in teacher_state.items():
        source = student_state[name].detach()
        if value.is_floating_point() or value.is_complex():
            value.mul_(momentum).add_(source, alpha=1.0 - momentum)
        else:
            value.copy_(source)
    teacher.requires_grad_(False)
    teacher.eval()
