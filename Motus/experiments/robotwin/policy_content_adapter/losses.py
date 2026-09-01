"""Losses for the Motus paired content stream."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
from torch.nn import functional as F

from .protocol import DEFAULT_TEMPERATURE


def _equality_mask(
    values: torch.Tensor | Sequence[Hashable],
    *,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1 or values.shape[0] != count:
            raise ValueError(f"{name} must have shape [{count}]")
        encoded = values.to(device=device)
    else:
        items = list(values)
        if len(items) != count:
            raise ValueError(f"{name} must contain {count} values")
        mapping: dict[Hashable, int] = {}
        try:
            encoded = torch.tensor(
                [mapping.setdefault(item, len(mapping)) for item in items],
                dtype=torch.long,
                device=device,
            )
        except TypeError as exc:
            raise TypeError(f"every {name} value must be hashable") from exc
    return encoded[:, None] == encoded[None, :]


def multi_positive_supcon_loss(
    embeddings: torch.Tensor,
    physical_state_ids: torch.Tensor | Sequence[Hashable],
    task_ids: torch.Tensor | Sequence[Hashable],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """Same-state positives and same-task/different-state negatives."""

    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be [N,D] with N>=2")
    if not torch.is_floating_point(embeddings):
        raise TypeError("embeddings must be floating point")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError("embeddings contains NaN or infinity")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    count = int(embeddings.shape[0])
    task_equal = _equality_mask(
        task_ids, count=count, device=embeddings.device, name="task_ids"
    )
    state_equal = _equality_mask(
        physical_state_ids,
        count=count,
        device=embeddings.device,
        name="physical_state_ids",
    )
    not_self = ~torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positive = task_equal & state_equal & not_self
    negative = task_equal & ~state_equal
    positive_count = positive.sum(dim=1)
    negative_count = negative.sum(dim=1)
    if bool((positive_count == 0).any().item()):
        raise ValueError("every anchor needs another view of the same state")
    if bool((negative_count == 0).any().item()):
        raise ValueError("every anchor needs a different state from the same task")

    normalized = F.normalize(embeddings, p=2.0, dim=-1)
    logits = (normalized @ normalized.transpose(0, 1)) / float(temperature)
    denominator = positive | negative
    logits = logits.masked_fill(~denominator, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive, 0.0).sum(dim=1)
    loss = -(positive_logits / positive_count - log_denominator).mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("contrastive loss is non-finite")
    return loss

