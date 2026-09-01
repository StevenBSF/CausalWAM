"""Small trainable representation head and its contrastive objective.

The FastWAM backbone is intentionally not referenced here.  Keeping this module
independent makes it straightforward for the experiment runner to freeze the
backbone and give the optimizer only ``ContrastiveContentHead.parameters()``.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_BACKBONE_DIM = 3072
DEFAULT_EMBED_DIM = 384
DEFAULT_NUM_QUERIES = 8
DEFAULT_NUM_HEADS = 8
DEFAULT_TEMPERATURE = 0.07
DEFAULT_PARAMETER_COUNT = 2_070_144


def count_trainable_parameters(module: nn.Module) -> int:
    """Return the exact number of scalar parameters that require gradients."""

    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class ContrastiveContentHead(nn.Module):
    """Pool visual tokens with learnable queries into a normalized embedding.

    Architecture::

        visual tokens -> Linear(D, E)
                      -> Q learnable queries cross-attend to the tokens
                      -> mean over queries
                      -> Linear(E, E) -> SiLU -> Linear(E, E)
                      -> L2 normalization

    ``nn.MultiheadAttention`` uses its standard trainable Q/K/V and output
    biases.  With the requested default dimensions this architecture contains
    exactly 2,070,144 trainable scalars.
    """

    def __init__(
        self,
        backbone_dim: int = DEFAULT_BACKBONE_DIM,
        embed_dim: int = DEFAULT_EMBED_DIM,
        num_queries: int = DEFAULT_NUM_QUERIES,
        num_heads: int = DEFAULT_NUM_HEADS,
        normalize_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if backbone_dim <= 0:
            raise ValueError(f"`backbone_dim` must be positive, got {backbone_dim}.")
        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if num_queries <= 0:
            raise ValueError(f"`num_queries` must be positive, got {num_queries}.")
        if num_heads <= 0 or embed_dim % num_heads != 0:
            raise ValueError(
                "`num_heads` must be positive and divide `embed_dim`, "
                f"got embed_dim={embed_dim}, num_heads={num_heads}."
            )
        if normalize_eps <= 0:
            raise ValueError(f"`normalize_eps` must be positive, got {normalize_eps}.")

        self.backbone_dim = int(backbone_dim)
        self.embed_dim = int(embed_dim)
        self.num_queries = int(num_queries)
        self.num_heads = int(num_heads)
        self.normalize_eps = float(normalize_eps)

        self.token_projection = nn.Linear(self.backbone_dim, self.embed_dim)
        self.content_queries = nn.Parameter(torch.empty(self.num_queries, self.embed_dim))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            batch_first=True,
            bias=True,
        )
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        nn.init.normal_(self.content_queries, mean=0.0, std=self.embed_dim**-0.5)

    def trainable_parameter_count(self) -> int:
        """Return the exact number of trainable scalars in this head."""

        return count_trainable_parameters(self)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return one L2-normalized embedding per token sequence.

        Args:
            visual_tokens: Tensor shaped ``[batch, tokens, backbone_dim]``.
            key_padding_mask: Optional boolean tensor shaped ``[batch, tokens]``;
                ``True`` positions are ignored, following PyTorch attention
                semantics.  A sample may not mask every token.
        """

        if visual_tokens.ndim != 3:
            raise ValueError(
                "`visual_tokens` must have shape [batch, tokens, backbone_dim], "
                f"got {tuple(visual_tokens.shape)}."
            )
        batch_size, sequence_length, feature_dim = visual_tokens.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError(
                "`visual_tokens` must contain at least one batch item and token, "
                f"got {tuple(visual_tokens.shape)}."
            )
        if feature_dim != self.backbone_dim:
            raise ValueError(
                f"Expected visual token dim {self.backbone_dim}, got {feature_dim}."
            )
        if not torch.is_floating_point(visual_tokens):
            raise TypeError(f"`visual_tokens` must be floating point, got {visual_tokens.dtype}.")
        if not bool(torch.isfinite(visual_tokens).all().item()):
            raise ValueError("`visual_tokens` contains NaN or infinity.")

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "`key_padding_mask` must have shape "
                    f"{(batch_size, sequence_length)}, got {tuple(key_padding_mask.shape)}."
                )
            if key_padding_mask.dtype is not torch.bool:
                raise TypeError(
                    f"`key_padding_mask` must have dtype bool, got {key_padding_mask.dtype}."
                )
            if bool(key_padding_mask.all(dim=1).any().item()):
                raise ValueError("Every sample must retain at least one unmasked visual token.")
            key_padding_mask = key_padding_mask.to(device=visual_tokens.device)

        projected_tokens = self.token_projection(visual_tokens)
        queries = self.content_queries.to(dtype=projected_tokens.dtype).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        content_tokens, _ = self.cross_attention(
            query=queries,
            key=projected_tokens,
            value=projected_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = content_tokens.mean(dim=1)
        embedding = self.mlp(pooled)
        return F.normalize(embedding, p=2.0, dim=-1, eps=self.normalize_eps)


def _labels_to_equality_mask(
    labels: torch.Tensor | Sequence[Hashable],
    *,
    count: int,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    """Encode arbitrary hashable labels and return their pairwise equality mask."""

    if isinstance(labels, torch.Tensor):
        if labels.ndim != 1 or labels.shape[0] != count:
            raise ValueError(f"`{name}` must have shape [{count}], got {tuple(labels.shape)}.")
        encoded = labels.to(device=device)
        return encoded[:, None] == encoded[None, :]

    values = list(labels)
    if len(values) != count:
        raise ValueError(f"`{name}` must contain {count} labels, got {len(values)}.")
    ids: dict[Hashable, int] = {}
    encoded_values: list[int] = []
    for value in values:
        try:
            encoded_values.append(ids.setdefault(value, len(ids)))
        except TypeError as exc:
            raise TypeError(f"Every `{name}` value must be hashable, got {value!r}.") from exc
    encoded = torch.tensor(encoded_values, dtype=torch.long, device=device)
    return encoded[:, None] == encoded[None, :]


def multi_positive_supcon_loss(
    embeddings: torch.Tensor,
    physical_state_ids: torch.Tensor | Sequence[Hashable],
    task_ids: torch.Tensor | Sequence[Hashable],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    negative_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute same-task, multi-positive supervised contrastive loss.

    Samples sharing both ``task_id`` and ``physical_state_id`` are positives.
    By default, every other sample from the same task is in the denominator.
    ``negative_mask`` can instead provide a caller-filtered admissible-negative
    relation, e.g. same-trajectory/far-timestep or same-task/other-trajectory
    pairs after a physical-state proximity filter.  Positives are always included
    in the denominator and cross-task samples are always excluded.  The function
    fails closed if any anchor has no positive or no admissible negative.
    """

    if embeddings.ndim != 2:
        raise ValueError(f"`embeddings` must have shape [N, D], got {tuple(embeddings.shape)}.")
    count, feature_dim = embeddings.shape
    if count < 2 or feature_dim <= 0:
        raise ValueError(f"`embeddings` must contain at least two samples, got {tuple(embeddings.shape)}.")
    if not torch.is_floating_point(embeddings):
        raise TypeError(f"`embeddings` must be floating point, got {embeddings.dtype}.")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError("`embeddings` contains NaN or infinity.")
    if temperature <= 0:
        raise ValueError(f"`temperature` must be positive, got {temperature}.")

    task_equal = _labels_to_equality_mask(
        task_ids,
        count=count,
        device=embeddings.device,
        name="task_ids",
    )
    state_equal = _labels_to_equality_mask(
        physical_state_ids,
        count=count,
        device=embeddings.device,
        name="physical_state_ids",
    )
    not_self = ~torch.eye(count, dtype=torch.bool, device=embeddings.device)
    comparison_mask = task_equal & not_self
    positive_mask = state_equal & comparison_mask
    positive_count = positive_mask.sum(dim=1)
    anchors_without_positive = torch.nonzero(positive_count == 0, as_tuple=False).flatten()
    if anchors_without_positive.numel() > 0:
        raise ValueError(
            "Every anchor must have a same-task sample with the same physical-state label; "
            f"missing anchors={anchors_without_positive.detach().cpu().tolist()}."
        )

    if negative_mask is None:
        admissible_negative_mask = comparison_mask & ~state_equal
    else:
        if not isinstance(negative_mask, torch.Tensor):
            raise TypeError("`negative_mask` must be a torch.Tensor or None.")
        if negative_mask.shape != (count, count):
            raise ValueError(
                f"`negative_mask` must have shape {(count, count)}, "
                f"got {tuple(negative_mask.shape)}."
            )
        if negative_mask.dtype is not torch.bool:
            raise TypeError(
                f"`negative_mask` must have dtype bool, got {negative_mask.dtype}."
            )
        admissible_negative_mask = negative_mask.to(device=embeddings.device)
        if bool((admissible_negative_mask & ~comparison_mask).any().item()):
            raise ValueError("`negative_mask` contains self or cross-task pairs.")
        if bool((admissible_negative_mask & state_equal).any().item()):
            raise ValueError("`negative_mask` marks a positive pair as negative.")
    negative_count = admissible_negative_mask.sum(dim=1)
    anchors_without_negative = torch.nonzero(
        negative_count == 0, as_tuple=False
    ).flatten()
    if anchors_without_negative.numel() > 0:
        raise ValueError(
            "Every anchor must have an admissible same-task physical-state negative; "
            f"missing anchors={anchors_without_negative.detach().cpu().tolist()}."
        )

    normalized = F.normalize(embeddings, p=2.0, dim=-1)
    logits = (normalized @ normalized.transpose(0, 1)) / float(temperature)
    denominator_mask = positive_mask | admissible_negative_mask
    logits = logits.masked_fill(~denominator_mask, -torch.inf)
    log_denominator = torch.logsumexp(logits, dim=1)
    positive_logits = logits.masked_fill(~positive_mask, 0.0).sum(dim=1)
    mean_positive_log_probability = positive_logits / positive_count - log_denominator
    loss = -mean_positive_log_probability.mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("Multi-positive contrastive loss is not finite.")
    return loss


# The experiment specification uses both SupCon and InfoNCE terminology for the
# same multi-positive objective.  Keep an explicit alias for CLI readability.
multi_positive_info_nce_loss = multi_positive_supcon_loss


__all__ = [
    "ContrastiveContentHead",
    "DEFAULT_BACKBONE_DIM",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_NUM_HEADS",
    "DEFAULT_NUM_QUERIES",
    "DEFAULT_PARAMETER_COUNT",
    "DEFAULT_TEMPERATURE",
    "count_trainable_parameters",
    "multi_positive_info_nce_loss",
    "multi_positive_supcon_loss",
]
