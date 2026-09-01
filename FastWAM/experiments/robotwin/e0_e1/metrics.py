"""Metrics for background-invariance representation experiments.

All operations are implemented with PyTorch and the Python standard library.
The caller is responsible for extracting representations and for establishing
the underlying data correspondence; this module validates the sample grouping
again before computing metrics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch


STYLE_ORDER = ("r1", "r2", "r3")
REQUIRED_VARIANTS = ("clean", *STYLE_ORDER)
RESULT_COLUMNS = (
    "task",
    "layer",
    "experiment",
    "style_distance",
    "state_distance",
    "state_style_ratio",
    "clean_r1_distance",
    "clean_r2_distance",
    "clean_r3_distance",
    "retrieval_r1",
    "retrieval_r5",
    "positive_similarity",
    "negative_similarity",
    "num_samples",
)

_VARIANT_ALIASES = {
    "clean": "clean",
    "c": "clean",
    "r1": "r1",
    "random_1": "r1",
    "random1": "r1",
    "style_00_seed_0": "r1",
    "r2": "r2",
    "random_2": "r2",
    "random2": "r2",
    "style_01_seed_1": "r2",
    "r3": "r3",
    "random_3": "r3",
    "random3": "r3",
    "style_02_seed_2": "r3",
}


@dataclass(frozen=True)
class RepresentationRecord:
    """Identity fields associated with one representation.

    ``physical_state_id`` must identify the exact aligned timestep or clip, not
    merely its trajectory.  ``trajectory_id`` and ``timestep`` are used only to
    prioritize default same-task state negatives.
    """

    task: str
    physical_state_id: Hashable
    trajectory_id: Hashable
    timestep: int
    variant: str


def _field(record: RepresentationRecord | Mapping[str, Any] | Any, name: str) -> Any:
    if isinstance(record, Mapping):
        if name not in record:
            raise KeyError(f"Sample record is missing required field `{name}`: {record!r}")
        return record[name]
    if not hasattr(record, name):
        raise AttributeError(f"Sample record is missing required field `{name}`: {record!r}")
    return getattr(record, name)


def _canonical_variant(value: Any) -> str:
    key = str(value).strip().lower()
    if key not in _VARIANT_ALIASES:
        raise ValueError(
            f"Unknown representation variant {value!r}; expected one of {sorted(_VARIANT_ALIASES)}."
        )
    return _VARIANT_ALIASES[key]


def _coerce_records(
    records: Sequence[RepresentationRecord | Mapping[str, Any] | Any],
) -> list[RepresentationRecord]:
    coerced: list[RepresentationRecord] = []
    for index, record in enumerate(records):
        task = str(_field(record, "task")).strip()
        if not task:
            raise ValueError(f"Sample record {index} has an empty task name.")
        physical_state_id = _field(record, "physical_state_id")
        trajectory_id = _field(record, "trajectory_id")
        try:
            hash(physical_state_id)
            hash(trajectory_id)
        except TypeError as exc:
            raise TypeError(
                f"Record {index} physical_state_id and trajectory_id must be hashable."
            ) from exc
        timestep = int(_field(record, "timestep"))
        coerced.append(
            RepresentationRecord(
                task=task,
                physical_state_id=physical_state_id,
                trajectory_id=trajectory_id,
                timestep=timestep,
                variant=_canonical_variant(_field(record, "variant")),
            )
        )
    return coerced


def _validate_normalized_embeddings(
    embeddings: torch.Tensor,
    *,
    expected_count: int,
    normalization_tolerance: float,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError(f"`embeddings` must have shape [N, D], got {tuple(embeddings.shape)}.")
    if embeddings.shape[0] != expected_count:
        raise ValueError(
            f"Embedding/record count mismatch: {embeddings.shape[0]} vs {expected_count}."
        )
    if embeddings.shape[1] <= 0:
        raise ValueError("Embedding feature dimension must be positive.")
    if not torch.is_floating_point(embeddings):
        raise TypeError(f"`embeddings` must be floating point, got {embeddings.dtype}.")
    if not bool(torch.isfinite(embeddings).all().item()):
        raise ValueError("`embeddings` contains NaN or infinity.")
    if normalization_tolerance <= 0:
        raise ValueError(
            f"`normalization_tolerance` must be positive, got {normalization_tolerance}."
        )
    norms = embeddings.float().norm(p=2, dim=-1)
    max_error = (norms - 1.0).abs().max()
    if float(max_error.item()) > normalization_tolerance:
        raise ValueError(
            "Metrics require L2-normalized embeddings; "
            f"maximum norm error is {float(max_error.item()):.6g}."
        )


def _validate_groups(
    records: Sequence[RepresentationRecord],
    *,
    required_variants: Sequence[str] = REQUIRED_VARIANTS,
) -> dict[tuple[str, Hashable], dict[str, int]]:
    groups: dict[tuple[str, Hashable], dict[str, int]] = defaultdict(dict)
    for index, record in enumerate(records):
        key = (record.task, record.physical_state_id)
        if record.variant in groups[key]:
            raise ValueError(
                f"Duplicate `{record.variant}` representation for task/state {key!r}."
            )
        groups[key][record.variant] = index

    required_tuple = tuple(required_variants)
    required = set(required_tuple)
    for key, variants in groups.items():
        actual = set(variants)
        if actual != required:
            raise ValueError(
                f"Task/state {key!r} must contain exactly {required_tuple}; "
                f"missing={sorted(required - actual)}, extra={sorted(actual - required)}."
            )
        indices = list(variants.values())
        trajectory_ids = {records[index].trajectory_id for index in indices}
        timesteps = {records[index].timestep for index in indices}
        if len(trajectory_ids) != 1 or len(timesteps) != 1:
            raise ValueError(
                f"Variants for task/state {key!r} are not trajectory/timestep aligned."
            )
    return dict(groups)


def _cosine(embeddings: torch.Tensor, left: int, right: int) -> torch.Tensor:
    # Float32 reduction can exceed the mathematical cosine range by a few ULPs
    # even for unit vectors.  Clamp before turning cosine into a distance.
    return (embeddings[left].float() * embeddings[right].float()).sum().clamp(-1.0, 1.0)


def _expand_state_negative_pairs(
    state_negative_pairs: (
        Mapping[int, int | Sequence[int]] | Iterable[tuple[int, int]] | None
    ),
) -> list[tuple[int, int]] | None:
    if state_negative_pairs is None:
        return None
    expanded: list[tuple[int, int]] = []
    if isinstance(state_negative_pairs, Mapping):
        for anchor, candidates in state_negative_pairs.items():
            if isinstance(candidates, Integral):
                expanded.append((int(anchor), int(candidates)))
            else:
                expanded.extend((int(anchor), int(candidate)) for candidate in candidates)
    else:
        for pair in state_negative_pairs:
            if len(pair) != 2:
                raise ValueError(f"Every state-negative pair must have two indices, got {pair!r}.")
            expanded.append((int(pair[0]), int(pair[1])))
    if not expanded:
        raise ValueError("`state_negative_pairs` was supplied but contains no pairs.")
    return expanded


def _default_negative_pairs_for_task(
    clean_indices: Sequence[int],
    records: Sequence[RepresentationRecord],
    *,
    min_temporal_gap: int,
) -> list[tuple[int, int]]:
    """Choose one deterministic, prioritized clean negative for each anchor."""

    pairs: list[tuple[int, int]] = []
    for anchor in clean_indices:
        anchor_record = records[anchor]
        valid = [
            candidate
            for candidate in clean_indices
            if records[candidate].physical_state_id != anchor_record.physical_state_id
        ]
        same_trajectory = [
            candidate
            for candidate in valid
            if records[candidate].trajectory_id == anchor_record.trajectory_id
            and abs(records[candidate].timestep - anchor_record.timestep) >= min_temporal_gap
        ]
        if same_trajectory:
            # The furthest available timestep is least likely to be a false negative.
            candidate = max(
                same_trajectory,
                key=lambda index: (abs(records[index].timestep - anchor_record.timestep), -index),
            )
        else:
            different_trajectory = [
                candidate
                for candidate in valid
                if records[candidate].trajectory_id != anchor_record.trajectory_id
            ]
            if not different_trajectory:
                raise ValueError(
                    f"No valid same-task state negative exists for clean sample index {anchor}."
                )
            candidate = max(
                different_trajectory,
                key=lambda index: (abs(records[index].timestep - anchor_record.timestep), -index),
            )
        pairs.append((anchor, candidate))
    return pairs


def _validate_negative_pairs_for_task(
    pairs: Sequence[tuple[int, int]],
    records: Sequence[RepresentationRecord],
    *,
    task: str,
) -> list[tuple[int, int]]:
    valid: list[tuple[int, int]] = []
    count = len(records)
    for anchor, candidate in pairs:
        if anchor < 0 or anchor >= count or candidate < 0 or candidate >= count:
            raise IndexError(
                f"State-negative pair {(anchor, candidate)} is outside [0, {count})."
            )
        anchor_record = records[anchor]
        candidate_record = records[candidate]
        if anchor_record.task != task:
            continue
        if candidate_record.task != task:
            raise ValueError(
                f"State-negative pair {(anchor, candidate)} crosses tasks: "
                f"{anchor_record.task!r} vs {candidate_record.task!r}."
            )
        if anchor_record.variant != "clean" or candidate_record.variant != "clean":
            raise ValueError(
                f"State-negative pair {(anchor, candidate)} must compare clean representations."
            )
        if anchor_record.physical_state_id == candidate_record.physical_state_id:
            raise ValueError(
                f"State-negative pair {(anchor, candidate)} refers to the same physical state."
            )
        valid.append((anchor, candidate))
    if not valid:
        raise ValueError(f"No state-negative pairs are available for task {task!r}.")
    return valid


def _retrieval_accuracy(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    target_gallery_indices: torch.Tensor,
    *,
    k: int,
) -> float:
    if query_embeddings.shape[0] == 0 or gallery_embeddings.shape[0] == 0:
        raise ValueError("Retrieval requires non-empty query and gallery embeddings.")
    if k <= 0:
        raise ValueError(f"Retrieval k must be positive, got {k}.")
    top_k = min(int(k), int(gallery_embeddings.shape[0]))
    similarity = query_embeddings.float() @ gallery_embeddings.float().transpose(0, 1)
    retrieved = similarity.topk(top_k, dim=1, largest=True, sorted=True).indices
    correct = (retrieved == target_gallery_indices[:, None]).any(dim=1)
    return float(correct.float().mean().item())


def compute_representation_metrics(
    embeddings: torch.Tensor,
    records: Sequence[RepresentationRecord | Mapping[str, Any] | Any],
    *,
    layer: str,
    experiment: str,
    state_negative_pairs: (
        Mapping[int, int | Sequence[int]] | Iterable[tuple[int, int]] | None
    ) = None,
    min_temporal_gap: int = 8,
    ratio_epsilon: float = 1e-8,
    normalization_tolerance: float = 5e-3,
    include_average: bool = True,
    style_order: Sequence[str] = STYLE_ORDER,
    required_variants: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Compute E0/E1 metrics per task and, optionally, their macro average.

    Args:
        embeddings: L2-normalized tensor ``[N, D]``.
        records: One record per embedding.  Dataclasses, mappings, and objects
            exposing the five ``RepresentationRecord`` attributes are accepted.
        layer: Candidate backbone layer name stored in each result row.
        experiment: For example ``E0-RawBackbone`` or ``E1-InitHead``.
        state_negative_pairs: Optional caller-filtered clean-clean pairs, either
            as an iterable of ``(anchor_index, negative_index)`` or as a mapping
            from anchor index to one/many candidate indices.  When omitted, a
            deterministic same-trajectory/far-timestep-first policy is used.
        min_temporal_gap: Minimum gap for default same-trajectory negatives.
    """

    if not str(layer).strip():
        raise ValueError("`layer` must be a non-empty string.")
    if not str(experiment).strip():
        raise ValueError("`experiment` must be a non-empty string.")
    if min_temporal_gap < 0:
        raise ValueError(f"`min_temporal_gap` must be non-negative, got {min_temporal_gap}.")
    if ratio_epsilon <= 0:
        raise ValueError(f"`ratio_epsilon` must be positive, got {ratio_epsilon}.")

    canonical_styles = tuple(_canonical_variant(style) for style in style_order)
    if not canonical_styles:
        raise ValueError("`style_order` must contain at least one randomized rendering.")
    if "clean" in canonical_styles or len(set(canonical_styles)) != len(canonical_styles):
        raise ValueError("`style_order` must contain unique non-clean variants.")
    canonical_required = (
        ("clean", *canonical_styles)
        if required_variants is None
        else tuple(_canonical_variant(variant) for variant in required_variants)
    )
    if (
        not canonical_required
        or canonical_required[0] != "clean"
        or len(set(canonical_required)) != len(canonical_required)
        or set(canonical_required) != {"clean", *canonical_styles}
    ):
        raise ValueError(
            "`required_variants` must contain clean followed by exactly the "
            "variants named in `style_order`."
        )

    normalized_records = _coerce_records(records)
    _validate_normalized_embeddings(
        embeddings,
        expected_count=len(normalized_records),
        normalization_tolerance=normalization_tolerance,
    )
    groups = _validate_groups(
        normalized_records, required_variants=canonical_required
    )
    explicit_pairs = _expand_state_negative_pairs(state_negative_pairs)

    tasks = sorted({record.task for record in normalized_records})
    rows: list[dict[str, Any]] = []
    for task in tasks:
        task_groups = [
            (key, variants)
            for key, variants in groups.items()
            if key[0] == task
        ]
        task_groups.sort(key=lambda item: min(item[1].values()))
        if len(task_groups) < 2:
            raise ValueError(
                f"Task {task!r} needs at least two physical states for state-distance metrics."
            )

        style_similarities: dict[str, list[torch.Tensor]] = {
            style: [] for style in canonical_styles
        }
        clean_indices: list[int] = []
        style_indices: dict[str, list[int]] = {
            style: [] for style in canonical_styles
        }
        for _, variants in task_groups:
            clean_index = variants["clean"]
            clean_indices.append(clean_index)
            for style in canonical_styles:
                style_index = variants[style]
                style_indices[style].append(style_index)
                style_similarities[style].append(_cosine(embeddings, clean_index, style_index))

        clean_gallery = embeddings[clean_indices]
        target_gallery_indices = torch.arange(
            len(clean_indices), dtype=torch.long, device=embeddings.device
        )
        retrieval_at_1: dict[str, float] = {}
        retrieval_at_5: dict[str, float] = {}
        for style in canonical_styles:
            queries = embeddings[style_indices[style]]
            retrieval_at_1[style] = _retrieval_accuracy(
                queries,
                clean_gallery,
                target_gallery_indices,
                k=1,
            )
            retrieval_at_5[style] = _retrieval_accuracy(
                queries,
                clean_gallery,
                target_gallery_indices,
                k=5,
            )

        if explicit_pairs is None:
            task_negative_pairs = _default_negative_pairs_for_task(
                clean_indices,
                normalized_records,
                min_temporal_gap=min_temporal_gap,
            )
        else:
            task_negative_pairs = _validate_negative_pairs_for_task(
                explicit_pairs,
                normalized_records,
                task=task,
            )
        # Default pairs have already been structurally selected, but validate
        # them through the same contract as caller-provided pairs.
        task_negative_pairs = _validate_negative_pairs_for_task(
            task_negative_pairs,
            normalized_records,
            task=task,
        )

        style_similarity_by_variant = {
            style: float(torch.stack(values).mean().item())
            for style, values in style_similarities.items()
        }
        positive_similarity = sum(style_similarity_by_variant.values()) / len(
            canonical_styles
        )
        clean_style_distances = {
            style: 1.0 - similarity
            for style, similarity in style_similarity_by_variant.items()
        }
        style_distance = max(
            0.0, sum(clean_style_distances.values()) / len(canonical_styles)
        )

        negative_values = [
            _cosine(embeddings, anchor, negative)
            for anchor, negative in task_negative_pairs
        ]
        negative_similarity = float(torch.stack(negative_values).mean().item())
        state_distance = max(0.0, 1.0 - negative_similarity)
        state_style_ratio = state_distance / (style_distance + float(ratio_epsilon))

        average_retrieval_at_1 = sum(retrieval_at_1.values()) / len(canonical_styles)
        average_retrieval_at_5 = sum(retrieval_at_5.values()) / len(canonical_styles)
        clean_distance_columns = {
            f"clean_{style}_distance": clean_style_distances.get(style)
            for style in STYLE_ORDER
        }
        retrieval_columns = {
            f"{style}_to_clean_retrieval_at{k}": (
                retrieval_at_1.get(style) if k == 1 else retrieval_at_5.get(style)
            )
            for style in STYLE_ORDER
            for k in (1, 5)
        }
        rows.append(
            {
                "task": task,
                "layer": str(layer),
                "experiment": str(experiment),
                "style_distance": style_distance,
                "state_distance": state_distance,
                "state_style_ratio": state_style_ratio,
                **clean_distance_columns,
                # Required result names: retrieval_r1/r5 mean Retrieval@1/@5.
                "retrieval_r1": average_retrieval_at_1,
                "retrieval_r5": average_retrieval_at_5,
                **retrieval_columns,
                "positive_similarity": positive_similarity,
                "negative_similarity": negative_similarity,
                "num_samples": len(task_groups) * len(canonical_required),
                "num_physical_states": len(task_groups),
                "num_state_negative_pairs": len(task_negative_pairs),
                "evaluated_styles": list(canonical_styles),
                "required_variants": list(canonical_required),
            }
        )

    if include_average:
        rows.append(macro_average_metrics(rows))
    return rows


def macro_average_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return an unweighted task-level average row."""

    task_rows = [row for row in rows if not str(row.get("task", "")).endswith("-task-average")]
    if not task_rows:
        raise ValueError("At least one task metric row is required for macro averaging.")
    layers = {str(row["layer"]) for row in task_rows}
    experiments = {str(row["experiment"]) for row in task_rows}
    if len(layers) != 1 or len(experiments) != 1:
        raise ValueError("Macro averaging requires rows from one layer and one experiment.")

    non_average_keys = {
        "task",
        "layer",
        "experiment",
        "num_samples",
        "num_physical_states",
        "num_state_negative_pairs",
    }
    numeric_keys = [
        key
        for key, value in task_rows[0].items()
        if key not in non_average_keys and isinstance(value, (int, float))
    ]
    averaged: dict[str, Any] = {
        "task": f"{len(task_rows)}-task-average",
        "layer": next(iter(layers)),
        "experiment": next(iter(experiments)),
    }
    for key in numeric_keys:
        averaged[key] = sum(float(row[key]) for row in task_rows) / len(task_rows)
    # Protocol-specific evaluations deliberately leave non-active per-style
    # columns null.  Preserve those stable columns in the macro row so the
    # JSON/CSV schema remains rectangular without inventing measurements.
    for key, value in task_rows[0].items():
        if key in non_average_keys or key in numeric_keys or key in averaged:
            continue
        values = [row.get(key) for row in task_rows]
        if all(item is None for item in values):
            averaged[key] = None
        elif all(item == value for item in values):
            averaged[key] = value
    for key in ("num_samples", "num_physical_states", "num_state_negative_pairs"):
        averaged[key] = sum(int(row[key]) for row in task_rows)
    return averaged


def summarize_metric_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] = RESULT_COLUMNS,
) -> list[dict[str, Any]]:
    """Select a stable JSON/CSV-ready set of result columns."""

    summarized: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        missing = [column for column in columns if column not in row]
        if missing:
            raise KeyError(f"Metric row {row_index} is missing columns {missing}.")
        summarized.append({column: row[column] for column in columns})
    return summarized


__all__ = [
    "REQUIRED_VARIANTS",
    "RESULT_COLUMNS",
    "RepresentationRecord",
    "STYLE_ORDER",
    "compute_representation_metrics",
    "macro_average_metrics",
    "summarize_metric_rows",
]
