"""Strict aggregation for the release-base Policy Protocol v2.

C1 and C3 are the primary trained controls and use matched Stage-2 seeds
1/2/3.  The strict primary matrix therefore contains exactly 36 records:
2 controls x 3 seeds x 3 tasks x 2 domains.  The only primary comparison is
the within-seed C3-C1 delta.

The author release checkpoint may optionally be attached as one fixed C0
reference.  If present, all six task/domain cells must use the same locked
online protocol and seed bank as C1/C3.  A partial or separately evaluated C0
reference is rejected instead of being silently mixed into the primary table.

Only the official online Clean and Random domains are accepted.  R1/R2/R3 are
paired training views, never Policy evaluation domains.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 4
PROFILE = "c1_c3_primary"
TASKS = ("place_a2b_left", "open_microwave", "move_stapler_pad")
DOMAINS = ("clean", "official_random")
TRAINED_CONTROLS = ("c1_architecture_only", "c3_ours")
STRICT_CONTROLS = ("c0_base", *TRAINED_CONTROLS)
PRIMARY_COMPARISON = ("c3_ours", "c1_architecture_only")
OPTIONAL_REFERENCE_COMPARISON = ("c3_ours", "c0_base")
COMPARISONS = (PRIMARY_COMPARISON, OPTIONAL_REFERENCE_COMPARISON)
ANCESTRY_FIELDS = (
    "base_checkpoint_sha256",
    "dataset_stats_sha256",
    "base_lineage_manifest_sha256",
    "runtime_source_sha256",
)
STAGE2_FAIRNESS_FIELDS = (
    "policy_regime",
    "head_init_sha256",
    "gca_init_sha256",
    "stage2_recipe_sha256",
    "p_mode_selection_manifest_sha256",
    "official_sample_sequence_sha256",
    "paired_physical_state_sequence_sha256",
    "matched_stream_contract_sha256",
)


class EvaluationProtocolError(ValueError):
    """Online result records do not prove the release-base protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationProtocolError(message)


def _mean_std(values: Sequence[float]) -> dict[str, float | int]:
    _require(bool(values), "cannot summarize an empty value sequence")
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "normal_95ci_low": mean - half_width,
        "normal_95ci_high": mean + half_width,
    }


def _wilson_summary(successes: int, episodes: int) -> dict[str, float | int]:
    _require(episodes > 0 and 0 <= successes <= episodes, "invalid binomial counts")
    rate = successes / episodes
    z = 1.96
    denominator = 1.0 + z * z / episodes
    centre = (rate + z * z / (2.0 * episodes)) / denominator
    radius = (
        z
        * math.sqrt(rate * (1.0 - rate) / episodes + z * z / (4.0 * episodes * episodes))
        / denominator
    )
    return {
        "episodes": episodes,
        "successes": successes,
        "success_rate": rate,
        "wilson_95ci_low": max(0.0, centre - radius),
        "wilson_95ci_high": min(1.0, centre + radius),
    }


def _as_rate(record: Mapping[str, Any]) -> float:
    try:
        rate = float(record["success_rate"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EvaluationProtocolError("success_rate must be numeric") from exc
    _require(math.isfinite(rate) and 0.0 <= rate <= 1.0, "success_rate must be within [0, 1]")
    return rate


def _as_positive_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _as_sha256(value: Any, label: str) -> str:
    digest = str(value).strip().lower()
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} must be a lowercase 64-character SHA-256",
    )
    return digest


def _record_fairness_identity(
    record: Mapping[str, Any], *, control: str, record_index: int
) -> tuple[tuple[str, ...], tuple[str | None, ...]]:
    ancestry = tuple(
        _as_sha256(record.get(field), f"record {record_index} {field}")
        for field in ANCESTRY_FIELDS
    )
    if control == "c0_base":
        for field in STAGE2_FAIRNESS_FIELDS:
            _require(
                field in record and record[field] is None,
                f"record {record_index} C0 {field} must be null",
            )
        stage2: tuple[str | None, ...] = (None,) * len(STAGE2_FAIRNESS_FIELDS)
    else:
        regime = str(record.get("policy_regime", ""))
        _require(regime in {"p_v1", "p_v2"}, f"record {record_index} policy_regime must be p_v1/p_v2")
        stage2 = (
            regime,
            _as_sha256(record.get("head_init_sha256"), f"record {record_index} head_init_sha256"),
            _as_sha256(record.get("gca_init_sha256"), f"record {record_index} gca_init_sha256"),
            _as_sha256(record.get("stage2_recipe_sha256"), f"record {record_index} stage2_recipe_sha256"),
            _as_sha256(
                record.get("p_mode_selection_manifest_sha256"),
                f"record {record_index} p_mode_selection_manifest_sha256",
            ),
            _as_sha256(
                record.get("official_sample_sequence_sha256"),
                f"record {record_index} official_sample_sequence_sha256",
            ),
            _as_sha256(
                record.get("paired_physical_state_sequence_sha256"),
                f"record {record_index} paired_physical_state_sequence_sha256",
            ),
            _as_sha256(
                record.get("matched_stream_contract_sha256"),
                f"record {record_index} matched_stream_contract_sha256",
            ),
        )
    return ancestry, stage2


def audit_and_summarize(
    payload: Mapping[str, Any],
    *,
    training_seeds: Sequence[int] = (1, 2, 3),
    episodes_per_cell: int = 100,
) -> dict[str, Any]:
    """Validate and summarize paired C1/C3, with an optional complete C0."""

    _require(payload.get("schema_version") == SCHEMA_VERSION, "evaluation schema_version changed")
    _require(payload.get("profile") == PROFILE, f"evaluation profile must be {PROFILE!r}")
    raw_records = payload.get("records")
    _require(isinstance(raw_records, list) and raw_records, "records must be a non-empty list")
    expected_seeds = tuple(int(value) for value in training_seeds)
    _require(len(expected_seeds) == len(set(expected_seeds)), "training seeds contain duplicates")
    _require(expected_seeds == (1, 2, 3), "strict formal aggregate requires Stage-2 seeds exactly (1, 2, 3)")
    expected_episodes = _as_positive_int(episodes_per_cell, "episodes_per_cell")

    cells: dict[tuple[str, int | None, str, str], float] = {}
    protocol_ids: set[str] = set()
    seed_bank_ids: set[str] = set()
    fairness_by_control_seed: dict[
        tuple[str, int | None], tuple[tuple[str, ...], tuple[str | None, ...]]
    ] = {}
    for index, raw in enumerate(raw_records):
        _require(isinstance(raw, Mapping), f"record {index} must be an object")
        control = str(raw.get("control", ""))
        _require(control in STRICT_CONTROLS, f"record {index} has unsupported control {control!r}")
        raw_seed = raw.get("training_seed")
        if control == "c0_base":
            _require(raw_seed is None, f"record {index} fixed C0 training_seed must be null")
            seed: int | None = None
        else:
            _require(
                isinstance(raw_seed, int) and not isinstance(raw_seed, bool),
                f"record {index} Stage-2 training_seed must be an integer",
            )
            seed = int(raw_seed)
            _require(seed in expected_seeds, f"record {index} training_seed is outside strict Stage-2 seeds")
        raw_lambda = raw.get("lambda_contrastive")
        gradient_flag = raw.get("paired_contrastive_gradient_enabled")
        if control == "c0_base":
            _require(
                raw_lambda is None and gradient_flag is None,
                f"record {index} fixed C0 must not claim contrastive treatment",
            )
        else:
            try:
                lambda_contrastive = float(raw_lambda)
            except (TypeError, ValueError, OverflowError) as exc:
                raise EvaluationProtocolError(
                    f"record {index} lambda_contrastive must be numeric"
                ) from exc
            _require(
                math.isfinite(lambda_contrastive) and lambda_contrastive >= 0.0,
                f"record {index} lambda_contrastive is invalid",
            )
            if control == "c1_architecture_only":
                _require(
                    lambda_contrastive == 0.0 and gradient_flag is False,
                    f"record {index} C1 must have lambda=0 and no contrastive gradient",
                )
            else:
                _require(
                    lambda_contrastive > 0.0 and gradient_flag is True,
                    f"record {index} C3 must enable contrastive gradient",
                )
        fairness_identity = _record_fairness_identity(raw, control=control, record_index=index)
        control_seed = (control, seed)
        previous_identity = fairness_by_control_seed.setdefault(control_seed, fairness_identity)
        _require(previous_identity == fairness_identity, f"checkpoint fairness identity mismatch within {control_seed}")

        task = str(raw.get("task", ""))
        domain = str(raw.get("domain", ""))
        _require(task in TASKS, f"record {index} has unsupported task {task!r}")
        _require(domain in DOMAINS, f"record {index} has unsupported domain {domain!r}; R1/R2/R3 are not Policy tests")
        _require(
            _as_positive_int(raw.get("episodes"), f"record {index} episodes") == expected_episodes,
            f"record {index} must contain exactly {expected_episodes} episodes",
        )
        protocol_id = str(raw.get("rollout_protocol_id", "")).strip()
        seed_bank_id = str(raw.get("simulator_seed_bank_id", "")).strip()
        _require(protocol_id != "", f"record {index} lacks rollout_protocol_id")
        _require(seed_bank_id != "", f"record {index} lacks simulator_seed_bank_id")
        protocol_ids.add(protocol_id)
        seed_bank_ids.add(seed_bank_id)
        key = (control, seed, task, domain)
        _require(key not in cells, f"duplicate evaluation cell: {key}")
        rate = _as_rate(raw)
        successes = round(rate * expected_episodes)
        _require(
            math.isclose(rate, successes / expected_episodes, abs_tol=1e-12),
            f"record {index} success_rate is not an exact episode count",
        )
        cells[key] = rate

    _require(len(protocol_ids) == 1, f"rollout protocol mismatch: {sorted(protocol_ids)}")
    _require(len(seed_bank_ids) == 1, f"simulator seed-bank mismatch: {sorted(seed_bank_ids)}")
    required_primary_keys = {
        (control, seed, task, domain)
        for control in TRAINED_CONTROLS
        for seed in expected_seeds
        for task in TASKS
        for domain in DOMAINS
    }
    c0_keys = {
        ("c0_base", None, task, domain) for task in TASKS for domain in DOMAINS
    }
    present_c0_keys = set(cells).intersection(c0_keys)
    if present_c0_keys:
        _require(
            present_c0_keys == c0_keys,
            "optional C0 reference must be all-or-none with all six task/domain cells",
        )
    c0_included = present_c0_keys == c0_keys
    required_keys = required_primary_keys | (c0_keys if c0_included else set())
    missing = sorted(required_primary_keys - set(cells), key=str)
    _require(not missing, f"missing required evaluation cells: {missing[:8]}")
    _require(set(cells) == required_keys, "evaluation matrix contains unexpected cells")
    _require(
        len(required_primary_keys) == 36,
        "internal primary evaluation matrix size changed",
    )

    # B_release is fixed globally, not independently seeded.  Every record must
    # therefore carry exactly the same release ancestry.
    ancestry_values = {identity[0] for identity in fairness_by_control_seed.values()}
    _require(len(ancestry_values) == 1, "fixed B_release ancestry mismatch across controls/seeds")
    release_ancestry = next(iter(ancestry_values))

    fairness_by_seed: dict[str, Any] = {}
    global_selection_shas: set[str | None] = set()
    for seed in expected_seeds:
        c1_identity = fairness_by_control_seed[("c1_architecture_only", seed)][1]
        c3_identity = fairness_by_control_seed[("c3_ours", seed)][1]
        _require(c1_identity == c3_identity, f"C1/C3 Stage-2 fairness identity mismatch for training_seed={seed}")
        selection_sha = c1_identity[STAGE2_FAIRNESS_FIELDS.index("p_mode_selection_manifest_sha256")]
        global_selection_shas.add(selection_sha)
        fairness_by_seed[str(seed)] = {
            "release_ancestry": dict(zip(ANCESTRY_FIELDS, release_ancestry, strict=True)),
            "c1_c3_stage2": dict(zip(STAGE2_FAIRNESS_FIELDS, c1_identity, strict=True)),
            "status": "PASS",
        }
    _require(len(global_selection_shas) == 1, "formal seeds do not share one global P-mode selection manifest")

    by_control: dict[str, Any] = {}
    for control in TRAINED_CONTROLS:
        task_rows: dict[str, Any] = {}
        for task in TASKS:
            task_rows[task] = {
                domain: _mean_std([cells[(control, seed, task, domain)] for seed in expected_seeds])
                for domain in DOMAINS
            }
        macro_by_seed = {
            domain: [
                statistics.fmean(cells[(control, seed, task, domain)] for task in TASKS)
                for seed in expected_seeds
            ]
            for domain in DOMAINS
        }
        by_control[control] = {
            "training_seeds": list(expected_seeds),
            "tasks": task_rows,
            "macro_average": {domain: _mean_std(values) for domain, values in macro_by_seed.items()},
        }

    if c0_included:
        # Fixed C0: report episode-level Wilson intervals; never training-seed
        # std.  This remains supplementary to the paired C1/C3 result.
        c0_tasks: dict[str, Any] = {}
        for task in TASKS:
            c0_tasks[task] = {}
            for domain in DOMAINS:
                rate = cells[("c0_base", None, task, domain)]
                c0_tasks[task][domain] = _wilson_summary(
                    round(rate * expected_episodes), expected_episodes
                )
        c0_macro: dict[str, Any] = {}
        for domain in DOMAINS:
            successes = sum(
                round(cells[("c0_base", None, task, domain)] * expected_episodes)
                for task in TASKS
            )
            c0_macro[domain] = {
                **_wilson_summary(successes, expected_episodes * len(TASKS)),
                "aggregation": "equal-task pooled episodes; fixed checkpoint",
            }
        by_control["c0_base"] = {
            "fixed_checkpoint": True,
            "training_seed": None,
            "tasks": c0_tasks,
            "macro_average": c0_macro,
        }

    comparisons: dict[str, Any] = {}
    active_comparisons = (PRIMARY_COMPARISON,) + (
        (OPTIONAL_REFERENCE_COMPARISON,) if c0_included else ()
    )
    for lhs, rhs in active_comparisons:
        comparison_id = f"{lhs}_minus_{rhs}"
        task_rows: dict[str, Any] = {}
        for task in TASKS:
            task_rows[task] = {}
            for domain in DOMAINS:
                deltas = [
                    cells[(lhs, seed, task, domain)]
                    - cells[(rhs, seed if rhs != "c0_base" else None, task, domain)]
                    for seed in expected_seeds
                ]
                task_rows[task][domain] = _mean_std(deltas)
        macro_rows: dict[str, Any] = {}
        for domain in DOMAINS:
            deltas = [
                statistics.fmean(cells[(lhs, seed, task, domain)] for task in TASKS)
                - statistics.fmean(
                    cells[(rhs, seed if rhs != "c0_base" else None, task, domain)]
                    for task in TASKS
                )
                for seed in expected_seeds
            ]
            macro_rows[domain] = _mean_std(deltas)
        comparisons[comparison_id] = {
            "tasks": task_rows,
            "macro_average": macro_rows,
            "role": "primary" if (lhs, rhs) == PRIMARY_COMPARISON else "supplementary_reference",
            "pairing": (
                "matched Stage-2 seed"
                if rhs != "c0_base"
                else "fixed C0 reused; uncertainty reflects Stage-2 seeds only"
            ),
        }

    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "profile": PROFILE,
        "record_count": len(cells),
        "required_primary_record_count": len(required_primary_keys),
        "optional_c0_reference": {
            "included": c0_included,
            "all_or_none": True,
            "record_count_if_included": len(c0_keys),
            "same_rollout_protocol_and_seed_bank_required": True,
        },
        "tasks": list(TASKS),
        "domains": list(DOMAINS),
        "stage2_training_seeds": list(expected_seeds),
        "seed_pairing": {
            "field": "training_seed",
            "semantics": (
                "C1_i and C3_i share Stage-2 seed i; optional fixed C0 has null seed"
            ),
            "paired_controls": list(TRAINED_CONTROLS),
            "c0_training_seed": None if c0_included else "not_included",
            "strict_formal": True,
        },
        "treatment_contract": {
            "c0_base": {"lambda_contrastive": None, "paired_contrastive_gradient_enabled": None},
            "c1_architecture_only": {"lambda_contrastive": 0.0, "paired_contrastive_gradient_enabled": False},
            "c3_ours": {"lambda_contrastive": "positive_locked_value", "paired_contrastive_gradient_enabled": True},
            "only_permitted_c1_c3_treatment_difference": "paired contrastive coefficient/gradient",
        },
        "fairness_identity_audit": {
            "status": "PASS",
            "release_ancestry_fields": list(ANCESTRY_FIELDS),
            "release_ancestry": dict(zip(ANCESTRY_FIELDS, release_ancestry, strict=True)),
            "trained_stage2_fields": list(STAGE2_FAIRNESS_FIELDS),
            "by_training_seed": fairness_by_seed,
        },
        "episodes_per_cell": expected_episodes,
        "rollout_protocol_id": next(iter(protocol_ids)),
        "simulator_seed_bank_id": next(iter(seed_bank_ids)),
        "controls": by_control,
        "comparisons": comparisons,
        "primary_comparison": "c3_ours_minus_c1_architecture_only",
        "c0_reference": by_control.get("c0_base"),
    }


def load_payload(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EvaluationProtocolError(f"cannot read evaluation JSON {resolved}: {exc}") from exc
    _require(isinstance(value, dict), "evaluation JSON root must be an object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output")
    parser.add_argument("--episodes-per-cell", type=int, default=100)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = audit_and_summarize(load_payload(args.input), episodes_per_cell=args.episodes_per_cell)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()


__all__ = [
    "ANCESTRY_FIELDS",
    "COMPARISONS",
    "DOMAINS",
    "EvaluationProtocolError",
    "OPTIONAL_REFERENCE_COMPARISON",
    "PRIMARY_COMPARISON",
    "PROFILE",
    "SCHEMA_VERSION",
    "STRICT_CONTROLS",
    "STAGE2_FAIRNESS_FIELDS",
    "TASKS",
    "TRAINED_CONTROLS",
    "audit_and_summarize",
    "load_payload",
]
