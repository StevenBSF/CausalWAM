"""Frozen, deployment-equivalent FastWAM representation extraction.

The action inference path does not call ``video_expert.blocks[i].forward``.
Instead, :class:`fastwam.models.wan22.mot.MoT` expands every block into its
attention and post-attention pieces while it builds the video K/V cache.  A
normal PyTorch forward hook on a video block is therefore ineffective.  This
module mirrors that small prefill loop, captures the *updated* video tokens at
selected layers, and never invokes action denoising or a policy rollout.

Layer numbers exposed here are one-based.  The default candidates, 8/16/24,
are the early-middle, middle, and late-middle blocks of the 30-layer released
RoboTwin FastWAM video expert.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_SRC = PROJECT_ROOT / "src"
DEFAULT_CONFIG_NAME = "sim_robotwin.yaml"
DEFAULT_TASK_CONFIG = "robotwin_uncond_3cam_384_1e-4"
DEFAULT_LAYERS = (8, 16, 24)
DEPLOYMENT_IMAGE_SHAPE = (3, 384, 320)
PROPRIO_MODES = ("observed", "constant_zero_normalized")
ProprioMode = Literal["observed", "constant_zero_normalized"]
DEPLOYMENT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


class FrozenBackboneError(RuntimeError):
    """The extractor cannot prove that its frozen-backbone contract holds."""


@dataclass(frozen=True)
class CheckpointAudit:
    """Strict, JSON-serializable identity of the loaded release checkpoint."""

    path: str
    size_bytes: int
    mtime_ns: int
    step: int | None
    declared_torch_dtype: str | None
    mot_tensor_count: int
    proprio_tensor_count: int
    sha256: str | None = None


@dataclass(frozen=True)
class FrozenBackboneOutput:
    """CPU cache from one deployment-equivalent observation batch.

    ``tokens_by_layer`` retains the spatial tokens in their checkpoint dtype.
    ``pooled_by_layer`` is always float32 and contains mean-pooled, L2-normalized
    vectors.  Both dictionaries use one-based video-block indices.
    """

    tokens_by_layer: dict[int, torch.Tensor]
    pooled_by_layer: dict[int, torch.Tensor]
    provenance: dict[str, Any]


def format_deployment_prompt(instruction: str) -> str:
    """Format an instruction exactly like FastWAM RoboTwin deployment."""

    value = str(instruction).strip()
    if not value:
        raise FrozenBackboneError("instruction must be non-empty")
    return DEPLOYMENT_PROMPT.format(task=value)


def _canonical_layers(layers: Sequence[int], num_layers: int) -> tuple[int, ...]:
    result = tuple(int(layer) for layer in layers)
    if not result:
        raise FrozenBackboneError("at least one capture layer is required")
    if len(set(result)) != len(result):
        raise FrozenBackboneError(f"capture layers must be distinct, got {result}")
    invalid = [layer for layer in result if layer < 1 or layer > num_layers]
    if invalid:
        raise FrozenBackboneError(
            f"capture layers {invalid} are outside the one-based range 1..{num_layers}"
        )
    return result


def run_video_prefill_with_captures(
    mot: torch.nn.Module,
    *,
    video_tokens: torch.Tensor,
    video_freqs: torch.Tensor,
    video_t_mod: torch.Tensor,
    video_context_payload: Mapping[str, torch.Tensor | None] | None,
    video_attention_mask: torch.Tensor,
    capture_layers: Sequence[int] = DEFAULT_LAYERS,
) -> tuple[dict[int, torch.Tensor], list[dict[str, torch.Tensor]]]:
    """Mirror ``MoT.prefill_video_cache`` and retain selected updated tokens.

    This intentionally calls the same private MoT primitives, in the same
    order, as the checked-out FastWAM implementation.  It is kept here rather
    than changing the policy/model code so E0/E1 cannot alter the action path.
    The caller must put the module in eval mode and control autograd.
    """

    if bool(mot.training):
        raise FrozenBackboneError("video prefill capture requires mot.eval()")
    if video_tokens.ndim != 3:
        raise FrozenBackboneError(
            f"video_tokens must be [B,S,D], got {tuple(video_tokens.shape)}"
        )
    if video_attention_mask.ndim != 2:
        raise FrozenBackboneError(
            "video_attention_mask must be a rank-two square matrix"
        )
    sequence_length = int(video_tokens.shape[1])
    if tuple(video_attention_mask.shape) != (sequence_length, sequence_length):
        raise FrozenBackboneError(
            "video attention mask/token mismatch: "
            f"{tuple(video_attention_mask.shape)} vs S={sequence_length}"
        )
    if not hasattr(mot, "mixtures") or "video" not in mot.mixtures:
        raise FrozenBackboneError("MoT has no video expert")

    expert = mot.mixtures["video"]
    num_layers = int(mot.num_layers)
    if len(expert.blocks) != num_layers:
        raise FrozenBackboneError(
            f"video expert has {len(expert.blocks)} blocks but MoT reports {num_layers}"
        )
    layers = _canonical_layers(capture_layers, num_layers)
    layer_set = set(layers)

    x = video_tokens
    captures: dict[int, torch.Tensor] = {}
    kv_cache: list[dict[str, torch.Tensor]] = []
    context_payload = (
        None if video_context_payload is None else dict(video_context_payload)
    )
    for zero_based_index in range(num_layers):
        block = expert.blocks[zero_based_index]
        (
            q,
            k,
            v,
            residual_x,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            use_gradient_checkpointing,
        ) = mot._build_expert_attention_io(  # noqa: SLF001 - mirrors native prefill
            expert=expert,
            block=block,
            x=x,
            freqs=video_freqs,
            t_mod=video_t_mod,
        )
        mixed = mot._mixed_attention(  # noqa: SLF001 - mirrors native prefill
            q_cat=q,
            k_cat=k,
            v_cat=v,
            attention_mask=video_attention_mask,
        )
        x = mot._apply_post_with_optional_checkpoint(  # noqa: SLF001
            block=block,
            residual_x=residual_x,
            gate_msa=gate_msa,
            shift_mlp=shift_mlp,
            scale_mlp=scale_mlp,
            gate_mlp=gate_mlp,
            use_gradient_checkpointing=use_gradient_checkpointing,
            mixed_slice=mixed,
            context_payload=context_payload,
        )
        one_based_layer = zero_based_index + 1
        if one_based_layer in layer_set:
            captures[one_based_layer] = x
        kv_cache.append({"k": k, "v": v})

    if set(captures) != layer_set:
        raise FrozenBackboneError(
            f"capture loop returned layers {sorted(captures)}, expected {sorted(layer_set)}"
        )
    return {layer: captures[layer] for layer in layers}, kv_cache


def assert_kv_cache_equivalent(
    captured_cache: Sequence[Mapping[str, torch.Tensor]],
    native_cache: Sequence[Mapping[str, torch.Tensor]],
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Fail if the mirrored loop diverges from native video prefill."""

    if len(captured_cache) != len(native_cache):
        raise FrozenBackboneError(
            f"K/V cache lengths differ: {len(captured_cache)} vs {len(native_cache)}"
        )
    for layer_index, (captured, native) in enumerate(
        zip(captured_cache, native_cache), start=1
    ):
        if set(captured) != {"k", "v"} or set(native) != {"k", "v"}:
            raise FrozenBackboneError(
                f"layer {layer_index} K/V cache keys are not canonical"
            )
        for key in ("k", "v"):
            try:
                torch.testing.assert_close(
                    captured[key], native[key], rtol=rtol, atol=atol
                )
            except AssertionError as exc:
                raise FrozenBackboneError(
                    f"mirrored prefill differs from native cache at layer "
                    f"{layer_index}/{key}: {exc}"
                ) from exc


def _state_dict_shape_audit(
    expected: Mapping[str, torch.Tensor],
    supplied: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    expected_keys = set(expected)
    supplied_keys = set(supplied)
    missing = sorted(expected_keys - supplied_keys)
    unexpected = sorted(supplied_keys - expected_keys)
    if missing or unexpected:
        raise FrozenBackboneError(
            f"{label} checkpoint keys differ: missing={missing[:20]}, "
            f"unexpected={unexpected[:20]}"
        )
    bad_types = sorted(key for key, value in supplied.items() if not torch.is_tensor(value))
    if bad_types:
        raise FrozenBackboneError(
            f"{label} contains non-tensor values at keys {bad_types[:20]}"
        )
    mismatches = [
        (key, tuple(expected[key].shape), tuple(supplied[key].shape))
        for key in sorted(expected)
        if tuple(expected[key].shape) != tuple(supplied[key].shape)
    ]
    if mismatches:
        raise FrozenBackboneError(
            f"{label} checkpoint tensor shapes differ: {mismatches[:20]}"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load_release_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    compute_sha256: bool = False,
) -> CheckpointAudit:
    """Load a FastWAM checkpoint only after exact key and shape validation.

    The normal FastWAM loader uses ``strict=False`` for ``mot``.  That is useful
    for legacy training compatibility but unsafe for a representation baseline:
    an accidentally partial video backbone would silently change E0.  This
    loader requires exact ``mot`` and ``proprio_encoder`` schemas before calling
    ``load_state_dict(..., strict=True)``.
    """

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FastWAM checkpoint not found: {path}")
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except Exception as exc:
        raise FrozenBackboneError(f"cannot load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise FrozenBackboneError(
            f"checkpoint root must be a mapping, got {type(payload).__name__}"
        )

    mot_state = payload.get("mot")
    if not isinstance(mot_state, Mapping):
        raise FrozenBackboneError("release checkpoint is missing mapping key 'mot'")
    if not hasattr(model, "mot"):
        raise FrozenBackboneError("model has no mot module")
    expected_mot = model.mot.state_dict()
    _state_dict_shape_audit(expected_mot, mot_state, label="mot")

    proprio_module = getattr(model, "proprio_encoder", None)
    checkpoint_proprio = payload.get("proprio_encoder")
    if proprio_module is None:
        if checkpoint_proprio is not None:
            raise FrozenBackboneError(
                "checkpoint contains proprio_encoder but model has it disabled"
            )
        proprio_count = 0
    else:
        if not isinstance(checkpoint_proprio, Mapping):
            raise FrozenBackboneError(
                "model requires proprio_encoder but checkpoint does not contain it"
            )
        expected_proprio = proprio_module.state_dict()
        _state_dict_shape_audit(
            expected_proprio, checkpoint_proprio, label="proprio_encoder"
        )
        proprio_count = len(checkpoint_proprio)

    incompatible = model.mot.load_state_dict(mot_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise FrozenBackboneError(
            "strict mot load unexpectedly reported incompatible keys: "
            f"{incompatible}"
        )
    if proprio_module is not None:
        incompatible = proprio_module.load_state_dict(checkpoint_proprio, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise FrozenBackboneError(
                "strict proprio load unexpectedly reported incompatible keys: "
                f"{incompatible}"
            )

    stat = path.stat()
    step_value = payload.get("step")
    step = None if step_value is None else int(step_value)
    declared_dtype = payload.get("torch_dtype")
    audit = CheckpointAudit(
        path=str(path),
        size_bytes=int(stat.st_size),
        mtime_ns=int(stat.st_mtime_ns),
        step=step,
        declared_torch_dtype=(
            None if declared_dtype is None else str(declared_dtype)
        ),
        mot_tensor_count=len(mot_state),
        proprio_tensor_count=proprio_count,
        sha256=_sha256_file(path) if compute_sha256 else None,
    )
    del payload
    return audit


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    byte_view = tensor.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@contextlib.contextmanager
def _temporary_environment(name: str, value: str):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _ensure_current_checkout_import() -> str:
    expected = PROJECT_SRC.resolve()
    expected_text = str(expected)
    if expected_text not in sys.path:
        sys.path.insert(0, expected_text)
    import fastwam  # pylint: disable=import-outside-toplevel

    actual = Path(fastwam.__file__).resolve()
    if expected not in actual.parents:
        raise FrozenBackboneError(
            "wrong FastWAM checkout imported: "
            f"{actual}; expected a module under {expected}. Start the command with "
            f"PYTHONPATH={expected_text}."
        )
    return str(actual)


class FrozenFastWAMExtractor:
    """Extract current-observation video tokens with a permanently frozen model."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        processor: Any,
        checkpoint_audit: CheckpointAudit,
        capture_layers: Sequence[int] = DEFAULT_LAYERS,
        config_provenance: Mapping[str, Any] | None = None,
        verify_native_prefill: bool = False,
    ) -> None:
        self._model = model
        self.processor = processor
        self.checkpoint_audit = checkpoint_audit
        if not hasattr(model, "mot"):
            raise FrozenBackboneError("FastWAM model has no mot module")
        self.capture_layers = _canonical_layers(
            capture_layers, int(model.mot.num_layers)
        )
        self.config_provenance = dict(config_provenance or {})
        self.verify_native_prefill = bool(verify_native_prefill)
        self._freeze_and_clear_gradients()

    @classmethod
    def from_release_checkpoint(
        cls,
        checkpoint_path: str | Path,
        dataset_stats_path: str | Path,
        *,
        task_config: str = DEFAULT_TASK_CONFIG,
        config_name: str = DEFAULT_CONFIG_NAME,
        config_root: str | Path = PROJECT_ROOT / "configs",
        model_base_path: str | Path | None = None,
        device: str = "cuda",
        model_dtype: torch.dtype = torch.bfloat16,
        capture_layers: Sequence[int] = DEFAULT_LAYERS,
        verify_native_prefill: bool = False,
        compute_checkpoint_sha256: bool = False,
    ) -> "FrozenFastWAMExtractor":
        """Compose the real RoboTwin Hydra task and load the official weights."""

        fastwam_source = _ensure_current_checkout_import()
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from fastwam.datasets.lerobot.utils.normalizer import (
            load_dataset_stats_from_json,
        )

        configs = Path(config_root).expanduser().resolve()
        if not configs.is_dir():
            raise FileNotFoundError(f"FastWAM config root not found: {configs}")
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(configs)):
            cfg = compose(config_name=config_name, overrides=[f"task={task_config}"])

        model_cfg = OmegaConf.create(
            OmegaConf.to_container(cfg.model, resolve=True)
        )
        # These are the released-checkpoint inference semantics.  The large
        # experts are initialized structurally and then strictly overwritten.
        model_cfg.load_text_encoder = True
        model_cfg.skip_dit_load_from_pretrain = True
        model_cfg.action_dit_pretrained_path = None

        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if model_base_path is None:
            env_base = os.environ.get("DIFFSYNTH_MODEL_BASE_PATH")
            if env_base:
                resolved_model_base = Path(env_base).expanduser().resolve()
            elif checkpoint.parent.name == "fastwam_release":
                resolved_model_base = checkpoint.parent.parent
            else:
                resolved_model_base = (PROJECT_ROOT / "checkpoints").resolve()
        else:
            resolved_model_base = Path(model_base_path).expanduser().resolve()
        if not resolved_model_base.is_dir():
            raise FileNotFoundError(
                "Wan/VAE/text model base not found: "
                f"{resolved_model_base}. Pass model_base_path explicitly."
            )

        with _temporary_environment(
            "DIFFSYNTH_MODEL_BASE_PATH", str(resolved_model_base)
        ):
            model = instantiate(
                model_cfg,
                model_dtype=model_dtype,
                device=str(device),
            )
        audit = strict_load_release_checkpoint(
            model,
            checkpoint,
            compute_sha256=compute_checkpoint_sha256,
        )

        stats_path = Path(dataset_stats_path).expanduser().resolve()
        if not stats_path.is_file():
            raise FileNotFoundError(f"dataset stats not found: {stats_path}")
        processor = instantiate(cfg.data.train.processor).eval()
        processor.set_normalizer_from_stats(
            load_dataset_stats_from_json(str(stats_path))
        )

        video_cfg = OmegaConf.to_container(
            model_cfg.video_dit_config, resolve=True
        )
        config_provenance = {
            "project_root": str(PROJECT_ROOT),
            "fastwam_import": fastwam_source,
            "config_root": str(configs),
            "config_name": str(config_name),
            "task_config": str(task_config),
            "model_config_sha256": _json_sha256(
                OmegaConf.to_container(model_cfg, resolve=True)
            ),
            "video_dit_config": video_cfg,
            "dataset_stats_path": str(stats_path),
            "dataset_stats_sha256": _sha256_file(stats_path),
            "model_base_path": str(resolved_model_base),
        }
        return cls(
            model,
            processor=processor,
            checkpoint_audit=audit,
            capture_layers=capture_layers,
            config_provenance=config_provenance,
            verify_native_prefill=verify_native_prefill,
        )

    @property
    def model(self) -> torch.nn.Module:
        """Read-only access for architecture inspection; extraction re-audits it."""

        return self._model

    def _freeze_and_clear_gradients(self) -> None:
        self._model.eval()
        self._model.requires_grad_(False)
        for parameter in self._model.parameters():
            parameter.grad = None
        self.assert_frozen()

    def assert_frozen(self) -> None:
        """Fail closed if external code enabled training or backbone gradients."""

        training_modules = [
            name
            for name, module in self._model.named_modules()
            if bool(module.training)
        ]
        trainable = [
            name
            for name, parameter in self._model.named_parameters()
            if bool(parameter.requires_grad)
        ]
        gradients = [
            name
            for name, parameter in self._model.named_parameters()
            if parameter.grad is not None
        ]
        if training_modules or trainable or gradients:
            raise FrozenBackboneError(
                "frozen backbone invariant failed: "
                f"training_modules={training_modules[:10]}, "
                f"trainable={trainable[:10]}, gradients={gradients[:10]}"
            )

    def normalize_proprio(
        self, proprio_raw: torch.Tensor, *, batch_size: int
    ) -> torch.Tensor:
        """Apply the released RoboTwin processor's state normalization."""

        value = torch.as_tensor(proprio_raw, dtype=torch.float32, device="cpu")
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.ndim != 2:
            raise FrozenBackboneError(
                f"proprio_raw must be [D] or [B,D], got {tuple(value.shape)}"
            )
        if value.shape[0] == 1 and batch_size > 1:
            value = value.expand(batch_size, -1).clone()
        if value.shape[0] != batch_size:
            raise FrozenBackboneError(
                f"proprio batch {value.shape[0]} does not match images {batch_size}"
            )
        if not bool(torch.isfinite(value).all()):
            raise FrozenBackboneError("raw proprio contains NaN or infinity")
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise FrozenBackboneError(
                "released RoboTwin processor must have exactly one state key"
            )
        state_key = state_meta[0]["key"]
        processor_dim = int(self.processor.proprio_output_dim)
        raw_dim = int(state_meta[0].get("raw_shape", processor_dim))
        if int(value.shape[1]) != raw_dim:
            raise FrozenBackboneError(
                f"raw proprio must have dimension {raw_dim}, got {value.shape[1]}"
            )
        model_dim = getattr(self._model, "proprio_dim", None)
        if model_dim is None or int(model_dim) != processor_dim:
            raise FrozenBackboneError(
                "released model/processor proprio dimensions disagree: "
                f"model={model_dim}, processor={processor_dim}"
            )
        state_batch = {"state": {state_key: value}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        normalized = state_batch["state"][state_key]
        if (
            normalized.ndim != 2
            or tuple(normalized.shape) != (batch_size, processor_dim)
            or not torch.isfinite(normalized).all()
        ):
            raise FrozenBackboneError("normalized proprio is malformed or non-finite")
        return normalized

    def _prepare_proprio_condition(
        self,
        proprio_raw: torch.Tensor,
        *,
        batch_size: int,
        proprio_mode: ProprioMode = "observed",
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return observed and effective normalized proprio for one extraction.

        ``constant_zero_normalized`` deliberately intervenes after the released
        state processor and immediately before the frozen proprio encoder.  It
        therefore removes physical-state values without deleting the encoder's
        context token (whose learned bias is intentionally preserved).
        """

        mode = str(proprio_mode)
        if mode not in PROPRIO_MODES:
            raise FrozenBackboneError(
                f"proprio_mode must be one of {PROPRIO_MODES}, got {mode!r}"
            )
        if getattr(self._model, "proprio_encoder", None) is None:
            raise FrozenBackboneError(
                "released FastWAM proprio encoder must remain enabled"
            )

        observed = self.normalize_proprio(proprio_raw, batch_size=batch_size)
        if mode == "observed":
            effective = observed
        else:
            effective = torch.zeros_like(observed)

        if (
            effective.shape != observed.shape
            or effective.dtype != observed.dtype
            or effective.device != observed.device
        ):
            raise FrozenBackboneError(
                "proprio intervention changed tensor shape, dtype, or device"
            )
        all_zero = bool(torch.count_nonzero(effective).item() == 0)
        if mode == "observed" and not torch.equal(effective, observed):
            raise FrozenBackboneError("observed proprio mode changed normalized state")
        if mode == "constant_zero_normalized" and not all_zero:
            raise FrozenBackboneError("constant-zero normalized proprio is not exact zero")

        provenance = {
            "mode": mode,
            "intervention_point": "post_normalizer_pre_proprio_encoder",
            "observed_normalized_sha256": _tensor_sha256(observed),
            "effective_normalized_sha256": _tensor_sha256(effective),
            "shape": list(effective.shape),
            "dtype": str(effective.dtype),
            "all_zero": all_zero,
            "proprio_token_preserved": True,
        }
        return observed, effective, provenance

    def encode_instruction(
        self, instruction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one fixed deployment prompt for reuse across observations.

        Formal extraction repeatedly sees the same task instruction.  Caching
        its frozen text-encoder output avoids rerunning the text encoder for
        every physical timestep while preserving the exact deployment prompt.
        Proprio is deliberately appended later, once per observation batch.
        """

        self.assert_frozen()
        prompt = format_deployment_prompt(instruction)
        with torch.inference_mode():
            context, context_mask = self._model.encode_prompt(prompt)
        if context.ndim != 3 or context_mask.ndim != 2:
            raise FrozenBackboneError(
                "text encoder returned malformed context/context_mask"
            )
        if context.shape[:2] != context_mask.shape or context.shape[0] != 1:
            raise FrozenBackboneError(
                "cached task prompt must have one aligned context batch"
            )
        if not torch.isfinite(context).all():
            raise FrozenBackboneError("text encoder returned non-finite context")
        self.assert_frozen()
        return (
            context.detach().to(device="cpu").contiguous(),
            context_mask.detach().to(device="cpu", dtype=torch.bool).contiguous(),
        )

    def _prepare_context(
        self,
        *,
        batch_size: int,
        normalized_proprio: torch.Tensor,
        instruction: str | None,
        context: torch.Tensor | None,
        context_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        use_instruction = instruction is not None
        use_context = context is not None or context_mask is not None
        if use_instruction == use_context:
            raise FrozenBackboneError(
                "provide exactly one of instruction or context/context_mask"
            )
        if use_instruction:
            prompt = format_deployment_prompt(str(instruction))
            encoded, mask = self._model.encode_prompt(prompt)
            context_provenance = {
                "kind": "deployment_prompt",
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
        else:
            if context is None or context_mask is None:
                raise FrozenBackboneError(
                    "precomputed context and context_mask must be provided together"
                )
            encoded = torch.as_tensor(context)
            mask = torch.as_tensor(context_mask, dtype=torch.bool)
            if encoded.ndim == 2:
                encoded = encoded.unsqueeze(0)
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
            if encoded.ndim != 3 or mask.ndim != 2:
                raise FrozenBackboneError(
                    "context/context_mask must be [B,L,D]/[B,L]"
                )
            if encoded.shape[:2] != mask.shape:
                raise FrozenBackboneError("context and context_mask shapes disagree")
            context_provenance = {
                "kind": "precomputed_context",
                "context_sha256": _tensor_sha256(encoded),
                "context_mask_sha256": _tensor_sha256(mask),
            }
            encoded = encoded.to(
                device=self._model.device, dtype=self._model.torch_dtype
            )
            mask = mask.to(device=self._model.device, dtype=torch.bool)

        if encoded.shape[0] == 1 and batch_size > 1:
            encoded = encoded.expand(batch_size, -1, -1)
            mask = mask.expand(batch_size, -1)
        if encoded.shape[0] != batch_size or mask.shape[0] != batch_size:
            raise FrozenBackboneError(
                "text context batch does not match current observation batch"
            )
        text_length = int(encoded.shape[1])
        text_dim = int(encoded.shape[2])
        text_context = encoded
        text_mask = mask
        encoded, mask = self._model._append_proprio_to_context(  # noqa: SLF001
            context=encoded,
            context_mask=mask,
            proprio=normalized_proprio.to(
                device=self._model.device, dtype=self._model.torch_dtype
            ),
        )
        if tuple(encoded.shape) != (batch_size, text_length + 1, text_dim):
            raise FrozenBackboneError(
                "proprio conditioning must append exactly one context token"
            )
        if tuple(mask.shape) != (batch_size, text_length + 1):
            raise FrozenBackboneError(
                "proprio conditioning changed context-mask structure"
            )
        if mask.dtype != torch.bool or not bool(mask[:, -1].all()):
            raise FrozenBackboneError("appended proprio context token is not enabled")
        if not torch.equal(encoded[:, :text_length], text_context) or not torch.equal(
            mask[:, :text_length], text_mask
        ):
            raise FrozenBackboneError(
                "proprio conditioning changed the text-context prefix"
            )
        context_provenance["text_context_shape"] = [
            batch_size,
            text_length,
            text_dim,
        ]
        context_provenance["conditioned_context_shape"] = list(encoded.shape)
        context_provenance["appended_proprio_tokens"] = 1
        context_provenance["proprio_context_token_sha256"] = _tensor_sha256(
            encoded[:, -1:, :]
        )
        return encoded, mask, context_provenance

    def extract_current_observations(
        self,
        images: torch.Tensor,
        proprio_raw: torch.Tensor,
        *,
        instruction: str | None = None,
        context: torch.Tensor | None = None,
        context_mask: torch.Tensor | None = None,
        proprio_mode: ProprioMode = "observed",
    ) -> FrozenBackboneOutput:
        """Extract selected video layers from current deployment observations.

        ``images`` must be the native uint8 RoboTwin deployment composite:
        head camera at 256x320 above two 128x160 wrist cameras, resulting in
        ``[B,3,384,320]``.  No future image, action token, noisy
        future latent, or action denoising step is constructed here.
        """

        self.assert_frozen()
        image_batch = torch.as_tensor(images)
        if image_batch.ndim == 3:
            image_batch = image_batch.unsqueeze(0)
        if image_batch.ndim != 4 or tuple(image_batch.shape[1:]) != DEPLOYMENT_IMAGE_SHAPE:
            raise FrozenBackboneError(
                "images must be deployment composites [B,3,384,320], got "
                f"{tuple(image_batch.shape)}"
            )
        if image_batch.dtype != torch.uint8:
            raise FrozenBackboneError(
                f"deployment images must preserve uint8 pixels, got {image_batch.dtype}"
            )
        pixel_minimum = int(image_batch.min().item())
        pixel_maximum = int(image_batch.max().item())
        batch_size = int(image_batch.shape[0])

        with torch.inference_mode():
            (
                _observed_normalized_proprio,
                effective_normalized_proprio,
                proprio_provenance,
            ) = self._prepare_proprio_condition(
                proprio_raw,
                batch_size=batch_size,
                proprio_mode=proprio_mode,
            )
            encoded_context, encoded_mask, context_provenance = (
                self._prepare_context(
                    batch_size=batch_size,
                    normalized_proprio=effective_normalized_proprio,
                    instruction=instruction,
                    context=context,
                    context_mask=context_mask,
                )
            )
            context_provenance["proprio"] = proprio_provenance
            image_batch = image_batch.to(
                device=self._model.device, dtype=self._model.torch_dtype
            )
            # Match deploy_policy._build_robotwin_image_tensor operation order
            # exactly: uint8 -> model dtype -> scale to [-1,1].
            image_batch = image_batch * (2.0 / 255.0) - 1.0
            minimum = float(image_batch.min().item())
            maximum = float(image_batch.max().item())
            if minimum < -1.0001 or maximum > 1.0001:
                raise FrozenBackboneError(
                    f"normalized deployment image range is invalid: [{minimum},{maximum}]"
                )
            # WanVideoVAE.encode iterates over B and deterministically returns
            # the posterior mean.  T=1 is exactly infer_action's first frame.
            first_frame_latents = self._model.vae.encode(
                [image.unsqueeze(1) for image in image_batch],
                device=self._model.device,
                tiled=False,
            )
            if first_frame_latents.shape[0] != batch_size:
                raise FrozenBackboneError("VAE changed the observation batch size")

            timestep_video = torch.zeros(
                (batch_size,),
                dtype=first_frame_latents.dtype,
                device=self._model.device,
            )
            video_pre = self._model.video_expert.pre_dit(
                x=first_frame_latents,
                timestep=timestep_video,
                context=encoded_context,
                context_mask=encoded_mask,
                action=None,
                fuse_vae_embedding_in_latents=bool(
                    self._model.video_expert.fuse_vae_embedding_in_latents
                ),
            )
            video_tokens = video_pre["tokens"]
            video_sequence_length = int(video_tokens.shape[1])
            tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
            video_mask = self._model.video_expert.build_video_to_video_mask(
                video_seq_len=video_sequence_length,
                video_tokens_per_frame=tokens_per_frame,
                device=video_tokens.device,
            )
            captures, captured_cache = run_video_prefill_with_captures(
                self._model.mot,
                video_tokens=video_tokens,
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_mask,
                capture_layers=self.capture_layers,
            )
            if self.verify_native_prefill:
                native_cache = self._model.mot.prefill_video_cache(
                    video_tokens=video_tokens,
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=video_mask,
                )
                assert_kv_cache_equivalent(captured_cache, native_cache)

            cpu_tokens: dict[int, torch.Tensor] = {}
            cpu_pooled: dict[int, torch.Tensor] = {}
            for layer, tokens in captures.items():
                if tokens.ndim != 3 or tokens.shape[0] != batch_size:
                    raise FrozenBackboneError(
                        f"block {layer} returned malformed tokens {tuple(tokens.shape)}"
                    )
                if not torch.isfinite(tokens).all():
                    raise FrozenBackboneError(f"block {layer} tokens are non-finite")
                pooled = tokens.float().mean(dim=1)
                norms = pooled.norm(p=2, dim=-1)
                if not torch.isfinite(norms).all() or bool((norms <= 1e-12).any()):
                    raise FrozenBackboneError(
                        f"block {layer} mean pooling is zero or non-finite"
                    )
                pooled = F.normalize(pooled, p=2, dim=-1)
                cpu_tokens[layer] = tokens.detach().to(device="cpu").contiguous()
                cpu_pooled[layer] = pooled.detach().to(device="cpu").contiguous()

        self.assert_frozen()
        provenance = {
            "extractor": "FrozenFastWAMExtractor",
            "representation_source": "current_observation_video_prefill",
            "layer_numbering": "one_based",
            "capture_point": "updated_video_tokens_after_complete_dit_block",
            "capture_layers": list(self.capture_layers),
            "pooling": "float32_mean_over_all_current_frame_spatial_tokens_then_l2",
            "uses_future_video": False,
            "uses_action_denoising": False,
            "uses_policy_rollout": False,
            "deployment_image_layout": "head_256x320_over_left_right_128x160",
            "image_shape": list(image_batch.shape),
            "image_source_dtype": "torch.uint8",
            "image_source_range": [pixel_minimum, pixel_maximum],
            "image_input_range": [minimum, maximum],
            "vae_latent_shape": list(first_frame_latents.shape),
            "video_token_shape": list(video_tokens.shape),
            "tokens_per_frame": tokens_per_frame,
            "token_shapes": {
                str(layer): list(value.shape) for layer, value in cpu_tokens.items()
            },
            "token_dtypes": {
                str(layer): str(value.dtype) for layer, value in cpu_tokens.items()
            },
            "checkpoint": asdict(self.checkpoint_audit),
            "condition": context_provenance,
            # Backward-compatible name: this is the normalized proprio actually
            # supplied to the frozen encoder.  In the default observed mode it
            # is identical to the historical value.
            "normalized_proprio_sha256": _tensor_sha256(
                effective_normalized_proprio
            ),
            "proprio_mode": proprio_provenance["mode"],
            "native_prefill_verified": self.verify_native_prefill,
            **self.config_provenance,
        }
        return FrozenBackboneOutput(
            tokens_by_layer=cpu_tokens,
            pooled_by_layer=cpu_pooled,
            provenance=provenance,
        )


__all__ = [
    "CheckpointAudit",
    "DEFAULT_LAYERS",
    "FrozenBackboneError",
    "FrozenBackboneOutput",
    "FrozenFastWAMExtractor",
    "PROPRIO_MODES",
    "ProprioMode",
    "assert_kv_cache_equivalent",
    "format_deployment_prompt",
    "run_video_prefill_with_captures",
    "strict_load_release_checkpoint",
]
