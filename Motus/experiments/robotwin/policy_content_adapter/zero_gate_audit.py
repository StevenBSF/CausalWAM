"""Full Motus inference equivalence gate before adapter training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import MotusPolicyContentConditioner
from .observation_content import extract_observation_visual_tokens
from .official_data import MotusOfficialDataset
from .paired_data import sha256_file
from .runtime import instantiate_author_release, load_lineage
from .task_text_cache import load_task_embeddings, validate_task_text_cache
from .vlm_processor import load_qwen3_vl_processor


AUDIT_SCHEMA = "motus_policy_content_adapter_zero_gate_audit"


def run_zero_gate_audit(
    *,
    lineage_path: str | Path,
    official_manifest_path: str | Path,
    task_text_cache_dir: str | Path,
    output_path: str | Path,
    local_cuda_index: int,
    inference_seed: int = 123,
) -> dict:
    from data.utils.image_utils import tensor_to_pil
    from utils.vlm_utils import preprocess_vlm_messages

    lineage_path = Path(lineage_path).resolve()
    official_manifest_path = Path(official_manifest_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    lineage = load_lineage(lineage_path, verify_files=True)
    validate_task_text_cache(task_text_cache_dir, verify_encoder_assets=False)
    embeddings = load_task_embeddings(task_text_cache_dir)
    dataset = MotusOfficialDataset(
        official_manifest_path,
        task_embeddings=embeddings,
        training_seed=1,
    )
    sample = dataset[0]
    processor = load_qwen3_vl_processor(lineage["vlm"]["root"])
    vlm_inputs = preprocess_vlm_messages(
        sample["text_instruction"],
        tensor_to_pil(sample["first_frame"]),
        processor,
    )
    model = instantiate_author_release(
        lineage,
        batch_size=1,
        local_cuda_index=local_cuda_index,
        strict=True,
    )
    model.eval()
    first_frame = sample["first_frame"].unsqueeze(0)
    state = sample["initial_state"].unsqueeze(0)
    language = [sample["language_embedding"]]
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed(inference_seed)
    with torch.no_grad():
        native_video, native_action = model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=10,
            language_embeddings=language,
            vlm_inputs=[vlm_inputs],
        )
    conditioner = MotusPolicyContentConditioner().to(
        device=model.device, dtype=model.dtype
    )
    if conditioner.adapter.gate.detach().item() != 0.0:
        raise RuntimeError("zero-gate audit conditioner is not exactly zero")
    model.set_policy_content_conditioner(conditioner)
    visual = extract_observation_visual_tokens(
        model,
        first_frame=first_frame,
        language_embeddings=language,
        capture_layer=conditioner.capture_layer,
    )
    content = conditioner.content_tokens(visual)
    torch.manual_seed(inference_seed)
    torch.cuda.manual_seed(inference_seed)
    with torch.no_grad():
        adapter_video, adapter_action = model.inference_step(
            first_frame=first_frame,
            state=state,
            num_inference_steps=10,
            language_embeddings=language,
            vlm_inputs=[vlm_inputs],
            policy_content_tokens=content,
        )
    action_equal = torch.equal(native_action, adapter_action)
    video_equal = torch.equal(native_video, adapter_video)
    if not action_equal or not video_equal:
        raise RuntimeError(
            "zero-gate Motus output is not bit-exact: "
            f"action={action_equal}, video={video_equal}"
        )
    audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": 1,
        "status": "PASS",
        "lineage_manifest": {
            "path": str(lineage_path),
            "size_bytes": lineage_path.stat().st_size,
            "sha256": sha256_file(lineage_path),
        },
        "official_manifest": {
            "path": str(official_manifest_path),
            "size_bytes": official_manifest_path.stat().st_size,
            "sha256": sha256_file(official_manifest_path),
        },
        "sample": {
            "task": sample["task"],
            "domain": sample["domain"],
            "episode_index": sample["episode_index"],
            "condition_frame_index": sample["condition_frame_index"],
        },
        "inference_seed": inference_seed,
        "inference_steps": 10,
        "capture_layer": conditioner.capture_layer,
        "visual_token_shape": list(visual.shape),
        "content_token_shape": list(content.shape),
        "action_shape": list(native_action.shape),
        "video_shape": list(native_video.shape),
        "action_bit_exact": True,
        "video_bit_exact": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--official-manifest", required=True)
    parser.add_argument("--task-text-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-cuda-index", type=int, default=0)
    parser.add_argument("--inference-seed", type=int, default=123)
    args = parser.parse_args()
    result = run_zero_gate_audit(
        lineage_path=args.lineage,
        official_manifest_path=args.official_manifest,
        task_text_cache_dir=args.task_text_cache,
        output_path=args.output,
        local_cuda_index=args.local_cuda_index,
        inference_seed=args.inference_seed,
    )
    print(json.dumps({"status": result["status"], "output": str(Path(args.output).resolve()), "action_bit_exact": result["action_bit_exact"]}, sort_keys=True))


if __name__ == "__main__":
    main()
