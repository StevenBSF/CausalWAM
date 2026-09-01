import json
from pathlib import Path
from experiments.robotwin.policy_content_adapter.vlm_processor import (
    load_qwen3_vl_processor,
)

VLM = Path("/mnt/cpfs-E/baoshifeng/Motus/pretrained_models/Qwen3-VL-2B-Instruct")


def test_released_qwen_processor_preserves_every_added_token_id():
    processor = load_qwen3_vl_processor(VLM)
    config = json.loads((VLM / "tokenizer_config.json").read_text())
    for key, item in config["added_tokens_decoder"].items():
        assert processor.tokenizer.convert_tokens_to_ids(item["content"]) == int(key)
    rendered = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        tokenize=False,
    )
    assert "<|im_start|>user" in rendered and "hello" in rendered
