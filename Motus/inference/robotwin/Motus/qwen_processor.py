"""Qwen3-VL processor compatibility for the released Motus environment."""

import json
from pathlib import Path


def load_qwen3_vl_processor(root):
    from transformers import AddedToken, AutoImageProcessor, AutoVideoProcessor
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    directory = Path(root).resolve()
    vocab = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    config = json.loads(
        (directory / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    merges = []
    for line in (directory / "merges.txt").read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2:
            raise ValueError("Qwen merge row is not a token pair")
        merges.append(tuple(parts))
    tokenizer = Qwen2Tokenizer(
        vocab=vocab,
        merges=merges,
        unk_token=None,
        bos_token=None,
        eos_token=None,
        pad_token=None,
        add_prefix_space=config.get("add_prefix_space"),
    )
    for key, item in sorted(
        config.get("added_tokens_decoder", {}).items(),
        key=lambda pair: int(pair[0]),
    ):
        tokenizer.add_tokens(
            [AddedToken(**item)], special_tokens=bool(item.get("special"))
        )
        if tokenizer.convert_tokens_to_ids(item["content"]) != int(key):
            raise ValueError("Qwen added-token ID changed")
    for name in ("unk_token", "bos_token", "eos_token", "pad_token"):
        if config.get(name) is not None:
            setattr(tokenizer, name, config[name])
    image = AutoImageProcessor.from_pretrained(
        directory, local_files_only=True, trust_remote_code=True
    )
    video = AutoVideoProcessor.from_pretrained(
        directory, local_files_only=True, trust_remote_code=True
    )
    return Qwen3VLProcessor(
        image_processor=image,
        tokenizer=tokenizer,
        video_processor=video,
        chat_template=config.get("chat_template"),
    )
