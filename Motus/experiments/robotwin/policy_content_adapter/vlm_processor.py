"""Load Qwen3-VL processor with exact token IDs under Transformers 5.0rc0."""

from __future__ import annotations

import json
from pathlib import Path


class VlmProcessorError(RuntimeError):
    pass


def load_qwen3_vl_processor(root: str | Path):
    """Reconstruct the released tokenizer without changing its vocabulary.

    Transformers 5.0.0rc0 ignores ``vocab_file``/``merges_file`` in the new
    Qwen2 TokenizersBackend and passes list-of-lists to tokenizers 0.22 BPE.
    Building from the same two released files with tuple merges is equivalent
    and lets us assert every added token keeps its published integer ID.
    """

    from transformers import AddedToken, AutoImageProcessor, AutoVideoProcessor
    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    directory = Path(root).resolve()
    required = [
        directory / "vocab.json",
        directory / "merges.txt",
        directory / "tokenizer_config.json",
        directory / "preprocessor_config.json",
    ]
    if not all(path.is_file() for path in required):
        raise VlmProcessorError("Qwen3-VL processor files are incomplete")
    vocab = json.loads(required[0].read_text(encoding="utf-8"))
    config = json.loads(required[2].read_text(encoding="utf-8"))
    merges = []
    for line in required[1].read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        parts = text.split()
        if len(parts) != 2:
            raise VlmProcessorError("Qwen merge row is not a token pair")
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
    decoder = config.get("added_tokens_decoder", {})
    for key, item in sorted(decoder.items(), key=lambda pair: int(pair[0])):
        tokenizer.add_tokens(
            [AddedToken(**item)], special_tokens=bool(item.get("special"))
        )
        actual = tokenizer.convert_tokens_to_ids(item["content"])
        if actual != int(key):
            raise VlmProcessorError("Qwen added-token ID changed")
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
