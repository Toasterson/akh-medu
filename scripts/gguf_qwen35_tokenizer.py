#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["gguf==0.17.1", "tokenizers==0.22.2"]
# ///
"""Build a Candle tokenizer from GGUF vocabulary, preserving qwen35 semantics.

Usage: uv run scripts/gguf_qwen35_tokenizer.py MODEL --output TOKENIZER.json
No model weights are loaded. Output files must not already exist.
The pinned reference is checked for active IDs, merges, regex, and template;
its stale tokenizer.json pre-tokenizer and NFC normalizer are NOT copied.
"""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from gguf import GGUFReader
from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

REFERENCE = (
    "https://huggingface.co/JonathanColetti/Qwen3.8-27B-Uncensored/resolve/"
    "5bb7aa90f0efef548e87005b1fb7658e522b6b7f/"
)
# qwen35 in ggml-org/llama.cpp src/llama-vocab.cpp; also pinned reference config.
QWEN35_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+"
    r"|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def fetch(name):
    with urlopen(REFERENCE + name, timeout=120) as response:
        return response.read()


def build(reader):
    def field(name):
        return reader.fields[name].contents()

    if (
        field("tokenizer.ggml.model") != "gpt2"
        or field("tokenizer.ggml.pre") != "qwen35"
    ):
        raise ValueError("Only GPT2 BPE with GGUF pre=qwen35 is supported")
    if field("tokenizer.ggml.add_bos_token"):
        raise ValueError("Automatic BOS requires a separate post-processor")
    tokens = field("tokenizer.ggml.tokens")
    types = field("tokenizer.ggml.token_type")
    if len(tokens) != len(set(tokens)) or len(tokens) != len(types):
        raise ValueError("Duplicate tokens or inconsistent token types")
    if set(types) - {1, 3, 4, 5}:
        raise ValueError("Unsupported GGUF token type")
    merges = [tuple(pair.split(" ")) for pair in field("tokenizer.ggml.merges")]
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: index for index, token in enumerate(tokens)},
            merges=merges,
            byte_fallback=False,
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex(QWEN35_REGEX), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    # CONTROL tokens are special; USER_DEFINED tokens remain visible on decode.
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind in (3, 4):
            tokenizer.add_tokens(
                [AddedToken(token, normalized=False, special=kind == 3)]
            )
        if tokenizer.token_to_id(token) != index:
            raise ValueError(f"Token ID changed at {index}")
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    reader = GGUFReader(str(args.model))
    fields = {
        key: value.contents()
        for key, value in reader.fields.items()
        if key.startswith("tokenizer.")
    }
    tokenizer = build(reader)
    reference_bytes = fetch("tokenizer.json")
    reference = json.loads(reference_bytes)
    config = json.loads(fetch("tokenizer_config.json"))
    template = fetch("chat_template.jinja")
    if config["pretokenize_regex"] != QWEN35_REGEX:
        raise ValueError("Pinned reference no longer confirms qwen35 regex")
    vocab = reference["model"]["vocab"] | {
        token["content"]: token["id"] for token in reference["added_tokens"]
    }
    tokens = fields["tokenizer.ggml.tokens"]
    types = fields["tokenizer.ggml.token_type"]
    for index, (token, kind) in enumerate(zip(tokens, types)):
        if kind != 5 and vocab.get(token) != index:
            raise ValueError(f"Reference active token mismatch at {index}")
    if len(vocab) != sum(kind != 5 for kind in types):
        raise ValueError("Reference has unexpected additional tokens")
    merges = [
        " ".join(pair) if isinstance(pair, list) else pair
        for pair in reference["model"]["merges"]
    ]
    if merges != fields["tokenizer.ggml.merges"]:
        raise ValueError("Reference merge order differs")
    if template.decode() != fields["tokenizer.chat_template"]:
        raise ValueError("Reference template differs")
    reference_tokenizer = Tokenizer.from_str(reference_bytes.decode())
    probes = [
        "Hello, world!",
        "café",
        "cafe\u0301",
        "हिन्दी",
        "العَرَبِيَّة",
        "a\u0301\u0327",
        "中文 🏺\n\t12345",
        "<|im_start|>assistant\n<think>\n",
        "<tool_call>\n<function=lookup>\n<parameter=query>\nPtah\n</parameter>\n</function>\n</tool_call>",
    ]
    samples = []
    for probe in probes:
        ids = tokenizer.encode(probe, add_special_tokens=False).ids
        if tokenizer.decode(ids, skip_special_tokens=False) != probe:
            raise ValueError(f"Lossy roundtrip for {probe!r}")
        samples.append(
            {
                "text": probe,
                "gguf_ids": ids,
                "published_json_ids": reference_tokenizer.encode(probe).ids,
            }
        )
    payload = (tokenizer.to_str(pretty=True) + "\n").encode()
    report = {
        "model": str(args.model),
        "reference": REFERENCE,
        "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(payload).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template).hexdigest(),
        "vocabulary_size": len(tokens),
        "active_tokens": len(vocab),
        "unused_tokens": types.count(5),
        "merges": len(merges),
        "control_tokens": {
            token: i for i, (token, kind) in enumerate(zip(tokens, types)) if kind == 3
        },
        "user_defined_tokens": {
            token: i for i, (token, kind) in enumerate(zip(tokens, types)) if kind == 4
        },
        "add_bos": fields["tokenizer.ggml.add_bos_token"],
        "eos_id": fields["tokenizer.ggml.eos_token_id"],
        "regex": QWEN35_REGEX,
        "normalizer": None,
        "published_json_regex": reference["pre_tokenizer"]["pretokenizers"][0][
            "pattern"
        ],
        "published_json_normalizer": reference["normalizer"],
        "samples": samples,
    }
    if args.output:
        outputs = {
            args.output: payload,
            args.output.with_suffix(".chat_template.jinja"): template,
            args.output.with_suffix(".verification.json"): (
                json.dumps(report, indent=2, ensure_ascii=False) + "\n"
            ).encode(),
        }
        if any(path.exists() for path in outputs):
            raise FileExistsError("Refusing to overwrite existing tokenizer artifacts")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for path, data in outputs.items():
            with path.open("xb") as stream:
                stream.write(data)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
