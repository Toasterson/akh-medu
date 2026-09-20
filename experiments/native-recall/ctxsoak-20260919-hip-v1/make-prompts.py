#!/usr/bin/env python3
"""Regenerate the long-context soak prompts under /tmp/ctxsoak.

The prompts are a fixed sentence repeated. Measured on the pinned Qwen3.8
tokenizer: 500 blocks came to 13,017 prompt tokens, so one block is about 26
tokens and the sizes below are approximate targets, not exact. (A first
version of this file assumed 20.8 per block and overshot the 08k prompt by a
quarter; the probe's prompt_tokens field is the only exact count.)
"""
import os, sys
BLOCK = ("The resident runtime holds attention state for every token it has seen, "
         "and the cost of that state is what this measurement is about. ")
SIZES = {"05k": 200, "08k": 400, "16k": 800, "32k": 1600, "48k": 2400, "64k": 3200}
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctxsoak"
os.makedirs(out, exist_ok=True)
for name, blocks in SIZES.items():
    text = (f"<|im_start|>user\n{BLOCK * blocks}\nIn one short sentence, say hello."
            f"<|im_end|>\n<|im_start|>assistant\n")
    path = os.path.join(out, f"prompt-{name}.txt")
    open(path, "w").write(text)
    print(f"{path}: {len(text)} chars, {blocks} blocks")
