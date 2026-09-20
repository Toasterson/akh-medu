#!/usr/bin/env python3
"""Regenerate the long-context soak prompts under /tmp/ctxsoak.

The prompts are a fixed sentence repeated: one block is about 20.8 tokens on
the pinned Qwen3.8 tokenizer, so the sizes below are approximate token
targets, not exact. The originals from 2026-09-19 were built the same way;
their exact counts (10,417 for 08k) came from this tokenizer and this block.
"""
import os, sys
BLOCK = ("The resident runtime holds attention state for every token it has seen, "
         "and the cost of that state is what this measurement is about. ")
SIZES = {"05k": 250, "08k": 500, "16k": 1000, "32k": 2000, "48k": 3000, "64k": 4000}
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ctxsoak"
os.makedirs(out, exist_ok=True)
for name, blocks in SIZES.items():
    text = (f"<|im_start|>user\n{BLOCK * blocks}\nIn one short sentence, say hello."
            f"<|im_end|>\n<|im_start|>assistant\n")
    path = os.path.join(out, f"prompt-{name}.txt")
    open(path, "w").write(text)
    print(f"{path}: {len(text)} chars, {blocks} blocks")
