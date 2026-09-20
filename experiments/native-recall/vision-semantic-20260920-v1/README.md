# Semantic reference comparison, 2026-09-20

The same weights and projector, run by two implementations, on the same
eleven images with the same instruction ("Describe this image in one
sentence."), greedy.

- **Reference:** ollama `ptah-qwen3.8:stable`, whose manifest pins model layer
  `sha256:3445102e...` and projector `sha256:add205b7...` - the exact blobs
  akhrtd loads. Started as a user process on the read-only system model
  store (`OLLAMA_MODELS=/usr/share/ollama/.ollama/models`), `think: false`,
  temperature 0, seed 0, stopped afterwards; the card was confirmed released
  (0.03 GiB) before ours ran. Its `params` layer sets temperature 1 / top_k
  20 / top_p 0.95, overridden per request. Output in
  `reference-ollama-ptah-qwen3.8-stable.jsonl`; server log in
  `ollama-serve.log`.
- **Ours:** `vision_describe_probe --device hip`, temperature 0. The seven
  geometry/placement images' outputs are this morning's
  `vision-e2e-20260919-hip-v1/placement-after-interleaved-mrope.log`; the
  four semantic images' outputs are in `reference-then-ours.log` and
  `ours-text-256.log`.

Images: the seven from the geometry set plus four built for meaning rather
than layout - rendered text `PTAH 42`, one green disc bottom-left, three
black discs in a row, a small blue triangle above a large red circle.

This is a behavioural comparison. ollama exposes no hidden states, so no
cosine is possible against it; what it establishes is whether the two
implementations reach the same reading of the same pixels.
