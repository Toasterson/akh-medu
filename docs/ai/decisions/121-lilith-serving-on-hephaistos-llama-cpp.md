# ADR 121 — Lilith is served on hephaistos with llama.cpp (Q8_0), Runpod is for training

- **Status:** Accepted (operator decision 2026-09-29; scripts and runbook on branch
  `feat/lilith-serve-hephaistos`; executed by the supervisor when hephaistos is powered on)
- **Date:** 2026-09-29
- **Relates:** [ADR 116](116-lilith-chief-of-staff-guided-rl.md) (Lilith; this refines Decision 7's
  serving placement), [ADR 113](113-full-akh-always.md) (Lilith is a workspace of the Roman core),
  [ADR 107](107-training-memory-turboquant-vs-recompute-attention.md) (TurboQuant is inference-side)
- **Anima:** WI `01a0e270-314e` (Lilith C2), Plan `01a0e26f-d032`
- **Plan sections used:** [local GPU qualification](../plans/2026-09-27-lilith-local-gpu-qualification.md)
  §2 (memory), §2a (KV), §4a (measured on this card)
- **Runbook:** [docs/ops/lilith-serve-hephaistos.md](../../ops/lilith-serve-hephaistos.md); scripts in
  `scripts/lilith/hephaistos/`

## Question

Until new hardware arrives, where and with what does Lilith (the frozen pinned parent
`google/gemma-4-12B-it@5926caa4…`, no adapter) run for interaction? She needs native OpenAI
`tool_calls` with Gemma 4's own call syntax, no reasoning leaking into `content`, a context of at
least 49152 tokens (her akh's `context_window`), quality close to bf16, and it must fit the
operator's Linux box, hephaistos (RX 7900 XTX, 24 GB, ROCm, gfx1100, 31 GiB RAM).

## Operator decision (2026-09-29)

Lilith is served for interaction on hephaistos. Runpod is used only for training from now on
(the Runpod serve pod of `docs/ops/lilith-serve.md` stays as a test surface and for training-era
checks). Hephaistos is not always on: nothing in the repo powers it on, and this ADR does not add
that.

## Decision 1 — llama.cpp `llama-server` (HIP build) with a Q8_0 GGUF of the pinned parent

| Choice | Value |
|---|---|
| Server | llama.cpp `llama-server`, tag `v0.5.0`, commit `7fe450e19305b828c199d602c23a8337aaa1f03b`, built `-DGGML_HIP=ON -DGPU_TARGETS=gfx1100` (static, no web UI, no TLS stack); Vulkan build of the same source as fallback |
| Weights | the pinned parent's `model.safetensors` (sha256 `5a84cb31…bff18d`) converted by the pinned `convert_hf_to_gguf.py` to **Q8_0**, about 12.7 GB |
| Template | the parent's own `chat_template.jinja` (sha256 `36e3a42e…ead3f0`), passed with `--jinja --chat-template-file`, never the GGUF-embedded copy |
| Tools / reasoning | native `tool_calls` (llama.cpp's `peg-gemma4` parser); `--reasoning-format deepseek` puts thoughts in `reasoning_content` |
| Context | `--parallel 2 --kv-unified-per-slot 65536`: two slots, each with 65536 tokens (the akh's window is 49152; the rest is output room and token-estimation error, as on the Runpod pod) |
| KV | f16 (default). Gemma 4's KV is small (plan §2a: 0.6 GB per sequence at 32K); `q8_0` is an env switch to measure, not a default |
| Sampling | the parent's `generation_config.json`: temperature 1.0, top-k 64, top-p 0.95, min-p 0 |
| Model name | `--alias lilith`: the akh's `model = "lilith"` does not change |

### Why llama.cpp and not vLLM on ROCm

1. **Q8_0 fits and BF16 does not.** BF16 weights are 23.8 GB and fill the 24 GB card by
   themselves (plan §2). To run vLLM on this card the weights must be quantized. RDNA3 has no
   FP8 matrix hardware, so the quantized formats vLLM offers there are 4-bit weight-only ones
   (GPTQ/AWQ class), which are not "close to bf16" the way Q8_0 is. Q8_0 at 12.7 GB plus about
   3 GB of KV for two 65536-token slots plus 1-2 GB of compute buffers leaves roughly 5 GB of
   headroom (an estimate from the plan's arithmetic; `verify.py --vram` measures it).
2. **vLLM does list the card, but nothing here proves it.** vLLM's ROCm install page names
   "Radeon RX 7900 series (gfx1100/1101)" with prebuilt ROCm 7.0/7.2.1 wheels. No vLLM run on this
   card exists in the repo. What does exist (C6b, plan §4a) is negative for the PyTorch/ROCm
   stack on gfx1100: HF bf16 at 5.19 tok/s; bitsandbytes' fast decode kernel numerically wrong
   (relative error 0.065 against 0.0017); forward nondeterminism (probe-logit drift 0.678, then
   0.364); an offload experiment that reset the GPU twice. vLLM is that same torch-ROCm stack.
3. **llama.cpp supports this exact model natively.** At `v0.5.0` the converter registers
   `Gemma4UnifiedForConditionalGeneration` (the parent's architecture, with a 2026-06 fix for
   Gemma 4 Unified conversion). `common/chat.cpp` selects the Gemma 4 format when the template
   contains `'<|tool_call>call:'` and treats a template with `{#- OpenAI Chat Completions:` as the
   current one; the pinned template contains both (checked), so no compatibility workaround is
   applied. The converter also marks `<|channel>`, `<channel|>`, `<|tool_call>`, `<tool_call|>`,
   `<|tool_response>`, `<tool_response|>` as visible tokens so the parser can read them.
4. **Auth and surface.** `--api-key-file` reads the key from a file, so it is on no command line;
   only `/health` is public (`server-http.cpp`); `--no-webui --no-slots` remove the UI and the
   slot/prompt inspection endpoint.
5. **Measured elsewhere.** The repo's own Vulkan runtime reaches 23-26 tok/s on Q4 on this card.
   llama.cpp HIP with Q8_0 is expected to be memory-bandwidth bound (960 GB/s over 12.7 GB is an
   upper bound near 75 tok/s); that is a bound, not a forecast. `verify.py` measures 8K and 32K.

### Tool calling for Gemma 4

Supported by the chosen server; no workaround is designed in. If `verify.py`'s tool checks fail on
the pinned build, the order of remedies is: (1) read the failure (it distinguishes "plain text",
"native syntax left in `content`" and "malformed arguments"); (2) test a later llama.cpp
commit with the same `verify.py`, then bump `pins.json` and this ADR; (3) serve from the Runpod
vLLM pod, whose `gemma4` parsers are proven by the C6 smoke, until (2) is done. Adding a
translation layer in the akh is rejected as larger and less reliable than pinning a fixed server.

### The converter needs a newer `transformers` than llama.cpp pins

llama.cpp's converter requirements pin `transformers==4.57.6`. That version's
`AutoTokenizer.from_pretrained` raises `AttributeError: 'list' object has no attribute 'keys'` on
the parent's `tokenizer_config.json`, whose `extra_special_tokens` is a list (a transformers 5
format; the parent was written by `5.10.0.dev0`). Found by running the pinned converter's
`--vocab-only` path on the parent's tokenizer files on 2026-09-29: 4.57.6 fails; 5.10.0 (yanked
upstream) and **5.17.0** export the vocabulary and the chat template. `pins.json` therefore pins
`converter_transformers = 5.17.0`, which `convert_parent.sh` installs over llama.cpp's pin, and the
converted GGUF's provenance records the environment's `pip freeze` hash.

## Decision 2 — A converted GGUF is provenance-bound and reusable from the share

`convert_parent.sh` verifies all seven parent files against `pins.json` (size, then sha256; a
mismatch is never converted), converts, and writes `PROVENANCE.json` (the source pins, the
converter commit and command, the output's size and sha256, the template's sha256). It publishes
to the runs share (`hephaistos-20260927/models/lilith-parent-5926caa4-q8_0-llamacpp-<commit12>/`)
with `PROVENANCE.json` last (it is the commit marker), reads the published GGUF back and verifies
it. A later run, or another machine, copies from the share instead of converting, and refuses a
file that does not match its provenance, the pinned parent, the pinned converter or the pinned
template. The GGUF itself is not bit-reproducible evidence of the parent; the hash chain from the
pinned parent to the recorded output hash is.

## Decision 3 — Plain HTTP on the LAN, only for a private IP, only with an explicit opt-in

Her base URL is `http://192.168.1.114:8080/v1`. `full_akh.py` `lilith_settings` previously required
`https`. No TLS path to hephaistos exists on the LAN: the lab gateway is plain HTTP too (the Board
and Seshat are `http://*.lab.home`), and the only mTLS path in the repo is for Postgres.
`config/lilith.json` gains `allow_insecure_lan_http` (boolean, default false). `http://` is
accepted only with it and only when the host is a literal IP address in RFC 1918, loopback or IPv6
unique-local space, never a DNS name (a name could resolve anywhere). `https` is unchanged.
akhomed needs no change: `OpenAiConfig` and `akh_config` do not restrict the scheme, and the
client is `ureq`, which speaks HTTP and HTTPS (the local Ollama backend already uses `http://`).

**Accepted risk:** the bearer key and Lilith's conversations cross the home LAN unencrypted. The
key grants only inference on this one server; it is rotated with `install_service.sh
--rotate-key`. Rejected alternatives: a TLS reverse proxy with a private CA (a new certificate
authority, trust distribution to the Mac's LaunchDaemon and a renewal path: worth doing only if
the LAN stops being trusted), and binding the server to loopback behind an SSH tunnel (a
long-lived tunnel process on the Mac in front of an always-on akh).

## Decision 4 — T3 health-checks hephaistos and never starts anything

`config/lilith.json` gains `serving` (`hephaistos` default, or `runpod`), rendered into
`config/t3-acp.json` as `lilith_serving`. For `hephaistos`, `t3_entrypoint.py` asks `/health`; when
it is down it writes one clear line to `logs/lilith-serve-start.log` (throttled to one per ten
minutes) and starts nothing: the repo has no wake or power mechanism for hephaistos, and inventing
one is out of scope. For `runpod` the previous behaviour (on-demand `lilith_serve.py start`) is
kept for training-era tests. An unknown value starts nothing.

## Decision 5 — The service is a locked-down systemd unit; the key never touches argv or the unit

`lilith-serve.service` runs as the system user `lilith-serve` (groups `render`, `video`),
`ExecStart` is an installed copy of `run_server.sh` (so the unit does not depend on a checkout),
non-secret settings are in `/etc/lilith-serve/server.env`, and the key is `/etc/lilith-serve/api-key`
(0600, owned by the service user, generated on the box through a pipe and never printed). The bind
address is the LAN address only; `0.0.0.0` is refused by `install_service.sh` and `run_server.sh`.
The GPU is shared with the operator's training experiments: the unit must be stopped before a
ROCm training run (a mixed-backend episode reset this card in September).

## Consequences

- Everyday interaction needs hephaistos powered on and the unit active. While it is off, her turns
  fail with the backend error and T3 logs one line; nothing bills.
- The Runpod serve pod (`docs/ops/lilith-serve.md`) is a test/training surface only; its budget
  limits stand.
- `full_akh.py` gains three settings (`serving`, `allow_insecure_lan_http`, a URL check);
  `t3_entrypoint.py` no longer starts a pod by default.
- Validated on the Mac on 2026-09-29, without the GPU or the real weights (evidence, not
  qualification):
  - the pinned llama.cpp `v0.5.0` builds hermetically (static, `-DLLAMA_USE_PREBUILT_UI=OFF
    -DLLAMA_OPENSSL=OFF`; the default build downloads a prebuilt web UI from Hugging Face);
  - `run_server.sh` starts it with `--jinja --chat-template-file <the parent's pinned template>`
    on the public Gemma 4 12B GGUF (`gemma4-12b.gguf`, a Q4-class file, not Lilith's), and
    `verify.py` passes every check: `tool_calls` well-formed and, streamed, in the standard OpenAI
    incremental form; a tool result round-trips; `reasoning_content` is populated and `content` is
    clean, with and without thinking; requests without the key get 401 on `/v1/*`;
  - the pinned converter turns a header-exact stand-in for the parent (the real safetensors header,
    sparse zero data) into 667 tensors and a **12.67 GB Q8_0** file (`Model architecture:
    Gemma4UnifiedForConditionalGeneration`), which is the size the memory arithmetic assumes.
- Not measured yet (hephaistos is off): the HIP build, the conversion of the real weights, tokens/s
  at 8K and 32K on the 7900 XTX, VRAM at 2 x 65536. `verify.py` records all of them on the first
  run; the numbers replace the bounds above in the runbook. Reasoning tokens count against
  `max_tokens`: the model reasons at length even without thinking mode (its thoughts are routed to
  `reasoning_content`), so a small `max_tokens` can end a turn before the answer or tool call.
- Not covered: the frozen parent only. Serving a certified adapter needs a LoRA-to-GGUF export
  (plan §4a) and a parity check against the served arithmetic; that is a later slice.

## Alternatives considered

- **vLLM on ROCm** (above): no measurement, wrong quantization options, fragile stack on this card.
- **BF16 HF/transformers on ROCm:** does not fit with KV, 5.19 tok/s measured.
- **The repo's Ptah/Candle Vulkan runtime:** proven at 23-26 tok/s on Q4, but it has no OpenAI
  server, no parser for `tool_calls`, and only a Q4 GGUF; kept as the fallback the plan named.
- **llama.cpp Vulkan (RADV) instead of HIP:** kept as `build_server.sh --backend vulkan`; HIP is the
  default because it is the maintained fast path on gfx1100.
- **Ollama:** the akh has an Ollama backend, but the requirement is to serve the parent's pinned
  template and a provenance-bound GGUF; `llama-server --chat-template-file` does that directly, with
  no second model registry to keep in step.
