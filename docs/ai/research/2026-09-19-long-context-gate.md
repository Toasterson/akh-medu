# The long-context gate: the runtime advertised 262,144 tokens and can serve 48,672

Date: 2026-09-19. Evidence:
[`experiments/native-recall/ctxgate-20260919-v1`](../../../experiments/native-recall/ctxgate-20260919-v1).

## What was wrong

The only admission check was against the declared window:

    if prompt_tokens.len() > self.context_length { ... }

`context_length` is read from the GGUF - 262,144 for the pinned Qwen3.8
artifact. That is what the model supports, not what this card can hold. At
131,072 bytes of KV per token, the full declared window needs 32 GiB on a card
with 23.98 GiB total, of which the resident weights already take about 16.8 GB.

So a request could be well inside the declared window, pass every check, and
then fail on an allocation part way through prefill - after the caller had
waited for it.

## What it does now

`memory_bound_context_limit()` = (free device memory - reserve) / KV bytes per
token, capped by the declared window. Measured on both devices, weights resident:

| device | memory source | free | limit | declared |
| --- | --- | --- | --- | --- |
| HIP | `hipMemGetInfo` | 8,527,020,032 | **48,672** | 262,144 |
| Vulkan | `amdgpu` sysfs `mem_info_vram` | 7,420,080,128 | **40,226** | 262,144 |

The runtime was advertising between 5.4x and 6.5x the context it can serve.

The two sources differ by about 1.1 GB because they measure different things:
`hipMemGetInfo` is the driver's view for this process, while sysfs counts
card-wide VRAM including the desktop compositor. Sysfs is the more conservative,
which is the right bias for an admission gate. Neither is wrong.

wgpu has no portable free-memory query, which is why Vulkan reads sysfs at all.
A device that can report neither yields `None` and the gate stands down rather
than inventing a limit.

## Verified in both directions

| case | exit | generation rounds | outcome |
| --- | --- | --- | --- |
| normal request, default reserve | 0 | 2 | limit 48,672, decode 20.26 steps/s |
| reserve consumes the whole budget | 1 | 0 | `ContextExceedsDeviceMemory` |

Decode on the passing path is 20.26 steps/s against 20.15 and 20.20 measured
before the change, so the gate costs nothing when it admits.

**The first verification of this was hollow and wrong.** The rejection case
appeared to pass because the run produced no generation lines - but that was the
grep pattern excluding them, and generation had completed normally with exit 0.
The gate had been wired into `generate_inner` and `prefill`, neither of which is
on the completion path: completions admit through `generate_completion_inner` via
`validate_completion_budget`. Absence of expected output is not evidence of
rejection. The check now asserts exit code and a generation-line count of zero.

There are four admission points, and the gate sits at all of them. It is taken as
one snapshot per request rather than per call site: each costs a driver query,
and a limit that shifted between two checks of the same request would be worse
than either answer alone.

## The diagnostic

Per the house rule, it tells the operator what to do rather than only what broke:
the limit that actually applies, the declared window it contradicts, the
per-token cost, the adjustable reserve, and the remedies - shorten the request,
free memory, or enable compressed KV paging.

It names compressed KV **with its caveat attached**: that it meets no agreed
quality criterion and fails the 0.9999 cosine gate at every width. A help string
recommending compression without that would quietly undo
[the finding](2026-09-19-vulkan-packed-kv.md) from the same day. The test asserts
on the rendered help text rather than the struct's fields, because the rendered
text is the thing that has to be right.

## Limits

- The 2 GiB reserve is a judgement call, overridable with
  `AKH_CANDLE_KV_HEADROOM_BYTES`. Prefill intermediates scale with prompt shape
  and no single constant is right for all of them. **It has not been calibrated**:
  no measurement here establishes that 2 GiB is enough at 40,000 tokens, only
  that the arithmetic and the refusal work.
- The limits above were read at one moment with one desktop state. Free memory
  moves; the gate reads it per request, which is correct, but means the limit a
  caller sees is not a constant.
- Nothing here was tested at a context anywhere near the limit. The largest
  prompt exercised is 15 tokens. That the gate admits correctly at 40,000 tokens
  is **unverified**, and a long-context soak is the obvious next measurement.
