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

## The soak falsified the gate above

That last caveat was the important one. The soak ran, and the gate as first
committed **admitted requests that then failed on allocation** - exactly what it
was built to prevent. Evidence:
[`ctxsoak-20260919-hip-v1`](../../../experiments/native-recall/ctxsoak-20260919-hip-v1).

| prompt | tokens | outcome under the first gate |
| --- | --- | --- |
| 8k | 10,417 | round 0 succeeded; identical round 1 refused, limit 48,672 -> 4,112 |
| 16k | 20,833 | **admitted, then `hipMalloc(235929600) failed`** |
| 32k | ~41,000 | **admitted, then `hipMalloc(240648192) failed`** |
| 48k | 62,433 | correctly refused |
| 64k | 83,233 | correctly refused |

### KV is 23% of what a token costs

A 10,417-token request consumed **5.44 GiB** of device memory. Its KV is 1.30
GiB. The remaining 4.14 GiB is prefill working space and allocator blocks not
returned between requests - 547.5 KiB per token in total against 128 KiB of KV,
a factor of **4.28**.

Attention scores alone scale with chunk x positions, so this overhead grows with
context. No fixed reserve can absorb a term that grows with the quantity being
gated, which is why the 2 GiB headroom failed rather than merely being too small.

The gate now sizes on total cost. The multiplier is **empirical, from a single
measurement**: 4.5, chosen to admit the largest request measured to work and
refuse the smallest measured to fail. It is not derived from the architecture.
`AKH_CANDLE_CONTEXT_COST_MULTIPLIER` overrides it, which is how a different model
or card gets a correct answer rather than this one's.

### After the fix

| case | tokens | result |
| --- | --- | --- |
| 16k prompt | 20,833 | refused cleanly, `fits_tokens: 10816`, nothing allocated |
| 8k prompt | 10,417 | admitted, completed in 285.9 s |

The serviceable context has now moved twice, both times downward, as the
measurement got more honest:

| basis | limit |
| --- | --- |
| declared window | 262,144 |
| KV-only gate | 48,672 |
| measured total cost | **10,816** |

Four percent of what the artifact advertises.

## Unresolved

- **The gate refuses a repeat of a request that just succeeded.** After the
  10,417-token run, driver-free memory reads 2.69 GiB and the limit collapses to
  913 tokens, because `hipMemGetInfo` excludes the HIP allocator's retained
  cache - memory that is reusable. `release_cached_memory` exists but its own
  contract restricts it to a serialized model-unload boundary, never between
  turns, so calling it from the gate would trade a wrong refusal for a possible
  correctness fault. This needs a real answer: either a pool-aware free figure or
  a supported reclaim point.
- **The multiplier rests on one measurement, one model, one card.** It brackets
  the two data points available and nothing more. A second model would need its
  own, and there is no mechanism that would notice if it were wrong other than
  another soak.
- **Prefill at length is the real limit before memory is.** 10,417 tokens took
  282-298 s to first token, about 35 tokens/s, against 0.86 s for a 15-token
  prompt. Whatever the memory gate permits, a context of this size is not
  currently usable interactively. That is an item-7 finding that arrived
  unbidden and is not addressed here.
- The limits are read per request against live free memory, so the figure a
  caller sees is not a constant.
