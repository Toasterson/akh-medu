# The HIP keyed decode anomaly is the Q8 activation GEMV

Source `9b834ca9` plus per-step instrumentation. Local RX 7900 XTX, HIP, exact
pinned Ptah model, one key, eight consecutive keyed turns, identical prompts
across both configurations.

Earlier measurement found keyed HIP decode non-monotonic in context position:
about 17 steps/s below 200 positions, 8.2 at 256, 13.0 at 322-381, then 6.9 and
5.8 at 440 and 510. Vulkan declined smoothly over the same sequence. The
mechanism was unidentified and the working hypothesis was a periodic expensive
step, such as a reallocation as the cache grows.

## Per-step timing rules out a periodic spike

`CandleCompletion` now carries `step_seconds`, one duration per sampled
position, measured at the sampling point because that reads the result back and
therefore includes the step's queued GPU work. The aggregate rate cannot
separate a periodic expensive step from a uniformly slower kernel; this can.

| End context | Rate, steps/s | Median step | Max/median | Steps over 2x median |
| ---: | ---: | ---: | ---: | ---: |
| 58 | 17.24 | 0.0577 s | 1.1x | 0 of 35 |
| 128 | 16.98 | 0.0586 s | 1.1x | 0 of 47 |
| 191 | 16.58 | 0.0592 s | 1.4x | 0 of 37 |
| 256 | 7.19 | 0.1644 s | 1.2x | 0 of 42 |
| 322 | 12.17 | 0.0810 s | 1.2x | 0 of 42 |
| 381 | 11.99 | 0.0837 s | 1.1x | 0 of 35 |
| 440 | 6.95 | 0.1731 s | 1.1x | 0 of 34 |
| 510 | 5.83 | 0.1705 s | 1.1x | 0 of 47 |

**No turn contains a single outlier step.** The worst step is at most 1.4x the
median and not one step anywhere exceeds twice it. The slowdown is uniform
across every step of an affected turn: the median itself moves, from 0.059 s to
0.164 s. A reallocation, a cache miss or any other periodic event is excluded.

## Correction: the single-run comparison was mostly noise

An earlier version of this document claimed, from one run per configuration,
that disabling the Q8 GEMV removed the anomaly at 256, 440 and 510 positions.
**That claim was wrong**, and reading the code should have prevented it:
`candle_backend.rs` already wraps every keyed completion in
`with_full_precision_activations`, which forces `q8_gemv_enabled()` false
regardless of the environment variable. The flag cannot affect most of the keyed
path, so a large apparent effect there should have been treated as suspect
rather than published.

Four interleaved runs, two per configuration, alternating so drift cannot favour
either:

| End context | Q8 on, pass 1 | Q8 off, pass 1 | Q8 on, pass 2 | Q8 off, pass 2 |
| ---: | ---: | ---: | ---: | ---: |
| 191 | 16.77 | 16.76 | 16.74 | 16.74 |
| 256 | 14.00 | 14.11 | 13.88 | 14.18 |
| 440 | 11.90 | 11.65 | 11.86 | 11.90 |
| 510 | **7.26** | **11.64** | **7.41** | **11.51** |

At 191, 256 and 440 positions the two configurations are **indistinguishable**,
within 2%. The earlier reading of 7.19 against 14.24 at 256 positions was
run-to-run variance, not an effect of the flag.

At 510 positions the difference **is** real and reproduces in both passes: about
7.3 with the quantized route against about 11.6 without it, a consistent 1.6x.
Why one context length is sensitive when the others are not is unexplained, and
that is now the narrow open question rather than the broad one.

Two further consequences. The morning's wild non-monotonic profile — 8.2 at 256
positions rising to 13.0 at 322 — **did not reproduce** in any of these four
runs, which all show a nearly monotonic decline. So keyed decode at larger
contexts is **unstable between processes**, and any single-run comparison of it,
including several recorded earlier today, cannot support a causal claim.

The per-step evidence below stands, because it is within-run structure rather
than a between-run comparison: no turn contains an outlier step, so whatever
varies moves the whole distribution rather than adding a periodic spike.

## Superseded: the original single-run comparison

Same eight turns with `AKH_HIP_Q8_GEMV=0`, which routes activations through f32
instead of the quantized GEMV:

| End context | Q8 GEMV on (default) | Q8 GEMV off |
| ---: | ---: | ---: |
| 58 | 17.24 | 17.23 |
| 128 | 16.98 | 16.95 |
| 191 | 16.58 | 16.71 |
| 256 | **7.19** | **14.24** |
| 322 | 12.17 | 12.30 |
| 381 | 11.99 | 12.03 |
| 440 | **6.95** | **11.88** |
| 510 | **5.83** | **11.68** |

With the Q8 GEMV disabled the curve is **smooth and monotonic**, 17.2 falling
gently to 11.7 across 58 to 510 positions, which is the shape Vulkan already
had. Every anomalous collapse disappears. At the affected lengths f32 is close
to **twice as fast** as the quantized path, and at no measured length is it
slower.

So the cause is the Q8 activation GEMV kernel degrading badly at particular
context lengths, not context growth, not memory pressure, not the scoped-f32
continuation guard, and not a periodic event.

## Consequence

`AKH_HIP_Q8_GEMV=1` is a **pessimization for keyed decode** on this model and
device. The probe recorder sets it explicitly, so every keyed throughput figure
recorded today was measured on the slower path.

## Where the Q8 GEMV does help: unkeyed decode

The same comparison on the unkeyed path, one 392-token prompt, two fresh calls:

| Measurement | Q8 GEMV on | Q8 GEMV off |
| --- | ---: | ---: |
| Time to first token | 3.913 s / 3.793 s | 3.911 s / 3.790 s |
| Unkeyed decode, steps/s | 18.37 / 18.34 | 16.02 / 14.69 |

Two things follow. **Prefill is unaffected** — the times agree to a millisecond,
so the kernel plays no part there. And **unkeyed decode is meaningfully faster
with it on**, by roughly 15 to 25 %, the opposite of its effect on keyed decode.

Both Q8 rounds returned byte-identical text with identical step counts, so the
quantized path is deterministic here. An earlier reading of a truncated probe
output as a determinism failure was wrong; it was an artifact of the reporting
pipeline, and this clean rerun supersedes it.

## Consequence, revised

The unkeyed measurements stand, since they compare the path the flag actually
governs: prefill is unaffected, and unkeyed decode is 15-25% faster with the
quantized GEMV on. Keeping it enabled for fresh generation is supported.

For keyed decode, no change is justified. The flag is already neutralized there
by the full-precision scope at every context measured except 510 positions, and
that single point needs explaining before anything is built on it.

What this leaves is a real but much narrower open question, and a methodological
one: keyed decode at large context varies enough between processes that
measurements need repetition and interleaving. Every keyed comparison recorded
earlier today was a single run per configuration.

## Not established

Why the kernel degrades at those sizes. The pattern is not a simple threshold:
256, 440 and 510 are slow while 322 and 381 are fast, which suggests a tiling or
alignment interaction with the key/value length rather than a size cut-off, but
nothing here inspects the kernel.

Whether a per-path selection is safe. The keyed route already carries a
scoped-f32 continuation guard for numerical reasons, so routing keyed
completions away from the quantized GEMV is consistent with that guard rather
than in tension with it, but no numerical gate has been rerun under a per-path
split.

The measurements are also narrow: one model, one device, one prompt shape per
path, and unkeyed decode measured over 26 steps. The unkeyed advantage is a
consistent 15-25 % across four samples, not a single reading, but it is not a
sweep across shapes.
