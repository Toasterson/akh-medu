# Long residency soak, keyed HIP

Source: `aed3b2d0`. Local RX 7900 XTX only; no shared service or remote host was
touched. The probe exited normally and passed its resident-idle gate.

## Run

160 keyed requests over 14m42s with the model owned throughout: one session key,
four turns, forty repeats from empty histories, 48-token cap. Every repeat issues
the same four prompt lengths in the same order, so any change across repeats is
drift in the resident process rather than a longer prompt.

```sh
python3 scripts/session-platform/resident-probe.py --device hip \
  --model <pinned Q4_K_M blob> --tokenizer <pinned tokenizer> \
  --binary target/release/examples/candle_session_probe \
  --output experiments/native-recall/soak-20260919-hip-long-v1 --timeout 2700 \
  --probe-arg=--keyed-benchmark --probe-arg=--keyed-mode --probe-arg=sequential \
  --probe-arg=--keyed-sessions --probe-arg=1 --probe-arg=--keyed-turns --probe-arg=4 \
  --probe-arg=--keyed-repeats --probe-arg=40 --probe-arg=--keyed-tokens --probe-arg=48
```

## Result: no unbounded drift

Every fourth repeat, decode steps/s per turn index:

| Turn (prompt tokens) | r0 | r4 | r12 | r20 | r28 | r36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 (23) | 17.2 | 17.1 | 17.0 | 17.1 | 16.9 | 17.1 |
| 1 (81) | 16.9 | 16.9 | 16.8 | 16.9 | 16.7 | 16.8 |
| 2 (154) | 16.5 | 16.6 | 16.5 | 16.6 | 16.4 | 16.6 |
| 3 (214) | 8.1 | 11.3 | 11.6 | 11.5 | 11.8 | 11.5 |

Decode is flat for 36 repeats after the first few. Turn 3 stays in the slow band
described in the [keyed measurement](2026-09-19-keyed-session-measurement.md);
that is the context-position effect, and it does not worsen with residency.

Process and device memory were flat for the whole run:

| Window | GPU busy | Power | Peak VRAM | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| 120–240 s | 93.5 % | 280.0 W | 24.46 GB | 0.600 GB |
| 480–600 s | 93.3 % | 279.4 W | 24.35 GB | 0.600 GB |
| 600–720 s | 94.1 % | 280.7 W | 24.49 GB | 0.600 GB |

RSS did not move off 0.600 GB after load. Device VRAM stayed within 24.35–24.49
GB with no trend. The resident-idle gate then passed at 0 % busy and 10.72 W
against an 11.0 W unloaded baseline, model still owned.

## Correction: the earlier "2.3x prefill drift" is warm-up

The six-repeat soak recorded earlier today showed time to first token growing
0.80 to 1.83 s for a fixed 23-token prompt and called it drift to pursue. With
forty repeats the shape is clear: it rises once over the first few repeats and
then holds flat.

| Repeat | 0 | 4 | 8 | 12 | 20 | 28 | 36 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Turn 0 time to first token, s | 0.80 | 1.88 | 1.91 | 1.94 | 1.92 | 1.92 | 1.92 |

That is a one-time warm-up cost of roughly 1.1 s, not accumulating growth. The
six-repeat run simply ended inside the transition. The same shape appears on
every turn index.

## Not established

Fifteen minutes and 160 requests on one key. This is not a multi-hour soak, not
a multi-session residency test, and not a daemon-level test: it exercises the
backend directly. VRAM figures are device-wide sysfs telemetry, not per-process
attribution, and RSS is sampled once per second rather than a high-water mark.
The unexplained context-position slowdown on HIP remains open and belongs to the
performance step.
