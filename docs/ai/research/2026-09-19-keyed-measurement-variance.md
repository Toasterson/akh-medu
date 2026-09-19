# What actually varied in today's keyed measurements

Seven runs of the same eight-turn sequential keyed benchmark were taken on
2026-09-19, on one machine with the exact pinned model. Pooling them explains
the anomaly that produced one retracted claim, and gives later measurements a
rule rather than an error bar.

## The data

Decode steps/s per turn, across every run that measured it:

| Turn | End context | n | Values | Spread |
| ---: | ---: | ---: | --- | ---: |
| 0 | 58 | 2 | 17.23, 17.24 | 1.00x |
| 1 | 128 | 2 | 16.95, 16.98 | 1.00x |
| 2 | 191 | 6 | 16.58 … 16.77 | 1.01x |
| 3 | 256 | 6 | **7.19**, 13.88, 14.00, 14.11, 14.18, 14.24 | 1.98x |
| 4 | 322 | 2 | 12.17, 12.30 | 1.01x |
| 5 | 381 | 2 | 11.99, 12.03 | 1.00x |
| 6 | 440 | 6 | **6.95**, 11.65, 11.86, 11.88, 11.90, 11.90 | 1.71x |
| 7 | 510 | 6 | 5.83, 7.26, 7.41, 11.51, 11.64, 11.68 | 2.00x |

## It was not random variance

The three bold outliers — 7.19 at turn 3, 6.95 at turn 6, 5.83 at turn 7 — are
**all from the same run**, the first of the day. Every other run agrees with
every other to within 1-2% at those turns. So keyed decode is not noisy; one
run was slow, in a way that is consistent within itself.

What distinguished it: it went through `scripts/session-platform/resident-probe.py`
rather than invoking the probe directly. The recorder runs the probe with
`--continuation` and a 64-token generation pair **before** the keyed benchmark,
and samples GPU sysfs once per second throughout. The benchmark therefore began
from a different resident state, after other keyed work, with a sampler running.

That is the confound. The retracted Q8 comparison set a **recorder** run against
a **direct** run and attributed the whole difference to the flag. Two variables
moved; only one was named.

## The one effect that survives

At turn 7, and only turn 7, the direct runs separate cleanly by configuration:
7.26 and 7.41 with the quantized GEMV enabled against 11.51, 11.64 and 11.68
with it disabled, three against three with no overlap. Turns 3 and 6 show no
such separation among direct runs. Why one context length is sensitive when its
neighbours are not remains unexplained, and it is now the whole of the open
question rather than a broad pattern.

## The rule this gives

Compare runs with **identical preambles**. A keyed benchmark preceded by
continuation checks is not comparable with one that starts cold, and the
difference is large enough to invent an effect that is not there. Either route
everything through the recorder or route nothing through it; do not mix, which
is what happened here.

Repetition alone would not have caught this. Repeating the recorder run would
have reproduced 7.19 consistently and made the false conclusion look robust.
What caught it was interleaving the two configurations in the same harness.

## Not established

One machine, one model, one device, one prompt sequence. The preamble
explanation is the only one consistent with all seven runs, but it was not
tested directly by rerunning the recorder path against a direct path with the
flag held fixed. That is the experiment that would confirm it.
