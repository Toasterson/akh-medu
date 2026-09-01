#!/usr/bin/env python3
"""Divergence-precursor analysis over a train_rule log tail (stdin).

Reads `step N: loss = X lr = Y (Z ms/step)` lines -- optionally with a
` gnorm = G` segment (the pre-clip global grad-norm gate) -- plus `batch
skipped` lines from stdin and prints a one-block verdict. Exit code: 0 =
OK/WARN, 2 = CRITICAL (precursor pattern strong enough to wake the operator
before the run dies).

Precursor logic (initial thresholds; refine against the literature sweep):
- variance blow-up: rolling std of the recent window vs the baseline window.
  Loss-noise inflation precedes spike cascades (edge-of-stability behavior).
- spike foreshocks: single-step loss deltas far above the recent noise floor.
- grad-norm foreshocks: the pre-clip grad norm spikes a few steps *before*
  the loss does (see `grug-train`'s `train_step_guarded`) -- only computed
  when the log carries the optional `gnorm` segment; logs without it (older
  runs, or runs predating the grad-norm gate) fall back to loss-only.
- skip pressure: the in-trainer EMA guard skipping more than a handful of
  batches per window means the guard is load-bearing, not idle insurance.
- LR context: all of the above matter more while LR is still rising (warmup).
"""
import re
import sys

STEP_RE = re.compile(
    r"^step\s+(\d+): loss = ([0-9.eE+-]+|NaN|inf) lr = ([0-9.eE+-]+)"
    r"(?: gnorm = ([0-9.eE+-]+|NaN|inf))?"
)

steps, losses, lrs, gnorms = [], [], [], []
skips = 0
for line in sys.stdin:
    m = STEP_RE.match(line.strip())
    if m:
        try:
            loss = float(m.group(2))
        except ValueError:
            loss = float("nan")
        steps.append(int(m.group(1)))
        losses.append(loss)
        lrs.append(float(m.group(3)))
        gnorm_str = m.group(4)
        if gnorm_str is None:
            gnorms.append(None)
        else:
            try:
                gnorms.append(float(gnorm_str))
            except ValueError:
                gnorms.append(float("nan"))
    elif "batch skipped" in line:
        skips += 1

if len(steps) < 30:
    print(f"PRECURSOR: warming up ({len(steps)} samples) skips_in_tail={skips}")
    sys.exit(0)


def mean(xs):
    return sum(xs) / len(xs)


def std(xs):
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2


finite = [(s, l) for s, l in zip(steps, losses) if l == l and l != float("inf")]
fl = [l for _, l in finite]
recent = fl[-10:]
baseline = fl[-40:-10] if len(fl) >= 40 else fl[:-10]

b_mean, b_std = mean(baseline), std(baseline)
r_mean, r_std = mean(recent), std(recent)
deltas = [abs(b - a) for a, b in zip(fl[:-1], fl[1:])]
d_base = sorted(deltas)[len(deltas) // 2] if deltas else 0.0
max_recent_delta = max(deltas[-10:]) if len(deltas) >= 10 else 0.0

warns, crits = [], []
if b_std > 1e-9 and r_std / b_std > 3.0:
    warns.append(f"variance x{r_std / b_std:.1f} vs baseline")
if b_std > 1e-9 and r_std / b_std > 6.0:
    crits.append(f"variance blow-up x{r_std / b_std:.1f}")
if d_base > 1e-9 and max_recent_delta / max(d_base, 1e-9) > 8.0:
    warns.append(f"foreshock delta {max_recent_delta:.3f} ({max_recent_delta / d_base:.0f}x median)")
if r_mean > b_mean * 1.5 and r_mean - b_mean > 0.2:
    warns.append(f"loss drifting up {b_mean:.3f}->{r_mean:.3f}")
if r_mean > b_mean * 3.0:
    crits.append(f"loss escaping {b_mean:.3f}->{r_mean:.3f}")
if skips >= 3:
    warns.append(f"{skips} guard skips in tail window")
if skips >= 10:
    crits.append(f"skip pressure {skips} in tail window")

# Grad-norm precursor (optional -- backward compatible with logs that predate
# the ` gnorm = X` segment, where every entry in `gnorms` is None).
raw_gnorms = [g for g in gnorms if g is not None]
g_recent_max = None
g_median = None
if raw_gnorms:
    g_tail = raw_gnorms[-40:] if len(raw_gnorms) >= 40 else raw_gnorms
    g_recent = raw_gnorms[-10:]
    g_tail_finite = [g for g in g_tail if g == g and g != float("inf")]
    g_recent_finite = [g for g in g_recent if g == g and g != float("inf")]
    any_nonfinite_recent = len(g_recent_finite) < len(g_recent)

    if g_tail_finite:
        g_median = median(g_tail_finite)
    if g_recent_finite:
        g_recent_max = max(g_recent_finite)

    if any_nonfinite_recent:
        crits.append("non-finite gnorm in tail window")
    elif g_median is not None and g_recent_max is not None and g_median > 1e-9:
        ratio = g_recent_max / g_median
        if ratio > 8.0:
            crits.append(f"gnorm blow-up x{ratio:.1f} vs tail median ({g_recent_max:.3f} vs {g_median:.3f})")
        elif ratio > 3.0:
            warns.append(f"gnorm x{ratio:.1f} vs tail median ({g_recent_max:.3f} vs {g_median:.3f})")

lr_rising = len(lrs) >= 2 and lrs[-1] > lrs[0]
phase = "warmup(LR rising)" if lr_rising else "decay"

gnorm_note = ""
if g_recent_max is not None and g_median is not None:
    gnorm_note = f" gnorm_max={g_recent_max:.3f} gnorm_median={g_median:.3f}"

print(
    f"PRECURSOR: step={steps[-1]} loss={fl[-1]:.4f} ema~{r_mean:.4f} "
    f"base_std={b_std:.4f} recent_std={r_std:.4f} skips={skips} phase={phase}{gnorm_note}"
)
if crits:
    print("CRITICAL: " + "; ".join(crits))
    sys.exit(2)
if warns:
    print("WARN: " + "; ".join(warns))
sys.exit(0)
