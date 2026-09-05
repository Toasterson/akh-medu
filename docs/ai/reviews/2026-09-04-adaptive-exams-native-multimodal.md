# GLM 5.3 and Kimi K3 review: adaptive exams and native multimodality

**Review date:** 2026-09-03/04  
**Subject:** research memo, ADR 100, and Phase 51 execution plan  
**Round 1 status:** completed independently against the same unamended draft  
**Round 2 status:** GLM 5.3 **PASS**; Kimi K3 **PASS**  
**Final implementation status:** GLM 5.3 **PASS**; Kimi K3 **PASS**

## Invocation evidence

Requested reviewers:

- `z-ai/glm-5.3` — completed, 2,091 completion tokens;
- `moonshotai/kimi-k3` — completed, 2,025 completion tokens.

The Ollama Cloud catalog advertised `glm-5.3` and `kimi-k3`, but its direct
`/api/chat` route returned HTTP 401 with the configured credential. The exact
models were therefore invoked through the already configured OpenRouter account.
No substitute model or Agent Board identity was used. This provider-path failure
also reinforces ADR 100's rule that external APIs cannot be the local OCR
runtime dependency.

## Convergent findings and dispositions

| Finding from both reviews | Disposition |
|---|---|
| A new seed from the same generator is still distribution leakage | Audit now requires a different renderer or real/human-verified data. Same-renderer seeds are rolling probes only. |
| Sibling generation had no operational near-duplicate proof | Added held-out-axis evidence, transcript n-gram Jaccard and perceptual-hash distance contracts; protected tiers cannot derive siblings. |
| EWMA point thresholds would chase noise | Added a 200-case family/window floor, Wilson intervals, two-window confirmation and a Benjamini-Hochberg family-matrix helper. OCR scoring includes deterministic case-bootstrap intervals. |
| Precursors were plausible but not proven to lead failure | First two real runs log precursors in shadow mode. A retrospective checkpoint lead-time study gates activation. |
| Fine patches can exhaust context; coarse patches can erase type | Token-budget preflight moved before V1. A stem-only resolution probe and explicit compact-conv/tiling/defer branch now gate decoder training. |
| EN/DE/FR did not justify a broad multilingual claim | Scope renamed Latin-script EN/DE/FR. Arabic and CJK are generator-independent audit probes until deliberately supported. |
| Host fonts/rasterizers break pixel replay | Bootstrap generator uses version-locked, compiled bitmap glyphs and hashes the exact glyph set. Production fonts remain content-hashed, licence-cleared artifacts. |
| Stop/resume and cumulative text regression were underspecified | `stop` retains model, optimizer, controller and evidence state; resume creates a reasoned run segment. Text regression is bounded per promotion and cumulatively. |

## Reviewer-specific emphasis retained

GLM 5.3 additionally requested replaying a frozen sample of old frontier items
to detect a controller chasing its own moving distribution. That becomes a
required rolling-probe family before adaptive training is enabled.

Kimi K3 additionally requested a deliberately planted leakage canary and a
stem-only small-type experiment that isolates representation from decoder
capacity. Both are pre-350M kill tests in the plan.

## Items that remain empirical, not paper-complete

The implementation supplies the controller, generator/scorer, token-budget
preflight, native patch stem, shared-decoder forward path and optimization
boundary. It does not manufacture evidence for experiments that have not run.
The following remain hard gates before a 350M multimodal allocation:

- precursor lead-time replay across saved checkpoints;
- controller false-steer simulation at the chosen exam sizes;
- stem-only glyph resolution and patch/tiling ablation;
- different-renderer plus real-photo audit corpus;
- planted-leakage detection;
- equal-text-token modality-interference runs.

No threshold may be loosened merely to advance a rung.

## Round 2 verification

Both exact models reviewed the revised ADR, plan, controller, scorer/audit,
native-vision implementation, CPU smoke result and token-budget result. Both
returned `VERDICT: PASS` with no unresolved blockers. They specifically agreed
that the capability-red smoke is honestly gated and that inventing an unrun
large-training result would be incorrect.

Invocation evidence:

- `z-ai/glm-5.3`: 19,819 prompt tokens, 1,251 completion tokens;
- `moonshotai/kimi-k3`: 19,873 prompt tokens, 1,306 completion tokens.

One Kimi observation was hardened immediately: trend intervals, EWMA and
confirmation streaks are now stored per capability **and per exam tier**, and
only frontier observations change frontier difficulty. A test proves sentinel
and frontier observations cannot contaminate each other's histories.

Remaining non-blocking observations are retained for the empirical stages:

- calibrate the current 0.80 transcript-overlap threshold on real sibling data;
- broaden held-out-axis automation beyond content-bank and renderer metadata;
- add run-segment/retention automation before precursor steering is enabled;
- consider wider image-neighbourhood distance checks after the bootstrap
  generator, whose exact-pixel overlap check already remains unconditional.

GLM suggested replacing adjacent-pair equality for metamorphic groups with
all-pairs equality. No change was made: requiring every adjacent pair in a
sequence to be equal is transitively equivalent to requiring all values to be
equal, and the existing test covers an inconsistent group.

## Final implementation audit and amendment

Both exact models received the same 274,278-byte implementation packet:

- SHA-256: `af943765442c18ef1578d663b22be05eb90f80858dbed2b55cb5763ee6008d93`;
- scope: final ADRs, plan, runbook, controller, scorer/audit, native vision
  implementation, A40 result, checkpoint metadata, offline replay and proposed
  Hugging Face packaging.

The first high-reasoning calls exhausted their completion windows without a
machine-usable verdict. A verdict-first retry produced:

- `z-ai/glm-5.3`: **PASS**, 65,606 prompt / 2,167 completion tokens;
- `moonshotai/kimi-k3`: **BLOCK**, 65,316 prompt / 2,209 completion tokens.

Kimi found that sentinel seed/fingerprint excluded checkpoint identity but
still consumed the mutable frontier difficulty. An intervening frontier
observation could therefore change the supposedly frozen sentinel case set.
That was a real blocker; publication remained held.

The amendment introduced a per-family `sentinel_difficulty`, initialized from
the registered initial difficulty and never mutated. Sentinel plans use it for
both seed construction and the emitted plan. Legacy state without the field
falls back to `CapabilitySpec::initial_difficulty`, never the already-adapted
frontier difficulty. A regression test creates a sentinel plan, advances the
frontier from difficulty 2 to 3, and proves the later sentinel difficulty,
seed and fingerprint are unchanged. Related hardening lowered the catastrophic
outcome stop default from 50% to 5%, made pass/catastrophic counts disjoint, and
made disabled precursor signals neutral to family priority.

Both exact reviewers then received the same amendment packet:

- SHA-256: `bccce588b032129e684dfa0f73739f658f2989b7a04582fb73f7192f65b024bd`;
- `z-ai/glm-5.3`: **PASS**, 1,443 prompt / 1,188 completion tokens;
- `moonshotai/kimi-k3`: **PASS**, 1,553 prompt / 885 completion tokens.

Both independently concluded that the sentinel blocker was closed at its
source and that the amendment introduced no new blocker. Their remaining
comments are non-blocking operational follow-ups: make legacy-state re-pinning
more visible, eventually replace the compatibility `Option<u8>` with a schema
bump, document the 5% catastrophic threshold calibration, and expand the
future independent-renderer audit. None changes the current empirical boundary:
the published checkpoint is still only a clean English training-overfit path
proof, while unseen OCR, multilingual/layout robustness, text retention and
precursor lead time remain gated.
