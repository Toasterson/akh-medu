# C5a — role fit of the harvested transcripts for Lilith

- **Date:** 2026-09-27
- **Status:** Complete (analysis). Anima WI `01a0e2a0-4488` (Lilith C5a).
- **Related:** [ADR 116](../decisions/116-lilith-chief-of-staff-guided-rl.md) Decision 6,
  [exam and evaluation workflow](../plans/2026-09-27-lilith-exam-and-evaluation-workflow.md) §3 (judging panel)
- **Private data:** the per-record ratings, the adjudications and `final.jsonl` are on NFS
  `akh-medu-runs/hephaistos-20260927/lilith/c5a/` and in `~/lilith-private/c5a/` on the Mac. This
  document holds aggregates only, no transcript text.

## Question

Operator: *"I don't know if the harvested transcripts have good input for the role. They need to be
evaluated, but especially for illumos she should be able to guide the implementors and the PM and PO
to do the right work."*

## Corpus

`admitted-v2.jsonl`: 559 prompt→reply records from 12 Claude Code sessions.
- Schema `akh/lilith-transcript-admission@1`; all `supervised_imitation`; 0 verified outcomes.
- Trajectories are empty: no tool calls or results survive the export.
- 60% of the records come from three sessions.

## Method (first use of the judging panel)

- An anchored 0–3 rubric covering: CoS value and class, illumos guidance, PM/PO steering, grounding,
  self-containedness, operator correction, privacy, and verdict.
- Six batches, each rated by two independent raters:

  | Batches | Raters |
  |---|---|
  | b1–b2 | Fable 5.1 + Opus 5.5 |
  | b3–b4 | Opus 5.5 + Codex Astra (gpt-6-astra) |
  | b5–b6 | Codex Astra + Fable 5.1 |

- The third rater adjudicated every disagreement on the reliable dimension (below). Cohen's κ is
  reported per dimension.
- **Reliability rule** (workflow §3): a dimension with κ < 0.6 does not count.

| Dimension | κ by batch (b1…b6) | Counts? |
|---|---|---|
| CoS-grade (`cos` ≥ 2, binary) | 0.63 · 0.78 · 0.67 · 0.20 · 0.83 · 0.62 | **Yes.** b4's 37 splits were adjudicated |
| Grounding | 0.23 · 0.01 · 0.02 · 0.24 · 0.09 · −0.02 | **No.** Not judgeable from pairs |
| Overall keep/drop verdict | 0.12 · 0.24 · 0.43 · 0.17 · 0.40 · 0.26 | **No.** It depends on grounding |
| PM/PO steering | 0.07–0.66 | Reported only |

Rater calibration: Opus was the strictest. On disagreements, the adjudicator sided with Astra 41
times, Fable 32 times and Opus 18 times.

## Findings

1. **193 of 559 records (35%) are Chief-of-Staff-grade** by panel decision: 136 by agreement and 57
   by adjudication. By class:

   | Class | Records |
   |---|---|
   | Run supervision | 69 |
   | Briefing | 41 |
   | Chase / coordination | 33 |
   | Triage | 28 |
   | Decision prep | 16 |
   | Commitment | 5 |
   | Implementation | 1 |

2. **Zero illumos content.** No rater scored illumos above 0 on any of the 559 records. The
   harvested material cannot teach illumos guidance of implementers, PM or PO at all.
3. **PM/PO steering is thin:** 67 records reach a majority score of 2 or more.
4. **Grounding cannot be judged from these pairs.** The export keeps only the prompt and the reply.
   The tool results a status claim rests on were earlier in the session and are gone. The strongest
   briefings were the ones the strict raters dropped. This is a data-format defect, not a behaviour
   defect, and it matters because receipts (ADR 116 D4 provenance) are Lilith's hard constraint.
5. **Concentrated.** Two sessions supply 127 of the 193 CoS-grade records (91 and 36), and the
   dominant class is run supervision of BabyLLM training. Trained as-is, Lilith would imitate one
   supervisor voice on one kind of job.
6. **Privacy.** 96 of the 193 need redaction, mostly usernames in tmp paths and LAN IPs.
7. **Operator corrections.** 10 CoS-grade records contain one. These are the most valuable input for
   corrected siblings.
8. **Split skew.** The "90/10" session split came out 359 / 200 because one 200-record session fell
   into validation. It has to be re-split for CoS use.

## Decision for the CoS corpus (C5)

- **Use the 193 as a candidate pool only.** Do not admit them in their current form.
- **Re-export those sessions with trajectories.** Pull the tool-call and tool-result events from the
  transcript harvest store (ADR 119) so that each reply carries its receipts. Grounding then becomes
  a deterministic receipt check, and the panel stops judging it.
- **Redact** usernames, paths and LAN IPs, and cap any one session at ≤ 25% of the admitted CoS set.
- **Rubric v2:** drop judged grounding; keep the judged CoS class and value. Add "operator correction
  handled" as a first-class label for corrected siblings.
- **Fill the gaps from other sources, not by relabelling:**
  - illumos guidance and PM/PO steering: teacher dojo trajectories on real illumos WIs (C4), work
    grounded in man pages and source, and Lilith's live work over WireGuard (C6a);
  - commitments: the C3 recorder plus live work.
- The **operator's own Claude Code / Codex sessions** remain the best source of corrected CoS
  behaviour. The operator authorized evaluation by Astra, Fable 5.1 and Opus 5.5; admission of new
  sessions still goes through this panel.
