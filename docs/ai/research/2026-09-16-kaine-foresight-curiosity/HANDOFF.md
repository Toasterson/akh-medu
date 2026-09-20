# Kaine: theory-based foresight and compression-progress curiosity

Prepared for Fable on 2026-09-16 at the user's request. This is a research harvest and handoff, not an implementation or an approved architecture change.

## Start here

Read the verbatim conversations in `transcripts/`, then the full paper texts in `papers/`. `context/` contains snapshots of ADR 110, ADR 111, and the P7 plan as harvested from the working tree. Seshat document identifiers and persistence evidence are in `seshat/`. The local root is `/Users/toasty/ws/akh-medu/docs/ai/research/2026-09-16-kaine-foresight-curiosity`.

The original t3code thread is `7e5cfbbd-084f-4cf8-96ab-16fcf9f39733`, titled **Theory-Based Foresight for Kaine**. Claude's session is `bfb0ce41-a104-407c-a0cc-6f5c801cc003`. The follow-up Codex session is `01a0ab7f-6f83-7ea2-8cba-8c938422c48c`. The thread was read without changing its snooze or session state.

## What the user wants

Bring forward-looking, theory-based thinking into Kaine. The user then supplied Schmidhuber's compression-progress paper as a reply to the tweet discussing the Oxford paper. Preserve both discussions and papers in Seshat and a local folder tree so the user can continue with Fable after exhausting Claude usage.

Tweet: https://x.com/perksverse/status/2099822502126219588

Oxford paper: Teppo Felin and Matthias Holweg, *Theory Is All You Need: AI, Human Cognition, and Causal Reasoning*, Strategy Science 9(4), 346–371 (2024), https://doi.org/10.1287/stsc.2024.0189.

Compression-progress paper: Juergen Schmidhuber, *Driven by Compression Progress: A Simple Principle Explains Essential Aspects of Subjective Beauty, Novelty, Surprise, Interestingness, Attention, Curiosity, Creativity, Art, Science, Music, Jokes*, arXiv:0812.4360 (2008, revised 2009). The archived complete PDF is the 35-page Dagstuhl 2009 edition, https://doi.org/10.4230/DagSemProc.09291.14, licensed CC BY 4.0. Do not silently equate edition bytes.

## Claude's proposal, not implemented

1. A `Theory` datatype connecting a problem, causal claims, evidence, and a discriminating experiment.
2. A `foresight` component scoring predictions committed before an action and checked after the action.
3. Theory-first training episodes, with corrected siblings for retrieval-only attempts where an experiment is needed.
4. Choosing experiments using reduction of uncertainty around live hypotheses.
5. Exposing the hypothesis explorer (called Phase 29f in the prior conversation).

Claude proposed placing this in ADR 110 / P7 corpus work rather than changing ADR 111's architecture ladder, and mentioned a possible P7.14 draft gated on P6. These are historical suggestions, not reserved item numbers or current tracker state. Claude explicitly said it wrote nothing to disk. Its original paper retrieval failed with HTTP 403; the original analysis used the abstract and prior knowledge. Full paper text was recovered during this harvest.

## Codex's additional proposal

Keep distinct signals for foresight (did the predicted effect occur?) and learning progress (did the experience improve understanding?). The combined experimental loop is:

`conjecture -> predict -> intervene -> observe -> revise -> select the next experiment`

A candidate intrinsic signal is `L_before(D) - L_after(D)`. Both predictors must be evaluated on the same data. A held-out generalization check is an additional proposed safeguard against memorization, not a claim that the paper's exact algorithm used it. Reward expected learnable structure rather than raw surprise. Compare familiar tasks, unfamiliar but learnable tasks, and irreducibly random tasks under matched budgets; measure later verified task performance as well as exploration choices.

ADR 110 currently describes weighted supervised learning on corrected targets and restricted sibling preference pairs, not unrestricted scalar-judge policy-gradient training. A first experiment could apply learning progress to curriculum or experiment selection while preserving those constraints. Adopting Schmidhuber's full RL controller would require a separate decision. Authority, action validity, provenance, budgets, and lifecycle gates remain engine responsibilities.

## Review questions and corrections for Fable

- Treat the tweet's universal impossibility claim and the reply's claim that one specific theory is necessary for creativity as claims to evaluate, not established results.
- The papers can motivate an engineering synthesis without agreeing philosophically: a computational curiosity controller is not itself proof that the authors' objection about causal theory formation is solved.
- Recheck all claimed Rust seams and implementation status. The harvest does not certify that the old Claude code descriptions are current or sufficient.
- A Dempster–Shafer belief/plausibility interval is not a calibrated probability distribution. Interval width is not automatically expected information gain or an executable experiment-selection score.
- Do not reward contrarianism or low prior probability alone. Use committed predictions, falsifiable alternatives, reproducible evidence, cost controls, and held-out performance. A disproved hypothesis can still yield learning progress.
- The historical statements about flight, Pearl's rungs, and the separate “Beyond Bayes” chapter need primary-source checking before reuse in an ADR.
- Preserve the distinction between theory records in engine memory and the model learning a procedure for proposing and testing theories.

## Fidelity

Conversation message strings are preserved exactly in JSON and included unchanged under role headings in Markdown, including original caveats, links, and corrections. System/developer prompts, private reasoning, and credentials are excluded. The Codex transcript has an explicit snapshot cutoff; the archival receipt is necessarily produced afterward.

Paper text is complete extraction without paraphrasing or editorial correction. Extraction can affect whitespace, ligatures, mathematical layout, figures, and hyphenation. The Oxford published edition is captured from all 2,405 lines of the web tool's 26-page PDF extraction, with the line map retained. Its binary PDF was blocked to the local downloader; the separately named BU August preprint PDF is not a substitute for the published edition. Schmidhuber's PDF and `pdftotext -layout` extraction are both retained. Consult the original PDFs for exact visual layout.

No training, infrastructure mutation, work-item dispatch, PR, or architecture amendment was performed by this harvest.
