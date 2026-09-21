# Jevlike: source-level findings and review handoff

## Evidence boundary

Inspected commit `94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452` of [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike/tree/94f5fd1b0b11d52bbdfdf4e0ee6aa96b568f8452). This is an independent architecture with Jev's broad text-plus-candidates interface, not a disclosed reconstruction of TypeSafe's private architecture or RLCD. Findings below are static inspection and author-reported results. No upstream code, tests, checkpoints, or training were executed during this harvest.

## What to take from the implementation

The [text head](upstream/jevlike/model.py) normalizes context tokens and option representations, projects queries/keys/values, lets each option attend over context, and scores the dot product of that option query with its attended context. Invalid padded candidates are masked. A downstream softmax turns logits into a closed-set distribution.

For ADAM, the reusable idea is a shared scorer over a variable menu of action descriptions. It separates scoring from language generation and does not require a permanent classifier output neuron for every possible text label. This is an experimental baseline; complete candidates and adequate observations still have to come from somewhere.

Important source details:

- `TinyScorer` averages byte embeddings for option text without option positional embeddings. Consequently, byte permutations with the same retained byte counts have identical option representations. It is unsuitable as-is when argument order or identifier structure distinguishes actions. Context has positional embeddings but no contextual text encoder in this tiny path.
- `FrozenTransformerScorer.forward` invokes its frozen encoder twice: once for context, once for the flattened batch of options. It then runs the shared attention head. “One pass” describes non-autoregressive decision scoring, not literally one encoder invocation or constant cost as the option list grows. This qualifies the earlier conversational shorthand.
- [Text training](upstream/jevlike/train.py) uses ordinary supervised cross-entropy and AdamW, choosing by validation loss. It is not an RLCD implementation. Each row has one chosen label; several valid actions and abstention need an explicit data/objective treatment.
- Default text limits are 192 context tokens and 32 option tokens; in the tiny path these are bytes. A software-action representation needs truncation checks that preserve discriminating evidence.
- [Evaluation](upstream/jevlike/eval.py) reports top-1, top-3, ten-bin ECE, and a batch-rolled context control. ECE is measurement, not calibration training. Shuffling does nothing for a batch of one and can pair similar contexts in other batches; a stronger control should establish mismatched examples. Top-3 is uninformative for menus of three or fewer choices.
- The text head has no built-in candidate adequacy detector, calibrated workflow-success predictor, or explicit abstention policy. Normalization over offered options does not supply these.

## Results worth preserving with their limits

The [upstream README](upstream/README.md) reports approximately 98% on synthetic menus; 26% next-click accuracy with frozen Qwen2.5-0.5B versus approximately 8% shuffled/random-encoder controls; and 29% for a model trained from scratch on 40,000 clicks. These are prior local experiments, explicitly not the supplied quickstart run or proof of parity with Jev.

Its approximately 100-fold latency comparison uses eight options against a small decoder forced to emit 400 tokens. It does not establish the same gain over a compact option-ID decoder or a logits-only baseline. Include candidate construction, tokenization, both encoder calls, and any caching policy in an end-to-end comparison.

## Newer visual examples

The pinned snapshot includes [Doom](upstream/examples/doom/README.md), [chess](upstream/examples/chess/README.md), a shared [visual scorer](upstream/jevlike/vision.py), supplied checkpoints, and a selected demo film. These additions were absent from the GitHub page initially fetched in the conversation.

The visual examples use image patches and a learned fixed table of 12 controller-option embeddings (seven Doom buttons, five chess keys). This demonstrates another input modality for option attention; it does not demonstrate arbitrary unseen action-description generalization in the visual model.

The author reports imitation and DAgger as more useful starting points than pure pixel PPO. Their chess procedure collects the policy's own visited states, asks a Stockfish-based teacher for corrective keys, and mixes those with teacher trajectories. That is relevant to recovery-state coverage in Kaine, but is not evidence that the existing Kaine objective should be replaced by PPO.

Preserve the negative results: the selected Doom demo's joint checkpoint averaged 0.60 kills and -97.50 reward in ten recorded episodes. The chess-only model recorded 4 wins, 46 draws, 0 losses against a random mover, but 0 wins, 2 draws, 48 losses against Stockfish level 0 in 50 games each. Random legal moves substitute after a 40-key budget failure, so game results include fallback behavior. The author says joint training lost much of the chess controller; sampled play also substantially outperformed greedy play, which could loop. These are author reports, not reproduced measurements.

## Proposed bounded comparison

Use the [earlier TypeSafe/Jev handoff](../2026-09-16-typesafe-jev-adam-kaine/README.md) as context, and re-read ADR 110, ADR 111, and P7 before implementation. This harvest does not certify current project deployment or lifecycle state.

1. Freeze an ADAM dataset of observations, full candidate actions, verifier outcomes, and provenance. Split by task family/entity/workflow to limit leakage. Represent multiple valid choices, omitted correct choices, and insufficient evidence explicitly.
2. Compare compact constrained generation, logits/candidate scoring, and this attention-head design under the same menus and observations. Test complete-action scoring before assuming factorized tool/entity/argument decisions remain valid when recombined.
3. Measure candidate recall, verified action and workflow success, latency and total compute, log loss, Brier score, reliability bins, and coverage/error under abstention. Fit any temperature on validation data only. Include shuffled options, reordered argument strings, changed identifiers, missing evidence, and long context.
4. For Kaine, consider a separate decision output sharing representations with generation. Evaluate both tasks for regressions, retain engine-side authority and verification, and preserve corrected-sibling supervision and reward-vector semantics. The joint-game regression makes interference a concrete review question.
5. Investigate recovery-state relabeling independently of the architecture choice. Keep failed behavior distinct from teacher-corrected targets and avoid labeling every unchosen action wrong when several are valid.

Review questions: What is the smallest useful ADAM menu? Which exact probability is being calibrated? What happens when all candidates are bad? Can byte-order collisions or truncation erase action differences? Does the gain remain after candidate construction and encoder cost? Does a shared Kaine head degrade generation or recovery?

## Preservation

The MIT license is retained with the full upstream source. Supplied binary checkpoints and media are archived, not loaded. Seshat receives this handoff, the verbatim conversation, the source post, and exact text-source bundles; binary originals remain in the repository harvest with hashes. See `seshat/` for import and retrieval evidence.
