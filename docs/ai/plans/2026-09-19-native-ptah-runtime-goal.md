# Native Ptah runtime — recovered handoff and goal brief (2026-09-19)

Astra's 2026-09-15 handoff was written to `/tmp/ptah-handoff.kVtTm6/HANDOFF.md`
and that directory no longer exists. The full text was recovered verbatim from
the Claude Code transcript
`~/.claude/projects/-home-toasty-ws-ptah/18288d3d-ecde-4da6-924a-669a78e9861f.jsonl`
and is reproduced in the appendix below so it stops living only in `/tmp`.

Read this brief together with, not instead of, the master ledger
`docs/ai/plans/2026-09-14-native-ptah-runtime.md`. That ledger has not been
updated since 2026-09-14 22:41 and its "latest checkpoint" section is stale;
three further working sessions landed after it.

## What happened after the handoff

The handoff named the **native tap -> projection -> retrieval -> prompt loop**
as the next feature. That loop was built and validated:

| Commit | Date | Content |
| --- | --- | --- |
| `e8d87782` | 09-15 | exact-model recall budget failure exposed on HIP and Vulkan |
| `bc52dd47` | 09-15 | ranked recall fitted without losing the top fact |
| `200e4548` | 09-15 | note interpretation isolated; explicit instructions validated |
| `fc1c5fe8` | 09-15 | native recall validated through five private daemons |
| `e291ab44` | 09-15 | native Candle HIP/Vulkan implementation checkpointed |
| `8f6f9d24` | 09-15 | clean-source builds and checkpoint review recorded |
| `9347805b` | 09-15 | Vulkan resident pipeline cache bounded with LRU eviction |
| `bb2c6a2a` | 09-18 | full-model HIP/Vulkan resident-idle acceptance recorded |

Evidence lives in `docs/ai/research/2026-09-15-*.md` and
`docs/ai/research/2026-09-18-resident-gpu-acceptance.md`. Current HEAD is
`bb2c6a2a`; nothing has been pushed; 17 untracked paths (research drafts,
experiments) remain in the tree and must be preserved.

Measured at `9347805b` with the exact pinned Q4_K_M artifact: model load 23.9 s
HIP / 30.6 s Vulkan; fresh-call warm decode 20.2 steps/s HIP, 13.5 Vulkan;
resident idle 10.0 W / 0 % busy on both, fans stopped. Those are **fresh unkeyed**
calls — keyed Qwen continuation still uses the scoped-f32 HIP guard and has never
been benchmarked. The old 30.7 steps/s llama-server reference came from a
different fixture and is not a like-for-like comparison.

## State after 2026-09-19

Updated at the end of the day. Commits `9489ed7f` through `2c4f5725` on `main`,
nothing pushed. The master ledger carries the per-slice evidence; this is the
categorised summary.

### Done, with committed evidence

1. **Keyed measurement.** Both devices measured; variance explained (one
   recorder run with a different preamble). Rule: compare identical preambles.
2. **Long residency.** 160 keyed requests over 14m42s, decode and RSS flat.
3. **Daemon acceptance.** Five daemons rebuilt, private canary passing, both
   devices, including recall after restart.
6. **Live KV compression and long-context gates - complete.** Compressed KV
   works on HIP and on Vulkan (packed three codes per `u32`, 2.94x, ~5% decode;
   the byte layout there had been failing closed on a missing cast). Fidelity is
   measured on the live path: worst cosine **0.99788 at 8 bits, 0.98540 at 4** -
   it clears the 0.9999 gate at **no** width, and greedy-token agreement was shown
   to be actively misleading (same codes, different tokens on two devices). Off
   by default; needs an acceptance criterion of its own, which is a decision.
   The context gate now sizes on **measured total cost** - KV is 23% of what a
   token costs - after the first version admitted requests that then failed on
   allocation: serviceable context is **10,816** tokens against 262,144 declared.
8. **Reproducible release bundle - done.** Manifest records binary digests,
   per-binary kernel-library pairings and toolchain from files; the relocatable
   bundle passes the canary run from its own `--bin-dir`.

### Blocked on a decision from the user

4. **ContextManager materialization.** Built and inert: the seam is de-risked
   and the fork evidenced in code, but **no party holds both history and
   policy**. Either give akhrtd a core client, or move the conversation into the
   session record.
5. **Vision - the runtime can take an image; the daemons refuse one.**
   Everything in the model is landed and gated: projector agreeing across
   CPU/HIP/Vulkan at cosine 1.0000000000; per-token rope positions; spliced
   embeddings; grid-derived positions; decode continuing from the rope cursor;
   a front end (`generate_with_image`) that decodes, resizes, expands the
   placeholder, splices and generates; and **2D rotary in every vision block**,
   which the tower had lacked - the first image came back "pixelated,
   fragmented" and now comes back "a solid red square ... a blue horizontal bar
   along the bottom edge". Text path bit-identical throughout.

   What remains is not model work:
   - **Transport.** Both daemons refuse turn artifacts by design, with a test
     pinning the refusal so nothing is dropped silently. akhrtd has no
     filesystem access to workspaces (its reads are brokered tools returning
     text). Three ways an image can arrive: inline bytes in the turn input,
     a bytes-returning brokered read, or workspace filesystem access for
     akhrtd. This is a topology choice of the same family as item 4.
   - **Semantic validation.** "No external consumer is installed" was wrong:
     ollama is installed with this exact model as `ptah-qwen3.8`. Its server is
     down; starting it changes machine state and contends for the GPU. Needs a
     go-ahead.
   - ~~Residual: a centred square called "upper right"~~ - closed 2026-09-20:
     the LLM-side M-RoPE sections were laid out contiguously (Qwen2-VL);
     Qwen3.5 interleaves them. Placement now reads correctly.
   - Multiple images per prompt: refused explicitly, not built.

### Open

7. **Performance - measured, ceiling known, default deliberately unchanged.**
   Prefill chunk 256 is **1.71x** faster than 128 (three interleaved pairs, clean
   separation), but peaks at 98.4% of the card at a context the gate admits, so
   the default stays 128 until the cap or the headroom moves - a trade to take
   deliberately. Removing the `repeat_kv` expansion was tried: numerically
   exact and **10% slower**, kept behind a flag with the number on it. Earlier
   single-run figures (49%, 1.95x) were retracted. The 510-position keyed effect
   remains unexplained.

### Not started

9. **Homelab canary and cutover.** Needs shared services; ask first.

## Resumed 2026-09-20: both pending verifications passed

Two fixes were committed on 2026-09-19 without GPU verification, the day
having been capped. Both were run first thing on resuming and **both passed**;
the commands are kept so the runs are repeatable:

1. **Interleaved M-RoPE for image tokens.** The LLM-side section mapping was
   Qwen2-VL's contiguous blocks; Qwen3.5 interleaves them (transformers
   `qwen3_5` `recomposition_frequencies`, llama.cpp `LLM_ARCH_QWEN35` ->
   `IMROPE`). Symptom before the fix: a red square placed left, centre or right
   was called "right" every time. Verify with the three placements and the four
   geometry images:

       cargo build --release --features hip --example vision_describe_probe
       for IMG in square_left square_centre square_right square_bar_bottom \
                  square_bar_left stripes_vertical stripes_horizontal; do
         ./target/release/examples/vision_describe_probe --device hip \
           --model $MODEL --tokenizer $TOK \
           --image experiments/native-recall/vision-e2e-20260919-hip-v1/images/$IMG.png \
           --tokens 64
       done

   Expected: left/centre/right each named correctly; the four geometry images
   at least as good as the post-rope run in `geometry-after-rope.log`.

   **Done 2026-09-20: passed.** Left/centre/right each correct; bar-bottom
   square now "centered in the upper portion". See
   `placement-after-interleaved-mrope.log`.

2. **Allocator-cache reclaim on the admission refuse path.** Before the fix a
   repeat of a 10,417-token request that had just succeeded was refused at a
   913-token limit, because `hipMemGetInfo` excludes the pool's reusable
   blocks. Verify that round 1 now admits and completes:

       cargo build --release --features hip --example candle_session_probe
       RUST_LOG=info ./target/release/examples/candle_session_probe --device hip \
         --model $MODEL --tokenizer $TOK --prompt-file /tmp/ctxsoak/prompt-08k.txt \
         --tokens 8 --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0

   Expected: two generation rounds and an `admitted after reclaiming the
   allocator cache` log line on the second. If `/tmp` was cleared, regenerate
   the prompts first with
   `experiments/native-recall/ctxsoak-20260919-hip-v1/make-prompts.py` - the
   ladder script only references them. (The note here originally said the
   ladder script regenerates them; it does not, and this was found on resuming.)

   **Done 2026-09-20: passed.** Round 1 admitted after reclaiming 5.33 GB,
   limit 910 -> 10,844, and completed. See `reclaim-verification-passed.log`.

With both verified, nothing unblocked remains. Next are the decisions listed
under "Blocked on a decision from the user".

## Standing constraints (unchanged)

- Implementation repo is `/home/toasty/ws/agi/akh-medu`; `/home/toasty/ws/ptah`
  is an unrelated TUI editor prototype. Read `AGENTS.md` first.
- No psql / direct Anima Postgres queries; tracker access via the Anima MCP only.
- Local commits authorized; pushing is not. Never blanket stage, reset or clean —
  other sessions edit this tree concurrently.
- No package/driver/kernel/sysctl changes. GPU probes run serially; no CPU
  compilation during throughput or idle acceptance runs; do not swap daemon
  binaries while the integration harness is running.
- Do not weaken the 0.9999 cosine gates on greedy-token agreement alone, and keep
  the HIP scoped-f32 continuation guard. HIP recurrence fusion stays rejected.
- Keep one cognitive core, bounded exchanges, authoritative history in the durable
  session loop. No raw hidden-state retention by default.
- Fable consultation: `/home/toasty/.local/bin/claude`, model `claude-fable-5`,
  tools/MCP disabled; treat its output as hypotheses to verify against source.
- Update the master ledger as work lands; it is the durable record, not chat.

---

## Appendix — recovered handoff, verbatim (2026-09-15)

> # Ptah native runtime handoff — 2026-09-15
>
> ## Objective and current request
>
> Finish the native resident Candle runtime in `akhrtd`: HIP and Vulkan, quiet GPU
> idle, exact Ptah model fidelity and quantization/compression, session-daemon
> integration, and eventual obsolete Ollama/llama-server/akhomed deployment cleanup.
> The project is **not finished**. The user wants autonomous work, features before
> performance, and Fable architectural consultation when useful. They requested
> this handoff so another agent can continue.
>
> The durable goal in this thread still reports `usageLimited`, including after
> the user's `/goal resume`. Recorded usage is 1,543,863 tokens; remaining allowance
> is null. Available goal tools cannot resume or reset allowance. Do not falsely
> complete/replace this unfinished goal to bypass that state or claim unattended
> execution is active. Ordinary explicitly requested work has remained possible.
>
> ## Workspace and authority
>
> - Actual Git repository: `/home/toasty/ws/agi/akh-medu`.
> - Supplied working directory `/home/toasty/ws/ptah` is not the implementation repo.
> - Read repository `AGENTS.md` first. Hard rule: **no psql/direct Anima Postgres queries**.
> - Local work only. Leave Archibald and Forgejo alone; other sessions own those.
> - Ptah downtime is allowed. Earlier shared session/core/sandbox restart approval
>   was followed by local-only direction; the current ledger explicitly pauses
>   shared-service changes. Reconcile scope before eventual shared cutover.
> - Local commits and rebases are explicitly authorized. No push authorization
>   was inferred. No rebase was needed: branch had no recorded upstream divergence.
> - Preserve all uncommitted/untracked work. Do not blanket stage, reset, or clean.
>   Other sessions may modify files. No packages/drivers/kernel/sysctl changes.
> - Run GPU probes serially. No CPU compilation during throughput/idle acceptance.
>   Do not replace daemon binaries during the integration harness: it restarts them.
> - Use `apply_patch` for edits, `/bin/bash` with `login:false` for shell calls.
>
> ## Read these existing artifacts; they contain the detailed evidence
>
> All paths below are relative to the implementation repository. Do not restart
> completed investigations or reconstruct evidence already recorded here.
>
> 1. `docs/ai/plans/2026-09-14-native-ptah-runtime.md` — master ledger, exact artifacts,
>    hardware, ordered acceptance gates, operational state and numerical results.
> 2. `docs/ai/research/2026-09-14-native-span-annex.md` — newest completed feature,
>    actual protocol/storage trust boundaries, tests and remaining wiring.
> 3. `docs/ai/research/2026-09-14-native-continuation.md` — prefix reuse, session
>    ownership, HIP precision regression, scoped fix and daemon/idle evidence.
> 4. `docs/ai/research/2026-09-14-native-same-pass-taps.md` — bounded pooled taps,
>    numerical gates and capture semantics.
> 5. `docs/ai/research/2026-09-14-native-vsa-space-assessment.md` and
>    `docs/ai/research/2026-09-14-native-taps-fable-assessment.md` — projection-space
>    and architecture decisions; important limitations on Fable's suggestions.
> 6. `docs/ai/plans/2026-08-24-candle-qwen38-rdna3-inference.md`, ADR 098 and ADR 065
>    (locate with `rg --files`) — accepted architecture and feature requirements.
> 7. `examples/native-ptah/README.md` — staged bundle/testing instructions, not a
>    production deployment. Related research docs cover Vulkan, tokenizer and perf.
>
> ## Git checkpoint and latest verification
>
> Verified HEAD: `9a3984e2 feat(core): add bounded session-scoped neural span annex`.
> Previous: `1a30a453 fix(observability): remove duplicate OTLP imports`.
> Both were created locally after user authorization; nothing pushed or rebased.
> The annex commit selectively staged only its shared-types hunks, leaving native
> provider configuration changes unstaged. The index is empty at handoff creation.
> There are still 32 modified tracked files and numerous untracked implementation,
> experiment, deployment-example and evidence files. `git diff --stat` excludes the
> untracked files: inspect both before staging any subsequent checkpoint.
>
> Latest rerun passed all 21 relevant tests and the session-daemon check:
>
> ```bash
> CARGO_BUILD_JOBS=2 cargo test --offline -p akh-session-types -p akh-service-store --lib
> env -u AKH_TRANSCRIPT_CAPTURE CARGO_BUILD_JOBS=2 cargo test --offline --lib --no-default-features --features core-server core_service::tests -- --nocapture --test-threads=1
> CARGO_BUILD_JOBS=2 cargo check --offline -p akh-session-services --no-default-features --features session --bin akhsessd
> ```
>
> Counts: 17 Store tests, 2 shared-types tests, 2 core HTTP tests. These are not a
> full project suite or an external PostgreSQL integration pass. Existing warnings
> remain. `git diff --check` passed. No builds/probes from this session are left
> running; recheck processes before new work because the workspace is shared.
>
> ## Where to continue
>
> The next feature is the **native tap → projection → retrieval → prompt loop**.
> The annex is real, tested storage/API infrastructure, not completed neural recall.
>
> - `crates/akh-session-types/src/neural_spans.rs` defines representation pins and
>   bounded wire types. `crates/akh-service-store/src/neural_spans.rs` stores them
>   in the existing core Store. `src/core_service.rs` exposes authenticated
>   capabilities, acceptance and query routes.
> - `crates/akh-session-services/src/server/sessions.rs` currently sends an empty
>   `span_captures` vector. Capability negotiation, native transport and delivery
>   are still unwired. Empty captures preserve the old wire payload/fingerprint.
> - `src/nlu/vendored/quantized_qwen35.rs` implements immutable shared weights,
>   private inference sessions/checkpoints, exact-token continuation and bounded
>   same-pass taps. `src/nlu/candle_backend.rs` and
>   `crates/akh-session-services/src/internal_native.rs` own generation integration.
> - Query staging needs actual design: current `/v1/context` precedes runtime
>   dispatch, but a hidden-state query only exists after prefill. Keep one cognitive
>   core, bounded exchanges, authoritative history and correct prompt templating.
>   Recall buckets include service-principal identity; direct runtime queries with
>   a different principal cannot see the session service's accepted spans.
> - Do not substitute a random projector for the learned symbolic ItemMemory
>   space and claim equivalent grounding. Version the complete representation and
>   validate semantic retrieval quality. Same-space neural-span hits are not
>   automatically symbolic concept matches. No raw hidden-state retention by default.
> - Cached requested tap spans return no taps instead of silently replaying prefill.
>   Respect this when designing query capture and accurate text/token boundaries.
> - Core operation retries with an interrupted/unknown outcome remain fail-closed
>   409, not automatically replayed. Pins identify recipes but are not attestation.
>   Capture opt-out gates access/writes; it does not retroactively erase stored data.
>
> After that loop: real ContextManager commit/materialization, exact vision path,
> live attention KV compression and long-context gates, then performance tuning,
> full validation and canonical deployment/cleanup. Consult the ledger rather than
> treating this list as a replacement specification.
>
> ## Critical numerical and operational cautions
>
> Keyed HIP continuation deliberately uses scoped f32 activation arithmetic:
> mixed Q8 decode/f32 prefill failed recurrent-state parity. The weight format
> remains Q4_K_M. Historical ~20 HIP / ~13 Vulkan decode figures predate this keyed
> guard and are **not current keyed performance measurements**. Near-reference
> performance remains unfinished. Do not remove the precision guard on the basis
> of greedy-token agreement alone. HIP recurrence fusion remains rejected by its
> combined-Q8 gate. Exact-device continuation, tap and resident-idle evidence is
> recorded in the referenced docs.
>
> The old local system `ptah-inference.service` and old local akhomed were stopped
> and disabled. No production native daemon was installed. Model, projector and
> draft artifacts remain preserved. No obsolete model/binary deletion has occurred.
> The deployed `/home/toasty/src/akh-medu` is a different, older non-Git tree.
> Earlier relocated bundle `/tmp/ptah-native-bundle.bKJjz5` predates continuation
> and taps; do not deploy it as the latest build. Release binaries also predate the
> new annex changes; rebuild and verify exact deployment artifacts before cutover.
>
> ## Fable consultation
>
> User explicitly requested Fable; it is available through the local Claude CLI
> at `/home/toasty/.local/bin/claude`, model `claude-fable-5`. Prior consultations
> used safe mode, no tools/MCP and no persistent sessions. Briefs and assessed
> responses are under `docs/ai/plans/2026-09-14-*-fable-brief.md` and the research
> paths above. Latest raw annex review: `/tmp/ptah-fable-span-annex-review.json`.
> Treat suggestions as hypotheses: its evicted-turn replay concern was disproved
> by a full HTTP regression because the durable journal retains request identity.
>
> ## Suggested skills
>
> - `implement` and `tdd`: implement the next vertical feature with focused tests.
> - `codebase-design`: place the staged retrieval boundary without a second Engine.
> - `diagnosing-bugs`: numerical/performance regressions and measured experiments.
> - `mattpocock-skills:research`: primary-source backend/compression research if needed.
> - `review`: review a defined checkpoint once implemented and tested.
> - `openai-docs`: only if troubleshooting the product goal controls again.
>
> Read each applicable skill fully before acting. This handoff was intentionally
> written to an OS temporary directory, not committed into the repository.
