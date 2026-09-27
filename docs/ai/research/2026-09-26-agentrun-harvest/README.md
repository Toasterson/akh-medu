# AgentRun harvest: let the agent do the job, then fire itself from it

Harvested 2026-09-26 from two public posts by Grep.ai. This is a bounded
reading of what they published, mapped onto what already exists in akh-medu.
It is not a claim about their private implementation and none of their numbers
were reproduced here.

Decision that came out of it: [ADR 114](../../decisions/114-knowledge-work-codification.md).
Plan: [Phase 54 — knowledge-work codification](../../plans/2026-09-26-phase54-knowledge-work-codification.md).

## Sources

| Source | Retrieved as | What it is |
| --- | --- | --- |
| [AJ Asver's post](https://x.com/_aj/status/2102061534956662818) (2026-09-21) | [sources/aj-post.fxtwitter.json](sources/aj-post.fxtwitter.json) | The headline claim: a harness that "learns the job as it runs, moving steps from LLM calls to code", quoting the article below. |
| [Miguel Ríos Berríos, "AgentRun: a harness for repetitive knowledge work"](https://x.com/MiguelriosEN/status/2101029313906987422) | [sources/agentrun-article.md](sources/agentrun-article.md) (verbatim block transcription), [sources/agentrun-article.fxtwitter.json](sources/agentrun-article.fxtwitter.json) (raw) | The full article. Images are referenced as `[media block]`; they were not retrieved. |

Direct X access returns 403; both were read through the FxTwitter mirror.
`SHA256SUMS` covers every file in this directory except itself.

## The mechanism, in their words

1. **Do it by hand first.** A frontier agent runs the job from the standard
   operating procedure, the expensive way. It selects tools, writes code when
   it needs to, and finishes.
2. **Leave notes about the job, not the case.** In learning mode the agent
   writes one to three sentences per run: which tool found what, which step
   was pointless, after which point nothing could change the answer.
3. **Write the program.** After several runs the agent reads the traces and
   notes and writes a workflow in a small DSL. Twelve node kinds: four agents
   (research, judge, report, operator), code, call, artifact, four typed
   questions (route, classify, sift, pick), and three shapes (chain,
   map/reduce, loop-until) plus escalate.
4. **Route a growing share to the program.** Fixed schedule: first batch all
   agent, then 50%, 75%, 90%, 100% with the agent as fallback. A candidate
   version goes live only after it beats the current one on a development set;
   a second set the author never sees decides promotion, because the author
   eventually matched every development label without improving on the
   reserved set.
5. **Keep the agent for the odd cases.** The program takes the ordinary path,
   the frontier agent handles what the program hands back, and a person
   reviews where the procedure requires it.

Their rule for the split: **the model decides what a thing is; code decides
what follows.** Closed questions become typed questions with a probability;
arithmetic, counting and thresholds are code; an agent step exists only where
something must be looked up or done.

## Numbers they report

These are author-reported, on their AML alert task, and not reproduced.

| Measure | Full agent | Workflow |
| --- | --- | --- |
| Tool calls per alert | 826 | ~30 |
| Wall time per alert | 51 min | ~3 min |
| Cost per alert | $2.89 | $0.25 |
| Cost for 100,000 alerts | $290K | $26K |
| Judge-scored accuracy | 90% (first 100) | 95% (last 300) |
| Cases handed back to the agent | n/a | ~1 in 20 early, ~1 in 70 late |
| Tuning run cost | | $1,049 total, $68 of it authoring |

Cost fell in steps, one per workflow version: replace judgment with typed
questions and code; triage and put the stoppers first; stop once nothing left
can change the rating; look the customer up only when a card needs it. The
saving came from pages never read, not from a smaller model reading the same
pages.

## Lessons worth keeping

- **Frequency is the trigger, the note is the seed.** They do not codify from
  the first run; they codify after several runs, from job-level shortcuts the
  agent wrote for the next agent. This is the unix-admin ladder: do it by hand,
  write down the trick, script it, cron it.
- **Replayable judgments.** Every typed answer and its probability is kept with
  the case. Changing the code that reads the answers re-runs a thousand cases
  in minutes with no model call. That property depends on retaining answers,
  not on which model produced them.
- **Evaluate one node at a time.** Each node has one model, one input contract,
  one output schema. Freeze the evidence and change only the suspect.
- **The full rubric per judge.** Splitting a workflow can leave a judge with
  half the policy; read the prompt the node actually receives.
- **Governance by construction.** Which model, on what inputs, under which
  policy version, and show me: each decision names its node, model, inputs and
  questions, and every version is fingerprinted.
- **Crash recovery scope.** Their recovery covers flat chains of code, calls,
  polling and artifacts; model nodes and nested workflows are not on that path.
  Unknown external outcomes are preserved for reconciliation before retry.

## What already exists in akh-medu

| AgentRun piece | akh-medu counterpart | State |
| --- | --- | --- |
| Traces per run | ADR 091 transcript episodic capture, `mutation_log`, akhexecd receipts | Live |
| Compile repeated traces into a procedure | Phase 11h chunking (`src/agent/chunking.rs`): `LearnedMethod` from repeated successful tool traces, HNSW method index, success/failure lifecycle, persisted under `agent:learned_methods` | Live, in-engine only, no code output |
| Latent skill detection in the background | Phase 31d skill discovery (`src/skills/discover.rs`): co-activation clusters → `SkillCandidate` | Live, declarative knowledge only |
| Procedural skill files | Phase 30 skillpacks: `actions.json`, `plans.json`, `procedures.md`; Phase 30c LLM authoring | Live |
| Code node | Typed TS cells in `akh-repl` (2026-09-22 typed-execution proposal, slice A2) | Proposed |
| Package | `.akhpack` (slice A1/A3), WASM tool runtime `wit/akh-tool.wit` | Proposed / feature-gated |
| Call node with receipt | akhexecd operations (Phase 53c) | Landing |
| Typed question | `RouteFilter` data-driven rules (53g), Laya shadow adapter (`experiments/typed-decisions/LAYA_DECISION_LAYER.md`) | Rules live; model retired (53g D1) |
| Escalate to the full agent | Reactor result "request a larger-model turn"; 53e delegation grants | Proposed / landing |
| Judge with the full rubric | Microtheories, argumentation, ACH (Phases 9, 18) | Live |
| Governance fingerprint | `TripleIdentity`, WAL, provenance `DerivationKind` | Live |

## What we take and what we leave

**Take.** The ladder and its trigger; retained decision records for replay;
per-node evaluation; the sealed held-out set as the promotion gate; the
fixed canary ramp with fallback; "code decides what follows".

**Leave for now.** A Jev-class typed-question model in the dispatch path. The
53g evidence stands: the Laya checkpoint answered `investigate` on 52 of 52
complete inputs with near-uniform scores, and the 288-decision replay in
`experiments/typed-decisions/LAYA_BENCHMARK.md` gave no controller-grade
result. The seam stays; the backend today is the engine's structural rules.

**Do not copy.** A new DSL. The typed-cell language plus packages plus reactor
route filters already gives us the twelve cards; inventing a second syntax
would split evaluation and provenance across two systems.
