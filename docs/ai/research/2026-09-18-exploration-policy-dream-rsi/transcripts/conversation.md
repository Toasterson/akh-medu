# Exploration policy in the loop + Dream-RSI implementability — verbatim conversation

Snapshot of user-facing messages of T3 thread `6adfb261-5682-4d80-89fc-4875480d9876` (Codex, `gpt-6-astra` lane) through 2026-09-18T12:50:58.755Z. Excludes system/developer instructions, private reasoning and tool traffic. Historical analysis is preserved without editorial changes.

## 1. user (user-facing) — 2026-09-18T12:49:33.629Z

Whats our exploration policy in the loop for cloud models and local llms? And can we implement deepminds rsi paper?

## 2. assistant (user-facing, commentary) — 2026-09-18T12:49:37.730Z

I’ll check how the loop currently explores with cloud and local models, then identify which DeepMind RSI paper fits the implementation question.

## 3. assistant (user-facing, commentary) — 2026-09-18T12:50:15.786Z

The code has several exploration mechanisms, but I haven’t found a single policy shared across them. I also found our Dream-RSI archive: its replay-based controller looks applicable to both cloud and local models because it changes search decisions while keeping model weights fixed. I’m checking the actual LLM loop so I can distinguish its behavior from the symbolic planner and model router.

## 4. assistant (user-facing, final_answer) — 2026-09-18T12:50:58.755Z

**Our cloud and local LLMs currently share a model-directed tool loop. We can implement Dream-RSI around that loop, but its replay-and-improvement machinery is not implemented today.** This is based on the current source, not a live deployment check.

For exploration, there are three distinct mechanisms:

| Layer | Current policy |
|---|---|
| **Cloud/local agentic LLM loop** | The configured model chooses its next tool calls from the conversation and results. The host enforces permissions, iteration/time budgets, repeated-call detection, and completion checks. It follows one evolving conversation; it does not systematically compare alternative branches. |
| **Symbolic OODA loop** | Default selection ranks tools using semantic relevance, novelty, recency penalties, previous success, and other context. Optional VSA-native selection uses UCB1 exploration to resolve ambiguous rankings. |
| **NLU backend router** | A contextual bandit explores backend choices with default `α = 0.25`. However, **its `CloudApi` and `BurnLocal` execution arms remain unimplemented and fall back to the cascade**. This is not a working cloud-versus-local exploration controller. |

The relevant code is in [agentic_chat.rs](/Users/toasty/ws/akh-medu/src/agent/agentic_chat.rs:839), [ooda.rs](/Users/toasty/ws/akh-medu/src/agent/ooda.rs:707), and [nlu/mod.rs](/Users/toasty/ws/akh-medu/src/nlu/mod.rs:451).

So today, **the LLM largely decides what to try; the harness decides what it may execute and when it must stop.** We do not yet learn whether branching, revisiting an earlier attempt, or abandoning a direction would improve results.

Assuming you mean **Dream-RSI**, the Google/DeepMind paper we already [archived](/Users/toasty/ws/akh-medu/docs/ai/research/2026-09-16-dream-rsi-adam-kaine/README.md), it fits that missing layer closely. It improves executable exploration-policy code using recorded discovery trees, while keeping the underlying models and evaluator fixed. That works conceptually with either cloud APIs or frozen local checkpoints; weight training is unnecessary. [Paper](https://arxiv.org/html/2609.14858v1)

I’d implement it through these concrete pieces:

1. **An explicit exploration controller:** choose which attempt to extend, when to branch, and when to stop.
2. **Replayable discovery records:** parent attempt, environment snapshot, model/prompt version, proposal, verified outcome, and cost.
3. **An offline replay evaluator:** expose only outcomes the candidate policy has reached; mark unrecorded branches as unknown.
4. **Policy improvement and promotion:** compare the current policy, a handwritten adaptive baseline, and replay-improved candidates on held-out tasks, then verify the winner on fresh executions.

For cloud/local comparisons, record model identity and costs from the start, but evaluate each backend separately first. Otherwise, a better model can masquerade as a better exploration policy.

**The hardest part is faithful replay of stateful tool execution.** Ordinary chat logs cannot establish what would have happened after changing an earlier action. Start with isolated code-repair or algorithm tasks where branches can be restored and objectively evaluated. Keep existing permission and completion gates authoritative.

The authors still list their full implementation and reproduction scripts as pending, so this would currently be our implementation of the method. [Repository](https://github.com/zhengkid/Dream-RSI/blob/main/README.md)
