# Is "selection-only" context management a real approach?

Asked while deciding whether native turns must be compacted. Primary source
read directly: Zhang et al., *A Survey on the Memory Mechanism of LLM-based
Agents*, arXiv:2404.13501, later ACM TOIS (doi 10.1145/3748302).

## What the literature actually separates

The survey factors an agent's memory into three operations, and selection and
summarization live in **different** ones:

- **Writing** `W({a,o})` — project raw observations into stored content.
- **Management** `P(M, m)` — "summarizing high-level concepts to make the agent
  more generalizable, merging similar information to reduce redundancy, and
  forgetting unimportant or irrelevant information". This is the rewriting
  family: reflection, consolidation, forgetting.
- **Reading** `R(M, c)` — "usually implemented by computing the similarity
  between M and c". In Generative Agents it is "three criteria including
  similarity, time interval, and importance".

Selection is memory **reading**, not a degenerate form of summarization. The
survey's taxonomy of textual memory forms makes the same split: content is kept
as *complete* interactions, *recent* interactions, *retrieved* interactions, or
external knowledge. "Recent" and "retrieved" are exactly selection policies.

Its comparison table shows retrieval-based selection is the mainstream choice —
MemoryBank, RET-LLM, ChatDB, TiM, Voyager, MemGPT, MemoChat, MPC, Generative
Agents, Synapse, RecAgent, S3 and MetaAgents all read by retrieval — while
several combine *recent* + *retrieved* (MemGPT, SCM, RecAgent, InteRecAgent).
Summarization is an operation many of them add on top, not the mechanism by
which context is chosen.

So a selection manager is not an unusual proposal. It is the common case; what
MemGPT adds is a *hierarchy* (an in-context FIFO working set plus external
storage it pages against) rather than replacing selection with rewriting.

## Why this matters specifically for a resident local runtime

There is a project-specific argument the survey does not make, because it does
not assume a resident KV cache.

Our keyed path reuses an exact token prefix. Today's measurements: with a reused
prefix, time to first token is **0.92 s on HIP and 1.43 s on Vulkan**; with a
full prefill it is **6.74 s and 9.89 s**. Prefix reuse is worth roughly 7x on
prefill.

Rewriting history destroys that prefix. Any compaction that rewrites earlier
turns changes the token sequence from the rewrite point onward, so the next turn
reuses nothing and pays a full prefill — the same cliff we already measured when
a capped reply drops session state. Selection that only *removes whole trailing
or leading spans* can preserve a reusable prefix; summarization cannot, by
construction.

That does not make compaction wrong. It makes it an **expensive, infrequent
operation** rather than a per-turn one — which is how MemGPT uses it, at
eviction time, not on every turn.

## Bearing on the decision

- Selection-only for native turns is well-founded, not a shortcut.
- A `VirtualContextManager`-style hierarchy is the richer target the user
  prefers, and it is *compatible* with selection: paging decides what is
  resident, selection decides what enters the prompt.
- Compaction should be modelled as a management operation triggered on
  eviction, with its prefill cost accounted for, not as the default assembly
  path for every native turn.

## Not established

One survey read directly plus our own measurements. No held-out quality
comparison between selection and summarization on this model, and no measurement
of how often compaction would actually trigger in a real session. The 7x prefill
figure is from three reused-prefix requests per device; it is an order of
magnitude, not a calibrated constant.
