# ADR 014 — Multi-Agent Communication with Capability Tokens

> Date: 2026-02-21
> Status: Accepted
> Phase: 12g

## Context

Phase 12a–12f established the CommChannel abstraction, grounded dialogue,
constraint checking, social KG, federation, and transparent reasoning. However,
all interaction was between a single agent and human interlocutors (operator,
social contacts, public channels). There was no protocol for structured
agent-to-agent communication — agents couldn't query each other's knowledge
graphs, propose goals, or share assertions with capability-scoped authorization.

## Decision

Introduce a capability token system (OCapN-inspired) with structured protocol
messages for agent-to-agent communication, carried through the existing
CommChannel infrastructure.

### Architecture

1. **CapabilityToken** — immutable scoped permission tokens granted by the
   operator. Each token specifies: target agent, holder agent, permitted scopes,
   optional expiry, and revocation flag. Tokens cannot be modified after
   creation — only revoked.

2. **CapabilityScope** (6 variants) — `QueryAll`, `QueryTopics(Vec<SymbolId>)`,
   `AssertTopics(Vec<SymbolId>)`, `ProposeGoals`, `Subscribe`, `ViewProvenance`.
   Topic-scoped variants restrict access to specific KG subjects.

3. **AgentProtocolMessage** (10 variants) — structured messages that bypass
   the NLP classifier:
   - `Query` / `QueryResponse` — KG queries with grounded triple responses
   - `Assert` — fact proposals with evidence and confidence
   - `ProposeGoal` — goal suggestions (require operator approval)
   - `Subscribe` / `Unsubscribe` — topic update notifications
   - `GrantCapability` / `RevokeCapability` — token management
   - `Ack` / `Error` — protocol-level responses

4. **TokenRegistry** — HashMap-indexed by token ID and by agent pair
   (holder, target). Validates incoming messages against their token:
   holder matches sender, token is valid (not expired/revoked), and
   action is within scope.

5. **InterlocutorKind** — `Human` | `Agent` enum on `InterlocutorProfile`,
   enabling the agent to distinguish human from agent interlocutors for
   protocol dispatch.

6. **Trust bootstrap** — agents introduced by the operator start at
   `ChannelKind::Trusted`; agents encountered via federation without
   introduction start at `ChannelKind::Public`. `GrantCapability` messages
   promote trust.

7. **Channel integration** — `MessageContent::AgentMessage` variant carries
   `AgentProtocolMessage` through CommChannel, classified as
   `UserIntent::AgentProtocol` (bypasses NLP). `can_propose_goals` capability
   flag added to `ChannelCapabilities`.

## Alternatives Considered

- **Unstructured text-based agent communication**: Rejected — the agent is
  LLM-free, so NLP-based intent parsing of agent messages would be fragile.
  Structured protocol messages are deterministic and type-safe.

- **Full OCapN/CapTP implementation**: Too complex for current needs. The
  simplified capability token model captures the essential authorization
  pattern without the full handoff/promise protocol.

- **Shared KG access (no protocol)**: Rejected — agents should have
  independent knowledge graphs with explicit information sharing. Shared
  state creates coupling and makes provenance tracking impossible.

## Consequences

- Agents can communicate through structured protocol messages with
  capability-scoped authorization.
- The operator retains full control — all capability tokens are granted
  by the operator, and goal proposals from agents require operator approval.
- Token validation is O(1) by ID lookup with O(n) pair queries.
- The system supports topic-scoped access control — an agent can be granted
  query access to specific KG subjects without exposing the full graph.
- InterlocutorKind enables future protocol dispatch based on whether the
  interlocutor is human or agent.
