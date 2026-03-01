# ADR-020: Predictive Planning & Epistemic Reasoning Research

- **Date**: 2026-02-22
- **Status**: Accepted
- **Context**: The akh-medu agent has working infrastructure (OODA loop, tools, goals, plans, knowledge graph, VSA, e-graph reasoning, communication channels, theory of mind microtheories) but lacks the ability to: (1) plan multi-step action sequences with predicted outcomes, (2) reason about the consequences of actions before executing them, (3) anticipate world state changes, (4) assess information trustworthiness and detect potential deception from sources with different goals.

## Research Questions

1. How should the agent plan sequences of actions and predict outcomes?
2. How should it anticipate world state changes and predict consequences of actions?
3. How should it analyse real-world information for trustworthiness and detect potential deception?
4. How do these capabilities map to the existing VSA+KG+e-graph architecture?

## Key Findings

### Planning & Prediction

#### 1. Model-Based Planning (STRIPS/PDDL/HTN)
Classical planning represents world state as propositional fluents. PDDL actions have preconditions and effects. HTN decomposes tasks into subtasks via methods. **Mapping**: KG triples *are* grounded predicates. The existing goal decomposition (Phase 8b) is a primitive HTN. Adding explicit precondition/effect triples per tool/action creates a PDDL-style domain model within the KG. E-graph rules can express domain axioms.

#### 2. Monte Carlo Tree Search (MCTS)
Builds search trees incrementally using random simulations. UCT formula balances exploration/exploitation. Recent R-MCTS (2024) adds reflection. **Mapping**: Each node = KG state snapshot. Edges = tool actions. Existing utility scoring = prior policy. Phase 8f reflection = R-MCTS reflection. Bounded by search budget per decision.

#### 3. World Models (DreamerV3, MuZero)
Internal models predict state transitions from actions. DreamerV3 uses Recurrent State-Space Models. MuZero learns representation + dynamics + prediction functions. **Mapping**: The KG is already an explicit symbolic world model. Missing piece: transition model predicting which triples change per action. VSA can learn state-action-outcome associations via binding.

#### 4. Causal Reasoning (Pearl)
Structural Causal Models with do-calculus. Three levels: association, intervention, counterfactual. **Mapping**: KG relations encode causal links. E-graph rules implement do-calculus (graph mutilation). Existing provenance chains are already causal explanations. The CausalKG framework (2022) shows how to embed interventional reasoning directly into knowledge graphs.

#### 5. Active Inference (Friston)
Agents minimize expected free energy = pragmatic value (reaching preferred states) + epistemic value (reducing uncertainty). Natural exploration-exploitation balance. **Mapping**: OODA loop already has the right structure. Utility scoring decomposes into pragmatic (base_score) and epistemic (novelty_bonus). Missing: formal prediction error tracking, surprise minimization.

#### 6. Event Calculus (Kowalski & Sergot)
Linear-time formalism for reasoning about events and fluent persistence. Core axiom: a fluent holds if some event initiated it and no event terminated it. **Mapping**: Complements Phase 13f calendar/Allen algebra. E-graph rules implement Initiates/Terminates/HoldsAt. Timestamps on triples already capture temporal dimension.

#### 7. Temporal Difference Learning
Online value estimation via bootstrapping: V(s) += alpha * [r + gamma*V(s') - V(s)]. Dyna-Q combines real experience with simulated model-based updates. **Mapping**: Each goal state gets a learned value estimate. Reward = goal progress. Enables long-horizon value estimation without full trajectory simulation.

#### 8. Game-Theoretic Planning
Multi-agent interaction modeling. Signaling games for communication under strategic incentives. Level-k thinking for bounded-rational opponents. **Mapping**: InterlocutorProfile + trust levels provide per-agent models. Argumentation framework (Phase 9) handles conflicting interests. Capability tokens (Phase 12g) manage cooperative trust.

### Epistemic Reasoning & Social Analysis

#### 9. Epistemic Logic (Kripke Models, DEL)
Modal operators K_i (knows), B_i (believes). Kripke models: possible worlds + accessibility relations. Dynamic Epistemic Logic models information change. **Mapping**: Per-interlocutor microtheories = possible worlds. E-graph rules for K-axiom, positive introspection, announcement updates. Recent "belief traps" (KR 2025) and GNN heuristics improve scalability.

#### 10. Dempster-Shafer Theory
Generalizes probability with explicit ignorance. Basic Probability Assignment on hypothesis subsets. Belief (lower bound) vs Plausibility (upper bound). Dempster's rule combines independent evidence sources. **Mapping**: Extend triple confidence from single value to (Belief, Plausibility) interval. Mass functions on {True, False, Unknown} per claim. Directly addresses "could this information be totally false?"

#### 11. Argumentation (ASPIC+, DeLP)
Structured argumentation: arguments are inference trees with strict/defeasible rules. Three attack types: undermining, rebutting, undercutting. Preferences resolve attacks into defeats. **Mapping**: Phase 9 already implements Dung frameworks. ASPIC+ extends with structured arguments from derivation chains. Source credibility = preference ordering. Proven in Netherlands Police criminal investigation.

#### 12. Source Reliability (Admiralty System + ACH)
NATO Admiralty Code: source reliability (A-F) x information credibility (1-6). Analysis of Competing Hypotheses: evidence-hypothesis matrix with disconfirmation focus. **Mapping**: Add Admiralty ratings to InterlocutorProfile. ACH as structured analysis tool. Very low computational cost.

#### 13. Deception Detection
Consistency checking against KG facts. Source behavior modeling over time. Linguistic cue detection. Causal chain plausibility. **Mapping**: Phase 12c constraint checking already does consistency checks. Extend with temporal source behavior tracking and argument evaluation.

#### 14. Bayesian Trust Networks
Probabilistic graphical models for trust: competence, benevolence, integrity dimensions. Dynamic Bayesian updating from interaction outcomes. **Mapping**: InterlocutorProfile statistics provide evidence nodes. Bayesian trust update after each prediction/claim verification cycle.

#### 15. Computational Theory of Mind
Level-k recursive belief modeling. BDI (Belief-Desire-Intention) architecture. Recent work: GPT-4 achieves 5th-6th order belief reasoning. **Mapping**: Per-interlocutor microtheories represent Level-1 ToM. Nested binding for Level-2+: bind(AGENT_I, bind(BELIEVES, bind(AGENT_J, bind(BELIEVES, proposition)))).

#### 16. Credibility Assessment
Eight signal categories (2024 survey of 175 papers): factuality, bias, persuasion techniques, check-worthiness, text quality, references, originality, toxicity. **Mapping**: Credibility signals as KG predicates. OnlineHD classifier pattern (Phase 13b) for credibility classification.

### VSA Integration

The Strathclyde 2024 work demonstrated a full OODA loop implementation using VSA for planning via cognitive maps, reasoning via semantic vector spaces, and communication via compact binary vectors. VSA cognitive map learners (CMLs) encode graph structures as hypervectors and can compose modularly (Tower of Hanoi solved by composing independently-trained CMLs).

**Key VSA planning techniques**:
- Sequence encoding: plan_vec = permute^0(step1) + permute^1(step2) + ...
- State prediction: bind(state, action) -> transition_memory lookup
- Epistemic states: world_vec = bind(FLUENT_ROLE, val) bundles

## Architectural Recommendation

Seven phases covering causal world model, predictive planning (MCTS), Dempster-Shafer evidence theory, source reliability/ACH, epistemic logic, active inference OODA enhancement, and game-theoretic social reasoning. Each phase builds on the previous, and all integrate through the existing VSA+KG+e-graph+OODA infrastructure.

See implementation plans: `docs/ai/plans/2026-02-22-phase15-causal-world-model.md` through `docs/ai/plans/2026-02-22-phase21-game-theoretic-reasoning.md`.

## Key References

- Geffner & Bonet, "A Concise Introduction to Models and Methods for Automated Planning" (2013)
- Silver et al., "MuZero: Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (2019)
- Hafner et al., "DreamerV3: Mastering Diverse Domains through World Models" (Nature 2025)
- Pearl, "Causality: Models, Reasoning, and Inference" (2009)
- Friston et al., "Active Inference on Discrete State Spaces: A Synthesis" (2020)
- Kowalski & Sergot, "A Logic-based Calculus of Events" (1986)
- van Ditmarsch et al., "Dynamic Epistemic Logic" (SEP)
- Shafer, "A Mathematical Theory of Evidence" (1976)
- Modgil & Prakken, "The ASPIC+ Framework for Structured Argumentation" (2014)
- Heuer, "Psychology of Intelligence Analysis" (1999)
- Kautz et al., "Cognitive Map Learners with HDC" (2024)
- Strathclyde, "VSA OODA Loop Demonstration" (2024)
- Belief Traps (KR 2025), GNN Epistemic Planning Heuristics (2025)
- R-MCTS / ExACT (2024), SPIRAL (2024)
- CausalKG (Jaimini & Kaul 2022)
