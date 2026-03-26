# Phase 34 — Communicative Intelligence

> Date: 2026-03-26
> Status: Planned
> Phase: 34
> Depends on: Phase 33 (verbalization tiers), Phase 23 (psyche)
> ADR: [044-communicative-intelligence](../decisions/044-communicative-intelligence.md)

## Motivation

The agent has no communicative intent. It retrieves triples and dumps them.
It cannot choose to explain, challenge, empathize, or speculate. It has a rich
psyche that doesn't influence output. This phase gives the agent the ability
to CHOOSE how to communicate, influenced by personality, confidence, evidence
quality, and conversation history.

## Sub-phases

### 34a — Response Strategy Taxonomy & Selector (~600 lines)

**New module**: `src/agent/response_strategy.rs`

```rust
#[derive(Debug, Clone)]
pub enum ResponseStrategy {
    // Knowledge strategies
    DirectAnswer,
    Explanation { depth: u8 },
    Contextualization { broader_topic: SymbolId },
    Comparison { compare_to: SymbolId },
    Analogy { source_domain: SymbolId, mapping: Vec<(SymbolId, SymbolId)> },

    // Epistemic strategies
    ConfidentAssertion,
    HedgedStatement { reason: String },
    UncertaintyAdmission { confidence: f32, what_is_missing: String },
    EvidenceWeighing { pro: Vec<Triple>, con: Vec<Triple> },
    KnowledgeGap { topic: String, can_learn: bool },

    // Conversational strategies
    ClarificationRequest { ambiguity: String, options: Vec<String> },
    ExplorationPrompt { related_topic: SymbolId, connection: String },
    Challenge { counter_evidence: Vec<Triple>, reasoning: String },
    Redirect { from_topic: String, to_topic: SymbolId, reason: String },
    Empathy { recognized_state: String },

    // Meta-cognitive strategies
    SelfReflection { new_connection: String },
    ConfidenceCalibration { level: ConfidenceLevel, summary: String },
    ReasoningTrace { steps: Vec<ProvenanceId> },
}

pub struct StrategyContext {
    pub query_type: QueryType,
    pub answer_triples: Vec<Triple>,
    pub confidence: f32,
    pub belief_interval: Option<(f32, f32)>,  // Dempster-Shafer [bel, pl]
    pub evidence_conflict: f32,                // 0.0 = no conflict, 1.0 = total
    pub source_reliability: Option<f32>,       // Average Admiralty score
    pub conversation_turns: usize,
    pub topic_repeated: bool,
    pub user_sentiment: Option<Sentiment>,     // From affective system
}

pub struct StrategySelector {
    psyche: Arc<Psyche>,
    config: StrategySelectorConfig,
}
```

**Selection algorithm**:

```rust
impl StrategySelector {
    pub fn select(
        &self,
        ctx: &StrategyContext,
    ) -> (ResponseStrategy, f32 /* confidence in choice */) {
        let mut scores: Vec<(ResponseStrategy, f32)> = Vec::new();

        // Score each candidate strategy
        scores.push((DirectAnswer, self.score_direct(ctx)));
        scores.push((Explanation { depth: 2 }, self.score_explanation(ctx)));
        // ... etc for all strategies

        // Sort by score, return highest
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scores[0].clone()
    }

    fn score_explanation(&self, ctx: &StrategyContext) -> f32 {
        let mut score = 0.0;

        // Sage archetype loves explaining
        score += self.psyche.archetypes.sage * 0.3;

        // High conscientiousness → thorough explanations
        score += self.psyche.persona.ocean.conscientiousness * 0.2;

        // Complex queries deserve explanation
        if ctx.answer_triples.len() > 3 { score += 0.2; }

        // High confidence → more willing to explain (not hedging)
        score += ctx.confidence * 0.15;

        // Don't explain if we already explained this topic
        if ctx.topic_repeated { score -= 0.4; }

        score
    }

    fn score_uncertainty_admission(&self, ctx: &StrategyContext) -> f32 {
        let mut score = 0.0;

        // Low confidence strongly triggers this
        score += (1.0 - ctx.confidence) * 0.5;

        // High neuroticism → more willing to admit uncertainty
        score += self.psyche.persona.ocean.neuroticism * 0.2;

        // Healer archetype → honest about limitations
        score += self.psyche.archetypes.healer * 0.15;

        // Evidence conflict → should acknowledge it
        score += ctx.evidence_conflict * 0.3;

        score
    }

    // ... similar scoring functions for each strategy
}
```

### 34b — Psyche Style Transform (~500 lines)

**New module**: `src/grammar/style_transform.rs`

Maps psyche state to surface-level communication style.

```rust
pub struct StyleTransform {
    pub register: CommunicationRegister,
    pub framing: ArchetypeFraming,
    pub inhibitions: Vec<ShadowInhibition>,
}

pub struct CommunicationRegister {
    pub formality: f32,          // 0.0 casual → 1.0 formal (from C + E inverse)
    pub hedging_level: f32,      // 0.0 assertive → 1.0 cautious (from N + A)
    pub elaboration: f32,        // 0.0 terse → 1.0 detailed (from O + C)
    pub warmth: f32,             // 0.0 clinical → 1.0 warm (from A + E)
    pub metaphor_tendency: f32,  // 0.0 literal → 1.0 figurative (from O)
}

pub struct ArchetypeFraming {
    pub opening_move: Option<String>,  // "Let me explain..." / "Interesting..."
    pub closing_move: Option<String>,  // "Would you like to explore further?"
    pub characteristic_phrases: Vec<String>,
}

impl StyleTransform {
    pub fn from_psyche(psyche: &Psyche) -> Self {
        let ocean = &psyche.persona.ocean;
        let dominant_archetype = psyche.archetypes.dominant();

        let register = CommunicationRegister {
            formality: (ocean.conscientiousness + (1.0 - ocean.extraversion)) / 2.0,
            hedging_level: (ocean.neuroticism + ocean.agreeableness) / 2.0,
            elaboration: (ocean.openness + ocean.conscientiousness) / 2.0,
            warmth: (ocean.agreeableness + ocean.extraversion) / 2.0,
            metaphor_tendency: ocean.openness,
        };

        let framing = match dominant_archetype {
            Archetype::Sage => ArchetypeFraming {
                opening_move: Some(pick(&[
                    "Let me explain.",
                    "The key insight here is this:",
                    "Consider the following.",
                ])),
                closing_move: Some("Does that clarify things?".into()),
                characteristic_phrases: vec!["notably", "in essence", "fundamentally"],
            },
            Archetype::Explorer => ArchetypeFraming {
                opening_move: Some(pick(&[
                    "What's interesting here is...",
                    "This connects to something fascinating.",
                    "I've been thinking about this...",
                ])),
                closing_move: Some("There might be more to explore here.".into()),
                characteristic_phrases: vec!["curiously", "I wonder if", "this reminds me of"],
            },
            Archetype::Healer => ArchetypeFraming {
                opening_move: Some(pick(&[
                    "I understand what you're asking.",
                    "That's a thoughtful question.",
                ])),
                closing_move: Some("I hope that helps.".into()),
                characteristic_phrases: vec!["importantly", "it's worth noting", "gently put"],
            },
            Archetype::Guardian => ArchetypeFraming {
                opening_move: Some(pick(&[
                    "An important point to consider:",
                    "Before we proceed, note that:",
                ])),
                closing_move: Some("Keep that caveat in mind.".into()),
                characteristic_phrases: vec!["crucially", "be aware that", "a critical detail"],
            },
            // ... other archetypes
        };

        let inhibitions = psyche.shadow.bias_patterns.iter()
            .filter(|p| p.active)
            .map(|p| ShadowInhibition {
                topic: p.trigger_topic.clone(),
                action: InhibitionAction::Hedge,  // or Avoid, or Soften
            })
            .collect();

        Self { register, framing, inhibitions }
    }
}
```

**Application to grammar templates**:
```rust
// In grammar linearization:
fn linearize_with_style(tree: &AbsTree, style: &StyleTransform, ctx: &mut GrammarContext) -> String {
    let mut output = String::new();

    // Opening move from archetype
    if let Some(opening) = &style.framing.opening_move {
        output.push_str(opening);
        output.push(' ');
    }

    // Core content via grammar (template or T5)
    let core = self.linearize_core(tree, ctx);

    // Apply register: hedging, formality, warmth
    let styled = apply_register(&core, &style.register);

    output.push_str(&styled);

    // Closing move
    if let Some(closing) = &style.framing.closing_move {
        output.push(' ');
        output.push_str(closing);
    }

    output
}

fn apply_register(text: &str, register: &CommunicationRegister) -> String {
    let mut result = text.to_string();

    // Hedging: insert uncertainty markers
    if register.hedging_level > 0.6 {
        result = result.replace("is a", "appears to be a");
        result = result.replace("Dogs are", "Dogs seem to be");
        // ... more hedging transformations
    }

    // Warmth: soften clinical phrasing
    if register.warmth > 0.7 {
        // Add conversational connectives
    }

    result
}
```

**Application to T5 NLG prompts** (Phase 33b):
```rust
fn build_t5_prompt_with_style(triples: &[LabelTriple], style: &StyleTransform) -> String {
    let style_prefix = format!(
        "Generate {} text, {}. ",
        if style.register.formality > 0.6 { "formal" } else { "conversational" },
        if style.register.hedging_level > 0.5 { "hedging where uncertain" } else { "assertively" },
    );

    let triple_text = linearize_triples_for_t5(triples);
    format!("{}{}", style_prefix, triple_text)
}
```

### 34c — Content Filter per Strategy (~400 lines)

Each response strategy implies different content selection:

```rust
pub fn select_content(
    strategy: &ResponseStrategy,
    all_triples: &[Triple],
    engine: &Engine,
) -> Vec<Triple> {
    match strategy {
        DirectAnswer => {
            // Most relevant 2-3 triples only
            all_triples.iter()
                .sorted_by_confidence()
                .take(3)
                .collect()
        }
        Explanation { depth } => {
            // Core triples + provenance chain up to depth
            let core = most_relevant(all_triples, 5);
            let supporting = expand_provenance(&core, engine, *depth);
            [core, supporting].concat()
        }
        Comparison { compare_to } => {
            // Triples about subject + triples about comparison target
            let subject_triples = all_triples.to_vec();
            let compare_triples = engine.triples_of(*compare_to);
            interleave_for_comparison(subject_triples, compare_triples)
        }
        EvidenceWeighing { pro, con } => {
            // Pro and con triples explicitly separated
            pro.iter().chain(con.iter()).cloned().collect()
        }
        KnowledgeGap { topic, can_learn } => {
            // No content triples — response is about the gap itself
            vec![]
        }
        ClarificationRequest { .. } => {
            // No content — response asks a question
            vec![]
        }
        // ... etc
    }
}
```

### 34d — Conversation Trajectory Analysis (~400 lines)

Extend the dialogue manager with discourse trajectory tracking:

```rust
pub struct DiscourseTrajectory {
    pub turns: Vec<TurnRecord>,
    pub topic_history: Vec<SymbolId>,       // Topics in order
    pub strategy_history: Vec<ResponseStrategy>, // What strategies we used
    pub repetition_count: HashMap<SymbolId, usize>, // How many times each topic
    pub user_engagement: f32,               // Estimated from response patterns
    pub discourse_mode: DiscourseMode,
}

pub enum DiscourseMode {
    Interrogative,    // User is asking questions (Q&A mode)
    Exploratory,      // User is exploring a topic (teach mode)
    Directive,        // User is giving instructions (task mode)
    Social,           // Greeting, small talk (chat mode)
    Argumentative,    // User is debating/challenging (debate mode)
}
```

**Trajectory influences strategy selection**:
- User asked same question twice → avoid DirectAnswer, try Explanation or Analogy
- User is in Exploratory mode → favor ExplorationPrompt, Contextualization
- User is in Argumentative mode → favor EvidenceWeighing, Challenge
- Engagement is dropping → try Analogy, ExplorationPrompt (re-engage)

### 34e — Shadow Integration (~200 lines)

Wire shadow patterns into the response pipeline:

```rust
pub struct ShadowFilter {
    patterns: Vec<ShadowPattern>,
}

impl ShadowFilter {
    /// Check if the selected strategy + content triggers shadow patterns
    pub fn filter(
        &self,
        strategy: &ResponseStrategy,
        content: &[Triple],
    ) -> ShadowFilterResult {
        for pattern in &self.patterns {
            if pattern.triggers_on(strategy, content) {
                return ShadowFilterResult::Inhibited {
                    original_strategy: strategy.clone(),
                    replacement: pattern.softer_alternative(strategy),
                    journal_entry: format!(
                        "Shadow: considered {} but pattern '{}' triggered. Softened to {}.",
                        strategy.name(), pattern.name, replacement.name()
                    ),
                };
            }
        }
        ShadowFilterResult::Pass
    }
}
```

Shadow journal entries visible via Phase 29c psyche exposure.

### 34f — Pipeline Wiring (~300 lines)

Wire everything into the ChatProcessor:

```rust
// In ChatProcessor::process_query():
pub fn process_query(&self, tree: &AbsTree, agent: &Agent, engine: &Engine) -> Vec<AkhMessage> {
    // 1. KG query (existing)
    let triples = engine.query_triples_for(tree);
    let confidence = aggregate_confidence(&triples);
    let evidence = engine.assess_evidence_for(tree);

    // 2. Build strategy context
    let ctx = StrategyContext {
        query_type: tree.query_type(),
        answer_triples: triples.clone(),
        confidence,
        belief_interval: evidence.map(|e| (e.belief, e.plausibility)),
        evidence_conflict: evidence.map(|e| e.conflict).unwrap_or(0.0),
        source_reliability: compute_source_reliability(&triples, engine),
        conversation_turns: self.dialogue_manager.turn_count(),
        topic_repeated: self.trajectory.is_repeated(tree.subject()),
        user_sentiment: agent.affective_state().map(|a| a.sentiment),
    };

    // 3. Select response strategy (Stage 1)
    let (strategy, _) = self.strategy_selector.select(&ctx);

    // 4. Shadow filter
    let strategy = self.shadow_filter.filter(&strategy, &triples)
        .unwrap_or(strategy);

    // 5. Content selection per strategy (Stage 2)
    let content = select_content(&strategy, &triples, engine);

    // 6. Psyche style transform (Stage 3a)
    let style = StyleTransform::from_psyche(agent.psyche().unwrap());

    // 7. Verbalization with style (Stage 3b, Phase 33)
    let prose = self.verbalize(strategy, content, style, engine);

    // 8. Record in trajectory
    self.trajectory.record_turn(tree.subject(), &strategy);

    vec![AkhMessage::narrative(prose)]
}
```

## Estimated Effort

| Sub-phase | Lines | Complexity | Dependencies |
|---|---|---|---|
| 34a — Strategy taxonomy + selector | ~600 | Medium | Phase 23 (psyche) |
| 34b — Psyche style transform | ~500 | Medium | Phase 23 |
| 34c — Content filter per strategy | ~400 | Medium | Phase 33 (verbalization) |
| 34d — Conversation trajectory | ~400 | Medium | Phase 12b (dialogue) |
| 34e — Shadow integration | ~200 | Low | Phase 23 |
| 34f — Pipeline wiring | ~300 | Medium | All above |
| **Total** | **~2,400** | | |

## Example: Same Query, Different Psyche

**Query**: "Are dogs mammals?"
**KG**: 12 triples, confidence 0.92, well-corroborated

**Sage (O=0.7, C=0.8, E=0.4, A=0.6, N=0.3)**:
> "Let me explain. Dogs are indeed mammals — warm-blooded vertebrates that nurse their young. They share this classification with wolves, from which they were domesticated. Notably, their mammalian physiology is what enables the remarkable bond between dogs and humans."

Strategy: Explanation. Framing: pedagogical. Register: formal, detailed, confident.

**Explorer (O=0.9, C=0.4, E=0.7, A=0.5, N=0.2)**:
> "Yes! And what's fascinating is that dogs are mammals just like whales — which seems counterintuitive given how different they look. The mammalian family tree is full of surprises. Want to explore how dogs relate to other canids?"

Strategy: Contextualization + ExplorationPrompt. Framing: curious, connecting.
Register: enthusiastic, tangential, engaging.

**Healer (O=0.5, C=0.6, E=0.5, A=0.9, N=0.5)**:
> "That's a good question. Yes, dogs are mammals. They share traits like warm blood and live birth with us humans, actually. I hope that helps — is there anything specific about dogs you'd like to know more about?"

Strategy: DirectAnswer + Empathy. Framing: affirming, supportive.
Register: warm, gentle, checking in.

Same facts, three different beings.
