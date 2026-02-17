# ADR-004: Personal Assistant Research — Email Processing, Planning, Preference Learning, and Structured Output

- **Date**: 2026-02-17
- **Status**: Accepted
- **Context**: Deep research for Phase 13, specifically how to build personal assistant capabilities (email, planning, preference learning, dashboards) using the VSA + KG + e-graph + OODA architecture without LLM dependency

## Research Question

How can a neuro-symbolic engine provide personal assistant capabilities — email processing, spam detection, task/calendar management, preference learning, and structured operator dashboards — entirely on-device, with full provenance and no LLM?

## Key Findings

### 1. Email Spam Detection via VSA (OnlineHD)

**OnlineHD** (DATE 2021) demonstrates single-pass online learning for hyperdimensional classification. The approach maintains class prototype vectors (spam/ham) and classifies by Hamming similarity. With 10,000-bit binary vectors, this provides high-capacity classification with single-pass incremental updates — no batch retraining needed.

**SpamBayes** (Fisher chi-square method) combines individual token probabilities using Fisher's method to produce a single spam score. Gary Robinson's "A Statistical Approach to the Spam Problem" (2003) refined this by using the geometric mean of probabilities, solving the rare-word problem.

**SpamAssassin** uses a hybrid approach: header analysis rules, body content rules, network tests, and Bayesian scoring. The multi-rule architecture maps directly to e-graph rewrite rules where all applicable rules fire simultaneously.

**Mapping to akh-medu**: Encode each email as a composite hypervector using role-filler bindings (SenderRole XOR sender_hv, SubjectRole XOR subject_hv, etc.). Maintain spam/ham prototype vectors. Classification uses existing `VsaOps::similarity` (Hamming distance). E-graph rules handle deterministic cases (trusted sender → ham, failed DKIM → likely spam). User feedback incrementally updates prototypes via OnlineHD adaptive weighting.

### 2. Email Priority and Triage (Gmail Priority Inbox / HEY)

**Gmail Priority Inbox** (Google Research) uses four feature categories: social features (interaction frequency, reply rate), content features (recently-acted-on terms), thread features (user started thread, previous replies), and label features. A two-layer model (global baseline + personal delta) achieves ~80% accuracy.

**HEY.com** uses explicit consent-based screening: first-time senders require user approval, then route to Imbox (important), Feed (newsletters), or Paper Trail (transactional). This demonstrates that explicit KG triples (sender → routing decision) are a powerful complement to statistical classification.

**Mapping to akh-medu**: Build a sender reputation subgraph in Oxigraph (reply rate, recency, frequency, relationship type). Encode sender interaction signatures as VSA composite vectors. Priority prototype vectors (built from emails the user acts on quickly) enable similarity-based importance scoring. HEY-style screening maps to operator approval via CommChannel.

### 3. JMAP as Primary Email Protocol

**JMAP** (RFC 8620 + RFC 8621) is the clear winner for programmatic email access: JSON-native (no MIME parsing on client), batched operations in single HTTP requests, efficient delta sync via `Email/changes`, and push notifications via EventSource/WebSocket. The stateless HTTP model fits the synchronous OODA loop.

**Practical constraint**: JMAP adoption is limited (Fastmail primary). Gmail and Outlook still require IMAP. Support both: JMAP primary, IMAP fallback.

**Rust crates**: `jmap-client` (Stalwart Labs, full RFC compliance), `imap` (sync), `mail-parser` (RFC 5322/MIME, safe Rust, zero-copy).

### 4. Email Threading (JWZ Algorithm)

The **JWZ threading algorithm** (Jamie Zawinski, basis of RFC 5256) builds conversation trees from Message-ID, In-Reply-To, and References headers. "Incredibly robust in the face of garbage input" — field-tested by ~10 million users. Thread participation is a strong priority signal.

### 5. Structured Extraction from Email

Rule-based NER using three complementary techniques:
- **Pattern matching**: Dates, times, phone numbers, tracking numbers (carrier-specific patterns)
- **Gazetteer lookup**: City names, organization names, month/day names
- **Contextual rules**: "Meeting with [NAME] on [DATE] at [TIME] in [LOCATION]"

These rules can be expressed as KG patterns and e-graph rewrite rules rather than hardcoded regex. Extracted entities become KG triples with provenance. Action items convert to agent goals.

### 6. GTD + Eisenhower + PARA for Task Management

**GTD** (David Allen) provides the workflow engine: Inbox → Clarify → Organize → Engage → Review. Five states map to well-known predicates: `task:status → {inbox, next, waiting, someday, reference}`.

**Eisenhower Matrix** (urgent × important) maps to VSA composite vectors: `task_priority = bundle(bind(V_urgent, level), bind(V_important, level))`. Similarity search finds tasks with similar priority profiles.

**PARA** (Tiago Forte) organizes by actionability: Projects (deadline), Areas (standard), Resources (reference), Archive (inactive). Maps to KG predicates with the existing compartment system.

**Mapping to akh-medu**: Define `PimPredicates` (personal information management) analogous to `AgentPredicates`. Task dependencies as petgraph DAG with topological sort for execution order. Recurring tasks via RRULE-style recurrence strings. Bullet Journal migration maps to the existing reflection system.

### 7. Calendar and Temporal Reasoning

**RFC 5545 (iCalendar)** defines VEVENT, VTODO, VALARM, RRULE. A CalDAV sync adapter parses `.ics` files into Entity symbols and triples, following the library parser pattern.

**Allen's Interval Algebra** (1983) defines 13 relations between time intervals (before, meets, overlaps, during, starts, finishes, equals + inverses). Each becomes a well-known predicate enabling SPARQL queries like "Find all events overlapping with my meeting."

**E-graph temporal rules**: Before-transitivity, meets-before composition, conflict detection (overlapping events requiring exclusive resources).

**Scheduling as CSP**: Hard constraints (physical impossibility), soft constraints (preferences), energy constraints (chronotype). The existing petgraph + SPARQL infrastructure handles personal scheduling without an external solver.

### 8. Smart Notifications (Interruption Research)

**Iqbal & Bailey (2008)**: Scheduling notifications at task breakpoints significantly reduces frustration and resumption lag. **OASIS** (Microsoft Research) implemented defer-to-breakpoint policies.

**Key principle**: Coarser breakpoints (between major activities) have lower interruption cost. Cognitive load limits: max 5-7 items per notification batch (Miller's Law, revised to ~4 by Cowan).

**Mapping to akh-medu**: Extend `TriggerCondition` with breakpoint, batch, urgency, and context conditions. VSA context matching for fuzzy trigger evaluation. Rate-limited notification delivery.

### 9. Preference Learning (HyperRec + Temporal Decay)

**HyperRec** (IEEE ASP-DAC 2021) uses hyperdimensional computing for recommendation: encode user-item interaction profiles as hypervectors, classify via prototype similarity. Achieves competitive accuracy with single-pass learning.

**Temporal dynamics**: Koren (CACM 2009) showed user preferences evolve. Adaptive collaborative filtering with personalized time decay weights recent interactions higher. VSA natively supports this: older interaction vectors get lower weight in the bundle.

**Cold start**: Addressed by multi-layered approach — onboarding goals, library ingestion as interest signal, goal-based bootstrapping, and VSA serendipity (HNSW near-miss matches).

**Mapping to akh-medu**: VSA user preference profiles with role-filler bindings (topic × interest-level). Implicit signals (reply speed, reading time, action taken) as KG triples. Explicit preferences from operator input. Temporal decay via weighted bundling.

### 10. Proactive Information Management

**Remembrance Agent** (Rhodes & Starner, MIT 1996): Continuous just-in-time information retrieval based on current context. Users "actually retrieve and use more information than they would with traditional search engines."

**Serendipity**: Research (RecSys 2023) requires three properties: relevant, novel, unexpected. HNSW naturally promotes serendipity — items at intermediate Hamming distance share some features but introduce novel dimensions. The "serendipity zone" (distance 0.3-0.6) captures meaningfully related but non-obvious connections.

**Graduated proactivity**: Research warns that unsolicited proactive behavior can feel threatening. Five-level model: (1) ambient suggestions, (2) gentle nudges, (3) contextual offers, (4) scheduled actions with approval, (5) autonomous execution. Default to Level 1; escalate only with explicit opt-in.

### 11. Structured Output for Operator Dashboards

**JSON-LD** is the natural bridge between the engine's RDF triples and dashboard consumption. Oxigraph already stores RDF; JSON-LD is its JSON serialization.

**Three-tier summarization**: Daily summary (quick hits, 5-7 items max), weekly digest (pattern synthesis), monthly review (strategic view).

**Compartment-organized dashboards**: Each compartment shows 4±1 top-level items (Cowan's limit). Progressive disclosure for drill-down.

**Notification structure**: Priority + context + action + provenance + expiry. The trigger system already implements event-driven behavior; notifications are a natural output format.

### 12. Delegated Agent Spawning

The operator can create specialized agents with their own identities for domain-specific assistance. Each delegated agent gets:
- **Own identity**: Name, description, capability set
- **Scoped knowledge**: Access to specific compartments from operator's KG
- **Dedicated CommChannel**: Forwarded messages route through operator
- **Lifecycle management**: Create, pause, terminate, merge knowledge back

This maps to: spawning a new `Agent` instance with a restricted `CompartmentManager` and a `CommChannel` configured with `ChannelKind::Trusted` capabilities.

### 13. Bidirectional Email (Compose + Send)

Email as a full CommChannel means the agent can compose and send, not just read and classify. The composition pipeline: grammar module generates prose from KG-grounded content → constraint checking validates before send → capability model controls approval flow:
- **Social level**: Draft for operator approval
- **Trusted level**: Send after approval
- **Delegated rule**: Auto-send for learned patterns (e.g., routine acknowledgments)

## Decision

Create Phase 13 with nine sub-phases:

### 13a — Email Channel (JMAP/IMAP + MIME)
CommChannel implementation for email. JMAP primary with IMAP fallback. MIME parsing via `mail-parser`. JWZ threading. Email entities with `email:` predicates.

### 13b — Spam & Relevance Classification
OnlineHD prototype-based spam/ham classification (VSA-native). Bayesian token probabilities (redb-durable). KG sender reputation graph. E-graph spam reasoning rules. User feedback training loop.

### 13c — Email Triage & Priority
Sender importance model (4-feature: social, content, thread, label). HEY-style screening via operator approval. VSA priority prototypes. SPARQL sender statistics. Priority inbox routing.

### 13d — Structured Extraction from Messages
Rule-based NER (dates, times, locations, tracking numbers). Calendar event extraction → KG entities. Action item extraction → agent goals. E-graph extraction rules.

### 13e — Personal Task & Project Management
GTD workflow states. PARA categorization. Eisenhower matrix with VSA urgency/importance. Task dependencies as petgraph DAG. Recurring tasks (RRULE). Bullet Journal-style migration in reflection.

### 13f — Calendar & Temporal Reasoning
iCalendar parser (RFC 5545). Allen interval algebra as KG predicates. Temporal e-graph rules. Scheduling as constraint satisfaction. Proactive scheduling via OODA.

### 13g — Preference Learning & Proactive Assistance
VSA preference profiles (HyperRec-style, temporal decay). Implicit feedback tracking. JITIR (Remembrance Agent pattern). Serendipity engine (near-miss HNSW). Context-aware reminders with breakpoint timing.

### 13h — Structured Output & Operator Dashboards
JSON-LD export. Daily/weekly/monthly briefing tiers. Notification system (priority/context/action/provenance/expiry). Compartment-organized dashboard data. Machine-readable SPARQL endpoints.

### 13i — Delegated Agent Spawning
Agent identity creation with scoped knowledge compartments. CommChannel forwarding for operator-routed messages. Agent-specific outbound identity. Lifecycle management (create, pause, terminate, merge knowledge). Email composition pipeline via grammar module + constraint checking.

## Sources

### Email Processing & Spam Detection
- [Graham, P. "A Plan for Spam" (2002)](https://paulgraham.com/spam.html)
- [Robinson, G. "A Statistical Approach to the Spam Problem" (2003)](https://www.linuxjournal.com/article/6467)
- [SpamBayes Project](https://spambayes.sourceforge.io/)
- [SpamAssassin Bayes Documentation](https://cwiki.apache.org/confluence/display/spamassassin/BayesInSpamAssassin)
- [Imani, M. et al. "OnlineHD" (DATE 2021)](https://ics.uci.edu/~mohseni/papers/DATE21_OnlineHD_Imani.pdf)
- [Najafabadi & Rahimi. "HDC for Text Classification"](https://www.semanticscholar.org/paper/35e5e934f8d427beb8bd23b336fdc235c04ff860)

### Email Prioritization & Triage
- [Aberdeen & Pacovsky. "Gmail Priority Inbox" (Google Research)](https://research.google.com/pubs/archive/36955.pdf)
- [HEY — How It Works](https://www.hey.com/how-it-works/)
- [Superhuman Split Inbox](https://help.superhuman.com/hc/en-us/articles/38449611367187-Split-Inbox-Basics)
- [Content-Based Email Triage Action Prediction (arXiv 2019)](https://arxiv.org/abs/1905.01991)
- [Enterprise Email Reply Behavior (SIGIR 2017)](https://cseweb.ucsd.edu/classes/fa17/cse291-b/reading/sigir17a_email.pdf)
- [Zawinski, J. "Message Threading" (JWZ Algorithm)](https://www.jwz.org/doc/threading.html)

### Email Protocols & Rust Crates
- [JMAP Specification (RFC 8620/8621)](https://jmap.io/spec-mail.html)
- [Fastmail JMAP Blog](https://www.fastmail.com/blog/jmap-new-email-open-standard/)
- [jmap-client Rust Crate (Stalwart Labs)](https://github.com/stalwartlabs/jmap-client)
- [imap Rust Crate](https://github.com/jonhoo/rust-imap)
- [mail-parser Rust Crate](https://docs.rs/mail-parser/)
- [Notmuch Mail](https://notmuchmail.org/)

### Task Management Methodologies
- [Getting Things Done (GTD)](https://en.wikipedia.org/wiki/Getting_Things_Done)
- [Eisenhower Matrix](https://www.eisenhower.me/eisenhower-matrix/)
- [PARA Method (Forte Labs)](https://fortelabs.com/blog/para/)
- [Bullet Journal Method](https://bulletjournal.com/pages/how-to-bullet-journal)
- [Building a Second Brain (CODE Framework)](https://www.buildingasecondbrain.com/)

### Calendar & Temporal Reasoning
- [RFC 5545 — iCalendar](https://datatracker.ietf.org/doc/html/rfc5545)
- [RFC 6638 — CalDAV Scheduling](https://datatracker.ietf.org/doc/html/rfc6638)
- [Allen, J.F. "Temporal Intervals" (CACM 1983)](https://cse.unl.edu/~choueiry/Documents/Allen-CACM1983.pdf)
- [OWL-Time Ontology](https://www.semantic-web-journal.net/system/files/swj1118.pdf)
- [Bartak, R. "Constraint Programming for Scheduling"](https://files.core.ac.uk/download/pdf/232825873.pdf)

### Interruption & Notification Research
- [Iqbal & Bailey. "Intelligent Notification Management" (CHI 2008)](https://interruptions.net/literature/Iqbal-CHI08.pdf)
- [Horvitz & Iqbal. "Disruption and Recovery" (CHI 2007)](http://erichorvitz.com/CHI_2007_Iqbal_Horvitz.pdf)
- [OASIS Notification Framework (Microsoft Research)](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/TOCHI-Oasis-final.pdf)
- [Miller, G.A. "The Magical Number Seven" (1956)](https://en.wikipedia.org/wiki/The_Magical_Number_Seven,_Plus_or_Minus_Two)

### Preference Learning & Recommendation
- [HyperRec (IEEE ASP-DAC 2021)](https://ieeexplore.ieee.org/abstract/document/9371652)
- [Koren. "Collaborative Filtering with Temporal Dynamics" (CACM 2009)](https://cacm.acm.org/research/collaborative-filtering-with-temporal-dynamics/)
- [KG-Based Recommendation Introduction](https://towardsdatascience.com/introduction-to-knowledge-graph-based-recommender-systems-34254efd1960/)
- [Serendipity in Recommender Systems (RecSys 2023)](https://dl.acm.org/doi/fullHtml/10.1145/3576840.3578310)

### Proactive & Context-Aware Computing
- [Rhodes & Starner. "Remembrance Agent" (MIT 1996)](https://www.bradleyrhodes.com/Papers/remembrance.html)
- [Pejovic & Musolesi. "Anticipatory Mobile Computing" (ACM CS 2015)](https://dl.acm.org/doi/10.1145/2693843)
- [Dey & Abowd. "Context Toolkit"](https://contexttoolkit.sourceforge.net/)
- [Proactive Information Retrieval (ACM TIIS)](https://dl.acm.org/doi/10.1145/3150975)

### Intelligent Assistant UX
- [NN/g. "Intelligent Assistants Have Poor Usability"](https://www.nngroup.com/articles/intelligent-assistant-usability/)
- [Luger & Sellen. "Like Having a Really Bad PA" (CHI 2016)](https://dl.acm.org/doi/10.1145/2858036.2858288)
- [Proactive AI Adoption Can Be Threatening (arXiv 2025)](https://arxiv.org/abs/2509.09309)
- [Privacy International — Future AI Assistants](https://privacyinternational.org/long-read/5555/your-future-ai-assistant-still-needs-earn-your-trust)

### Personal Information Management
- [NEPOMUK Project / Semantic Desktop](https://nepomuk.semanticdesktop.org/Project+Summary.html)
- [PIMO Ontology](https://oscaf.sourceforge.net/pimo.html)
- [TMO Task Management Ontology](https://oscaf.sourceforge.net/tmo.html)
- [Bush, V. "As We May Think" (1945)](https://www.w3.org/History/1945/vbush/vbush.shtml)

### Habit & Behavior Science
- [Fogg Behavior Model (B=MAP)](https://www.behaviormodel.org/)
- [James Clear, Habit Tracker Guide](https://jamesclear.com/habit-tracker)
- [Spaced Learning Enhances Episodic Memory (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6607761/)

### Integration Standards
- [JSON-LD Primer (W3C)](https://json-ld.org/primer/latest/)
- [Micropub Specification (W3C)](https://micropub.spec.indieweb.org/)
- [ActivityPub (W3C)](https://www.w3.org/TR/activitypub/)
- [SPARQL Transformer (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3184558.3188739)
- [The API of Me (Nordic APIs)](https://nordicapis.com/the-api-of-me/)

### HDC/VSA Surveys
- [VSA Survey Part I (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3538531)
- [VSA Survey Part II (ACM Computing Surveys)](https://dl.acm.org/doi/10.1145/3558000)
- [HDC as Computing Framework](https://link.springer.com/article/10.1186/s40537-024-01010-8)

## Consequences

- Phase 13 adds 9 sub-phases (13a–13i), estimated ~5,500–8,000 lines
- New `email` module needed alongside existing agent/library modules
- New `pim` module (personal information management) for task/calendar/habit tracking
- Email processing adds 3 crate dependencies: `jmap-client`, `imap`, `mail-parser`
- Calendar adds optional `icalendar` crate dependency
- VSA spam prototypes stored in tiered storage (hot for active classification, cold for durability)
- OnlineHD training is incremental — no batch retraining, fits synchronous OODA loop
- Delegated agent spawning reuses existing Agent infrastructure with restricted CompartmentManager
- Email composition pipeline reuses grammar module (AbsTree → linearize) + Phase 12c constraint checking
- All processing is local — no email content leaves the user's machine
- Preference learning is VSA-native (prototype vectors), not ML-based
