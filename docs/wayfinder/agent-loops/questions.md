# Session question contract

Runtime adapters normalize provider questions into `akh.questions.v1` inside the
existing `PendingInput.prompt`. Each field has an immutable ID, original prompt
and header, choices, and explicit multiple-selection/free-text/secret flags.
Optional previews are text data; clients must not execute provider HTML.

Answers contain `cancelled` and an `answers` map from field ID to string arrays.
Validation requires exactly the offered fields and cardinalities. Skip has an
empty answer map. Secret-marked forms accept Skip through this journaled path;
collecting secret answers requires a private channel. Tool permissions retain their
separate approval contract and cannot be manufactured from question answers.

Codex uses its native question IDs and receives nested answer arrays, matching
the installed 0.153.4 app-server JSON schema. Claude receives a map keyed by the
original question text, preserving arrays for multiple choices. The adapter rejects
duplicate Claude question text before presenting ambiguous fields. The SDK callback
retains the original questions alongside answers and denies an explicitly skipped
question. These mappings follow [Claude's documented input contract](https://code.claude.com/docs/en/agent-sdk/user-input).

Claude's brokered tool list includes `AskUserQuestion`, and its callback handles
that tool before the native workspace denial. Plan mode may ask questions too.
Native workspace operations remain denied and reach no executor through that
callback. This bridge change requires a fresh conformance witness for its hash
before native runtime admission; SDK fixture tests alone are not that witness.

Akh validates the normalized form before journaling it, and validates answers
before recording a decision. Existing opaque question records retain their old
contract for compatibility; clients show unsupported forms as pending status.
Anima consumes the versioned form through its independent wire DTO, with a matching
fixture at `anima-pwa/src/lib/fixtures/akh-question-form.json`. The published Akh
fixture is `crates/akh-session-types/tests/fixtures/questions.json`.

Anima's live-input reader verifies each pending input against its exact journal
source. Its permission reader no longer infers central pending state from an
activity tail. Responses use the authenticated actor, source sequence, original
turn and generation; retries retain the same command and answers.

Local checks cover the shared fixture, provider mapping/transport, SDK callbacks,
server decision validation, full-schema Anima HTTP recovery and 97 UI regressions.
The question work does not prove native CLI elicitation, live-model behavior,
browser runtime or deployed cross-host question delivery. The earlier native Codex
workspace-tool witness remains a separate measurement.
