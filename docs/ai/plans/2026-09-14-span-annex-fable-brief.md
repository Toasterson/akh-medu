# Span-annex correctness review

Review this bounded implementation design in at most 500 words. No tools,
code execution or remote operations. Advice is not test evidence.

Following your recommendation, a same-space neural-span annex now uses the
existing cognitive Store (no new Engine or database). AcceptedTurn has optional
span_captures: Vec, default empty, omitted when empty so old payloads and replay
fingerprints remain unchanged. New producers must first discover the authenticated
/v1/capabilities neural_spans_v1 flag. Current akhsessd still sends an empty list;
native projection and transport are not wired yet.

Each capture has four SHA256 pins (model, tokenizer, full tap/pooling/arithmetic
policy artifact, full projector artifact), source=user|assistant, UTF-8 byte
range in the actual accepted input/output, and exactly 1250 bytes of binary
vector. Maximum four captures/turn, 8192 bytes of materialized text per capture.
No raw hidden states. Range/pin/vector validation happens before operation admission.

The core already authenticates service credentials, scopes each principal to
allowed Akhs, rejects browser origins and limits bodies to 2MB. New query route
uses the same auth, rechecks transcript-capture policy, and scopes its bucket by
length-prefixed (principal ID, Akh, session ID). Exact space equality is required.
It scans at most 256 records, ranks by Hamming distance then stable turn ID, and
returns at most 16 text/span hits. Representation mismatch gives no hits.

Accepted-turn operation fingerprinting and existing unknown-outcome handling
remain. Transcript journal omits the capture vectors (avoids unbounded duplicate
retention). The journal rejects same turn ID with changed accepted content, before
the engine records it. After engine.persist succeeds, projected spans are appended
in the SAME Store transaction as the successful acceptance receipt. Capture-off
means no span writes and queries are forbidden. Receipt replay does not append.
Stored per-session bucket is capped at 256 spans, oldest evicted, with at most
four spans admitted each operation. Annex append additionally rejects conflicting
retained turn IDs and treats identical retained batches as no-ops. Evicted-turn
replays are protected by the existing core operation IDs / immutable journal,
not an unbounded second dedupe table.

Tests cover restart persistence, independently known Hamming distances, service/
Akh/session/space isolation, replay/conflict, invalid UTF-8 ranges, transaction
rollback, retention cap, old wire shape, capture policy and actual core HTTP routes.
No projected retrieval-quality claim, symbolic-space compatibility claim, or
production deployment. Question: identify a concrete correctness/security gap
in this first annex slice, distinguishing blockers from later lifecycle work.
