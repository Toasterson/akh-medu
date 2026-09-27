# Independent provenance audit

Read-only CPU audit performed after the completed run. No model execution,
generated-source execution, receipt rewriting, or protocol changes were made.

The existing `independent-report.json` is an independent replay of the bounded
artifact verifier. It reproduces the saved 4/8 Lilith and 6/8 Ptah scores, but
does not by itself authenticate collection through the saved session stack.

This audit checked every one of the 16 records against its corresponding
`*-request.json` and `*-session.json` artifacts. For every record:

- the record, request, and session use the same session ID;
- the record and request use the same turn ID, which exists in the session;
- the saved turn output and status exactly equal the record source and status;
- the saved turn input exactly equals the request prompt;
- the session Akh is the expected `lilith-compare` or `ptah-compare` identity;
- the runtime profile is the corresponding `lilith` or `ptah` profile; and
- the session exposes an empty tool list.

Result: **16 records checked, zero binding errors**.

The frozen protocol's collector-source, four binary, model, tokenizer, task,
verifier, and interpreter hashes matched the inspected artifacts. The Ptah
artifact remains the specifically pinned `JonathanColetti/Qwen3.8-27B-Uncensored`
derivative; this audit does not turn it into an upstream reference or establish
release parity.

The verifier remains intentionally limited. It interprets admitted Python under
a bounded AST grammar and finite authored cases. Grammar failures are contract
failures rather than proof of incorrect algorithms, and finite passes are not
general correctness proofs. The public pack has one observation per model and no
per-request seed. Therefore this evidence supports the recorded single-run
scores and collection bindings, not statistical superiority or certification.

The provenance check was performed manually from the frozen protocol and raw
artifacts. `lilith_stack_report.py` does not automate the request/session/model/
binary bindings and does not pin its own reporter hash, so
`independent-report.json` should continue to be described as an independent
score replay rather than a fully independent collection-provenance audit.
