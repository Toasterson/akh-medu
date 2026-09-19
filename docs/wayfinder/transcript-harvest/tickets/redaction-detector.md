---
type: prototype
status: closed
resolved: 2026-09-19
blocked-by: [event-normal-form.md]
---

## Question

What does the credential detector actually match, and what is its measured error
rate on the real corpus?

[The secrets posture](secrets-posture.md) fixed the direction — verbatim in,
redact outward, flag per line. This ticket makes the detector real, and it is a
prototype because the only honest way to set the pattern set is to run it against
the 68 MB already on disk and look at what it catches and misses.

1. Assemble the pattern set: provider API keys, OAuth bearer tokens, PATs
   (GitHub, Forgejo, Anima), SSH private keys, `.env` dumps, connection strings
   with embedded passwords, JWTs, and the high-entropy fallback.
2. **Run it on the local corpus and report real numbers**: how many lines flag,
   sampled precision, and — harder and more important — a deliberate hunt for
   false negatives.
3. Decide what "flagged" does downstream: withheld entirely, or emitted with the
   matched span masked. These differ for a line that is 3 KB of useful tool output
   with one token in the middle.
4. Decide where the detector runs: at derivation time (flag stored, cheap reads)
   or at read time (policy can change without re-deriving, every read pays).
5. State the residual risk plainly. A pattern-based detector will miss things.
   The posture is only honest if the store's access controls assume that.

Deliberately in scope to verify: that OpenCode's `account`/`credential` tables
are genuinely never touched by the adapter.

## Resolution (2026-09-19)

Prototype: `assets/scripts/redaction_prototype.py`, run over all 68,360 records.

### 1. Named patterns only — measured at 0.042%

Ten patterns: AWS access key, GitHub PAT/OAuth, Anthropic key, OpenAI key, Slack
token, JWT, PEM private key, `Bearer <token>`, Postgres URL with password, and
`secret|password|api_key|access_token = value` assignment.

**29 of 68,360 records flagged (0.042%)** — and they are not theoretical:

| pattern | hits | note |
|---|---|---|
| `github_pat` | 2 | **a real `gho_…` OAuth token sitting in a transcript** |
| `private_key` | 1 | **a real `-----BEGIN OPENSSH PRIVATE KEY-----`** |
| `assignment` | 14 | mixed; the sample is a markdown table false positive |
| `bearer` | 7 | includes `Bearer core-test-credential-…` (a test value) |
| `pg_url_password` | 5 | `postgres://akh:akh@` from config examples |

Two genuine credentials in 68k records ≈ **one per 34,000 records**. At the
measured 313,255 records/year/machine that is roughly **nine real credential
leaks per machine per year** landing in the store. The posture is not academic.

### 2. Entropy fallback — rejected, with numbers

A high-entropy base64 rule would add **10,392 records** — **358x the named-pattern
yield, about 15% of the corpus**. **68.4% of those sit on records containing
`signature` or `encrypted_content`**: Claude's thinking-block signatures and
Codex's encrypted reasoning, which are opaque *model* data and categorically not
secrets ([render prototype](render-format.md)).

A rule that flags one record in seven, two-thirds of them provably harmless, is
not a detector — it is a second copy of the corpus wearing a warning label.

**Corollary:** `signature` and `encrypted_content` fields are **excluded from
scanning** entirely. They are high-entropy by construction and will never contain
a user credential.

### 3. Cue-word broadening — rejected, with a domain trap

Records merely *mentioning* a credential cue word: **42,015, or 61.4%** — driven
by **"token" in 40,810 records**, because `token_count` and `token_usage_record`
telemetry mean *LLM tokens*. In an agent-transcript corpus the vocabulary of
secrets collides head-on with the vocabulary of inference. Any detector tuned on
cue words flags one record in six and means nothing.

This is the sharpest lesson of the prototype: generic secret-scanning heuristics
are calibrated for source repositories, and this corpus is not one.

### 4. What a flag does

- **Mask the matched span** by default, keeping the rest of the record. Most hits
  are one token inside otherwise useful output; withholding the whole record would
  lose the context that makes the transcript worth keeping.
- **Withhold the whole record** for `private_key`, alone among the patterns,
  because the match is the *opening line* of a multi-line body the pattern cannot
  bound. Masking a PEM header while emitting the key beneath it would be worse
  than doing nothing.
- The flag, its pattern name and its offsets are stored on the normalized row, so
  policy can be re-applied without re-scanning.

### 5. Where it runs

At **derivation time**, in [the derivation worker](derivation-worker.md), with the
flag stored — so reads stay cheap. Changing the pattern set or the mask/withhold
policy is a `derivation_version` bump, which re-pends the corpus through machinery
that already exists. That is precisely the flexibility the read-time alternative
was supposed to buy, obtained for free.

Verified in passing: the OpenCode adapter never touches `account` or `credential`
([secrets posture](secrets-posture.md)).

### 6. Residual risk, stated plainly

A pattern-based detector **will** miss novel and bespoke credential formats — and
this corpus already proves the point, since the two real hits were both
well-known shapes. Nothing here justifies treating the store as safe to expose.

The honest posture: the store is **private and credential-bearing**, its bearer
token is the real protection ([read surface](read-mcp-surface.md)), and redaction
reduces blast radius rather than eliminating it. The one genuine improvement over
the alternatives is that, having cut Seshat, **a missed secret is now retractable**
— delete the row, bump the version, re-derive.
