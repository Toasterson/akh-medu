# Continuation safety review

Review this bounded design; no tools, code changes or remote operations. Return
at most 600 words of concrete correctness issues. Advice is not test evidence.

We implemented your previous immutable-weight/owned-state split in the existing
Qwen35 forward. CPU and exact HIP/Vulkan tests prove two interleaved sessions
bit-identical to serial execution. Now adding one active native conversation.

InferenceSession privately tracks TextState { forwarded_tokens: Vec<u32>,
last_logits: Tensor }. resume_prompt(full_authoritative_tokens) only appends when
the ENTIRE saved forwarded-token vector is an exact prefix of the new prompt.
Divergence or shorter history causes fresh prefill. Equal history returns saved
logits. No longest-prefix KV slicing (recurrent state cannot be truncated).
Appending is bounded 128-query prefill at the saved position. forward_token adds
a token to the log only after successful forward. Sampling stays outside; the
last sampled token and EOS are not implicitly recorded. Tensor-based legacy
forward invalidates host-token tracking. Opaque snapshots save tokens, logits,
all mixer state and session ownership; other sessions cannot restore them.

Cancellation/forward failure discards this active cache and next call re-prefills
from durable history. This is simpler than keeping checkpoint cadence initially.
Existing explicit snapshots remain immutable/reusable by their own session, but
the daemon does not retain them. Context cap is checked on FULL prompt+reserved
output before reuse, not suffix length; the canary remains 2048 tokens.

CandleBackend will retain a single active key containing canonical session ID,
provider-conversation ID and runtime binding. Key mismatch creates an empty
InferenceSession sharing immutable weights. Model/device replacement drops the
entire backend and releases unused HIP cache as previously tested. The runtime
already validates durable configuration pins before entering inference. Shared
compute remains globally serialized on HIP default stream.

Generation uses one common loop: legacy unkeyed calls force fresh state; keyed
Qwen calls use resume_prompt/forward_token. Any generation error, cancelled
post-generation callback, token-budget exhaustion or invalid XML response clears
the active cache before returning. Tool approvals and durable receipts remain
outside this cache. No kernel optimization, streaming, image placeholders or
multi-session GPU pool is added. We will test exact-token extension, divergence,
identical prompts, cancellation, final-unforwarded tokens, eviction and real
daemon restart/tool loops on both backends.

What subtle ownership/accounting/security issue would you fix before wiring this
into the daemon? Distinguish a first-slice blocker from a future optimization.
