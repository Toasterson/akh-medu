# Event producers and MCP tools

`akheventd` normalizes input and durably forwards it. `akhsessd` owns automation
authorization, occurrence deduplication, targeting, concurrency and admission.
Neither daemon owns a cron clock. A scheduler retries the original occurrence;
it must not create a fresh ID or timestamp after an uncertain HTTP response.

## External cron and Kubernetes

Build `akh-trigger` with `--no-default-features --features trigger`. It accepts an
Endpoint JSON/TOML file containing `url`, `token_file`, and optional `ca_file`.
Credentials are read at request time. Use a source-scoped automation principal.

```sh
akh-trigger --endpoint-config /etc/akh/sessions.json --automation nightly \
  --template /etc/akh/nightly.json \
  --occurrence nightly-2026-09-12T01:00:00Z --occurred-at-ms 1789174800000
```

The template contains `source`, `type`, `subject`, `correlation_id`, and `data`.
Alternatively, `--event FILE` reads an immutable complete `EventEnvelope`.
Omit `--automation` to send through `akheventd`'s configured routes instead.
An admission receipt confirms acceptance, not completion of the agent turn.

`--template FILE --kubernetes-job --kubernetes-api-config FILE` reads Job metadata
using `AKH_JOB_NAME` and `AKH_JOB_NAMESPACE`. Inject these through the Downward API
from the `batch.kubernetes.io/job-name` label and `metadata.namespace`. Grant the
Job's service account namespaced `get` access to Jobs; no Pod listing or mutation
is needed. The API endpoint config uses its mounted service-account token and CA.

With the scheduled timestamp annotation and controller CronJob UID, the occurrence
is `cron:<namespace>/<cronjob-uid>/<scheduled-ms>`; duplicate Jobs for that tick
deduplicate too. Otherwise it uses `k8s:<namespace>/<job-uid>` with the scheduled
or creation timestamp, preserving identity across Pod retries. The scheduled
annotation is available from Kubernetes 1.32. This follows the upstream
[CronJob](https://kubernetes.io/docs/concepts/workloads/controllers/cron-jobs/)
and [Job](https://kubernetes.io/docs/concepts/workloads/controllers/job/) contracts.

## Native Djehuty

Configure `djehuty` sources with an endpoint name, source ID, thread UUIDs,
optional `ignore_authors`, `allowed_authors`, `message_types`, `recipient_id`,
`interval_secs`, and `page_size` (1–200). Each thread
has its own durable cursor. The adapter calls the native
`GET /v1/threads/{id}/messages?limit=N&after_message_id=UUID` API and consumes
`X-Next-Cursor`. It verifies message identity, thread, timestamps, ordering and
cursor progress before advancing. The configured credential represents this
integration's identity; the adapter never substitutes an agent identity, posts
messages, or acknowledges another agent's inbox.

Each message becomes `type: djehuty.message`, ID `<thread>/<message>`, with its
original message in `data` and the thread as `correlation_id`. Ignored/expired
messages are recorded before cursor advancement. Other messages reach the
durable ingress/outbox before advancement, so a crash replays through dedupe.
Changing the endpoint bound to an existing source/thread cursor is rejected.

Contract verification used the Board API source at
`d02c1a24f6ddb88d6b541e3da7676ea1b8a7dff4` and an HTTP fixture. It did not read a
live Board account. The older `boards` page adapter remains a distinct generic
integration and does not imply native Board compatibility.

## Streamable HTTP MCP

The session tool broker initializes and negotiates with each configured MCP
endpoint, advertises no client sampling or roots capability, sends the initialized
notification, and invokes `tools/call`. Protocol/result types come from rmcp
0.17. This client negotiates the SDK's 2025-06-18 version; it does not claim every
feature of the [current MCP transport specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports).

JSON and SSE responses are bounded. SSE reconnection uses a committed event ID
and `Last-Event-ID` on GET, never a second `tools/call` POST. Incomplete events do
not advance the cursor. Pings are answered; unsupported server requests receive
a JSON-RPC method error. A missing cursor or unconfirmed result becomes a durable
unknown outcome requiring reconciliation. Session deletion follows only a known
tool result. HTTP redirects are disabled and endpoint credentials/private CAs
remain scoped to their configured service.

Local tests exercise JSON, resumed SSE, partial-frame replay, server ping,
unrecoverable EOF, Djehuty cursor crash replay, and immutable scheduler identity.
These are contract fixtures, not live MCP or Board evidence.

## Native RabbitMQ verification

`python3 scripts/session-platform/rabbitmq-native-smoke.py` creates one disposable
Docker broker with loopback-only ports and generated test credentials. It passed
against RabbitMQ 4.3.5 with actual `akheventd` and `akhsessd` processes: the broker
acknowledges valid events after durable local receipt while sessions are offline;
malformed, unauthorized-source and conflicting-ID deliveries reach a dead-letter
queue. Restarting ingress preserves the original occurrence, which subsequently
admits exactly one central session and turn. Restarting the broker reconnects the
consumer, and an acknowledged redelivery creates no extra turn. The script removes
its own broker, volume and credentials. This does not test model execution,
workspace allocation, or a production broker installation.

## Thoth document and research delegation

Use a dedicated durable Board thread for Thoth requests. The Board inbox listing
marks messages read as a side effect and has no replay cursor, so it is not used
as the delivery queue. Configure the thread source with the participating Akh
UUIDs in `allowed_authors`, `coordination/delegation@v1` in `message_types`, and
Thoth's UUID in `recipient_id`. Filtered posts are audited before the cursor moves.

An Akh first posts a durable `board/note@v1` document anchor and uploads its
attachments with the upstream Board client. After upload completes, it posts a
`coordination/delegation@v1` to the request thread with `assignee_id`, `objective`
and `scope`; the objective includes exact attachment IDs and returned SHA-256
hashes or versioned Seshat references. Set `reply_to` and `parent_message_id` to
the anchor. Never send a readiness notification before its content is available.
The immutable delegation ID is the central occurrence/correlation source.

Install the unmodified Agent Board client/skill in each executor image and mount
only that Akh's Board credential at the client's configured private env-file path.
Run it through the brokered process tool, so downloads land in the assigned
workspace and their execution has a receipt. Thoth verifies attachment hashes,
treats supplied documents as task data, and uses his configured research/tools
and Seshat to produce a versioned result. He replies with `coordination/evidence@v1`
and `coordination/handoff@v1`, linking the delegation via `reply_to`. His response
protocols do not trigger another research turn. No agent overrides another
identity, and no credential is placed in a Board message or central prompt.
