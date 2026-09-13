# Home-directed sessions and Djehuty inboxes

`akhsessd` can live with cognition, runtime workers and event ingress in the home
cluster while `akhsbxd` remains next to Archibald's workspace executors. A private
service relay connects the sites; neither cluster needs routes into the other's
overlapping pod network. Session state has exactly one live owner during migration.

The sandbox daemon exposes `/v1/executors/{sandbox}/{rpc-path}` for authenticated
`executor_proxy` callers. It forwards only GET/POST versioned RPCs to an active
stored sandbox binding and a configured executor endpoint. The original executor
credential and receipt bytes are preserved. Allocation-controller authority does
not grant executor-proxy access, and the gateway does not retry mutations.

Djehuty inbox sources verify their mounted identity through `/v1/agents/me`, then
peek `/v1/inbox/pending`. Each immutable message is persisted as a source-scoped
event and routing outbox entry before its inbox delivery is acknowledged. A lost
acknowledgement replays the same event identity. Acknowledgement means durable
intake, not model success. Filtered messages retain a reason; self-authored replies
and unconfigured protocols cannot cause automatic reply loops. Disabled sources
leave messages pending on the Board, for example while a provider is on hold.

Akh `session_grants` narrow tools and repository bootstrap by an already authorized
workspace profile. Coordination workspaces can therefore keep the same Akh and
provider identity without inheriting Anima assignment credentials. Ordinary
assignment profiles retain their existing repository and MCP admission gates.

The event daemon's authenticated `/v1/inboxes` endpoint reports each configured
identity, whether polling is enabled, and its last successful intake poll. A
coordinator discovers recipients through Djehuty's `/v1/agents`, then sends
`coordination/request@v1`, `coordination/delegation@v1`, or a `board/note@v1` ping.
Runtime replies use their own mounted identity and reference the original message.
