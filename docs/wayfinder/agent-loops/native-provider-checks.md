# Native provider checks

Brokered runtime admission still requires an installed-profile conformance
document. A fixture or a process handshake cannot produce that certificate.

## Installed Codex with a local Responses fixture

Build and run:

```sh
cargo build -p akh-session-services --no-default-features --features runtime \
  --example native-provider-probe
cargo build -p akh-session-services --no-default-features --features executor \
  --bin akhexecd
python3 scripts/session-platform/codex-native-wire-smoke.py \
  --codex /absolute/path/to/codex
```

The test starts the installed Codex app-server, the repository's provider adapter,
a real executor process and a synthetic Responses endpoint. Its private profile
uses the documented custom-provider
[configuration](https://learn.chatgpt.com/docs/config-file/config-reference)
with authentication disabled and a loopback API URL. No user login or auth cache
is read or copied. All files, generated test credentials and processes belong to
the invocation. The provider daemon's admission gate is not involved or changed.

Codex CLI 0.153.4 passed on Darwin arm64 with thirteen synthetic model requests:

- Supplied write, read and process operations reached `akhexecd`; each duplicate
  operation preserved its authoritative receipt. The resulting files existed
  in the executor workspace and not in the runtime working directory.
- Plain and namespaced native `exec_command` calls were rejected as unsupported,
  as were native `apply_patch` and `view_image` calls. Those workspace tools were absent
  from every advertised model catalog, including after resume.
- A new app-server process resumed the exact provider thread and used its
  original supplied tools. A supplied-tool failure returned to the model, and a
  forced native fallback was rejected without creating its target file.
- Native planning/goal and skill-discovery tools remain advertised. The profile
  restricts native workspace execution; it does not remove all provider metadata
  tools. The report records the actual executable hash and platform.

The resume check caught an adapter defect: `thread/resume` has no `environments`
parameter, and disabling environment access on the original thread was not
preserved after an app-server restart. The adapter now sets `environments: []`
on every brokered `turn/start`. Without that change, the resumed model catalog
advertised native patch and image tools. Both the native check and the subprocess
regression tests require the turn-level restriction.

This verifies the installed harness's wire behavior, tool dispatch restrictions,
adapter continuation and local executor operations. It does not verify a live
model's tool selection, learning after refusal, production credentials, kernel
confinement, cross-host transport, or a deployable native conformance profile.
Fable's design review completed after the user repaired local keychain access;
see [the design review](fable-completion-review.md).

## Installed Claude with a local Messages fixture

```sh
python3 scripts/session-platform/claude-native-wire-smoke.py \
  --claude /absolute/path/to/claude --node /absolute/path/to/node
```

Claude Code 2.1.269 and the pinned Agent SDK 0.3.266 passed on Darwin arm64.
Twelve synthetic model requests exercised the actual SDK bridge, installed CLI
and executor. The supplied write, read and process tools reached `akhexecd`,
replayed their exact receipts, and changed only its workspace. Restarting the
SDK/CLI resumed the original provider session with the same supplied tools.
Forced native Bash, Write, Read and Edit calls returned error tool results;
those tools were absent from all advertised catalogs, including after resume.
An executor refusal also returned to the provider without permitting local
fallback or creating its target file.

The fixture uses the documented [gateway environment variables](https://code.claude.com/docs/en/env-vars)
and [Messages streaming protocol](https://platform.claude.com/docs/en/build-with-claude/streaming),
a throwaway API key, an isolated profile and disabled nonessential traffic.
It reads or copies no user login cache and creates no conformance certificate.
This is native harness/protocol evidence on one machine; live-model behavior,
kernel confinement, cross-host operation and deployment admission remain separate
checks. Fable's authenticated model review is also separate from this fixture.
