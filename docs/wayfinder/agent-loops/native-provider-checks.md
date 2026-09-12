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
Claude's native check and Fable's design review still require an accessible login
in the configured private profile.
