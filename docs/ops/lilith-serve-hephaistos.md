# Runbook — Lilith on hephaistos (llama.cpp, Q8_0)

> Anima WI `01a0e270-314e` (Lilith C2). Decision and reasons:
> [ADR 121](../ai/decisions/121-lilith-serving-on-hephaistos-llama-cpp.md); Lilith:
> [ADR 116](../ai/decisions/116-lilith-chief-of-staff-guided-rl.md).
> Code: `scripts/lilith/hephaistos/`. The Runpod serve pod is now test/training only:
> [lilith-serve.md](lilith-serve.md).

Since 2026-09-29 Lilith (the frozen pinned parent `google/gemma-4-12B-it@5926caa4…`, no adapter) is
served on **hephaistos**, the operator's Linux box (RX 7900 XTX, 24 GB, ROCm, gfx1100), until new
hardware arrives. The server is llama.cpp `llama-server` `v0.5.0`
(commit `7fe450e19305`), **the Vulkan build (RADV), not HIP**, serving a Q8_0 GGUF of
the parent with the parent's own chat template, on the LAN address `192.168.1.114:8080` over plain
HTTP with a bearer key. She is a workspace of the Roman core on the Mac; T3 talks to her through the
provider "Lilith · Chief of Staff".

**Live server: the Vulkan build of llama.cpp `7fe450e19305` (v0.5.0), until ROCm is upgraded.** The
first real build on hephaistos (2026-09-30; Zorin OS 17.3 = Ubuntu 22.04 "jammy", ROCm 6.3.4, gfx1100)
showed that the HIP build of this commit does not compile with ROCm 6.3.4 (see "Why Vulkan, not HIP"
below), so `build_server.sh` refuses HIP on ROCm 6.3.4 and older and the install path is
`/opt/lilith-serve/llama.cpp/7fe450e19305-vulkan/`. Expect the Vulkan build's tokens/s, not the HIP
bounds, in the verify output; `verify.py` records them. On 2026-09-29 the pinned llama.cpp was also
built on the Mac, `run_server.sh` and `verify.py` passed against it on a public Gemma 4 12B GGUF with
the parent's pinned template, and the pinned converter produced a 12.67 GB Q8_0 file from a
header-exact stand-in for the parent (ADR 121).

## What runs where

| Piece | Where |
|---|---|
| `llama-server` (pinned build) | `/opt/lilith-serve/llama.cpp/<commit12>-vulkan/` (live) or `-hip/`, `/opt/lilith-serve/current`; `BUILD.json` in it records the backend, the Vulkan SDK version + sha256 or the ROCm version |
| Model | `/var/lib/lilith-serve/models/<name>/{lilith-parent-q8_0.gguf,chat_template.jinja,PROVENANCE.json}`, `models/current` |
| Converted copy for reuse | the runs share: `hephaistos-20260927/models/lilith-parent-5926caa4-q8_0-llamacpp-<commit12>/` |
| Settings | `/etc/lilith-serve/server.env` (created once from `server.env.example`, never overwritten) |
| API key | `/etc/lilith-serve/api-key` (0600, owned by `lilith-serve`); copy on the Mac: `$ROOT/secrets/lilith-openai-key` |
| Unit | `lilith-serve.service` (user `lilith-serve`, groups `render`, `video`) |
| Work directory (operator) | `~/lilith-serve-work/`: llama.cpp source, build, conversion venv, staged parent, `vulkan-sdk/<version>/` (the pinned LunarG SDK) |

Memory (a *bound*, from plan §2/§2a): Q8_0 weights about 12.7 GB, KV about 1.6 GB per 65536-token
slot (two slots: about 3.2 GB), compute buffers 1-2 GB: about 18 GB of 24 GB.

## Tomorrow: the supervisor's steps

Everything below runs from the Mac unless it says "on hephaistos". `ROOT` is the live Roman
deployment, `~/Library/Application Support/roman-akhs`. Login:
`ssh -i ~/.ssh/id_ed25519_illumos_workspace -o IdentitiesOnly=yes toasty@192.168.1.114`
(a harmless `ollama-key.txt: No such file` line prints on every command). The operator powers the
box on; nothing here does.

### 0. Merge, then take the scripts from main

The scripts and pins are deploy tooling: take them from a commit that is on `origin/main`
(`.claude/rules/deploy-from-main.md`). After the PR for `feat/lilith-serve-hephaistos` is merged:

```sh
git -C ~/ws/akh-medu fetch origin
SHA=$(git -C ~/ws/akh-medu rev-parse --short=12 origin/main)
git -C ~/ws/akh-medu merge-base --is-ancestor "$SHA" origin/main && echo "on main: $SHA"
git -C ~/ws/akh-medu archive origin/main scripts/lilith/hephaistos \
  | ssh -i ~/.ssh/id_ed25519_illumos_workspace -o IdentitiesOnly=yes toasty@192.168.1.114 \
      'mkdir -p ~/lilith-serve-scripts && tar -x -C ~/lilith-serve-scripts'
```

Hephaistos's own checkout cannot fetch origin over ssh, hence `git archive`. Record `$SHA` in the
run report. Then on hephaistos: `cd ~/lilith-serve-scripts/scripts/lilith/hephaistos`.

### 1. Read-only preflight (on hephaistos)

```sh
uname -sr; rocminfo | grep -m1 gfx; free -g | head -2; df -h ~ /var/lib /opt | cat
ip -4 -o addr show | grep 192.168.1.114; ss -ltn 'sport = :8080'
rocm-smi --showmeminfo vram   # the GPU must be idle: no training run, no ollama holding VRAM
systemctl is-active ollama 2>/dev/null; command -v cmake git python3; cat /opt/rocm/.info/version
```

Need: `gfx1100` listed, 192.168.1.114 on this host, port 8080 free, at least 60 GB free under `~` and
`/var/lib` together (24 GB staged parent + 13 GB GGUF + venv/build), the GPU idle. If Ollama or a
training venv holds VRAM, stop it first.

### 2. Build the server (Vulkan, 10-25 minutes)

```sh
bash build_server.sh --backend vulkan --install-deps   # apt tools + libvulkan-dev mesa-vulkan-drivers, then the Vulkan build
bash build_server.sh --backend vulkan --check          # exits 0 when the build is complete and has every flag the unit uses
```

Idempotent: a complete build of the same commit and backend is reused. `--toolchain-only` resolves
and verifies the toolchain without building.

**Vulkan on Ubuntu 22.04 (jammy).** jammy has no `glslc` package, and its `libvulkan-dev` ships the
Vulkan 1.3.204 headers, which are too old for the pinned llama.cpp (`PFN_vkGetDeviceFaultInfoEXT does
not name a type`; VK_EXT_device_fault needs headers 1.3.246 or newer). So `--install-deps` apt-installs
only `libvulkan-dev mesa-vulkan-drivers`; if `glslc` is missing or the system headers are older than
the minimum in `pins.json`, it downloads the **pinned** LunarG Vulkan SDK (`pins.json` `vulkan_sdk`:
a versioned URL, never "latest", and its sha256, verified before unpacking) into
`~/lilith-serve-work/vulkan-sdk/<version>/`, sources its `setup-env.sh` for the build only, and starts
from a clean build directory whenever the toolchain changed. Nothing is installed system-wide. The SDK
version and sha256 are recorded in the build's `BUILD.json`. Without `--install-deps` the script never
downloads: it uses an already unpacked SDK or tells you to rerun with the flag. The SDK tarball is
about 350 MB and x86_64 only.

#### Why Vulkan, not HIP (2026-09-30)

- The HIP build of the pinned commit (`7fe450e19305`, v0.5.0) **fails to compile with ROCm 6.3.4 on
  gfx1100**: the `mmf` template instances stop with `error: Illegal instruction detected: Invalid
  dpp_ctrl value: wavefront shifts are not supported on GFX10+`. llama.cpp's docs name no minimum ROCm
  version, so the rule is: **ROCm 6.3.4 and older are known bad** (`pins.json`
  `llama_cpp.hip_rocm_known_bad_through`). `build_server.sh` reads `/opt/rocm/.info/version` and
  refuses HIP at or below that with `lilith::heph::rocm_version`, naming `--backend vulkan` and a ROCm
  upgrade as the ways out. Whether a newer ROCm (6.4 or 7.x) fixes it is not measured: upgrade, then
  prove it with a build (`LILITH_ALLOW_OLD_ROCM=1` bypasses the refusal to test a version you believe
  fixed; it is not for the live build). Until then the live server is the Vulkan build.
- HIP on jammy also needs two setup facts, handled by the script: `/opt/rocm/bin` is not on `PATH` (it
  is added when it exists, so `hipconfig` is found), and ROCm's clang uses the highest installed GCC's
  libstdc++, so the HIP configure fails with `cuda_wrappers/cmath: 'cmath' file not found` unless
  `libstdc++-<N>-dev` for that GCC is installed. `--install-deps` installs it for the highest GCC
  under `/usr/lib/gcc/<arch>-linux-gnu` (detected, never a fixed 12), and the preflight names the
  package when it is missing (`lilith::heph::libstdcxx`).

### 3. Convert and publish the model (about 30-90 minutes, mostly network)

```sh
bash convert_parent.sh
```

It mounts the runs share if needed (`192.168.1.81:/export/toast/akh-medu-runs`, read and written as
uid 501 through `setpriv`), verifies **all seven parent files against `pins.json`** while copying
them (a mismatch aborts: never convert an unpinned file), converts to Q8_0 with the pinned
converter, writes `PROVENANCE.json`, publishes to the share (the share writes at about 11 MB/s: about
20 minutes for 13 GB), reads it back and verifies it, and installs it locally. Reruns skip whatever
already verifies; if the share already holds a valid model it is copied instead of converted.
`--no-publish` keeps the result local only.

The conversion venv installs `transformers==5.17.0` over llama.cpp's pinned 4.57.6, which cannot read
the parent's tokenizer config (ADR 121). If `pip` cannot resolve `torch==2.11.0` on the box's Python,
set `LILITH_CONVERT_PYTHON` to a python >= 3.10 (for example one from `uv python install 3.12`).

### 4. Install and start the service

```sh
bash install_service.sh                  # user, dirs, key, unit; starts it; waits for /health
systemctl status lilith-serve --no-pager
journalctl -u lilith-serve -n 40 --no-pager
```

The first start loads 12.7 GB from disk (about 20 s with a warm page cache, minutes cold).
`install_service.sh` prints only the key's fingerprint, never the key.

### 5. Verify from the Mac

The key file on the Mac comes from hephaistos through a pipe (never echoed, never on a command line):

```sh
umask 077
ssh -i ~/.ssh/id_ed25519_illumos_workspace -o IdentitiesOnly=yes toasty@192.168.1.114 \
  'sudo cat /etc/lilith-serve/api-key' > "$ROOT/secrets/lilith-openai-key"
chmod 600 "$ROOT/secrets/lilith-openai-key"
python3 scripts/lilith/hephaistos/verify.py --base-url http://192.168.1.114:8080 \
  --key-file "$ROOT/secrets/lilith-openai-key" --json ~/lilith-private/serve/hephaistos-verify-$(date +%Y%m%d).json
```

(That file is also the key the akh reads: `daemon_credentials.py` maps it to `LILITH_OPENAI_KEY`.)
The checks and what they prove:

| Check | Passes when |
|---|---|
| `health` | `/health` answers 200 without a key |
| `auth` | `/v1/models` and `/v1/chat/completions` refuse a request without the key |
| `models` | `/v1/models` lists `lilith` |
| `context` | `/props` reports at least 49152 tokens per slot (expected 65536) |
| `tool_call` | the `get_time` probe of `lilith_serve.py` returns well-formed `tool_calls`, nothing left in `content` |
| `tool_roundtrip` | a `tool` message goes back and the model answers in text |
| `reasoning_leak` | no `<|channel>`, `<channel|>`, `<|tool_call>` … in `content`, with and without thinking; WARN if thinking produced no `reasoning_content` |
| `throughput_8192`, `throughput_32768` | prefill and decode tok/s (cache off) from the server's own timings |

On hephaistos add `--vram` (and run it there against the LAN address) to record the GPU memory in use
after load and after each probe. Expected: all PASS. Any FAIL: do not point the akh at it.
`--min-decode-tps 20` turns a slow decode into a WARN. Paste the throughput lines and VRAM into the
run report; they replace the bounds in ADR 121.

### 6. Point Lilith's akh at hephaistos (Mac, live core: follow CUTOVER step 12)

The Roman core is live. Never open its SQLite files; this step only writes render inputs and
re-renders the workspace file.

```sh
umask 077
printf '%s\n' 'http://192.168.1.114:8080/v1' > "$ROOT/secrets/lilith-base-url"
python3 - <<'PY'
import json, os
path = os.path.expanduser("~/Library/Application Support/roman-akhs/config/lilith.json")
config = json.load(open(path))
config.update({"serving": "hephaistos", "allow_insecure_lan_http": True})
json.dump(config, open(path, "w"), indent=2)
PY
python3 deploy/roman-mini/configure_core.py --plan --root "$ROOT"   # writes nothing; review the diff
```

The plan diff should show only Lilith's `[openai_endpoint] base_url` (plus the comment text) and
`config/t3-acp.json` gaining `"lilith_serving": "hephaistos"`. Then render and restart as CUTOVER
step 4/12 describe (from a clean `origin/main` checkout; restart = `kill` the core pid, KeepAlive
brings it back). Check:

```sh
curl -s -H "Authorization: Bearer $(cat "$ROOT/secrets/lilith-openai-key")" \
  http://192.168.1.114:8080/v1/models | python3 -c 'import json,sys;print([m["id"] for m in json.load(sys.stdin)["data"]])'
grep -E 'workspace="?lilith|Chief-of-Staff guard' "$ROOT"/logs/core.*.log | tail
```

**macOS Local Network privacy.** `akhomed` runs as a LaunchDaemon; if it has no Local Network
approval its connection to a LAN address fails instantly (`No route to host`, os error 65) although
the `curl` above, run from a login session, works. Her turns then fail with the backend error. The
fix is the one-time approval in CUTOVER "Local Network privacy (TCC)": the signed stable
`bin/akhomed` keeps it across deploys.

### 7. How the operator talks to her

In T3 choose the provider **"Lilith · Chief of Staff"** (`roman-lilith`; installed by CUTOVER step 12
"Chat with Lilith in T3"). When T3 starts her transport, the entrypoint asks
`http://192.168.1.114:8080/health`. If hephaistos is off or the unit is down it writes one line to
`$ROOT/logs/lilith-serve-start.log` and starts nothing; her turns fail with the backend error until
the unit is up. To see why: `tail -n 5 "$ROOT/logs/lilith-serve-start.log"`.

## Operating

```sh
sudo systemctl stop lilith-serve       # before any ROCm training run: the GPU cannot hold both
sudo systemctl start lilith-serve
journalctl -u lilith-serve -f          # request lines carry no prompts (no --verbose)
bash install_service.sh --rotate-key   # new key; then rewrite $ROOT/secrets/lilith-openai-key and restart the core
```

Tuning lives in `/etc/lilith-serve/server.env` (then `sudo systemctl restart lilith-serve`):
`LILITH_LLAMA_PARALLEL`/`..._CTX_PER_SLOT`, `LILITH_LLAMA_KV_TYPE=q8_0` (measure with `verify.py`
first), `LILITH_LLAMA_SLEEP_IDLE_SECONDS` (unload the model from the GPU when idle; the next request
pays a reload), `LILITH_LLAMA_EXTRA_ARGS`. If a firewall is active, allow the LAN to port 8080 only
(`sudo ufw allow from 192.168.1.0/24 to any port 8080 proto tcp`); the server binds only the LAN
address.

Reasoning tokens count against `max_tokens`: Lilith reasons at length even without thinking mode
(routed to `reasoning_content`, never into `content`), so a small `max_tokens` can end a turn before
the answer or the tool call. `verify.py`'s probes leave room for it.

Changing the pinned llama.cpp commit or converter is a deliberate edit of `pins.json` (and ADR 121)
followed by `build_server.sh`, `convert_parent.sh --force`, `install_service.sh` and a green
`verify.py`.

## Failure modes

| Symptom | Meaning | Action |
|---|---|---|
| `build_server.sh`: `lilith::heph::rocm_version` | HIP requested on ROCm 6.3.4 or older (known bad for this llama.cpp) | `--backend vulkan --install-deps` (the live build), or upgrade ROCm |
| `build_server.sh`: `lilith::heph::libstdcxx` | HIP: the highest GCC's `libstdc++-<N>-dev` is missing | `--install-deps`, or the `apt-get` line in the message |
| `build_server.sh`: `lilith::heph::vulkan_sdk` | Vulkan: no glslc / headers too old and the pinned SDK is not unpacked, or its download failed the sha256 | `--backend vulkan --install-deps`; a repeated sha256 mismatch means stop and report |
| `build_server.sh`: `lilith::heph::gpu` | `rocminfo` lists no gfx1100 | driver: `dmesg | grep amdgpu`; or `--backend vulkan` |
| `build_server.sh`: `lilith::heph::flags` | the built server lacks a flag the unit passes | rebuild the pinned commit; do not edit the unit |
| `convert_parent.sh`: `lilith::heph::stage_mismatch` / `parent_hash` | a parent file differs from its pin, or the NFS read was cut short | rerun once (copies are idempotent); a repeat means the share's copy is not the pinned parent: stop and report |
| `convert_parent.sh`: `lilith::heph::pip` | torch or transformers could not install | `LILITH_CONVERT_PYTHON`, network to pypi.org and download.pytorch.org |
| `convert_parent.sh`: `lilith::heph::convert` | converter failed | `~/lilith-serve-work/convert-<name>.log`; the pinned converter registers `Gemma4UnifiedForConditionalGeneration` |
| `install_service.sh`: `lilith::heph::bind` | the bind address is 0.0.0.0 or not on this host | set `LILITH_LLAMA_HOST` in `server.env` to the LAN address |
| unit fails at start, journal `lilith::heph::key` | the key file is missing or readable by others | rerun `install_service.sh` |
| unit fails, out-of-memory | another process holds VRAM | `rocm-smi`; stop Ollama or the training job |
| `verify.py` `tool_call` FAIL "unparsed" | template and parser do not match | ADR 121, "Tool calling": test a later commit, else the Runpod pod |
| `verify.py` `context` FAIL | started with a smaller `--kv-unified-per-slot` | fix `server.env`, restart |
| turns fail, `curl` from a login shell works | Local Network TCC for the LaunchDaemon | CUTOVER TCC section |
| T3 log: "not answering /health" | hephaistos off, or the unit is down | operator powers it on; `systemctl status lilith-serve` |

## Rollback

Point her back at the Runpod pod: restore `secrets/lilith-base-url` to the proxy URL
(`https://…proxy.runpod.net/v1`), set `"serving": "runpod"` and drop `allow_insecure_lan_http` in
`config/lilith.json`, render and restart as above. Stop the unit with `sudo systemctl disable --now
lilith-serve`. Nothing on hephaistos needs removal; the share keeps the converted model.

## Tests

```sh
uv run --with pytest python -m pytest -q scripts/lilith/hephaistos     # 58 tests, no GPU, no network, no root
cd deploy/roman-mini && python3 -m unittest discover -p 'test_*.py'
```

They cover the pin/provenance chain (tampering, wrong parent, wrong converter, wrong template), each
script's refusals against fakes (`LILITH_NO_SUDO=1`, a fake NFS, a fake `llama-server`), `run_server.sh`'s
exact command line and that the key is on no argv or environment, the unit file, the build toolchain
handling (ROCm PATH and version floor, libstdc++-dev, the pinned Vulkan SDK download, hash refusal and
`BUILD.json` provenance), and `verify.py`
against a mock server with every failure shape.
