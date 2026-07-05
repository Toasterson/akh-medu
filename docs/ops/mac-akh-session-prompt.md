# Mac mini (armini) akhs — Claude Code session prompt

Unlike Archibald (a Linux host running the prebuilt `:cloud` container), the Mac is
Apple Silicon with **native Rust and a Metal GPU**. A Metal GPU akh can't run in a
Linux container, so the Mac runs **akhomed natively** (one binary serves a Metal
GPU akh *and* cloud akhs). The Mac is on the LAN (192.168.1.111) and reaches
`akh-pg` directly at `192.168.1.16:5432` — no port-forward, LAN mTLS.

Already staged for you on the Mac: the client-cert bundle at `~/akh-certs/`
(`ca.crt`, `mac-client.crt`, `mac-client.key`). You still need `OLLAMA_CLOUD_KEY`
exported in the session shell, and — for the GPU akh — the `gemma4-12b.gguf` model
(see step 4; or fetch from hephaistos:
`scp toasty@192.168.1.114:akh-3akh-test/data/akh-medu/models/llm/gemma4-12b.gguf .`).

Native build needs only Xcode Command Line Tools (already installed) + cargo;
oxigraph is built without RocksDB so **no cmake is required**.

---

```
You are setting up akhs natively on this Apple-Silicon Mac (akh-medu distributed
agent system). akhomed is a Rust daemon; build it from source here so it can use
the Metal GPU. Each akh is stateless compute — durable truth lives in a shared
PostgreSQL ("akh-pg") on the LAN reached with mutual TLS, LLMs are local (Metal,
via Candle) or cloud (Ollama). The orchestrator "Anima" drives akhs over ACP on a
TCP port. Goal: build akhomed, run a Metal GPU akh + a cloud akh, verify both
connect to the truth store and are reachable by Anima.

Facts you can rely on:
- This Mac is on the LAN; akh-pg is at 192.168.1.16:5432 with MUTUAL TLS — no
  password, a client certificate is the credential. Bundle at ~/akh-certs/:
  ca.crt, mac-client.crt, mac-client.key. Because 192.168.1.16 is an IP (not a
  cert SAN), use sslmode=require (encrypt + client cert; no server-name check) —
  NOT verify-full. Native akhomed runs as you, so the 0600 key is readable (no
  container uid issue).
- Ollama cloud key is in env var OLLAMA_CLOUD_KEY (never print it).
- Build deps present: Xcode CLT + cargo. No cmake/brew needed (oxigraph has no
  rocksdb here).

Do this:
1. Get the source: if ~/akh-medu doesn't exist,
   `git clone https://code.aopc.cloud/toasterson/akh-medu ~/akh-medu`; then
   `cd ~/akh-medu && git checkout develop && git pull`.
2. Build native with Metal (this also includes the cloud/Ollama path):
   `cargo build --release --no-default-features --features server,candle-backend,metal --bin akhomed`
   If a macOS-specific compile error appears, show it and stop — don't hack around
   it. (A cloud-only fallback is --features server,delegation, no model needed.)
3. Config under ~/.config/akh-medu/ :
   - mkdir -p ~/.config/akh-medu/workspaces ~/.config/akh-medu/certs
   - cp ~/akh-certs/* ~/.config/akh-medu/certs/
   - config.toml:
       [postgres]
       url = "postgres://akhomed@192.168.1.16:5432/akhomed?sslmode=require&sslrootcert=/Users/<you>/.config/akh-medu/certs/ca.crt&sslcert=/Users/<you>/.config/akh-medu/certs/mac-client.crt&sslkey=/Users/<you>/.config/akh-medu/certs/mac-client.key"
       migrate_on_start = true
       [acp]
       unix = false
       tcp_bind = "0.0.0.0"
       tcp_port = 8201
       workspace = "thoth-mac"
     (use the absolute home path; ~ is not expanded inside the url)
   - workspaces/thoth-mac.toml  (cloud akh — works immediately):
       name = "thoth-mac"
       is_main = true
       backend = "ollama"
       model = "gpt-oss:120b"
       ollama_endpoint = "https://ollama.com"
       system_prompt = "You are Thoth-mac, a reasoning cloud akh on the Mac mini."
   - workspaces/gemma-mac.toml  (Metal GPU akh — needs the model in step 4):
       name = "gemma-mac"
       backend = "candle"
       model = "gemma4-12b"
       system_prompt = "You are Gemma-mac, a local akh on this Mac's Metal GPU."
4. GPU model (for gemma-mac): place gemma4-12b.gguf in
   ~/.local/share/akh-medu/models/llm/ . NOTE: this Mac has ~16 GB RAM; a 12B
   model (~7 GB q4) is tight under Metal (unified memory). If you hit OOM, use a
   smaller GGUF (e.g. a 4B) and set its name in gemma-mac.toml + AKH_LLM_MODEL.
5. Run the daemon (binds the [acp] workspace; serves the others via `akh -w`):
   `OLLAMA_CLOUD_KEY="$OLLAMA_CLOUD_KEY" AKH_LLM_MODEL=gemma4 AKH_NLU_LLM=0 \
     RUST_LOG=akh_medu=info ./target/release/akhomed`
   (AKH_NLU_LLM=0 avoids loading a second local model — important on 16 GB.)
6. Verify:
   - logs show a Postgres connection + migration and "ACP" serving on TCP 8201,
     with NO "requires a valid client certificate" (that means wrong cert paths);
   - the Metal GPU akh logs a Metal device (not "using CPU device") when gemma-mac
     is exercised: `echo "hi" | ./target/release/akh -w gemma-mac chat --headless`;
   - the cloud akh answers: `echo "Reply: MAC-OK" | ./target/release/akh -w thoth-mac chat --headless`;
   - the akh rows persist: they will appear in akh-pg's akhomed_akh table.
   Report: which akhs are up, the ACP endpoint (this Mac's IP:8201) for Anima,
   and whether Metal was selected for gemma-mac. If the build or a connection
   fails, show the exact error line and stop.

Constraints: never print OLLAMA_CLOUD_KEY or the .key contents. Don't touch
anything outside ~/akh-medu, ~/.config/akh-medu, ~/.local/share/akh-medu.
```

---

After it's up, point Anima at `192.168.1.111:8201`. The Metal GPU akh (gemma-mac)
runs on the Mac's GPU; the cloud akhs share the same truth store as Tecton/Thoth
and the Archibald akh — one distributed akh-medu over the mTLS Postgres.
