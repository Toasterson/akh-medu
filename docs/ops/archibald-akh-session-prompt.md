# Archibald cloud-akh — Claude Code session prompt

Copy everything in the fenced block below into a fresh Claude Code session running
on **Archibald** (Linux). Before you do, place two things on Archibald:

1. **Cert bundle** (from hephaistos) — run on hephaistos:
   ```bash
   ssh toasty@archibald 'mkdir -p ~/akh-certs'
   scp ~/akh-3akh-test/pg-certs/{ca.crt,archibald-client.crt,archibald-client.key} \
       toasty@archibald:~/akh-certs/
   ```
2. **Ollama cloud key** — ensure `OLLAMA_CLOUD_KEY` is exported in the Archibald
   session's shell (e.g. in `~/env-file`, `set -a; . ~/env-file; set +a`).

Also confirm the akh-pg endpoint is reachable from Archibald
(`nc -z akh-pg.wegmueller.it 5432`); if DNS for that name isn't set yet, use the
public IP and `sslmode=require` instead of `verify-full` (noted in the prompt).

---

```
You are setting up a "cloud akh" on this Linux host (Archibald), part of the
akh-medu distributed agent system. An akh is a container that runs the `akhomed`
daemon; it is stateless compute — its durable truth lives in a shared PostgreSQL
("akh-pg") reached over the Internet with mutual TLS, and its LLM is a cloud model
served via Ollama. The orchestrator "Anima" will drive this akh over ACP (Agent
Client Protocol) on a TCP port. Goal: run this akh so it connects to the truth
store and is reachable by Anima, then verify it.

Facts you can rely on:
- Image: code.aopc.cloud/toasterson/akhomed:cloud  (multi-arch; docker picks this
  host's arch). Run `docker login code.aopc.cloud` first if the pull is denied.
- Container runtime: prefer `docker`; if absent, use `podman` (same flags).
- Postgres endpoint: akh-pg.wegmueller.it:5432, MUTUAL TLS — no password, a client
  certificate is the credential. The bundle is at ~/akh-certs/:
  ca.crt, archibald-client.crt, archibald-client.key (key must be chmod 600).
  If `akh-pg.wegmueller.it` does NOT resolve yet, ask me for the public IP and use
  `sslmode=require` (drop sslrootcert/verify-full) against IP:5432 — still mTLS.
- Ollama cloud key is in the env var OLLAMA_CLOUD_KEY (do not print it).

Do this:
1. Verify reachability: `nc -zv akh-pg.wegmueller.it 5432`. Verify the cert bundle
   exists and the key is 0600.
2. Pull the image: `docker pull code.aopc.cloud/toasterson/akhomed:cloud`.
3. Create the config tree under ~/akh-thoth/config/akh-medu/ :
   - copy ~/akh-certs/* into ~/akh-thoth/config/akh-medu/certs/
   - config.toml:
       [postgres]
       url = "postgres://akhomed@akh-pg.wegmueller.it:5432/akhomed?sslmode=verify-full&sslrootcert=/config/akh-medu/certs/ca.crt&sslcert=/config/akh-medu/certs/archibald-client.crt&sslkey=/config/akh-medu/certs/archibald-client.key"
       migrate_on_start = true
       [acp]
       unix = false
       tcp_bind = "0.0.0.0"
       tcp_port = 8201
       workspace = "thoth"
   - workspaces/thoth.toml:
       name = "thoth"
       is_main = true
       backend = "ollama"
       model = "gpt-oss:120b"
       ollama_endpoint = "https://ollama.com"
       system_prompt = "You are Thoth, a reasoning akh running on Archibald. Help set up and test the OpenIndiana projects; ground decisions in your knowledge graph; delegate to tools when useful."
4. Run it (mount config read-only, persist /data, expose ACP, pass the key):
   docker run -d --name akh-thoth --restart unless-stopped \
     -v ~/akh-thoth/config:/config:ro \
     -v akh-thoth-data:/data \
     -e OLLAMA_CLOUD_KEY="$OLLAMA_CLOUD_KEY" \
     -e AKH_NLU_LLM=0 \
     -p 8201:8201 \
     code.aopc.cloud/toasterson/akhomed:cloud
5. Verify: `docker logs akh-thoth` shows the ACP listener up on 0.0.0.0:8201 and
   NO Postgres/TLS error (look for a successful migration / "serving"). If you see
   "connection requires a valid client certificate" the cert paths are wrong;
   "certificate verify failed" means use the IP + sslmode=require (DNS/SAN issue).
6. Report: the container status, the ACP TCP endpoint (this host:8201) for Anima,
   and whether migrations ran. If anything fails, show the exact `docker logs`
   line and stop — don't guess at credentials.

Constraints: never print OLLAMA_CLOUD_KEY or the contents of the .key file. Don't
modify anything outside ~/akh-thoth and the docker container.
```

---

After it's up, point Anima (cluster ns `anima`) at `archibald:8201` as the akh's
ACP endpoint. The OpenIndiana zone akh is a later step — same image + bundle, run
inside the zone, same config.
