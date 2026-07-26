# Running a cloud akh in a container

A **cloud akh** (Tecton, Thoth, an OpenIndiana-zone akh driven by Anima) is
stateless compute: its truth lives in the shared `akh-pg` Postgres (mTLS), and its
LLM is a cloud model over Ollama. So it ships as a small OCI image and can run
anywhere it can reach Postgres + Ollama — a cluster arm64 worker, the Mac mini, or
an OpenIndiana zone on Archibald.

Image: `code.aopc.cloud/toasterson/akhomed:cloud` (multi-arch amd64+arm64), built
from [`docker/akhomed-cloud.Dockerfile`](../../docker/akhomed-cloud.Dockerfile)
(features `server,delegation` — no Candle/GPU, no ONNX).

## Build + push (on an amd64 builder, e.g. hephaistos — arm64 is cross-compiled)

```bash
docker buildx create --name akh-builder --driver docker-container --bootstrap   # once
docker buildx build --builder akh-builder -f docker/akhomed-cloud.Dockerfile \
  --platform linux/amd64,linux/arm64 \
  -t code.aopc.cloud/toasterson/akhomed:cloud --push .
```

## Config layout (mounted read-only at /config)

```
config/akh-medu/
  config.toml            # [postgres] mTLS + [acp] tcp
  workspaces/thoth.toml  # the akh's persona + backend
  certs/                 # the mTLS bundle (from docs/ops/akh-pg-remote-mtls.md)
    ca.crt  akhomed-client.crt  akhomed-client.key
```

`config.toml`:

```toml
[postgres]
url = "postgres://akhomed@akh-pg.wegmueller.it:5432/akhomed?sslmode=verify-full&sslrootcert=/config/akh-medu/certs/ca.crt&sslcert=/config/akh-medu/certs/akhomed-client.crt&sslkey=/config/akh-medu/certs/akhomed-client.key"
migrate_on_start = true

[acp]
unix = false
tcp_bind = "0.0.0.0"   # so Anima (in-cluster) can reach this akh's ACP
tcp_port = 8201
workspace = "thoth"
```

`workspaces/thoth.toml` (a reasoning cloud akh):

```toml
name = "thoth"
is_main = true
backend = "ollama"
model = "gpt-oss:120b"
ollama_endpoint = "https://ollama.com"
system_prompt = "You are Thoth, a reasoning akh. Set up and test the OpenIndiana projects; ground decisions in your knowledge graph; delegate to tools when useful."
```

The Ollama cloud key is passed as an env var, not baked into the image
(`OLLAMA_CLOUD_KEY`).

**Cert permissions (gotcha):** the container runs as its own non-root user, which
won't match the host uid that owns the mounted cert files. The client key is
`0600` by default → unreadable inside the container → the akh silently falls back
to local-only (no Postgres). Make the bundle readable to the container user:

```bash
chmod 0644 config/akh-medu/certs/akhomed-client.key   # readable to the container's akh user
```

(sqlx/rustls doesn't enforce key-file perms the way libpq does, so `0644` is fine.
The security boundary is the server's mTLS requirement, not the local key mode.)
Alternatively run with `--user "$(id -u):$(id -g)"` and keep `0600`, but then the
`/data` volume must be writable by that uid.

## Run (docker / podman)

```bash
docker run -d --name akh-thoth --restart unless-stopped \
  -v "$PWD/config:/config:ro" \
  -v akh-thoth-data:/data \
  -e OLLAMA_CLOUD_KEY="$OLLAMA_CLOUD_KEY" \
  -e AKH_NLU_LLM=0 \
  -p 8201:8201 \
  code.aopc.cloud/toasterson/akhomed:cloud
```

- `8201` is the ACP TCP endpoint Anima connects to.
- The container connects to `akh-pg.wegmueller.it:5432` over mTLS (the cert bundle
  under `/config/.../certs/`), runs migrations, and persists to the shared store.
- Verify: `docker logs akh-thoth` shows the ACP listener up and no Postgres error;
  the akh row appears in `akhomed_akh` (query with the client cert per the mTLS
  runbook).

## OpenIndiana / Archibald

OpenIndiana runs the Linux image in an **lx-branded zone** (or a Linux VM). Either
way it's the same OCI image; in an lx zone use `podman`/`docker` inside the zone,
or import the image rootfs. The akh then reaches `akh-pg.wegmueller.it` over the
Internet (mTLS) and Anima drives it over ACP TCP — letting an akh set up and test
the OpenIndiana projects in place.

## Anima ACP wiring

Anima (cluster ns `anima`) is the ACP **client**; it connects out to the akh's ACP
TCP endpoint (`<host>:8201`). Ensure that endpoint is reachable from the cluster
(LAN, or via the port-forward for an off-site host). The akh is the ACP **agent**.
