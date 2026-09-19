# akh-transcripts — agent-session transcript harvest (ADR 108, plan
# docs/ai/plans/2026-09-19-transcript-harvest.md)
#
# One image, three binaries built from `crates/akh-transcripts`:
#   akh-harvest           — runs on every machine that produces sessions (thin,
#                            no ONNX; kept in this image for convenience, but the
#                            in-cluster deployments only run akh-derive/-mcp)
#   akh-derive            — in-cluster singleton: normalizes, renders, flags
#                            secrets, embeds (needs the baked-in ONNX model)
#   akh-transcripts-mcp   — in-cluster read surface (exact + semantic search)
#
# All three are configured entirely by environment variables — there is no
# config file to mount (see the chart's transcripts-*-deployment.yaml).
#
# Single-arch (amd64), same reasoning as seshd: the ONNX runtime asset baked
# into this image is linux-x64 only, and the chart pins transcripts pods to
# amd64 nodes.
#
# BuildKit cache mounts are keyed on TARGETPLATFORM so a future arm64 leg
# (should one ever get added) does not silently share cache with amd64.
#
# Build:  DOCKER_BUILDKIT=1 docker build -f docker/akh-transcripts.Dockerfile -t akh-transcripts .
# Run:    docker run -e AKH_TRANSCRIPTS_DATABASE_URL=... akh-transcripts akh-derive

# syntax=docker/dockerfile:1

ARG RUST_VERSION=1.97.1

# ── Stage 1: Build ────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM rust:${RUST_VERSION}-bookworm AS builder

ARG TARGETPLATFORM

WORKDIR /build

# Copy manifests first so dependency compilation caches as its own layer.
# akh-transcripts depends on akh-embed by path, so both manifests are needed
# up front, same shape as sokard needing sokar-client's.
COPY Cargo.toml Cargo.lock rust-toolchain.toml* ./
COPY crates/akh-transcripts/Cargo.toml crates/akh-transcripts/Cargo.toml
COPY crates/akh-embed/Cargo.toml crates/akh-embed/Cargo.toml
RUN mkdir -p src/bin crates/akh-transcripts/src/bin crates/akh-embed/src benches && \
    echo 'fn main() {}' > src/main.rs && \
    echo 'fn main() {}' > src/bin/akhomed.rs && \
    touch src/lib.rs && \
    touch crates/akh-transcripts/src/lib.rs && \
    echo 'fn main() {}' > crates/akh-transcripts/src/bin/akh-harvest.rs && \
    echo 'fn main() {}' > crates/akh-transcripts/src/bin/akh-derive.rs && \
    echo 'fn main() {}' > crates/akh-transcripts/src/bin/akh-transcripts-mcp.rs && \
    touch crates/akh-embed/src/lib.rs && \
    echo 'fn main() {}' > benches/vsa_ops.rs && \
    echo 'fn main() {}' > benches/phase26.rs

# Pre-build dependencies with the stub entry points, cached per target
# platform. `|| true` so the stub phase doesn't fail the build if anything
# goes sideways (the three `[[bin]]`/auto-discovered targets above are a
# best-effort stand-in until the real binaries land — see the Dockerfile
# header).
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-akh-transcripts-${TARGETPLATFORM} \
    cargo build --release -p akh-transcripts --bins || true

# Copy full source and build the real binaries.
# `touch` forces cargo to re-evaluate mtimes (docker COPY preserves them and
# can leave cargo thinking the stub sources are still current). `rm -f` on
# the cached stub binaries is belt-and-suspenders so we never cp a stub.
COPY . .
RUN --mount=type=cache,target=/usr/local/cargo/registry,id=cargo-registry-${TARGETPLATFORM} \
    --mount=type=cache,target=/build/target,id=cargo-target-akh-transcripts-${TARGETPLATFORM} \
    find src crates benches -name '*.rs' -exec touch {} + && \
    rm -f target/release/akh-harvest target/release/akh-derive target/release/akh-transcripts-mcp && \
    cargo build --release -p akh-transcripts --bins && \
    mkdir -p /build/out && \
    cp target/release/akh-harvest target/release/akh-derive target/release/akh-transcripts-mcp /build/out/

# ── Stage 2: ONNX Runtime (for sentence embedding) ───────────────────────
# Identical to seshd's ort-fetch stage — same asset, same version.
FROM debian:bookworm-slim AS ort-fetch

ARG ORT_VERSION=1.23.2

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | \
    tar xzf - --strip-components=1 -C /tmp && \
    mkdir -p /opt/ort/lib && \
    cp /tmp/lib/libonnxruntime.so* /opt/ort/lib/

# ── Stage 3: embedding model asset ───────────────────────────────────────
# Baked into the image (embedding-pipeline decision 4): both akh-derive and
# akh-transcripts-mcp need the same model in-cluster, and baking it in means
# no startup network dependency and a deterministic asset for both.
FROM debian:bookworm-slim AS model-fetch

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN set -eu; \
    mkdir -p /models/all-MiniLM-L6-v2; \
    curl -fSL -o /models/all-MiniLM-L6-v2/model.onnx \
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx?download=true"; \
    curl -fSL -o /models/all-MiniLM-L6-v2/tokenizer.json \
      "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json?download=true"; \
    # A truncated/HTML error-page download is the failure mode that bit the
    # embedding-pipeline research (see the plan) — assert plausible sizes
    # rather than trusting a 200 status alone.
    model_size=$(stat -c %s /models/all-MiniLM-L6-v2/model.onnx); \
    tok_size=$(stat -c %s /models/all-MiniLM-L6-v2/tokenizer.json); \
    echo "model.onnx: ${model_size} bytes, tokenizer.json: ${tok_size} bytes"; \
    if [ "$model_size" -le 52428800 ]; then \
      echo "model.onnx is ${model_size} bytes, expected > 50MB — looks like an error page, not the model" >&2; \
      exit 1; \
    fi; \
    if [ "$tok_size" -le 307200 ]; then \
      echo "tokenizer.json is ${tok_size} bytes, expected > 300KB — looks like an error page, not the tokenizer" >&2; \
      exit 1; \
    fi

# ── Stage 4: Runtime ─────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd -r -g 999 akh-transcripts && \
    useradd -r -u 999 -g akh-transcripts -d /data -s /sbin/nologin akh-transcripts && \
    mkdir -p /data /models && chown akh-transcripts:akh-transcripts /data /models

COPY --from=builder /build/out/akh-harvest /usr/local/bin/akh-harvest
COPY --from=builder /build/out/akh-derive /usr/local/bin/akh-derive
COPY --from=builder /build/out/akh-transcripts-mcp /usr/local/bin/akh-transcripts-mcp
COPY --from=ort-fetch /opt/ort/lib/ /usr/local/lib/
COPY --from=model-fetch /models/ /models/

RUN ldconfig

ENV ORT_DYLIB_PATH=/usr/local/lib/libonnxruntime.so \
    AKH_TRANSCRIPTS_MODEL_DIR=/models/all-MiniLM-L6-v2 \
    AKH_TRANSCRIPTS_BIND=0.0.0.0:9500 \
    AKH_TRANSCRIPTS_DERIVE_INTERVAL_SECS=30 \
    AKH_TRANSCRIPTS_DERIVE_BATCH=200 \
    RUST_LOG=akh_transcripts=info

# MCP + health/readiness all share this port (path-routed: /mcp, /healthz, /readyz).
EXPOSE 9500

USER akh-transcripts
WORKDIR /data

VOLUME ["/data", "/models"]

ENTRYPOINT ["tini", "--"]
# No default CMD — the chart's deployments each pick the binary
# (akh-derive / akh-transcripts-mcp) and pass --migrate where needed.
CMD ["akh-transcripts-mcp"]
