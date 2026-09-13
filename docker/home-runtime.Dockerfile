# syntax=docker/dockerfile:1
# Native ARM64 provider workers and deterministic ingress. Model tools remain
# brokered to akhexecd; the provider container never mounts a workspace.
ARG RUST_VERSION=1.94.1
FROM rust:${RUST_VERSION}-bookworm AS build
ARG TARGETPLATFORM
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,id=akh-home-runtime-${TARGETPLATFORM},sharing=locked \
    cargo build --locked --release -p akh-session-services --no-default-features \
      --features runtime,events,executor,helpers,session \
      --bin akhrtd --bin akheventd --bin akhsessd --bin akhexecd --bin akh-runtime-profile --bin akh-service \
      --example native-provider-probe && \
    mkdir -p /out/examples && \
    cp target/release/akhrtd target/release/akheventd target/release/akhsessd target/release/akhexecd target/release/akh-runtime-profile target/release/akh-service /out/ && \
    cp target/release/examples/native-provider-probe /out/examples/

FROM node:22-bookworm-slim AS runtime
ARG CODEX_VERSION=0.153.4
ARG CLAUDE_VERSION=2.1.269
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates tini python3 git ripgrep procps && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 akh && \
    useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home akh && \
    mkdir -p /data /config /workspaces /opt/providers /opt/akh/scripts/agent-backends && \
    chown akh:akh /data /workspaces && \
    npm install --prefix /opt/providers --no-audit --no-fund \
      "@openai/codex@${CODEX_VERSION}" "@anthropic-ai/claude-code@${CLAUDE_VERSION}" && \
    find /opt/providers/node_modules -type f -name codex -exec ln -s '{}' /usr/local/bin/codex \; && \
    find /opt/providers/node_modules -type f -name codex-code-mode-host -exec ln -s '{}' /usr/local/bin/codex-code-mode-host \; && \
    find /opt/providers/node_modules -type f -name claude -exec ln -s '{}' /usr/local/bin/claude \; && \
    codex --version && claude --version
COPY scripts/agent-backends/package.json scripts/agent-backends/package-lock.json /opt/akh/scripts/agent-backends/
RUN cd /opt/akh/scripts/agent-backends && npm ci --omit=dev --no-audit --no-fund
COPY scripts/agent-backends/claude.mjs /opt/akh/scripts/agent-backends/
COPY scripts/session-platform/codex-native-wire-smoke.py scripts/session-platform/claude-native-wire-smoke.py scripts/session-platform/native-cancel-smoke.py /opt/akh/scripts/session-platform/
COPY --from=build /out/ /usr/local/bin/
USER 10001:10001
WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["/usr/local/bin/akhrtd", "--config", "/config/service.json"]

# Open-model workers and event ingress do not need provider CLIs or their SDKs.
FROM debian:bookworm-slim AS services-base
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini python3 && \
    rm -rf /var/lib/apt/lists/* && groupadd --gid 10001 akh && \
    useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home akh && \
    mkdir -p /data /config && chown akh:akh /data
USER 10001:10001
WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--"]

FROM services-base AS control
COPY --from=build /out/akhsessd /out/akheventd /usr/local/bin/
CMD ["/usr/local/bin/akhsessd", "--config", "/config/service.json"]

FROM services-base AS services
COPY --from=build /out/ /usr/local/bin/
CMD ["/usr/local/bin/akhrtd", "--config", "/config/service.json"]
