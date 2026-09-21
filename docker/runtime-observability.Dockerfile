# syntax=docker/dockerfile:1
# Preserve provider CLIs/SDKs and helpers; pair runtime changes with fresh measurements.
ARG RUNTIME_BASE
FROM rust:1.94.1-bookworm AS build
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,id=akh-runtime-otlp-arm64,sharing=locked \
    cargo build --locked --release -p akh-session-services --no-default-features \
      --features runtime --bin akhrtd --example native-provider-probe && \
    cp target/release/akhrtd /tmp/akhrtd && \
    cp target/release/examples/native-provider-probe /tmp/native-provider-probe
FROM ${RUNTIME_BASE}
COPY --from=build /tmp/akhrtd /usr/local/bin/akhrtd

COPY --from=build /tmp/native-provider-probe /usr/local/bin/examples/native-provider-probe
COPY scripts/agent-backends/claude.mjs /opt/akh/scripts/agent-backends/claude.mjs
