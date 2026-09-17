# syntax=docker/dockerfile:1
# Preserve the deployed provider/helper image and replace only akhrtd.
ARG RUNTIME_BASE
FROM rust:1.94.1-bookworm AS build
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
RUN --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    --mount=type=cache,target=/src/target,id=akh-runtime-otlp-arm64,sharing=locked \
    cargo build --locked --release -p akh-session-services --no-default-features \
      --features runtime --bin akhrtd && cp target/release/akhrtd /tmp/akhrtd
FROM ${RUNTIME_BASE}
COPY --from=build /tmp/akhrtd /usr/local/bin/akhrtd
