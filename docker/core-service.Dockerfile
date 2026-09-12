# syntax=docker/dockerfile:1
# Native build of the cognitive service, with no embedded loop/ACP profile.
ARG RUST_VERSION=1.94.1
FROM rust:${RUST_VERSION}-bookworm AS build
ARG TARGETPLATFORM
RUN apt-get update && apt-get install -y --no-install-recommends \
    clang libclang-dev protobuf-compiler libssl-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=0 CARGO_INCREMENTAL=0
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,id=akh-core-service-${TARGETPLATFORM} \
    cargo build --locked --release --no-default-features --features core-server --bin akhomed && \
    cp target/release/akhomed /akhomed
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini && \
    rm -rf /var/lib/apt/lists/* && groupadd --gid 10001 akh && \
    useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home akh && \
    mkdir -p /data /config && chown akh:akh /data
COPY --from=build /akhomed /usr/local/bin/akhomed
ENV AKH_HOME=/data AKH_CORE_CONFIG=/config/core.json RUST_LOG=info
USER 10001:10001
WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/akhomed"]
