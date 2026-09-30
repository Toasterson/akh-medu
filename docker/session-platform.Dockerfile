# syntax=docker/dockerfile:1
# Native per-platform build: pass --platform to buildx; do not share target caches
# with the embedded cognitive engine or another build profile.
ARG RUST_VERSION=1.94.1
FROM rust:${RUST_VERSION}-bookworm AS build
ARG SERVICE=akhsessd
ARG PROFILE=session
ARG TARGETPLATFORM
WORKDIR /src
COPY . .
ENV CARGO_BUILD_JOBS=2 CARGO_PROFILE_RELEASE_DEBUG=line-tables-only CARGO_INCREMENTAL=0
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target,id=akh-session-platform-${TARGETPLATFORM} \
    case "$SERVICE:$PROFILE" in \
      akhsessd:session|akhrtd:runtime|akhexecd:executor|akhsbxd:sandbox|akheventd:events|akh-trigger:trigger|akh-connect:client|akh-acp:client|akh-kagent:kagent) ;; \
      *) exit 2 ;; esac && \
    cargo build --locked --release -p akh-session-services --no-default-features --features "$PROFILE,profiling" --bin "$SERVICE" && \
    mkdir /out && cp "target/release/$SERVICE" /out/ && \
    case "$SERVICE" in \
      akhrtd|akhexecd) cargo build --locked --release -p akh-session-services --no-default-features --features helpers --bin akh-service && cp target/release/akh-service /out/ ;; \
    esac && \
    if [ "$SERVICE" = akhrtd ]; then cargo build --locked --release -p akh-session-services --no-default-features --features runtime --bin akh-runtime-profile && cp target/release/akh-runtime-profile /out/; fi && \
    ln -s "$SERVICE" /out/akh-entrypoint
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tini python3 && \
    rm -rf /var/lib/apt/lists/* && groupadd --gid 10001 akh && \
    useradd --uid 10001 --gid 10001 --home-dir /data --no-create-home akh && \
    mkdir -p /data /config /workspaces && chown akh:akh /data /workspaces
COPY --from=build /out/ /usr/local/bin/
USER 10001:10001
WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/akh-entrypoint"]
CMD ["--config", "/config/service.json"]
