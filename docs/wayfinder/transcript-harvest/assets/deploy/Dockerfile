# CloudNativePG image for the transcript-harvest store: upstream CNPG base plus
# TimescaleDB. pgvector is already present in the base (verified: `vector` shows
# in pg_available_extensions on every cluster in the fleet).
#
# Built the same way as the existing code.aopc.cloud/toasterson/cnpg-age image,
# so it slots into the established registry + Flux workflow.
#
#   docker buildx build --platform linux/amd64,linux/arm64 \
#     -t code.aopc.cloud/toasterson/cnpg-timescale:18-2.25.1 --push .
#
# arm64 matters here: the fleet's compute tier is arm64 and Timescale publishes
# trixie packages for both architectures.

ARG CNPG_BASE=ghcr.io/cloudnative-pg/postgresql:18.3-system-trixie
FROM ${CNPG_BASE}

ARG PG_MAJOR=18
# Pin explicitly. TimescaleDB gained PostgreSQL 18 support in 2.23; 2.25.1 is the
# trixie/pg18 build confirmed present in the packagecloud repo.
ARG TIMESCALE_VERSION=2.25.1
# Set by buildx per target platform. Cache mounts are keyed on it so parallel
# amd64/arm64 builds never share (and corrupt) one apt cache.
ARG TARGETPLATFORM

USER root

# docker-clean deletes .debs post-install, which defeats a cache mount.
RUN rm -f /etc/apt/apt.conf.d/docker-clean \
 && echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-debs

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked,id=apt-cache-${TARGETPLATFORM} \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked,id=apt-lists-${TARGETPLATFORM} \
    set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release; \
    install -d -m 0755 /usr/share/keyrings; \
    curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey \
      | gpg --dearmor -o /usr/share/keyrings/timescaledb.gpg; \
    echo "deb [signed-by=/usr/share/keyrings/timescaledb.gpg] \
https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -c -s) main" \
      > /etc/apt/sources.list.d/timescaledb.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
      "timescaledb-2-postgresql-${PG_MAJOR}=${TIMESCALE_VERSION}~debian13*" \
      "timescaledb-2-loader-postgresql-${PG_MAJOR}=${TIMESCALE_VERSION}~debian13*"; \
    apt-get purge -y --auto-remove curl gnupg lsb-release; \
    rm -rf /var/lib/apt/lists/*

# Fail the build rather than ship an image that silently lacks the extension —
# this is exactly the failure mode that left akhomed_mutation_log a plain table
# for 769k rows while ADR 091 documented it as a hypertable.
RUN set -eux; \
    test -f "/usr/lib/postgresql/${PG_MAJOR}/lib/timescaledb.so"; \
    test -e "/usr/share/postgresql/${PG_MAJOR}/extension/timescaledb.control"; \
    test -e "/usr/share/postgresql/${PG_MAJOR}/extension/vector.control"

# CNPG runs as uid 26.
USER 26
