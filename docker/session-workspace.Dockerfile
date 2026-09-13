# syntax=docker/dockerfile:1
# Add Akh's executor/helper to a selected, unmodified upstream workspace image.
# Both input images must be pinned by digest by the image pipeline. They must
# have the same architecture and a glibc >= Debian bookworm (2.36).
ARG EXECUTOR_IMAGE
ARG WORKSPACE_IMAGE
FROM ${EXECUTOR_IMAGE} AS executor
FROM ${WORKSPACE_IMAGE}
COPY --from=executor /usr/local/bin/akhexecd /usr/local/bin/akhexecd
COPY --from=executor /usr/local/bin/akh-service /usr/local/bin/akh-service
# The provisioner supplies UID, writable data/workspace mounts and scoped config.
# This image installs no runtime credentials, T3 connection DB, or patched editor.
ENTRYPOINT ["/usr/local/bin/akhexecd"]
CMD ["--config", "/config/service.json"]
