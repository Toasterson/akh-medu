# syntax=docker/dockerfile:1
# Vesta sandbox image (akh-medu 53g PR 8). Built on the fleet workspace image
# pinned in clusters/archibald/akh-platform/fleet.json (images.workspace), so
# akhexecd, git, python3, jq and tini are the exact binaries the fleet's
# Sandbox pods already run (no glibc/runtime drift for akhexecd). Adds what
# the retired roman-vesta-gitops container had on top (akh-medu
# deploy/roman-mini/vesta/Dockerfile: yamllint, kubectl) and the SSH client
# the seed clone and the vesta/* push need. One image serves both profiles;
# the profile decides credentials (config/sandbox-service.json).
#
# Build, push, then pin the digest in config/sandbox-service.json (README.md
# "Rollout" step 2):
#   docker buildx build --platform linux/amd64 --build-arg KUBECTL_VERSION=v1.32.2 \
#     -t code.aopc.cloud/toasterson/vesta-sandbox:vesta-v2 .
FROM code.aopc.cloud/toasterson/akh-workspace@sha256:adbabeee9f973768c9cb8c9bc29d3a1115194cb6d554c57e89ad03381bb21cd4
ARG KUBECTL_VERSION
USER 0:0
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client yamllint && \
    rm -rf /var/lib/apt/lists/* && \
    arch="$(dpkg --print-architecture)" && \
    curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl" && \
    curl -fsSLo /tmp/kubectl.sha256 "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl.sha256" && \
    printf '%s  /usr/local/bin/kubectl\n' "$(cat /tmp/kubectl.sha256)" | sha256sum -c - && \
    chmod 755 /usr/local/bin/kubectl && rm /tmp/kubectl.sha256 && \
    mkdir -p /seed && chown 10001:10001 /seed && \
    mkdir -p /usr/local/libexec && mv /usr/local/bin/kubectl /usr/local/libexec/kubectl && \
    printf '%s\n' '#!/bin/sh' \
      '# akhexecd runs commands with only LANG/PATH/PWD, so KUBECONFIG from the' \
      '# container env never reaches kubectl; the ops profile mounts its home-cluster' \
      '# kubeconfig here (akh-medu 53g PR 8).' \
      'if [ -z "$KUBECONFIG" ] && [ -r /run/vesta/kubeconfig/config ]; then' \
      '  exec /usr/local/libexec/kubectl --kubeconfig=/run/vesta/kubeconfig/config "$@"' \
      'fi' \
      'exec /usr/local/libexec/kubectl "$@"' > /usr/local/bin/kubectl && \
    chmod 755 /usr/local/bin/kubectl && \
    printf '%s\n' '# Environment-independent git settings (akhexecd drops the container env).' \
      '[user]' '    name = Vesta' '    email = vesta@akh.lab.home' \
      '[core]' '    sshCommand = /usr/bin/ssh -i /run/vesta/git/identity -o IdentitiesOnly=yes -o UserKnownHostsFile=/run/vesta/git/known_hosts -o StrictHostKeyChecking=yes' \
      > /etc/gitconfig
USER 10001:10001
