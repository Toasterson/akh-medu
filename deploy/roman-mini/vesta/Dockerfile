FROM python:3.12-slim-bookworm
ARG KUBECTL_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates jq yamllint && rm -rf /var/lib/apt/lists/*
RUN arch="$(dpkg --print-architecture)" && curl -fsSLo /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl" && curl -fsSLo /tmp/kubectl.sha256 "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${arch}/kubectl.sha256" && printf '%s  /usr/local/bin/kubectl\n' "$(cat /tmp/kubectl.sha256)" | sha256sum -c - && chmod 755 /usr/local/bin/kubectl && rm /tmp/kubectl.sha256
ENV HOME=/home/vesta
WORKDIR /workspace
CMD ["sleep", "infinity"]
