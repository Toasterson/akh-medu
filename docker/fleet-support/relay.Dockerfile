FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends openssh-client python3 curl tini ca-certificates && \
    rm -rf /var/lib/apt/lists/* && groupadd -g 10001 akh && useradd -u 10001 -g 10001 -M -d /tmp akh
USER 10001:10001
ENTRYPOINT ["/usr/bin/tini", "--", "python3", "/config/relay.py"]
