# syntax=docker/dockerfile:1
ARG EXECUTOR_IMAGE
ARG RUST_IMAGE
ARG NODE_IMAGE
FROM ${EXECUTOR_IMAGE} AS executor
FROM ${NODE_IMAGE} AS node
FROM ${RUST_IMAGE} AS git-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4-openssl-dev libexpat1-dev zlib1g-dev libssl-dev xz-utils && \
    curl --proto '=https' --tlsv1.2 -fsSL https://www.kernel.org/pub/software/scm/git/git-2.55.0.tar.xz -o /tmp/git.tar.xz && \
    printf '%s  %s\n' 457fdb04dc8728e007d4688695e6912e6f680727920f2a40bf11eacc17505357 /tmp/git.tar.xz | sha256sum -c - && \
    tar -xJf /tmp/git.tar.xz -C /tmp && \
    make -C /tmp/git-2.55.0 -j2 prefix=/usr NO_GETTEXT=YesPlease NO_TCLTK=YesPlease all && \
    make -C /tmp/git-2.55.0 prefix=/usr DESTDIR=/git-install NO_GETTEXT=YesPlease NO_TCLTK=YesPlease install && \
    strip /git-install/usr/bin/git
FROM ${RUST_IMAGE} AS workspace
RUN apt-get update && apt-get install -y --no-install-recommends \
    git python3 curl ca-certificates tini clang libclang-dev cmake pkg-config \
    libssl-dev libcurl4 libexpat1 protobuf-compiler libprotobuf-dev ripgrep unzip jq && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 akh && useradd --uid 10001 --gid 10001 --home-dir /data/home akh
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -s /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    npm install --global t3@0.0.40 && npm cache clean --force
COPY --from=executor /usr/local/bin/akhexecd /usr/local/bin/akhexecd
COPY --from=executor /usr/local/bin/akh-service /usr/local/bin/akh-service
COPY --from=git-build /git-install/usr/ /usr/
RUN /usr/bin/git --no-lazy-fetch --version
RUN curl --proto '=https' --tlsv1.2 -fsSL https://github.com/mozilla/sccache/releases/download/v0.17.0/sccache-v0.17.0-x86_64-unknown-linux-musl.tar.gz -o /tmp/sccache.tar.gz && \
    printf '%s  %s\n' 67c4a96dd237c1f518f6b36083f270f9976d516f1e57fce891755ea782e50006 /tmp/sccache.tar.gz | sha256sum -c - && \
    tar -xzf /tmp/sccache.tar.gz -C /tmp && \
    install -m 0755 /tmp/sccache-v0.17.0-x86_64-unknown-linux-musl/sccache /usr/local/bin/sccache && \
    rm -rf /tmp/sccache.tar.gz /tmp/sccache-v0.17.0-x86_64-unknown-linux-musl && \
    /usr/local/bin/sccache --version
COPY agent-board-coordination /opt/agent-board-coordination
RUN chmod -R a+rX /opt/agent-board-coordination
COPY rust-tool /usr/local/bin/akh-rust-tool
RUN chmod 755 /usr/local/bin/akh-rust-tool && \
    for tool in cargo rustc rustfmt cargo-fmt clippy-driver cargo-clippy; do ln -s /usr/local/bin/akh-rust-tool /usr/local/bin/$tool; done && \
    mkdir -p /data /workspaces /config /usr/local/share/akh && chown -R 10001:10001 /data /workspaces && \
    python3 -c 'import hashlib,json,pathlib,subprocess; p=pathlib.Path("/usr/bin/git"); print(json.dumps({"git_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"git_version":subprocess.check_output([str(p),"--version"],text=True).strip(),"rust_version":subprocess.check_output(["/usr/local/cargo/bin/rustc","--version"],text=True).strip(),"node_version":subprocess.check_output(["node","--version"],text=True).strip(),"t3_version":"0.0.40","sccache_version":subprocess.check_output(["/usr/local/bin/sccache","--version"],text=True).strip()}))' > /usr/local/share/akh/workspace-identity.json
USER 10001:10001
WORKDIR /data
ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/akhexecd"]
CMD ["--config", "/config/service.json"]
FROM scratch AS identity
COPY --from=workspace /usr/local/share/akh/workspace-identity.json /workspace-identity.json
