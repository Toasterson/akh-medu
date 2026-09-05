# syntax=docker/dockerfile:1.7

FROM rust:1.97-bookworm AS builder
WORKDIR /src
COPY . .
RUN cargo build --locked --release -p grug-train \
    --example eval_native_ocr \
    --example certify_multimodal_model

FROM debian:bookworm-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates python3 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /src/target/release/examples/eval_native_ocr /usr/local/bin/eval_native_ocr
COPY --from=builder /src/target/release/examples/certify_multimodal_model /usr/local/bin/certify_multimodal_model
COPY training/scripts/run_tool_native_adaptive_loop.py /opt/akh-medu/run_tool_native_adaptive_loop.py
COPY training/scripts/run_native_ocr_offline_gate.py /opt/akh-medu/run_native_ocr_offline_gate.py
ENV PYTHONDONTWRITEBYTECODE=1
ENTRYPOINT ["python3", "/opt/akh-medu/run_native_ocr_offline_gate.py", \
    "--eval-binary", "/usr/local/bin/eval_native_ocr", \
    "--certify-binary", "/usr/local/bin/certify_multimodal_model"]
