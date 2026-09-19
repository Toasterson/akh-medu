set -u
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
export AKH_VULKAN_BATCH_ENCODER=1 AKH_VULKAN_BATCH_PASS=1 AKH_VULKAN_BATCH_SUBMIT=1
run() {
  echo "=== $1 ==="
  env $2 timeout 1800 ./target/release/examples/candle_session_probe \
    --device vulkan --model "$MODEL" --tokenizer "$TOK" \
    --tokens 128 --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0
  echo "=== end $1 rc=$? ==="
}
run "B-byte-layout"   "AKH_QWEN35_KV_QUANT_BITS=4 AKH_QWEN35_KV_QUANT_RECENT=8 AKH_QWEN35_KV_PACK=byte"
run "C-packed-triple" "AKH_QWEN35_KV_QUANT_BITS=4 AKH_QWEN35_KV_QUANT_RECENT=8 AKH_QWEN35_KV_PACK=packed"
