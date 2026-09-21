set -u
DEV="$1"; shift
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
export AKH_VULKAN_BATCH_ENCODER=1 AKH_VULKAN_BATCH_PASS=1 AKH_VULKAN_BATCH_SUBMIT=1
export AKH_QWEN35_KV_PAGE_FIDELITY=1
for BITS in 8 4 3 2; do
  echo "=== $DEV bits=$BITS ==="
  AKH_QWEN35_KV_QUANT_BITS=$BITS AKH_QWEN35_KV_QUANT_RECENT=8 \
  timeout 2400 ./target/release/examples/candle_session_probe \
    --device "$DEV" --model "$MODEL" --tokenizer "$TOK" \
    --tokens 128 --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0 \
    2>&1 | grep -E 'kv_pages|Error|"text"' | sed 's/.*"text":"\([^"]*\)".*/TEXT: \1/'
  echo "=== end $DEV bits=$BITS rc=$? ==="
done
