set -u
DEV="$1"
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
export AKH_VULKAN_BATCH_ENCODER=1 AKH_VULKAN_BATCH_PASS=1 AKH_VULKAN_BATCH_SUBMIT=1
for SIZE in 08k 16k 32k 48k 64k; do
  echo "=== $DEV prompt=$SIZE ==="
  timeout 3600 ./target/release/examples/candle_session_probe \
    --device "$DEV" --model "$MODEL" --tokenizer "$TOK" \
    --prompt-file "/tmp/ctxsoak/prompt-$SIZE.txt" \
    --tokens 16 --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0
  echo "=== end $DEV $SIZE rc=$? ==="
done
