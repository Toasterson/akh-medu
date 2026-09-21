set -u
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
for CHUNK in 128 256 512 1024; do
  echo "=== chunk=$CHUNK ==="
  AKH_QWEN35_PREFILL_CHUNK=$CHUNK AKH_CANDLE_CONTEXT_COST_MULTIPLIER=4.5 \
  timeout 2400 ./target/release/examples/candle_session_probe \
    --device hip --model "$MODEL" --tokenizer "$TOK" \
    --prompt-file /tmp/ctxsoak/prompt-08k.txt \
    --tokens 8 --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0 \
    2>&1 | sed 's/"text":"[^"]*"/"text":"<elided>"/' | grep -E 'time_to_first_token|kv_pages|Error|memory_after'
  echo "=== end chunk=$CHUNK rc=$? ==="
done
