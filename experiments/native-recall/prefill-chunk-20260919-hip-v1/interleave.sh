set -u
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
for PASS in 1 2 3; do
  for MODE in off on; do
    if [ "$MODE" = on ]; then export AKH_QWEN35_GROUPED_ATTENTION=1; else unset AKH_QWEN35_GROUPED_ATTENTION; fi
    T=$(timeout 2400 ./target/release/examples/candle_session_probe \
      --device hip --model "$MODEL" --tokenizer "$TOK" \
      --prompt-file /tmp/ctxsoak/prompt-05k.txt --tokens 4 \
      --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0 2>&1 \
      | grep -oE '"time_to_first_token_seconds":[0-9.]*' | head -1 | cut -d: -f2)
    echo "pass=$PASS grouped=$MODE ttft=$T"
  done
done
