set -u
MODEL=/usr/share/ollama/.ollama/models/blobs/sha256-3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec
TOK=/home/toasty/.local/share/akh-medu/models/llm/ptah-qwen38.tokenizer.json
VRAM=/sys/class/drm/card0/device/mem_info_vram_used
for CHUNK in 128 256; do
  AKH_QWEN35_PREFILL_CHUNK=$CHUNK timeout 2400 ./target/release/examples/candle_session_probe \
    --device hip --model "$MODEL" --tokenizer "$TOK" \
    --prompt-file /tmp/ctxsoak/prompt-08k.txt --tokens 4 \
    --idle-samples 1 --settle-seconds 0 --baseline-settle-seconds 0 >/dev/null 2>&1 &
  PROBE=$!
  PEAK=0
  while kill -0 $PROBE 2>/dev/null; do
    U=$(cat $VRAM 2>/dev/null || echo 0)
    [ "$U" -gt "$PEAK" ] && PEAK=$U
    sleep 2
  done
  echo "chunk=$CHUNK peak_vram_used_bytes=$PEAK"
done
