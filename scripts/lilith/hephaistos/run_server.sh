#!/usr/bin/env bash
# ExecStart of lilith-serve.service (ADR 121): build llama-server's command line from the
# environment file and exec it. Installed to $PREFIX/bin/run_server.sh by install_service.sh; can
# also be run by hand after `set -a; . /etc/lilith-serve/server.env; set +a`.
#
# The API key is never read here: llama-server reads it from LILITH_API_KEY_FILE itself
# (--api-key-file), so it is on no command line and in no environment.
#
# Required environment (server.env.example documents all of them):
#   LLAMA_SERVER  LILITH_GGUF  LILITH_CHAT_TEMPLATE  LILITH_API_KEY_FILE
#   LILITH_LLAMA_HOST  LILITH_LLAMA_PORT  LILITH_LLAMA_PARALLEL  LILITH_LLAMA_CTX_PER_SLOT
# Optional: LILITH_MODEL_NAME (lilith) LILITH_LLAMA_KV_TYPE (f16) LILITH_LLAMA_CACHE_RAM_MIB (4096)
#   LILITH_LLAMA_SLEEP_IDLE_SECONDS (0 = never unload) LILITH_LLAMA_EXTRA_ARGS (space-separated flags)
set -euo pipefail

fail() { # fail CODE MESSAGE HELP
  printf 'run_server: error[%s]: %s\n  help: %s\n' "$1" "$2" "$3" >&2
  exit 1
}
need_var() { [[ -n "${!1:-}" ]] || fail lilith::heph::env "$1 is not set" "set it in /etc/lilith-serve/server.env (see server.env.example)"; }

for name in LLAMA_SERVER LILITH_GGUF LILITH_CHAT_TEMPLATE LILITH_API_KEY_FILE LILITH_LLAMA_HOST \
  LILITH_LLAMA_PORT LILITH_LLAMA_PARALLEL LILITH_LLAMA_CTX_PER_SLOT; do
  need_var "$name"
done
[[ -x "$LLAMA_SERVER" ]] || fail lilith::heph::server "$LLAMA_SERVER is not an executable" \
  "run scripts/lilith/hephaistos/build_server.sh"
[[ -f "$LILITH_GGUF" ]] || fail lilith::heph::model "$LILITH_GGUF does not exist" \
  "run scripts/lilith/hephaistos/convert_parent.sh"
[[ -f "$LILITH_CHAT_TEMPLATE" ]] || fail lilith::heph::template "$LILITH_CHAT_TEMPLATE does not exist" \
  "convert_parent.sh installs the parent's pinned chat_template.jinja next to the GGUF"
[[ -s "$LILITH_API_KEY_FILE" ]] || fail lilith::heph::key "$LILITH_API_KEY_FILE is missing or empty" \
  "run scripts/lilith/hephaistos/install_service.sh (it creates the key, 0600, owned by the service user)"
mode="$(stat -c '%a' "$LILITH_API_KEY_FILE" 2>/dev/null || stat -f '%Lp' "$LILITH_API_KEY_FILE")"
[[ "$mode" == 600 || "$mode" == 400 ]] || fail lilith::heph::key "$LILITH_API_KEY_FILE has mode $mode" \
  "chmod 600 $LILITH_API_KEY_FILE; the server refuses to start with a key readable by others"
case "$LILITH_LLAMA_HOST" in
  0.0.0.0 | :: | "") fail lilith::heph::bind "LILITH_LLAMA_HOST=$LILITH_LLAMA_HOST binds every interface" \
    "bind the LAN address only (192.168.1.114); the endpoint is plain HTTP" ;;
esac
[[ "$LILITH_LLAMA_PARALLEL" =~ ^[1-9][0-9]*$ ]] || fail lilith::heph::env "LILITH_LLAMA_PARALLEL=$LILITH_LLAMA_PARALLEL" "a positive integer"
[[ "$LILITH_LLAMA_CTX_PER_SLOT" =~ ^[1-9][0-9]*$ ]] || fail lilith::heph::env "LILITH_LLAMA_CTX_PER_SLOT=$LILITH_LLAMA_CTX_PER_SLOT" "a positive integer"

args=(
  --model "$LILITH_GGUF"
  --alias "${LILITH_MODEL_NAME:-lilith}"
  --host "$LILITH_LLAMA_HOST" --port "$LILITH_LLAMA_PORT"
  --api-key-file "$LILITH_API_KEY_FILE"
  # Native OpenAI tool_calls with the parent's pinned template: llama.cpp recognises the Gemma 4
  # template (peg-gemma4 parser); reasoning goes to `reasoning_content`, never into `content`.
  --jinja --chat-template-file "$LILITH_CHAT_TEMPLATE" --reasoning-format deepseek
  # Slots share one KV pool sized parallel x per-slot context.
  --parallel "$LILITH_LLAMA_PARALLEL" --kv-unified-per-slot "$LILITH_LLAMA_CTX_PER_SLOT"
  -ngl 99 --flash-attn auto
  --cache-type-k "${LILITH_LLAMA_KV_TYPE:-f16}" --cache-type-v "${LILITH_LLAMA_KV_TYPE:-f16}"
  --cache-ram "${LILITH_LLAMA_CACHE_RAM_MIB:-4096}"
  # The parent's generation_config.json: temperature 1.0, top_k 64, top_p 0.95 (no min_p).
  --temp 1.0 --top-k 64 --top-p 0.95 --min-p 0.0
  # Nothing but the API: no web UI, no slot/prompt inspection endpoint.
  --no-webui --no-slots
)
if [[ "${LILITH_LLAMA_SLEEP_IDLE_SECONDS:-0}" != 0 ]]; then
  args+=(--sleep-idle-seconds "$LILITH_LLAMA_SLEEP_IDLE_SECONDS")
fi
if [[ -n "${LILITH_LLAMA_EXTRA_ARGS:-}" ]]; then
  read -r -a extra <<<"$LILITH_LLAMA_EXTRA_ARGS"
  args+=("${extra[@]}")
fi

echo "run_server: $LLAMA_SERVER ${args[*]}" >&2
exec "$LLAMA_SERVER" "${args[@]}"
