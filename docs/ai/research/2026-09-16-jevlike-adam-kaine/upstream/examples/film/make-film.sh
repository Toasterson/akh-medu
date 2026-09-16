#!/bin/sh
set -eu
if [ "$#" -lt 2 ]; then
  echo "usage: $0 TRACE.json OUTPUT.mp4 [SECONDS] [EPISODE@START]" >&2
  exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
trace=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
output=$(mkdir -p -- "$(dirname -- "$2")" && cd -- "$(dirname -- "$2")" && pwd)/$(basename -- "$2")
seconds=${3:-10}
window=${4:-}
cd "$root"
node build-film.mjs "$trace" "$seconds" "$window"
node render-film.mjs "$output"
