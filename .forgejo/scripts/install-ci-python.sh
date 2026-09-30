#!/bin/sh
# Ubuntu 22.04's Python 3.10 cannot run tomllib-based Roman contracts.
set -eu
mkdir -p "$HOME/.local/bin"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/0.8.22/install.sh | sh
fi
uv_bin=$(command -v uv || printf '%s/.local/bin/uv' "$HOME")
"$uv_bin" python install 3.12
python_bin=$("$uv_bin" python find 3.12)
ln -sf "$python_bin" "$HOME/.local/bin/akh-ci-python"
"$HOME/.local/bin/akh-ci-python" -c 'import sys,tomllib; assert sys.version_info >= (3,11); print(sys.version)'
