#!/bin/sh
# Python is installed in the enrolled Rust runner and platform prerequisites.
exec python3 "$(dirname "$0")/run-logged.py" "$@"
