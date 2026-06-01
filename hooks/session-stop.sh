#!/bin/bash
# Stop hook: approve stopping and trigger asynchronous journal ingest.
#
# If the `akh` binary is available and a journal directory exists (either via
# AKH_JOURNAL_DIR or the default XDG path), we kick off `akh memory
# ingest-journals` in the background so fresh memory/*.md files are indexed
# into the episodic store without blocking the session stop.

# Trigger journal ingest in the background if akh binary is on PATH.
if command -v akh >/dev/null 2>&1; then
    # Determine journal directory: env override → ./memory → XDG default.
    if [ -n "${AKH_JOURNAL_DIR}" ] && [ -d "${AKH_JOURNAL_DIR}" ]; then
        JOURNAL_DIR="${AKH_JOURNAL_DIR}"
    elif [ -d "./memory" ]; then
        JOURNAL_DIR="./memory"
    fi

    if [ -n "${JOURNAL_DIR}" ]; then
        akh memory ingest-journals \
            --workspace "${AKH_WORKSPACE:-default}" \
            --dir "${JOURNAL_DIR}" \
            >/dev/null 2>&1 &
    fi
fi

echo '{"decision": "approve", "reason": "session complete"}'
exit 0
