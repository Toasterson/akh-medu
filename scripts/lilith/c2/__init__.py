"""Lilith C2 (ADR 116; Anima WI 01a0e270-314e): operator sessions -> corrections + baseline notes.

- `marker`            the in-chat correction convention (a language-neutral marker symbol)
- `acp_client`        a minimal ACP/TCP client for the read-only `_akh/transcripts/list`
- `harvest_sessions`  transcripts -> NFS `sessions/<id>.json` + `corrections.jsonl`
- `c2_notes`          baseline notes: fabrications, missed items, autonomy noise

Entry points put `scripts/lilith` on `sys.path` so `dojo` and `c2` import as top-level packages.
"""

C2_VERSION = "lilith/c2@1"
SESSION_SCHEMA = "lilith/c2-session@1"
