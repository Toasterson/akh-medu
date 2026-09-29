"""Builders for captured-turn rows as `_akh/transcripts/list` returns them."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

T0 = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)
WI_A = "01a0e270-314e-7000-8000-00000000000a"
WI_B = "01a0e270-475e-7000-8000-00000000000b"
WI_C = "01a0e270-137c-7000-8000-00000000000c"
WI_GHOST = "01a0ffff-0000-7000-8000-0000000000ff"


def call(tool: str, args: str = "{}", output: str = "ok", ok: bool = True, wrote: bool = False, ms: int = 120) -> dict:
    return {"tool": tool, "ok": ok, "wrote": wrote, "args": args, "output": output, "duration_ms": ms}


def row(seq: int, minutes: float, prompt: str, answer: str, tools: list[dict] | None = None, *,
        dispatch: bool = False, model: str = "gemma-4-12b") -> dict[str, Any]:
    when = (T0 + timedelta(minutes=minutes)).isoformat()
    payload: dict[str, Any] = {"v": 1, "ts": when, "workspace": "lilith", "model": model, "user_prompt": prompt,
                               "agent_answer": answer, "tools": tools or []}
    if dispatch:
        payload["dispatch"] = True
    return {"seq": seq, "applied_at": when, "payload": payload}
