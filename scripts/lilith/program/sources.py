"""Source readers that need no dojo replay: character corpus, historical lessons, cos-style transcripts,
C2 sessions, rehearsal records and generated records.

Only structural fields are read. The persona sentence of the character corpus is imported from
`training/character/convert_corpus.py` (it is data, not a matcher), and the role prompt is the repo's own
`deploy/roman-mini/roles/lilith.md`.
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from program import schema as S
from program.schema import ProgramError, assistant, assistant_calls, canonical, make_record, sha256_hex, tool_message, user
from program.splits import UUID, Exposure, uuids_in, work_item_uuids

REPO_ROOT = Path(__file__).resolve().parents[3]
KATA_PATH = REPO_ROOT / "data" / "phase51" / "character" / "kata-drills.jsonl"
SPARRING_PATH = REPO_ROOT / "data" / "phase51" / "character" / "sparring-green.jsonl"
CONVERT_PATH = REPO_ROOT / "training" / "character" / "convert_corpus.py"
ROLE_PATH = REPO_ROOT / "deploy" / "roman-mini" / "roles" / "lilith.md"
VERDICTS_PATH = Path(__file__).with_name("c2-turn-verdicts.json")
ROLE_SERVERS = "anima, djehuty, seshat"  # the servers lilith.md names for her own identities
UNDETERMINED = "und"  # BCP 47: the source records no language, and none is assumed

KATA_BELT_STAGE = {"white": "white", "yellow": "yellow"}
COS_CLASS_FAMILY = {"briefing": "briefing", "commitment": "commitment", "triage": "park-triage",
                    "chase_coordination": "chase", "run_supervision": "run_supervision",
                    "decision_prep": "decision_prep", "implementation": "implementation"}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ProgramError(f"{path}:{number}: not JSON ({error.msg})") from None
    return rows


def file_sha(path: Path) -> str:
    return sha256_hex(Path(path).read_bytes())


def _convert_module():
    spec = importlib.util.spec_from_file_location("character_convert_corpus", CONVERT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # guarded by __main__; defines PERSONA and the prompt composers
    return module


def role_prompt() -> str:
    return ROLE_PATH.read_text().replace(S_MCP_PLACEHOLDER, ROLE_SERVERS).rstrip("\n")


S_MCP_PLACEHOLDER = "{mcp_servers}"


# -- character corpus: kata, sparring, lineage ----------------------------------------------------------


def kata_records(path: Path = KATA_PATH) -> list[dict]:
    convert = _convert_module()
    out = []
    for line_no, row in enumerate(read_jsonl(path), 1):
        stage = KATA_BELT_STAGE.get(row["belt"])
        if stage is None:
            continue  # only white and yellow drills are SFT forms; other belts are not in the corpus
        voice = S.canonical_voice(row["voice"])
        sys_text = convert.kata_system_prompt({**row, "voice": voice})
        out.append(make_record(
            source="kata_drill", source_ref={"path": "data/phase51/character/kata-drills.jsonl", "line": line_no},
            stage=stage, families=[row["kata"]], kata=row["kata"], voice=voice, language=row["lang"],
            messages=[S.system(sys_text), user(row["situation"]), assistant(row["move_chosen"])],
            provenance={"source_sha256": file_sha(path), "source_voice": row["voice"], "belt": row["belt"],
                        "source_row_sha256": sha256_hex(row), "source_text_sha256": [sha256_hex(row["situation"]), sha256_hex(row["move_chosen"])]},
            verification={"structural": "single-turn drill; the supervised target is move_chosen"}))
    return out


def _dialogue_messages(row: Mapping[str, Any], system_text: str) -> list[dict]:
    return [S.system(system_text)] + [{"role": t["role"], "content": t["text"]} for t in row["dialogue"]]


def sparring_records(path: Path = SPARRING_PATH) -> list[dict]:
    """One window per assistant turn: the exact prompt the model saw, only that turn supervised."""
    convert = _convert_module()
    out = []
    for line_no, row in enumerate(read_jsonl(path), 1):
        sys_text = convert.sparring_system_prompt(row)
        full = _dialogue_messages(row, sys_text)
        for i, m in enumerate(full):
            if m["role"] != "assistant":
                continue
            out.append(make_record(
                source="sparring", source_ref={"path": "data/phase51/character/sparring-green.jsonl", "line": line_no, "turn": i},
                stage="green", families=list(row["kata_under_test"]), language=row["lang"],
                messages=full[: i + 1],
                provenance={"source_sha256": file_sha(path), "source_row_sha256": sha256_hex(row),
                            "adversary": row["adversary"], "window_turn": i,
                            "source_text_sha256": [sha256_hex(x["text"]) for x in row["dialogue"]],
                            "lineage_withheld": ["shadow_avoided", "why_this_move", "observed_effect_on_person"]},
                verification={"structural": "dialogue window; the last assistant turn is supervised, earlier turns are context"}))
    return out


def lineage_records(kata_path: Path = KATA_PATH, sparring_path: Path = SPARRING_PATH) -> list[dict]:
    """Four-field lineage as Naming-the-Move transmissions: the assistant target is the commentary."""
    convert = _convert_module()
    out = []
    for line_no, row in enumerate(read_jsonl(kata_path), 1):
        voice = S.canonical_voice(row["voice"])
        prompt = canonical({"situation": row["situation"], "move_chosen": row["move_chosen"]})
        target = canonical({"why_this_move": row["why_this_move"],
                            "observed_effect_on_person": row["observed_effect_on_person"]})
        out.append(make_record(
            source="lineage", source_ref={"path": "data/phase51/character/kata-drills.jsonl", "line": line_no},
            stage="black", families=[row["kata"]], kata=row["kata"], voice=voice, language=row["lang"],
            messages=[S.system(convert.kata_system_prompt({**row, "voice": voice})), user(prompt), assistant(target)],
            provenance={"source_sha256": file_sha(kata_path), "belt": row["belt"], "lineage": "kata",
                        "source_row_sha256": sha256_hex(row), "source_text_sha256": [sha256_hex(row["situation"]), sha256_hex(row["move_chosen"])]},
            verification={"structural": "four-field record: situation + move_chosen in, commentary out"}))
    for line_no, row in enumerate(read_jsonl(sparring_path), 1):
        prompt = canonical({"dialogue": row["dialogue"]})
        target = canonical({"why_this_move": row["why_this_move"],
                            "observed_effect_on_person": row["observed_effect_on_person"],
                            "shadow_avoided": row.get("shadow_avoided")})
        out.append(make_record(
            source="lineage", source_ref={"path": "data/phase51/character/sparring-green.jsonl", "line": line_no},
            stage="black", families=list(row["kata_under_test"]), language=row["lang"],
            messages=[S.system(convert.sparring_system_prompt(row)), user(prompt), assistant(target)],
            provenance={"source_sha256": file_sha(sparring_path), "adversary": row["adversary"], "lineage": "sparring",
                        "source_row_sha256": sha256_hex(row), "source_text_sha256": [sha256_hex(x["text"]) for x in row["dialogue"]]},
            verification={"structural": "sparring dialogue in, why/effect/shadow commentary out"}))
    return out


# -- historical receipt lessons (White) --------------------------------------------------------------------


def history_records(corpus: Path) -> tuple[list[dict], Exposure, dict]:
    """The 636 train lessons (tokens reused as-is, re-wrapped) and the exposure of dev + historical_eval."""
    corpus = Path(corpus)
    receipts = {r["receipt_id"]: r for r in read_jsonl(corpus / "receipts.jsonl")}
    tokens = {t["record_id"]: t for t in read_jsonl(corpus / "train.tokens.jsonl")}
    pin = json.loads((corpus / "tokenization.json").read_text())
    train_sha = file_sha(corpus / "train.jsonl")
    out = []
    for line_no, row in enumerate(read_jsonl(corpus / "train.jsonl"), 1):
        receipt = receipts[row["expected"]["receipt_id"]]
        rec = make_record(
            source="historical_receipt", source_ref={"path": str(corpus / "train.jsonl"), "line": line_no,
                                                    "original_id": row["record_id"]},
            stage="white", families=["historical-receipt:" + row["family"]], language="en", weight=float(row["weight"]),
            messages=[*row["messages"], assistant(row["answer"])],
            provenance={"session_ids": [row["session"]], "subject_ids": [row["work_item_id"]],
                        "observation_hashes": [sha256_hex(receipt)], "force_split": "train", "train_sha256": train_sha,
                        "historical_record_id": row["record_id"], "target_source": row["target_source"],
                        "memory_seed": row["memory_seed"], "tokenization": pin["files"].get("train.tokens.jsonl")},
            verification={"structural": "deterministic receipt projection; memory_revision seeds are synthetic"})
        rec["_tokens"] = tokens[row["record_id"]]
        out.append(rec)
    exposure = Exposure()
    for name in ("dev", "historical_eval"):
        for row in read_jsonl(corpus / f"{name}.jsonl"):
            receipt = receipts[row["expected"]["receipt_id"]]
            exposure.add(origin=f"historical_{name}", work_items=[row["work_item_id"]],
                         sessions=[row["session"]], responses=[sha256_hex(receipt), sha256_hex(receipt["facts"])])
    return out, exposure, {"train_lessons": len(out), "tokenization_pins": pin.get("tokenizer_pins")}


def behavior_exposure(path: Path) -> Exposure:
    exposure = Exposure()
    for case in json.loads(Path(path).read_text()):
        exposure.add(origin="behavior_spot", work_items=uuids_in(case), responses=[sha256_hex(case), sha256_hex(case["prompt"])])
    return exposure


# -- cos-style transcripts (Yellow) ------------------------------------------------------------------------


def _loads_args(text: str | None) -> Any:
    if text is None:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text  # passed through to the template verbatim
    return value


def cos_messages(row: Mapping[str, Any], system_text: str) -> tuple[list[dict], dict]:
    items: list[list] = []
    for s in row["trajectory"]["steps"]:
        if s["kind"] == "tool_call":
            items.append(["call", s["name"], _loads_args(s.get("args")), bool(s["train"]), None])
        elif s["kind"] == "tool_result":
            items.append(["result", s.get("result_text")])
        elif s["kind"] == "assistant_text":
            items.append(["text", s["text"], bool(s["train"])])
    # FIFO pairing of results to calls; a call without an observation is dropped, never given a made-up one.
    open_calls: list[int] = []
    for i, it in enumerate(items):
        if it[0] == "call":
            open_calls.append(i)
        elif it[0] == "result" and open_calls and it[1] is not None:
            call = items[open_calls.pop(0)]
            call[4] = i
    dropped = sum(1 for it in items if it[0] == "call" and it[4] is None)
    msgs: list[dict] = [S.system(system_text), user(row["input"])]
    n_calls = 0
    text_buf: list[tuple[str, bool]] = []
    group: list[tuple[str, str, Any, bool]] = []

    def flush_group():
        nonlocal group, text_buf
        if not group:
            return
        text = "\n\n".join(t for t, _ in text_buf)
        flags = [tr for _, _, _, tr in group] + [tr for _, tr in text_buf]
        msgs.append(assistant_calls([(cid, name, args) for cid, name, args, _ in group], text, train=all(flags)))
        text_buf, group = [], []

    answered: dict[int, str] = {}
    for i, it in enumerate(items):
        if it[0] == "call":
            if it[4] is None:
                continue
            n_calls += 1
            cid = f"c{n_calls}"
            answered[it[4]] = cid
            group.append((cid, it[1], it[2], it[3]))
        elif it[0] == "result":
            if i in answered:
                flush_group()
                msgs.append(tool_message(answered[i], it[1]))
        else:
            flush_group()
            text_buf.append((it[1], it[2]))
    flush_group()
    if text_buf:
        text = "\n\n".join(t for t, _ in text_buf)
        msgs.append(assistant(text, train=all(tr for _, tr in text_buf)))
    final = row["turn_final_reply"]["text"]
    if msgs[-1]["role"] != "assistant" or msgs[-1].get("tool_calls"):
        msgs.append(assistant(final))
    return msgs, {"calls_dropped_without_observation": dropped}


def cos_records(train_path: Path, validation_path: Path) -> list[dict]:
    system_text = role_prompt()
    out = []
    for path, forced in ((train_path, "train"), (validation_path, "dev")):
        for line_no, row in enumerate(read_jsonl(path), 1):
            msgs, notes = cos_messages(row, system_text)
            if not any(S.trained(m) for m in msgs):
                continue
            cls = row["labels"]["cos_class"]
            args_ids = uuids_in([m.get("tool_calls") for m in msgs if m.get("tool_calls")]) | uuids_in(row["input"])
            out.append(make_record(
                source="cos_transcript", source_ref={"path": str(path), "line": line_no, "original_id": row["record_id"]},
                stage="yellow", families=[COS_CLASS_FAMILY.get(cls, cls)], language=UNDETERMINED, weight=1.0,
                messages=msgs,
                provenance={"session_ids": [row["provenance"]["session_sha256"]], "subject_ids": sorted(args_ids),
                            "force_split": forced, "cos_class": cls, "operator_correction": row["labels"]["operator_correction"],
                            "schema": row["schema"], "source_split": row["split"], "tools_declared": False,
                            "weight_basis": "panel-admitted imitation at default weight 1.0 (no per-record weight is recorded)"},
                verification={"structural": "train=true steps supervised; tool results and subagent steps are context", **notes}))
    return out


def cos_exposure(recipe_path: Path, trajectories_path: Path) -> Exposure:
    """The cos-style session(s) already excluded for holdout overlap: ids, session and tool-result texts."""
    recipe = json.loads(Path(recipe_path).read_text())
    held = set(recipe.get("style_sessions_excluded_for_historical_holdout_overlap", []))
    exposure = Exposure()
    for row in read_jsonl(trajectories_path):
        session = row["provenance"]["session_sha256"]
        if session not in held:
            continue
        results = [sha256_hex(s["result_text"]) for s in row["trajectory"]["steps"] if s.get("result_text")]
        exposure.add(origin="cos_style_excluded_session", work_items=work_item_uuids(row), sessions=[session], responses=results)
    return exposure


# -- C2 sessions (White) --------------------------------------------------------------------------------------

TAG_BLOCK = re.compile(r"\s*<([A-Za-z_][\w-]*)>.*?</\1>\s*$", re.DOTALL)  # trailing client-injected tag blocks


def operator_text(turn: Mapping[str, Any]) -> str:
    text = turn.get("operator_text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    prompt = str(turn.get("user_prompt") or "")
    while True:
        stripped = TAG_BLOCK.sub("", prompt)
        if stripped == prompt:
            return prompt.strip()
        prompt = stripped


def load_verdicts(path: Path = VERDICTS_PATH) -> dict:
    return json.loads(Path(path).read_text())


def c2_turn_messages(turn: Mapping[str, Any], system_text: str) -> list[dict]:
    msgs: list[dict] = [S.system(system_text), user(operator_text(turn))]
    for n, call in enumerate(turn.get("tool_calls") or [], 1):
        cid = f"c{n}"
        msgs.append(assistant_calls([(cid, call["tool"], _loads_args(call.get("args")))]))
        msgs.append(tool_message(cid, call.get("output") or ""))
    msgs.append(assistant(turn["agent_answer"]))
    return msgs


def c2_records(sessions_dir: Path, verdicts: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """Positive C2 turns (verdict correct) at the round weight; every other turn is returned as an exclusion.

    Known-wrong turns return no SFT record; the ones with a corrected sibling become rejected sides in
    `dojo_sources` (the `raw_of` link), the rest are excluded here."""
    system_text = role_prompt()
    rounds = verdicts["round_of_session"]
    weights = verdicts["round_weight"]
    by_turn = {(t["session"], t["seq"]): t for t in verdicts["turns"]}
    out, excluded = [], []
    for session_id in sorted(rounds):
        path = Path(sessions_dir) / f"{session_id}.json"
        if not path.is_file():
            excluded.append({"source": "c2_session", "reason": "session_not_cached", "detail": session_id})
            continue
        session = json.loads(path.read_text())
        for turn in session["turns"]:
            key = (session_id, int(turn["seq"]))
            note = by_turn.get(key)
            if note is None or note["verdict"] != "correct":
                excluded.append({"source": "c2_session", "reason": "verdict_" + (note["verdict"] if note else "missing"),
                                 "detail": f"{session_id}#{turn['seq']}", "raw_of": (note or {}).get("raw_of")})
                continue
            msgs = c2_turn_messages(turn, system_text)
            out.append(make_record(
                source="c2_session", source_ref={"session": session_id, "seq": turn["seq"], "index": turn["index"]},
                stage="white", families=[note.get("family", "c2")], language=UNDETERMINED,
                weight=float(weights[str(rounds[session_id])]), messages=msgs,
                provenance={"session_ids": [session_id], "subject_ids": sorted(uuids_in([t.get("args") for t in turn["tool_calls"]])),
                            "round": rounds[session_id], "verdict": note["verdict"], "analysis_reason": note["reason"],
                            "analysis": verdicts["analysis_file"], "weight_basis": verdicts["weight_basis"][str(rounds[session_id])]},
                verification={"structural": "operator turn judged correct in the round analysis; real tool outputs kept"}))
    return out, excluded


# -- rehearsal (inherited ADAM / Kaine capabilities) -----------------------------------------------------------

REHEARSAL_SOURCES = (
    "training/lilith/rl-20260921/tasks.json",
    "training/lilith/rl-20260922/tasks.json",
)


def rehearsal_records(repo: Path = REPO_ROOT) -> list[dict]:
    """Coding references that exist on main as chat-shaped input/output pairs. Nothing is invented: only
    `training_targets` of train-split tasks, each with its source hash."""
    out = []
    for rel in REHEARSAL_SOURCES:
        path = Path(repo) / rel
        doc = json.loads(path.read_text())
        split = {t["id"]: t["split"] for t in doc["tasks"]}
        for t in doc["training_targets"]:
            if split.get(t["id"]) != "train":
                continue
            out.append(make_record(
                source="rehearsal", source_ref={"path": rel, "task": t["id"]}, stage="white", families=[t["family"]],
                language="en", weight=1.0, messages=[user(t["input"]), assistant(t["output"])],
                provenance={"capability": "coding", "origin": t.get("origin"), "source_sha256": file_sha(path),
                            "task_source_sha256": t.get("source_sha256")},
                verification={"structural": "assistant-authored reference with a replay receipt on main; not a generated solution"}))
    return out


def stage_copy(rec: Mapping[str, Any], stage: str) -> dict:
    """The same rehearsal content for another stage (record_id differs: the stage is part of the identity)."""
    copy = {k: v for k, v in rec.items() if k != "record_id"}
    copy["stage"] = stage
    copy["record_id"] = sha256_hex(S.identity_payload(copy))
    return copy


# -- generated records (other agent, program-record@1) -----------------------------------------------------------

GENERATED_SOURCES = ("variation", "guidance", "lineage", "sparring")


OPTIONAL_DEFAULTS = {"chosen_messages": None, "rejected_messages": None, "tools": [], "voice": None, "kata": None,
                     "provenance": {}, "verification": {}, "component_id": "", "split": "train"}


def normalize_generated(rec: dict) -> dict:
    """Producers may omit empty optional fields. Missing ones take their defaults; the record id is re-derived by
    this exporter's identity function (a producer id that differs is kept as provenance.generated_record_id)."""
    out = {**{k: (list(v) if isinstance(v, list) else dict(v) if isinstance(v, dict) else v) for k, v in OPTIONAL_DEFAULTS.items()}, **rec}
    if out.get("voice"):
        try:
            out["voice"] = S.canonical_voice(out["voice"])
        except ProgramError:
            pass  # left as given: validation reports it
    own = sha256_hex(S.identity_payload(out))
    if out.get("record_id") != own:
        out["provenance"] = {**out["provenance"], "generated_record_id": out.get("record_id")}
        out["record_id"] = own
    return out


def generated_records(directory: Path | None) -> tuple[list[dict], list[dict]]:
    """Whatever exists under `generated/*.jsonl`: validated, never repaired. Invalid lines are reported."""
    accepted, rejected = [], []
    if directory is None or not Path(directory).is_dir():
        return accepted, rejected
    for path in sorted(Path(directory).glob("*.jsonl")):
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            ref = {"file": path.name, "line": line_no}
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                rejected.append({"source": "generated", "reason": "not_json", "detail": ref})
                continue
            if isinstance(rec, dict):
                rec = normalize_generated(rec)
            problems = S.validate_record(rec) if isinstance(rec, dict) else ["not an object"]
            if not problems and rec["source"] not in GENERATED_SOURCES:
                problems.append(f"source {rec['source']!r} is not a generated source {GENERATED_SOURCES}")
            if problems:
                rejected.append({"source": "generated", "reason": "invalid_record", "detail": {**ref, "problems": problems[:5]}})
                continue
            prov = dict(rec.get("provenance") or {})
            prov.setdefault("generated_from", ref)
            if rec.get("component_id"):
                prov["link_keys"] = [*prov.get("link_keys", []), rec["component_id"]]
            rec = {**rec, "provenance": prov, "split": "train"}
            accepted.append(rec)
    return accepted, rejected
