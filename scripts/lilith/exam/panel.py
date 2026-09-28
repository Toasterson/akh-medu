"""Judging-panel runner (workflow §3).

The panel is Fable 5.1, Opus 5.5 and Codex Astra, coordinated by the supervisor as
subagents / `codex exec`; this module never calls a model. It:

1. assigns two blinded raters per item from {fable, opus, astra}, rotating the pair every
   `batches_per_pair` batches (default 2: b1–b2 Fable+Opus, b3–b4 Opus+Astra, b5–b6
   Astra+Fable, the C5a first use);
2. emits work orders: one JSONL batch per (batch, judge) plus the frozen rubric and the
   judge instructions, and a supervisor-only unblinding map;
3. ingests the judges' JSONL results, validating blind id, judge assignment, rubric hash,
   verdict, every dimension in 0..3 and cited turns;
4. emits adjudication orders for the third judge on a verdict disagreement or any
   dimension gap >= 2;
5. reports Cohen's κ per dimension (and on the verdict); dimensions with κ < 0.6 are
   excluded from gates until the rubric is revised.

Judges score only social effect and guidance quality (ADR 110 rule 6), citing turns.
Structural items (IDs, receipts, schemas, recipients, hindsight labels) are scored
deterministically elsewhere and are stripped from nothing here: the rubric simply does not
ask for them.
"""
from __future__ import annotations

import dataclasses
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dojo.snapshot import canonical_json, sha256_hex
from exam.errors import PanelError

JUDGES = ("fable", "opus", "astra")
PAIRS = (("fable", "opus"), ("opus", "astra"), ("astra", "fable"))
KAPPA_MIN = 0.6
ADJUDICATION_GAP = 2
SCORE_RANGE = range(0, 4)
VERDICTS = ("pass", "fail")
ORDER_SCHEMA = "lilith/panel-order@1"
RESULT_SCHEMA = "lilith/panel-result@1"
# Keys that would unblind arm, voice label or the other judge. Removed at any depth.
BLIND_KEYS = frozenset({"arm", "voice", "voice_label", "voice_shown", "voice_called_for", "model", "checkpoint",
                        "adapter", "adapter_sha256", "run_id", "judge", "rater", "teacher", "candidate", "parent"})
DISPATCH = {
    "fable": "Claude subagent, model fable (Fable 5.1)",
    "opus": "Claude subagent, model opus (Opus 5.5)",
    "astra": "codex exec -m gpt-6-astra (stdin from /dev/null)",
}
INSTRUCTIONS = """You are one of two independent judges on each item. You do not know which system produced it and
you will not see the other judge's rating.

Score only social effect and guidance quality (ADR 110 rule 6). Do not score IDs, receipts, schemas,
recipients or hindsight labels; those are checked deterministically elsewhere.

For every line of the batch file, write one JSON line to your result file:
  {"schema": "lilith/panel-result@1", "blind_id": <from the order>, "judge": <your judge id>,
   "rubric_sha256": <from the order>, "verdict": "pass" | "fail",
   "scores": {<every rubric dimension>: 0..3}, "cited_turns": [<turn ids you rely on>], "note": <one line>}
Use the anchors in rubric.json. Every score must cite at least one turn.
"""


@dataclasses.dataclass(frozen=True)
class Dimension:
    id: str
    question: str
    anchors: Mapping[int, str]  # 0..3, written anchors

    def to_dict(self) -> dict:
        return {"id": self.id, "question": self.question, "anchors": {str(k): v for k, v in sorted(self.anchors.items())}}


@dataclasses.dataclass(frozen=True)
class Rubric:
    id: str
    version: int
    verdict_question: str
    dimensions: tuple[Dimension, ...]

    def __post_init__(self) -> None:
        ids = [d.id for d in self.dimensions]
        if not ids or len(set(ids)) != len(ids):
            raise PanelError(f"rubric {self.id}: dimensions must be non-empty and unique, got {ids}")
        for d in self.dimensions:
            if set(d.anchors) != set(SCORE_RANGE):
                raise PanelError(f"rubric {self.id}: dimension {d.id} needs written anchors for 0..3")

    def to_dict(self) -> dict:
        return {"schema": "lilith/panel-rubric@1", "id": self.id, "version": self.version,
                "verdict_question": self.verdict_question, "dimensions": [d.to_dict() for d in self.dimensions]}

    @property
    def rubric_sha256(self) -> str:
        return sha256_hex(canonical_json(self.to_dict()))

    @property
    def dimension_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.dimensions)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Rubric":
        return cls(data["id"], int(data["version"]), data["verdict_question"],
                   tuple(Dimension(d["id"], d["question"], {int(k): v for k, v in d["anchors"].items()})
                         for d in data["dimensions"]))


@dataclasses.dataclass(frozen=True)
class PanelItem:
    item_id: str
    batch: str
    arm: str  # candidate | frozen-parent | teacher; never shown to judges
    content: Mapping[str, Any]  # transcript turns etc.; blinded before it leaves


@dataclasses.dataclass(frozen=True)
class Assignment:
    item_id: str
    batch: str
    arm: str
    blind_id: str
    judges: tuple[str, str]

    @property
    def adjudicator(self) -> str:
        return next(j for j in JUDGES if j not in self.judges)


@dataclasses.dataclass(frozen=True)
class Rating:
    blind_id: str
    judge: str
    verdict: str
    scores: Mapping[str, int]
    cited_turns: tuple[str, ...]
    note: str = ""


def blind(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: blind(v) for k, v in value.items() if k not in BLIND_KEYS}
    if isinstance(value, list):
        return [blind(v) for v in value]
    return value


def cohen_kappa(a: Sequence[Any], b: Sequence[Any]) -> float | None:
    """Unweighted Cohen's κ. None with no pairs. When chance agreement is 1 (both raters used
    one identical category throughout) κ is undefined; we return 1.0 for perfect agreement
    and 0.0 otherwise, and the report flags the dimension as degenerate."""
    if len(a) != len(b):
        raise PanelError("κ needs paired ratings")
    n = len(a)
    if n == 0:
        return None
    categories = sorted(set(a) | set(b), key=str)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    pe = sum((list(a).count(c) / n) * (list(b).count(c) / n) for c in categories)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


class Panel:
    def __init__(self, rubric: Rubric, salt: str, batches_per_pair: int = 2):
        if not salt:
            raise PanelError("a blinding salt is required (keep it in the supervisor's run directory)")
        if batches_per_pair <= 0:
            raise PanelError("batches_per_pair must be positive")
        self.rubric = rubric
        self.salt = salt
        self.batches_per_pair = batches_per_pair
        self.assignments: dict[str, Assignment] = {}  # by blind id

    # -- assignment and work orders -----------------------------------------------------------

    def blind_id(self, item_id: str) -> str:
        return "b-" + sha256_hex(f"{self.salt}:{item_id}".encode())[:16]

    def assign(self, items: Sequence[PanelItem]) -> list[Assignment]:
        batches = sorted({i.batch for i in items})
        seen: set[str] = set()
        out: list[Assignment] = []
        for item in items:
            if item.item_id in seen:
                raise PanelError(f"item {item.item_id} listed twice")
            seen.add(item.item_id)
            pair = PAIRS[(batches.index(item.batch) // self.batches_per_pair) % len(PAIRS)]
            a = Assignment(item.item_id, item.batch, item.arm, self.blind_id(item.item_id), pair)
            self.assignments[a.blind_id] = a
            out.append(a)
        return out

    def _order(self, a: Assignment, judge: str, content: Mapping[str, Any], role: str) -> dict:
        return {"schema": ORDER_SCHEMA, "blind_id": a.blind_id, "judge": judge, "role": role,
                "rubric_sha256": self.rubric.rubric_sha256, "content": blind(content)}

    def _write_orders(self, out_dir: Path, rows: dict[tuple[str, str], list[dict]], kind: str) -> dict:
        orders = out_dir / "orders"
        orders.mkdir(parents=True, exist_ok=True)
        (out_dir / "rubric.json").write_text(json.dumps(self.rubric.to_dict(), indent=2, sort_keys=True) + "\n")
        (orders / "INSTRUCTIONS.txt").write_text(INSTRUCTIONS)
        index = []
        for (batch, judge), lines in sorted(rows.items()):
            lines.sort(key=lambda r: r["blind_id"])  # interleaves arms deterministically
            name = f"{kind}-{batch}-{judge}.jsonl"
            data = "".join(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n" for line in lines)
            (orders / name).write_text(data)
            index.append({"file": f"orders/{name}", "judge": judge, "batch": batch, "kind": kind, "items": len(lines),
                          "sha256": sha256_hex(data.encode()), "dispatch": DISPATCH[judge],
                          "result_file": f"results/{kind}-{batch}-{judge}.jsonl"})
        return {"schema": "lilith/panel-orders@1", "rubric_sha256": self.rubric.rubric_sha256, "orders": index}

    def work_orders(self, items: Sequence[PanelItem], out_dir: Path) -> dict:
        """Writes `rubric.json`, `orders/*.jsonl`, `orders/INSTRUCTIONS.txt`, `orders/index.json`
        and the supervisor-only `supervisor/unblind.json`. Returns the index."""
        assignments = self.assign(items)
        by_id = {i.item_id: i for i in items}
        rows: dict[tuple[str, str], list[dict]] = {}
        for a in assignments:
            for judge in a.judges:
                rows.setdefault((a.batch, judge), []).append(self._order(a, judge, by_id[a.item_id].content, "primary"))
        index = self._write_orders(out_dir, rows, "primary")
        (out_dir / "orders" / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        supervisor = out_dir / "supervisor"
        supervisor.mkdir(parents=True, exist_ok=True)
        (supervisor / "unblind.json").write_text(json.dumps(
            {a.blind_id: {"item_id": a.item_id, "batch": a.batch, "arm": a.arm, "judges": list(a.judges)}
             for a in assignments}, indent=2, sort_keys=True) + "\n")
        return index

    # -- ingestion ----------------------------------------------------------------------------

    def parse_result(self, line: Mapping[str, Any]) -> Rating:
        blind_id = line.get("blind_id")
        a = self.assignments.get(blind_id) if isinstance(blind_id, str) else None
        where = f"result for {blind_id!r} by {line.get('judge')!r}"
        if a is None:
            raise PanelError(f"{where}: unknown blind id")
        judge = line.get("judge")
        if judge not in (*a.judges, a.adjudicator):
            raise PanelError(f"{where}: judge not assigned to this item")
        if line.get("rubric_sha256") != self.rubric.rubric_sha256:
            raise PanelError(f"{where}: rubric hash {line.get('rubric_sha256')} is not the frozen rubric")
        if line.get("verdict") not in VERDICTS:
            raise PanelError(f"{where}: verdict must be one of {VERDICTS}")
        scores = line.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(self.rubric.dimension_ids):
            raise PanelError(f"{where}: scores must cover exactly {self.rubric.dimension_ids}")
        if any(not isinstance(v, int) or isinstance(v, bool) or v not in SCORE_RANGE for v in scores.values()):
            raise PanelError(f"{where}: every score must be an integer 0..3")
        turns = line.get("cited_turns")
        if not isinstance(turns, list) or not turns:
            raise PanelError(f"{where}: no cited turns")
        return Rating(blind_id, judge, line["verdict"], dict(scores), tuple(str(t) for t in turns), str(line.get("note", "")))

    def ingest(self, paths: Iterable[Path]) -> list[Rating]:
        ratings: list[Rating] = []
        seen: set[tuple[str, str]] = set()
        for path in paths:
            for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
                if not raw.strip():
                    continue
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError as err:
                    raise PanelError(f"{path}:{number}: not JSON ({err})") from None
                rating = self.parse_result(line)
                key = (rating.blind_id, rating.judge)
                if key in seen:
                    raise PanelError(f"{path}:{number}: second rating by {rating.judge} for {rating.blind_id}")
                seen.add(key)
                ratings.append(rating)
        return ratings

    # -- adjudication and report ------------------------------------------------------------

    @staticmethod
    def _by_item(ratings: Iterable[Rating]) -> dict[str, dict[str, Rating]]:
        out: dict[str, dict[str, Rating]] = {}
        for r in ratings:
            out.setdefault(r.blind_id, {})[r.judge] = r
        return out

    def needs_adjudication(self, a: Assignment, got: Mapping[str, Rating]) -> bool:
        first, second = (got.get(j) for j in a.judges)
        if first is None or second is None:
            return False
        if first.verdict != second.verdict:
            return True
        return any(abs(first.scores[d] - second.scores[d]) >= ADJUDICATION_GAP for d in self.rubric.dimension_ids)

    def adjudication_orders(self, items: Sequence[PanelItem], ratings: Sequence[Rating], out_dir: Path) -> dict:
        by_item = self._by_item(ratings)
        by_id = {i.item_id: i for i in items}
        rows: dict[tuple[str, str], list[dict]] = {}
        for a in self.assignments.values():
            got = by_item.get(a.blind_id, {})
            if self.needs_adjudication(a, got) and a.adjudicator not in got:
                rows.setdefault((a.batch, a.adjudicator), []).append(
                    self._order(a, a.adjudicator, by_id[a.item_id].content, "adjudicator"))
        index = self._write_orders(out_dir, rows, "adjudication")
        (out_dir / "orders" / "adjudication-index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        return index

    def report(self, ratings: Sequence[Rating], arm_thresholds: Mapping[str, float] | None = None) -> dict:
        """`arm_thresholds`: {"verdict": min pass rate, <dimension>: min mean score} applied to
        the candidate arm; only gated (κ >= 0.6) dimensions can block."""
        by_item = self._by_item(ratings)
        dims = self.rubric.dimension_ids
        pairs: dict[str, tuple[list[Any], list[Any]]] = {d: ([], []) for d in (*dims, "verdict")}
        finals: list[dict] = []
        pending: list[str] = []
        missing: list[str] = []
        for a in sorted(self.assignments.values(), key=lambda x: x.blind_id):
            got = by_item.get(a.blind_id, {})
            first, second = (got.get(j) for j in a.judges)
            if first is None or second is None:
                missing.append(a.blind_id)
                continue
            for d in dims:
                pairs[d][0].append(first.scores[d])
                pairs[d][1].append(second.scores[d])
            pairs["verdict"][0].append(first.verdict)
            pairs["verdict"][1].append(second.verdict)
            if self.needs_adjudication(a, got):
                third = got.get(a.adjudicator)
                if third is None:
                    pending.append(a.blind_id)
                    continue
                trio = (first, second, third)
                verdict = max(VERDICTS, key=lambda v: sum(r.verdict == v for r in trio))
                scores = {d: float(statistics.median(r.scores[d] for r in trio)) for d in dims}
                adjudicated = True
            else:
                verdict = first.verdict
                scores = {d: (first.scores[d] + second.scores[d]) / 2 for d in dims}
                adjudicated = False
            finals.append({"blind_id": a.blind_id, "item_id": a.item_id, "arm": a.arm, "batch": a.batch,
                           "verdict": verdict, "scores": scores, "adjudicated": adjudicated})
        kappa = {d: cohen_kappa(*pairs[d]) for d in (*dims, "verdict")}
        degenerate = sorted(d for d in kappa if pairs[d][0] and len(set(pairs[d][0]) | set(pairs[d][1])) == 1)
        gated = [d for d in dims if kappa[d] is not None and kappa[d] >= KAPPA_MIN]
        excluded = {d: kappa[d] for d in dims if d not in gated}
        per_arm: dict[str, dict] = {}
        for f in finals:
            arm = per_arm.setdefault(f["arm"], {"n": 0, "verdict_pass": 0, "sums": {d: 0.0 for d in dims}})
            arm["n"] += 1
            arm["verdict_pass"] += f["verdict"] == "pass"
            for d in dims:
                arm["sums"][d] += f["scores"][d]
        arms = {name: {"n": v["n"], "verdict_pass_rate": v["verdict_pass"] / v["n"],
                       "dimension_means": {d: v["sums"][d] / v["n"] for d in dims}} for name, v in sorted(per_arm.items())}
        blockers: list[str] = []
        if missing:
            blockers.append(f"{len(missing)} items lack both primary ratings")
        if pending:
            blockers.append(f"{len(pending)} items await third-judge adjudication")
        verdict_gated = kappa["verdict"] is not None and kappa["verdict"] >= KAPPA_MIN
        candidate = arms.get("candidate")
        for key, minimum in (arm_thresholds or {}).items():
            if candidate is None:
                blockers.append("no candidate-arm items were rated")
                break
            if key == "verdict":
                if not verdict_gated:
                    blockers.append(f"verdict κ {kappa['verdict']} < {KAPPA_MIN}: revise the rubric before it counts")
                elif candidate["verdict_pass_rate"] < minimum:
                    blockers.append(f"candidate verdict pass rate {candidate['verdict_pass_rate']:.3f} < {minimum}")
            elif key in gated and candidate["dimension_means"][key] < minimum:
                blockers.append(f"candidate {key} mean {candidate['dimension_means'][key]:.3f} < {minimum}")
        return {"schema": "lilith/panel-report@1", "rubric_sha256": self.rubric.rubric_sha256,
                "items": len(self.assignments), "rated": len(finals), "missing": missing, "pending": pending,
                "kappa": kappa, "kappa_min": KAPPA_MIN, "degenerate_kappa": degenerate, "gated_dimensions": gated,
                "excluded_dimensions": excluded, "verdict_gated": verdict_gated, "per_arm": arms,
                "finals": finals, "blockers": blockers}
