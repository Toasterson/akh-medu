"""Structural checks for generated records. No natural-language word lists: every check is on ids, shapes,
set overlap or character statistics learned from the corpus itself.

- `extract_json`        the teacher's fenced JSON header (or a bare object) parsed structurally
- `jaccard`             normalized token-set overlap (different from the seed?)
- `LanguageProfiles`    char-trigram cosine profiles learned from labelled corpus text (is it the requested language?)
- `alternates`          user/assistant alternation of a dialogue
- `CitationResolver`    man pages and source files resolved against a local illumos-gate checkout or HTTPS
"""
from __future__ import annotations

import collections
import dataclasses
import json
import math
import os
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

_FENCE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n(.*?)```", re.S)
_WORD = re.compile(r"\w+", re.UNICODE)


@dataclasses.dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def failed(checks: Iterable[Check]) -> list[Check]:
    return [c for c in checks if not c.passed]


def all_passed(checks: Iterable[Check]) -> bool:
    return not failed(checks)


# --- JSON header ------------------------------------------------------------------------------------------


def extract_json(text: str) -> dict | None:
    """The last fenced block that parses as a JSON object; else the first bare object in the text."""
    for block in reversed(_FENCE.findall(text or "")):
        try:
            value = json.loads(block)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    decoder = json.JSONDecoder()
    start = (text or "").find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        if isinstance(value, dict):
            return value
        start = text.find("{", start + 1)
    return None


# --- overlap ----------------------------------------------------------------------------------------------


def tokens(text: str) -> frozenset[str]:
    return frozenset(m.group(0).lower() for m in _WORD.finditer(text or ""))


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    if not ta and not tb:
        return 1.0
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def non_empty(text: object) -> bool:
    return isinstance(text, str) and bool(text.strip())


# --- language (statistical, learned from labelled corpus text) -------------------------------------------


def _trigrams(text: str) -> collections.Counter:
    chars = "".join(ch.lower() if ch.isalpha() else " " for ch in text)
    chars = " ".join(chars.split())
    padded = f"  {chars} "
    return collections.Counter(padded[i : i + 3] for i in range(len(padded) - 2))


def _cosine(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


class LanguageProfiles:
    """Trigram profile per language, built from corpus text labelled with a language code."""

    def __init__(self, profiles: Mapping[str, Mapping[str, float]]):
        self.profiles = dict(profiles)

    @classmethod
    def from_texts(cls, texts_by_language: Mapping[str, Iterable[str]]) -> "LanguageProfiles":
        profiles: dict[str, collections.Counter] = {}
        for language, texts in texts_by_language.items():
            counter: collections.Counter = collections.Counter()
            for text in texts:
                counter.update(_trigrams(text))
            if counter:
                profiles[language] = counter
        return cls(profiles)

    def scores(self, text: str) -> dict[str, float]:
        grams = _trigrams(text)
        return {lang: _cosine(grams, profile) for lang, profile in self.profiles.items()}

    def matches(self, text: str, language: str, tolerance: float = 0.85) -> Check:
        """The requested language scores within `tolerance` of the best-scoring one."""
        if language not in self.profiles:
            return Check("language_detected", True, f"no profile for {language}; not checked")
        scores = self.scores(text)
        best_lang, best = max(scores.items(), key=lambda kv: kv[1])
        ok = scores[language] >= best * tolerance
        return Check("language_detected", ok, f"best={best_lang}:{best:.3f} requested={language}:{scores[language]:.3f}")


# --- dialogue shape ---------------------------------------------------------------------------------------


def alternates(messages: Sequence[Mapping[str, object]], *, first: str = "user", last: str = "assistant") -> Check:
    roles = [m.get("role") for m in messages]
    if not roles:
        return Check("alternating_roles", False, "no turns")
    expected = first
    for index, role in enumerate(roles):
        if role != expected:
            return Check("alternating_roles", False, f"turn {index} is {role!r}, expected {expected!r}")
        expected = "assistant" if expected == "user" else "user"
    if roles[-1] != last:
        return Check("alternating_roles", False, f"ends with {roles[-1]!r}, expected {last!r}")
    return Check("alternating_roles", True, f"{len(roles)} turns")


def turns_non_empty(messages: Sequence[Mapping[str, object]], role: str | None = None) -> Check:
    bad = [i for i, m in enumerate(messages) if (role is None or m.get("role") == role) and not non_empty(m.get("content"))]
    return Check("turns_non_empty" if role is None else f"{role}_turns_non_empty", not bad, f"empty turns at {bad}" if bad else "")


# --- citations --------------------------------------------------------------------------------------------

_MAN_NAME_SECTION = re.compile(r"^(?P<name>[\w.+-]+)\((?P<sec>\d\w*)\)$")
_MAN_COMMAND = re.compile(r"^man\s+(?:-s\s*)?(?P<sec>\d\w*)\s+(?P<name>[\w.+-]+)$")
_MAN_SUFFIX = re.compile(r"^(?P<name>[\w+-]+(?:\.[A-Za-z_][\w+-]*)*)\.(?P<sec>\d[A-Za-z0-9]*)$")
_MAN_URL = re.compile(r"^https?://illumos\.org/man/(?P<sec>\d\w*)/(?P<name>[\w.+-]+)/?$")
_LINE_SUFFIX = re.compile(r"(?::\d+(?:-\d+)?|#L\d+(?:-L?\d+)?)$")


@dataclasses.dataclass(frozen=True)
class ManRef:
    name: str
    section: str


def parse_man(reference: str) -> ManRef | None:
    text = (reference or "").strip()
    for pattern in (_MAN_NAME_SECTION, _MAN_COMMAND, _MAN_URL, _MAN_SUFFIX):
        match = pattern.match(text)
        if match:
            return ManRef(match.group("name"), match.group("sec"))
    return None


def parse_source_path(reference: str) -> str | None:
    """A repository-relative illumos-gate path (`usr/src/...`), line suffixes and URL prefixes stripped."""
    text = (reference or "").strip()
    text = re.sub(r"^https?://[^/]+/illumos/illumos-gate/(?:blob|raw)/[^/]+/", "", text)
    text = re.sub(r"^https?://raw\.[^/]+/illumos/illumos-gate/[^/]+/", "", text)
    text = _LINE_SUFFIX.sub("", text)
    text = re.sub(r"^(?:\./|/)+", "", text)
    text = re.sub(r"^illumos-gate/", "", text)
    if not text or ".." in text.split("/") or " " in text:
        return None
    if not text.startswith("usr/") or "/" not in text[4:]:
        return None
    return text


@dataclasses.dataclass(frozen=True)
class Resolution:
    reference: str
    kind: str
    ok: bool
    method: str  # local-man | https-man | local-source | https-source | unparseable | unsupported-kind | not-found | error
    detail: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


HttpGet = Callable[[str], tuple[int, str]]  # url -> (status, first bytes of the body decoded)


def http_get(url: str, timeout: float = 25.0) -> tuple[int, str]:
    request = urllib.request.Request(url, headers={"User-Agent": "akh-medu-lilith-generator/1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - fixed https hosts
            return response.status, response.read(4000).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        return error.code, ""


_TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


class CitationResolver:
    """Resolves `man-page` and `source` citations. A local illumos-gate checkout answers first; HTTPS decides
    the rest (illumos.org/man pages, raw files of the illumos-gate `master`). Results are cached."""

    MAN_URL = "https://illumos.org/man/{section}/{name}"
    RAW_URL = "https://raw.githubusercontent.com/illumos/illumos-gate/master/{path}"

    def __init__(self, gate: Path | None = None, get: HttpGet = http_get):
        self.gate = Path(gate) if gate is not None and Path(gate).is_dir() else None
        self._get = get
        self._lock = threading.Lock()
        self._cache: dict[tuple[str, str], Resolution] = {}
        self._man_index: dict[str, list[str]] | None = None

    def _local_man(self, name: str) -> list[str]:
        if self.gate is None:
            return []
        with self._lock:
            if self._man_index is None:
                index: dict[str, list[str]] = {}
                root = self.gate / "usr" / "src" / "man"
                if root.is_dir():
                    for section_dir in sorted(root.iterdir()):
                        if section_dir.is_dir() and section_dir.name.startswith("man"):
                            for entry in os.scandir(section_dir):
                                stem = entry.name.rsplit(".", 1)[0]
                                index.setdefault(stem, []).append(f"{section_dir.name}/{entry.name}")
                self._man_index = index
            return self._man_index.get(name, [])

    def resolve(self, kind: str, reference: str) -> Resolution:
        key = (kind, reference)
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        result = self._resolve(kind, reference)
        with self._lock:
            self._cache[key] = result
        return result

    def _resolve(self, kind: str, reference: str) -> Resolution:
        if kind == "man-page":
            man = parse_man(reference)
            if man is None:
                return Resolution(reference, kind, False, "unparseable", "not name(section), man <sec> <name> or an illumos.org URL")
            local = self._local_man(man.name)
            if local:
                return Resolution(reference, kind, True, "local-man", local[0])
            try:
                status, body = self._get(self.MAN_URL.format(section=man.section, name=man.name))
            except OSError as error:
                return Resolution(reference, kind, False, "error", str(error)[:120])
            title = _TITLE.search(body)
            title_text = title.group(1).strip() if title else ""
            if status == 200 and man.name.lower() in title_text.lower():
                return Resolution(reference, kind, True, "https-man", title_text[:80])
            return Resolution(reference, kind, False, "not-found", f"HTTP {status} title={title_text[:60]!r}")
        if kind == "source":
            path = parse_source_path(reference)
            if path is None:
                return Resolution(reference, kind, False, "unparseable", "not a usr/... illumos-gate path")
            if self.gate is not None and (self.gate / path).is_file():
                return Resolution(reference, kind, True, "local-source", path)
            try:
                status, _ = self._get(self.RAW_URL.format(path=path))
            except OSError as error:
                return Resolution(reference, kind, False, "error", str(error)[:120])
            if status == 200:
                return Resolution(reference, kind, True, "https-source", path)
            return Resolution(reference, kind, False, "not-found", f"HTTP {status} {path}")
        return Resolution(reference, kind, False, "unsupported-kind", "only man-page and source citations are verified here")

    def check_all(self, citations: Sequence[Mapping[str, object]]) -> tuple[Check, list[Resolution]]:
        if not citations:
            return Check("citations_resolve", False, "no citations"), []
        resolutions = [self.resolve(str(c.get("kind")), str(c.get("reference"))) for c in citations]
        bad = [r for r in resolutions if not r.ok]
        detail = "; ".join(f"{r.kind}:{r.reference} -> {r.method}" for r in bad)
        return Check("citations_resolve", not bad, detail), resolutions
