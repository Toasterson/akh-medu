from pathlib import Path

from program.generate import checks


def test_extract_json_prefers_last_fenced_object():
    text = 'noise {"a": 1}\n```json\n{"a": 2}\n```\nmore\n```json\n{"a": 3}\n```'
    assert checks.extract_json(text) == {"a": 3}


def test_extract_json_falls_back_to_bare_object_and_rejects_garbage():
    assert checks.extract_json('prefix {"k": [1, 2]} suffix') == {"k": [1, 2]}
    assert checks.extract_json("no json here") is None
    assert checks.extract_json("```json\n[1, 2]\n```") is None


def test_jaccard_is_unicode_and_case_insensitive():
    assert checks.jaccard("Räum die Dateien", "räum DIE dateien") == 1.0
    assert checks.jaccard("a b c d", "e f g h") == 0.0
    assert 0 < checks.jaccard("a b c d", "a b x y") < 1


def test_alternates():
    ok = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    assert checks.alternates(ok).passed
    assert not checks.alternates(list(reversed(ok))).passed
    assert not checks.alternates(ok + [{"role": "user", "content": "z"}]).passed  # must end with the assistant
    assert not checks.alternates([]).passed


def test_language_profiles_learn_from_labelled_text():
    profiles = checks.LanguageProfiles.from_texts({
        "en": ["the quick brown fox jumps over the lazy dog and keeps running through the field"],
        "de": ["der schnelle braune fuchs springt über den faulen hund und läuft durch das feld"],
    })
    assert profiles.matches("the dog keeps running over the field", "en").passed
    assert not profiles.matches("the dog keeps running over the field", "de").passed
    assert profiles.matches("irgendein text", "fr").passed  # no profile: not judged


def test_parse_man_forms():
    assert checks.parse_man("svcadm(8)") == checks.ManRef("svcadm", "8")
    assert checks.parse_man("man 7 smf_method") == checks.ManRef("smf_method", "7")
    assert checks.parse_man("https://illumos.org/man/1M/svcadm") == checks.ManRef("svcadm", "1M")
    assert checks.parse_man("smf_method.7") == checks.ManRef("smf_method", "7")
    assert checks.parse_man("see the svcadm page") is None


def test_parse_source_path():
    assert checks.parse_source_path("usr/src/uts/common/os/cpu.c:120") == "usr/src/uts/common/os/cpu.c"
    assert checks.parse_source_path("illumos-gate/usr/src/cmd/svc/svcadm/svcadm.c#L10") == "usr/src/cmd/svc/svcadm/svcadm.c"
    assert checks.parse_source_path("https://github.com/illumos/illumos-gate/blob/master/usr/src/lib/libc/port/gen/x.c") == "usr/src/lib/libc/port/gen/x.c"
    assert checks.parse_source_path("usr/src/../etc/passwd") is None
    assert checks.parse_source_path("README.md") is None


def test_resolver_local_then_https(tmp_path):
    gate = tmp_path / "gate"
    (gate / "usr/src/man/man8").mkdir(parents=True)
    (gate / "usr/src/man/man8/svcadm.8").write_text("x")
    (gate / "usr/src/uts/common/os").mkdir(parents=True)
    (gate / "usr/src/uts/common/os/cpu.c").write_text("x")
    calls = []

    def get(url):
        calls.append(url)
        if url == "https://illumos.org/man/7/smf_method":
            return 200, "<html><title>illumos: manual page: smf_method.7</title>"
        if url.endswith("usr/src/lib/libc/x.c"):
            return 200, ""
        return 404, ""

    resolver = checks.CitationResolver(gate, get)
    assert resolver.resolve("man-page", "svcadm(1M)") == checks.Resolution("svcadm(1M)", "man-page", True, "local-man", "man8/svcadm.8")
    assert resolver.resolve("man-page", "smf_method(7)").method == "https-man"
    assert not resolver.resolve("man-page", "nosuch(7)").ok
    assert resolver.resolve("source", "usr/src/uts/common/os/cpu.c").method == "local-source"
    assert resolver.resolve("source", "usr/src/lib/libc/x.c").method == "https-source"
    assert not resolver.resolve("source", "usr/src/uts/common/os/nosuch.c").ok
    assert resolver.resolve("vm-check", "anything").method == "unsupported-kind"
    n = len(calls)
    resolver.resolve("man-page", "smf_method(7)")
    assert len(calls) == n  # cached


def test_check_all_requires_every_citation():
    resolver = checks.CitationResolver(None, get=lambda url: (200, "<title>x svcadm</title>") if "svcadm" in url else (404, ""))
    good = [{"kind": "man-page", "reference": "svcadm(8)"}]
    bad = good + [{"kind": "man-page", "reference": "ghost(8)"}]
    assert resolver.check_all(good)[0].passed
    check, resolutions = resolver.check_all(bad)
    assert not check.passed and len(resolutions) == 2
    assert not resolver.check_all([])[0].passed
