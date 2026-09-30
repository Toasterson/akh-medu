#!/usr/bin/env python3
"""Pin, hash and provenance helpers for the hephaistos serving scripts (ADR 121).

Standard library only, Python 3.10+ (hephaistos is Ubuntu 22.04). The shell scripts
in this directory call these subcommands; they are kept here so that they can be tested:

  pins get KEY                      print one value from pins.json (dotted key)
  pins file NAME                    print "<sha256> <size>" of one pinned parent file
  sha256 FILE|-                     print the sha256 of a file (or of stdin)
  version-le A B                    exit 0 when dotted version A <= B, 2 when not (a "-76" suffix is ignored)
  verify-dir DIR [--files a,b]      check size + sha256 of the pinned parent files in DIR
  stage --dst PATH --sha256 H --size N
                                    copy stdin to PATH, verified, atomic (partial + rename)
  provenance ...                    write PROVENANCE.json for a converted GGUF
  check-model DIR                   a converted model directory is reusable (hash chain intact)

Every failure is `error[code]: message` plus a `help:` line and a non-zero exit, the same
shape as the other scripts under scripts/lilith/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_PINS = HERE / "pins.json"
CHUNK = 8 * 1024 * 1024
PROVENANCE_SCHEMA = "lilith/hephaistos-gguf-provenance@1"


class HephError(Exception):
    """A refusal with a stable code and the way out."""

    def __init__(self, code: str, message: str, help: str):
        super().__init__(message)
        self.code, self.message, self.help = code, message, help


def load_pins(path: Path = DEFAULT_PINS) -> dict[str, Any]:
    try:
        pins = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HephError("lilith::heph::pins", f"cannot read {path}: {error}",
                        "restore scripts/lilith/hephaistos/pins.json from the repository") from error
    if pins.get("schema") != "lilith/hephaistos-pins@1":
        raise HephError("lilith::heph::pins", f"{path} has an unknown schema {pins.get('schema')!r}",
                        "use the pins.json shipped with this checkout")
    return pins


def version_tuple(text: str) -> tuple[int, ...]:
    """'6.3.4-76' -> (6, 3, 4). Only the leading dotted-numeric part counts."""
    head = text.strip().split("-", 1)[0]
    parts = head.split(".")
    if not head or not all(part.isdigit() for part in parts):
        raise HephError("lilith::heph::version", f"'{text.strip()}' is not a dotted numeric version",
                        "expected something like 6.3.4 or 6.3.4-76")
    return tuple(int(part) for part in parts)


def version_le(left: str, right: str) -> bool:
    """left <= right, padding the shorter version with zeros (6.3 == 6.3.0)."""
    a, b = version_tuple(left), version_tuple(right)
    width = max(len(a), len(b))
    return a + (0,) * (width - len(a)) <= b + (0,) * (width - len(b))


def sha256_stream(handle) -> str:
    digest = hashlib.sha256()
    while chunk := handle.read(CHUNK):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with open(path, "rb") as handle:
        return sha256_stream(handle)


def verify_dir(directory: Path, pins: dict[str, Any], only: list[str] | None = None) -> dict[str, str]:
    """Every pinned parent file in `directory` has the pinned size and sha256.

    Returns {file: sha256}. Raises HephError naming the first file that is missing, has the
    wrong size or the wrong hash (size first: it is free and catches a truncated copy).
    """
    files = pins["parent"]["files"]
    names = only if only is not None else sorted(files)
    unknown = [n for n in names if n not in files]
    if unknown:
        raise HephError("lilith::heph::pins", f"{unknown} are not pinned files of the parent",
                        f"pinned files: {sorted(files)}")
    receipt: dict[str, str] = {}
    for name in names:
        path = directory / name
        expected = files[name]
        if not path.is_file():
            raise HephError("lilith::heph::parent_missing", f"{path} is missing",
                            "stage the parent again (convert_parent.sh) or check the NFS mount")
        size = path.stat().st_size
        if size != expected["size"]:
            raise HephError("lilith::heph::parent_size",
                            f"{path} is {size} bytes, the pin says {expected['size']}",
                            "a truncated or different file; delete it and stage again from the share")
        actual = sha256_file(path)
        if actual != expected["sha256"]:
            raise HephError("lilith::heph::parent_hash",
                            f"{path} sha256 {actual} != pinned {expected['sha256']}",
                            "this is not the pinned parent (google/gemma-4-12B-it@"
                            f"{pins['parent']['revision'][:8]}); never convert or serve it")
        receipt[name] = actual
    return receipt


def stage_stdin(dst: Path, sha256: str, size: int, source=None) -> str:
    """Copy stdin to `dst` through `dst.partial`; keep it only if size and hash match."""
    source = source if source is not None else sys.stdin.buffer
    dst.parent.mkdir(parents=True, exist_ok=True)
    partial = dst.with_name(dst.name + ".partial")
    digest = hashlib.sha256()
    written = 0
    with open(partial, "wb") as out:
        while chunk := source.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            written += len(chunk)
        out.flush()
        os.fsync(out.fileno())
    if written != size or digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise HephError("lilith::heph::stage_mismatch",
                        f"copy to {dst} read {written} bytes, sha256 {digest.hexdigest()}; "
                        f"expected {size} bytes, {sha256}",
                        "the source changed or the read was cut short (NFS drop?); rerun, the copy is idempotent")
    os.replace(partial, dst)
    return digest.hexdigest()


def write_provenance(out: Path, *, pins: dict[str, Any], gguf: Path, receipt: dict[str, str],
                     llama_commit: str, convert_command: str, pip_freeze_sha256: str,
                     template: Path, now: dt.datetime | None = None) -> dict[str, Any]:
    """PROVENANCE.json: the source pins, the converter, and the output's own sha256.

    Written next to the GGUF; `check_model` refuses a model whose file no longer matches.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    body = {
        "schema": PROVENANCE_SCHEMA,
        "created_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "repo": pins["parent"]["repo"],
            "revision": pins["parent"]["revision"],
            "files_sha256": receipt,
        },
        "converter": {
            "llama_cpp_repo": pins["llama_cpp"]["repo"],
            "llama_cpp_tag": pins["llama_cpp"]["tag"],
            "llama_cpp_commit": llama_commit,
            "command": convert_command,
            "pip_freeze_sha256": pip_freeze_sha256,
            "host": platform.node(),
            "python": platform.python_version(),
        },
        "output": {
            "file": gguf.name,
            "quant": pins["gguf"]["quant"],
            "size": gguf.stat().st_size,
            "sha256": sha256_file(gguf),
        },
        "chat_template": {"file": template.name, "sha256": sha256_file(template)},
    }
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n")
    return body


def check_model(directory: Path, pins: dict[str, Any]) -> dict[str, Any]:
    """A converted model directory is reusable when its provenance still matches everything.

    Checks, in order: PROVENANCE.json parses; it names this parent revision, the pinned
    source hashes and the pinned llama.cpp commit; the GGUF has the recorded size and
    sha256; the chat template is the pinned one. Returns the provenance body.
    """
    provenance_path = directory / "PROVENANCE.json"
    try:
        body = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise HephError("lilith::heph::provenance", f"{provenance_path}: {error}",
                        "convert the parent again (convert_parent.sh --force)") from error
    if body.get("schema") != PROVENANCE_SCHEMA:
        raise HephError("lilith::heph::provenance", f"unknown provenance schema {body.get('schema')!r}",
                        "convert the parent again with this checkout (convert_parent.sh --force)")
    source = body.get("source", {})
    pinned_hashes = {n: f["sha256"] for n, f in pins["parent"]["files"].items()}
    if source.get("revision") != pins["parent"]["revision"] or source.get("files_sha256") != pinned_hashes:
        raise HephError("lilith::heph::provenance_source",
                        "the GGUF was not converted from the pinned parent revision and files",
                        "never serve it; convert the pinned parent (convert_parent.sh --force)")
    commit = body.get("converter", {}).get("llama_cpp_commit")
    if commit != pins["llama_cpp"]["commit"]:
        raise HephError("lilith::heph::provenance_converter",
                        f"converted with llama.cpp {commit}, this checkout pins "
                        f"{pins['llama_cpp']['commit']}",
                        "reconvert with the pinned converter (convert_parent.sh --force), or bump the pin "
                        "deliberately in pins.json and the ADR")
    output = body.get("output", {})
    gguf = directory / str(output.get("file", ""))
    if not gguf.is_file() or gguf.stat().st_size != output.get("size"):
        raise HephError("lilith::heph::gguf_size", f"{gguf} is missing or has the wrong size",
                        "copy or convert the model again")
    if sha256_file(gguf) != output.get("sha256"):
        raise HephError("lilith::heph::gguf_hash", f"{gguf} does not match the sha256 in its provenance",
                        "the file is corrupt or was replaced; copy or convert it again")
    template = directory / str(body.get("chat_template", {}).get("file", ""))
    pinned_template = pins["parent"]["files"]["chat_template.jinja"]["sha256"]
    if not template.is_file() or sha256_file(template) != pinned_template:
        raise HephError("lilith::heph::template", f"{template} is missing or is not the pinned chat template",
                        "the server must run the parent's pinned template; copy chat_template.jinja from the parent")
    return body


def _fail(error: HephError) -> int:
    print(f"heph_tools: error[{error.code}]: {error.message}\n  help: {error.help}", file=sys.stderr)
    return 1


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    sub = parser.add_subparsers(dest="command", required=True)
    get = sub.add_parser("pins")
    get.add_argument("action", choices=["get", "file"])
    get.add_argument("key", help="dotted key for `get`; a parent file name for `file` (prints 'sha256 size')")
    sha = sub.add_parser("sha256")
    sha.add_argument("file", type=Path)
    le = sub.add_parser("version-le")
    le.add_argument("left")
    le.add_argument("right")
    verify = sub.add_parser("verify-dir")
    verify.add_argument("directory", type=Path)
    verify.add_argument("--files", default=None, help="comma-separated subset")
    stage = sub.add_parser("stage")
    stage.add_argument("--dst", type=Path, required=True)
    stage.add_argument("--sha256", required=True)
    stage.add_argument("--size", type=int, required=True)
    prov = sub.add_parser("provenance")
    prov.add_argument("--gguf", type=Path, required=True)
    prov.add_argument("--template", type=Path, required=True)
    prov.add_argument("--out", type=Path, required=True)
    prov.add_argument("--parent-dir", type=Path, required=True)
    prov.add_argument("--llama-commit", required=True)
    prov.add_argument("--convert-command", required=True)
    prov.add_argument("--pip-freeze-sha256", required=True)
    check = sub.add_parser("check-model")
    check.add_argument("directory", type=Path)
    args = parser.parse_args(argv)
    try:
        pins = load_pins(args.pins)
        if args.command == "pins" and args.action == "file":
            entry = pins["parent"]["files"].get(args.key)
            if entry is None:
                raise HephError("lilith::heph::pins", f"{args.key} is not a pinned parent file",
                                f"pinned files: {sorted(pins['parent']['files'])}")
            print(entry["sha256"], entry["size"])
        elif args.command == "pins":
            value: Any = pins
            for part in args.key.split("."):
                if not isinstance(value, dict) or part not in value:
                    raise HephError("lilith::heph::pins", f"no pin named {args.key}", "see pins.json")
                value = value[part]
            print(value if isinstance(value, str) else json.dumps(value))
        elif args.command == "sha256":
            if str(args.file) == "-":
                print(sha256_stream(sys.stdin.buffer))
            else:
                print(sha256_file(args.file))
        elif args.command == "version-le":
            return 0 if version_le(args.left, args.right) else 2
        elif args.command == "verify-dir":
            only = args.files.split(",") if args.files else None
            print(json.dumps(verify_dir(args.directory, pins, only), indent=2, sort_keys=True))
        elif args.command == "stage":
            print(stage_stdin(args.dst, args.sha256, args.size))
        elif args.command == "provenance":
            receipt = verify_dir(args.parent_dir, pins)
            write_provenance(args.out, pins=pins, gguf=args.gguf, receipt=receipt,
                             llama_commit=args.llama_commit, convert_command=args.convert_command,
                             pip_freeze_sha256=args.pip_freeze_sha256, template=args.template)
            print(args.out)
        elif args.command == "check-model":
            body = check_model(args.directory, pins)
            print(json.dumps({"ok": True, "output": body["output"]}, sort_keys=True))
    except HephError as error:
        return _fail(error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
