#!/usr/bin/env python3
"""Record exactly what a native Ptah build consists of. Builds nothing, deploys nothing.

A bundle is only reproducible if its identity is written down: which commit, which
binaries, which kernel libraries those binaries dlopen at runtime, and which model
artifacts they are pinned to. Hashing the executables alone is not enough — the HIP
and Vulkan kernels live in separate shared objects produced by build scripts, and a
binary paired with a different kernel library is a different runtime.

Refuses to emit a manifest from a dirty tree, because a hash that cannot be tied to
a commit is not evidence.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--binary",
        action="append",
        default=[],
        help="Path to an executable to record; repeatable.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="A pinned model artifact to record by digest; repeatable.",
    )
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and not args.allow_dirty:
        print(
            "refusing to write a manifest from a dirty tree; commit or pass "
            f"--allow-dirty\n{dirty}",
            file=sys.stderr,
        )
        return 1

    binaries = {}
    for name in args.binary:
        path = (args.root / name).resolve()
        if not path.exists():
            print(f"missing binary {path}", file=sys.stderr)
            return 1
        binaries[path.name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    # Kernel libraries are dlopened at runtime, so a binary alone does not
    # identify the runtime. Listing every .so the build produced does not either:
    # a tree accumulates same-named libraries from earlier builds, and six
    # identically named candidates identify nothing.
    #
    # Resolution order is an explicit AKH_HIP_KERNEL_DIR, then lib/ beside the
    # executable, then a compile-time path baked into the binary. Absent an
    # override or a bundle, the baked path is what loads, so read it back out of
    # each binary and hash exactly that.
    kernels = {}
    for name, entry in binaries.items():
        blob = Path(entry["path"]).read_bytes()
        found = {}
        # Paths are embedded as bare bytes, not necessarily null-delimited, so
        # scan for absolute paths rather than splitting on a separator.
        for match in re.finditer(rb"/[\w./+-]*(?:hip_kernels|hip_quant)[\w./+-]*\.so", blob):
            lib = Path(match.group().decode())
            if lib.exists():
                found[str(lib)] = {"sha256": sha256(lib), "bytes": lib.stat().st_size}
        kernels[name] = found

    # Same-named libraries elsewhere in the build tree, so a reader can see the
    # ambiguity rather than trip over it.
    candidates = {}
    for lib in sorted((args.root / "target/release/build").rglob("*.so")):
        candidates.setdefault(lib.name, []).append(str(lib.relative_to(args.root)))
    ambiguous = {name: paths for name, paths in candidates.items() if len(paths) > 1}

    artifacts = {}
    for entry in args.artifact:
        name, _, raw = entry.partition("=")
        path = Path(raw)
        if not path.exists():
            print(f"missing artifact {path}", file=sys.stderr)
            return 1
        artifacts[name] = {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}

    print(
        json.dumps(
            {
                "schema": "native-ptah-release-manifest/1",
                "commit": git("rev-parse", "HEAD"),
                "commit_subject": git("log", "-1", "--format=%s"),
                "tree_clean": not dirty,
                "binaries": binaries,
                "kernel_libraries_by_binary": kernels,
                "ambiguous_same_named_libraries": ambiguous,
                "artifacts": artifacts,
                "note": "identity only; this records a build, it does not install one",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
