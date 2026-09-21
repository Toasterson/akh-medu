#!/usr/bin/env python3
"""Assemble a relocatable native Ptah bundle. Builds nothing, installs nothing.

Copies the executables and the kernel libraries each one actually loads into a
single directory, so the bundle runs from anywhere without an
AKH_HIP_KERNEL_DIR override: lookup prefers lib/ beside the executable.

Records the toolchain the kernels were compiled against, which an executable
hash does not capture and which a bundle moved between machines depends on.
Refuses to overwrite an existing bundle, so evidence is never silently replaced.
"""
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kernel_paths(binary: Path) -> list[Path]:
    """The kernel libraries this binary resolves at runtime."""
    blob = binary.read_bytes()
    found = []
    for match in re.finditer(rb"/[\w./+-]*(?:hip_kernels|hip_quant)[\w./+-]*\.so", blob):
        lib = Path(match.group().decode())
        if lib.exists() and lib not in found:
            found.append(lib)
    return found


def toolchain() -> dict:
    """Versions the kernels were built against.

    Read from files rather than from tools on PATH: `hipcc` and `rocm-smi` are
    not necessarily on it even where ROCm is installed, and a bundle that
    records `null` because a command was missing is worse than one that records
    nothing, because it looks like an answer.
    """
    out = {}
    version = Path("/opt/rocm/.info/version")
    out["rocm"] = version.read_text().strip() if version.exists() else None
    # The amdgpu module exposes srcversion rather than a version file on this
    # kernel, so record whichever is present.
    for key, path in [
        ("amdgpu_version", Path("/sys/module/amdgpu/version")),
        ("amdgpu_srcversion", Path("/sys/module/amdgpu/srcversion")),
    ]:
        out[key] = path.read_text().strip() if path.exists() else None
    try:
        out["kernel"] = subprocess.check_output(["uname", "-r"], text=True).strip()
    except (OSError, subprocess.SubprocessError):
        out["kernel"] = None
    missing = [k for k, v in out.items() if v is None]
    if missing:
        out["unrecorded"] = missing
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # Exclusive creation: a bundle that silently replaced another is not evidence.
    args.output.mkdir(parents=True, exist_ok=False)
    lib_dir = args.output / "lib"
    lib_dir.mkdir()

    manifest = {"binaries": {}, "libraries": {}, "toolchain": toolchain()}
    manifest["commit"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    for name in args.binary:
        source = Path(name).resolve(strict=True)
        target = args.output / source.name
        shutil.copy2(source, target)
        manifest["binaries"][source.name] = {
            "sha256": sha256(target),
            "source": str(source),
        }
        for lib in kernel_paths(source):
            copied = lib_dir / lib.name
            if copied.exists() and sha256(copied) != sha256(lib):
                print(
                    f"conflict: two different {lib.name} required by this bundle",
                    file=sys.stderr,
                )
                return 1
            if not copied.exists():
                shutil.copy2(lib, copied)
            manifest["libraries"][lib.name] = {
                "sha256": sha256(copied),
                "source": str(lib),
            }

    (args.output / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bundle": str(args.output), **{k: len(v) for k, v in manifest.items() if isinstance(v, dict)}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
