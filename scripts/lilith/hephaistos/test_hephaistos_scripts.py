"""Tests for the hephaistos serving scripts (ADR 121).

No GPU, no root, no network, no llama.cpp: the shell scripts run against fakes in temp directories
(LILITH_NO_SUDO=1, LILITH_NFS_ACCESS=direct, a fake llama-server), verify.py against a local HTTP
server that speaks the llama-server subset it uses.

    python3 -m unittest scripts/lilith/hephaistos/test_hephaistos_scripts.py
"""
from __future__ import annotations

import hashlib
import http.server
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


tools = load("heph_tools", HERE / "heph_tools.py")
verify = load("heph_verify", HERE / "verify.py")

BASH = shutil.which("bash") or "/bin/bash"
SCRIPTS = ["common.sh", "build_server.sh", "convert_parent.sh", "install_service.sh", "run_server.sh"]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fake_pins(parent_files: dict[str, bytes], *, commit: str = "a" * 40) -> dict:
    return {
        "schema": "lilith/hephaistos-pins@1",
        "parent": {"repo": "google/gemma-4-12B-it", "revision": "5926caa4ec0cac5cbfadaf4077420520de1d5205",
                   "nfs_relative_dir": "runs/parent",
                   "files": {name: {"sha256": sha(data), "size": len(data)} for name, data in parent_files.items()}},
        "llama_cpp": {"repo": "https://example.invalid/llama.cpp.git", "tag": "vtest", "commit": commit,
                      "gpu_target": "gfx1100"},
        "gguf": {"quant": "q8_0", "filename": "lilith-parent-q8_0.gguf", "nfs_relative_parent": "runs/models"},
    }


PARENT = {"chat_template.jinja": b"{{ template }}\n", "config.json": b'{"a": 1}\n',
          "generation_config.json": b"{}\n", "model.safetensors": b"W" * 4096,
          "processor_config.json": b"{}\n", "tokenizer.json": b"{}\n", "tokenizer_config.json": b"{}\n"}


class PinsTests(unittest.TestCase):
    def test_pins_json_is_wellformed_and_consistent_with_the_runpod_bootstrap(self):
        pins = tools.load_pins()
        files = pins["parent"]["files"]
        self.assertEqual(sorted(files), sorted(["chat_template.jinja", "config.json", "generation_config.json",
                                                "model.safetensors", "processor_config.json", "tokenizer.json",
                                                "tokenizer_config.json"]))
        for entry in files.values():
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size"], 0)
        self.assertRegex(pins["llama_cpp"]["commit"], r"^[0-9a-f]{40}$")
        self.assertRegex(pins["llama_cpp"]["converter_transformers"], r"^5\.\d+\.\d+$")
        self.assertIn("4.57.6", pins["llama_cpp"]["converter_transformers_reason"])
        self.assertEqual(pins["parent"]["revision"], "5926caa4ec0cac5cbfadaf4077420520de1d5205")
        bootstrap = (REPO / "scripts/lilith/runpod/bootstrap.sh").read_text()
        for name, variable in (("model.safetensors", "WEIGHTS_SHA256"), ("config.json", "CONFIG_SHA256"),
                               ("tokenizer.json", "TOKENIZER_SHA256")):
            declared = re.search(rf'^{variable}="([0-9a-f]{{64}})"', bootstrap, re.M).group(1)
            self.assertEqual(files[name]["sha256"], declared, f"{name} pin differs from bootstrap.sh")
        revision = re.search(r'^PARENT_REVISION="([0-9a-f]{40})"', bootstrap, re.M).group(1)
        self.assertEqual(pins["parent"]["revision"], revision)

    def test_the_build_pins_name_the_rocm_floor_and_a_hashed_versioned_vulkan_sdk(self):
        pins = tools.load_pins()
        self.assertEqual(pins["llama_cpp"]["hip_rocm_known_bad_through"], "6.3.4")
        self.assertIn("dpp_ctrl", pins["llama_cpp"]["hip_rocm_known_bad_reason"])
        sdk = pins["vulkan_sdk"]
        self.assertRegex(sdk["version"], r"^1\.\d+\.\d+\.\d+$")
        self.assertRegex(sdk["sha256"], r"^[0-9a-f]{64}$")
        self.assertIn(sdk["version"], sdk["url"], "the URL is versioned, never 'latest'")
        self.assertNotIn("latest", sdk["url"])
        self.assertGreater(sdk["system_headers_min_patch"], 204, "jammy's 1.3.204 headers are known too old")

    def test_the_serving_defaults_meet_the_context_requirement(self):
        env = (HERE / "server.env.example").read_text()
        per_slot = int(re.search(r"^LILITH_LLAMA_CTX_PER_SLOT=(\d+)$", env, re.M).group(1))
        akh = (REPO / "deploy/roman-mini/full_akh.py").read_text()
        window = int(re.search(r"^LILITH_CONTEXT_WINDOW = (\d+)", akh, re.M).group(1))
        self.assertGreaterEqual(per_slot, 49152)
        self.assertGreaterEqual(per_slot, window)
        self.assertEqual(verify.MIN_CTX, 49152)
        model = re.search(r'^LILITH_MODEL = "([^"]+)"', akh, re.M).group(1)
        self.assertEqual(re.search(r"^LILITH_MODEL_NAME=(\S+)$", env, re.M).group(1), model)
        self.assertEqual(verify.DEFAULT_MODEL, model)


class VersionTests(unittest.TestCase):
    def test_version_ordering_ignores_the_package_suffix_and_pads_zeros(self):
        self.assertTrue(tools.version_le("6.3.4-76", "6.3.4"))
        self.assertTrue(tools.version_le("6.2.0", "6.3.4"))
        self.assertTrue(tools.version_le("6.3", "6.3.0"))
        self.assertFalse(tools.version_le("6.4.0", "6.3.4"))
        self.assertFalse(tools.version_le("6.10.0", "6.3.4"), "numeric, not lexicographic")
        self.assertTrue(tools.version_le("7.0.0", "7.0"))

    def test_a_non_numeric_version_is_a_refusal(self):
        with self.assertRaises(tools.HephError):
            tools.version_le("none", "6.3.4")

    def test_the_cli_exit_codes(self):
        run = lambda *a: subprocess.run([sys.executable, str(HERE / "heph_tools.py"), "version-le", *a],
                                        capture_output=True, text=True).returncode
        self.assertEqual(run("6.3.4", "6.3.4"), 0)
        self.assertEqual(run("6.4.0", "6.3.4"), 2)
        self.assertEqual(run("banana", "6.3.4"), 1)


class HephToolsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.parent = self.dir / "parent"
        self.parent.mkdir()
        for name, data in PARENT.items():
            (self.parent / name).write_bytes(data)
        self.pins = fake_pins(PARENT)

    def tearDown(self):
        self.temp.cleanup()

    def test_verify_dir_accepts_the_pinned_parent(self):
        receipt = tools.verify_dir(self.parent, self.pins)
        self.assertEqual(receipt["config.json"], sha(PARENT["config.json"]))
        self.assertEqual(sorted(receipt), sorted(PARENT))

    def test_verify_dir_refuses_each_kind_of_difference(self):
        (self.parent / "config.json").write_bytes(b'{"a": 2}\n')          # same size, other bytes
        with self.assertRaises(tools.HephError) as caught:
            tools.verify_dir(self.parent, self.pins)
        self.assertEqual(caught.exception.code, "lilith::heph::parent_hash")
        (self.parent / "config.json").write_bytes(b"short")
        with self.assertRaises(tools.HephError) as caught:
            tools.verify_dir(self.parent, self.pins, ["config.json"])
        self.assertEqual(caught.exception.code, "lilith::heph::parent_size")
        (self.parent / "config.json").unlink()
        with self.assertRaises(tools.HephError) as caught:
            tools.verify_dir(self.parent, self.pins, ["config.json"])
        self.assertEqual(caught.exception.code, "lilith::heph::parent_missing")
        with self.assertRaises(tools.HephError):
            tools.verify_dir(self.parent, self.pins, ["not-a-pinned-file"])

    def test_stage_keeps_only_a_verified_copy(self):
        data = PARENT["model.safetensors"]
        dst = self.dir / "stage" / "model.safetensors"
        tools.stage_stdin(dst, sha(data), len(data), io.BytesIO(data))
        self.assertEqual(dst.read_bytes(), data)
        self.assertFalse(dst.with_name(dst.name + ".partial").exists())
        bad = self.dir / "stage" / "bad.bin"
        with self.assertRaises(tools.HephError) as caught:
            tools.stage_stdin(bad, sha(data), len(data), io.BytesIO(data[:-1] + b"X"))
        self.assertEqual(caught.exception.code, "lilith::heph::stage_mismatch")
        self.assertFalse(bad.exists())
        self.assertFalse(bad.with_name("bad.bin.partial").exists())
        with self.assertRaises(tools.HephError):
            tools.stage_stdin(bad, sha(data), len(data), io.BytesIO(data[:100]))

    def make_model(self, **overrides) -> Path:
        out = self.dir / "model"
        out.mkdir(exist_ok=True)
        gguf = out / "lilith-parent-q8_0.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 1000)
        (out / "chat_template.jinja").write_bytes(PARENT["chat_template.jinja"])
        receipt = tools.verify_dir(self.parent, self.pins)
        tools.write_provenance(out / "PROVENANCE.json", pins=self.pins, gguf=gguf, receipt=receipt,
                               llama_commit=overrides.get("commit", self.pins["llama_cpp"]["commit"]),
                               convert_command="convert", pip_freeze_sha256="f" * 64,
                               template=out / "chat_template.jinja")
        return out

    def test_a_converted_model_verifies_and_tampering_is_caught(self):
        out = self.make_model()
        body = tools.check_model(out, self.pins)
        self.assertEqual(body["output"]["quant"], "q8_0")
        self.assertEqual(body["source"]["revision"], self.pins["parent"]["revision"])
        # the output file changes
        (out / "lilith-parent-q8_0.gguf").write_bytes(b"GGUF" + b"\1" * 1000)
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, self.pins)
        self.assertEqual(caught.exception.code, "lilith::heph::gguf_hash")
        (out / "lilith-parent-q8_0.gguf").write_bytes(b"GGUF")
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, self.pins)
        self.assertEqual(caught.exception.code, "lilith::heph::gguf_size")

    def test_a_model_from_another_parent_or_converter_or_template_is_refused(self):
        out = self.make_model()
        other = dict(self.pins, parent=dict(self.pins["parent"], files={
            **self.pins["parent"]["files"], "config.json": {"sha256": "0" * 64, "size": 8}}))
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, other)
        self.assertEqual(caught.exception.code, "lilith::heph::provenance_source")
        moved = dict(self.pins, llama_cpp=dict(self.pins["llama_cpp"], commit="b" * 40))
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, moved)
        self.assertEqual(caught.exception.code, "lilith::heph::provenance_converter")
        (out / "chat_template.jinja").write_bytes(b"{{ another }}\n")
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, self.pins)
        self.assertEqual(caught.exception.code, "lilith::heph::template")
        (out / "PROVENANCE.json").unlink()
        with self.assertRaises(tools.HephError) as caught:
            tools.check_model(out, self.pins)
        self.assertEqual(caught.exception.code, "lilith::heph::provenance")

    def test_cli_errors_have_a_code_and_a_help_line(self):
        pins_file = self.dir / "pins.json"
        pins_file.write_text(json.dumps(self.pins))
        (self.parent / "config.json").write_bytes(b'{"a": 9}\n')
        result = subprocess.run([sys.executable, str(HERE / "heph_tools.py"), "--pins", str(pins_file),
                                 "verify-dir", str(self.parent)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::parent_hash]", result.stderr)
        self.assertIn("help:", result.stderr)
        self.assertEqual(subprocess.run([sys.executable, str(HERE / "heph_tools.py"), "--pins", str(pins_file),
                                         "pins", "file", "config.json"], capture_output=True, text=True).stdout.split(),
                         [self.pins["parent"]["files"]["config.json"]["sha256"], str(len(PARENT["config.json"]))])


class ShellTests(unittest.TestCase):
    def test_every_script_parses(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                result = subprocess.run([BASH, "-n", str(HERE / name)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    @unittest.skipUnless(shutil.which("shellcheck"), "shellcheck is not installed")
    def test_shellcheck_is_clean(self):
        result = subprocess.run(["shellcheck", "-x", "-S", "warning", *[str(HERE / n) for n in SCRIPTS]],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout)


class RunServerTests(unittest.TestCase):
    """run_server.sh against a fake llama-server that records its argv and environment."""

    KEY = "k3y-" + "z" * 40

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.record = self.dir / "argv.json"
        server = self.dir / "llama-server"
        server.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                          f"json.dump({{'argv': sys.argv[1:], 'env': dict(os.environ)}}, open({str(self.record)!r}, 'w'))\n")
        server.chmod(0o755)
        (self.dir / "model.gguf").write_bytes(b"GGUF")
        (self.dir / "template.jinja").write_text("t")
        key = self.dir / "api-key"
        key.write_text(self.KEY + "\n")
        key.chmod(0o600)
        self.env = {**os.environ, "LLAMA_SERVER": str(server), "LILITH_GGUF": str(self.dir / "model.gguf"),
                    "LILITH_CHAT_TEMPLATE": str(self.dir / "template.jinja"), "LILITH_API_KEY_FILE": str(key),
                    "LILITH_LLAMA_HOST": "192.168.1.114", "LILITH_LLAMA_PORT": "8080",
                    "LILITH_LLAMA_PARALLEL": "2", "LILITH_LLAMA_CTX_PER_SLOT": "65536"}

    def tearDown(self):
        self.temp.cleanup()

    def run_it(self, **env):
        return subprocess.run([BASH, str(HERE / "run_server.sh")], env={**self.env, **env}, capture_output=True, text=True)

    def flag(self, argv, name):
        return argv[argv.index(name) + 1]

    def test_the_command_line_serves_lilith_with_tools_and_no_secret(self):
        result = self.run_it()
        self.assertEqual(result.returncode, 0, result.stderr)
        seen = json.loads(self.record.read_text())
        argv = seen["argv"]
        self.assertEqual(self.flag(argv, "--host"), "192.168.1.114")
        self.assertEqual(self.flag(argv, "--port"), "8080")
        self.assertEqual(self.flag(argv, "--alias"), "lilith")
        self.assertEqual(self.flag(argv, "--api-key-file"), str(self.dir / "api-key"))
        self.assertIn("--jinja", argv)
        self.assertEqual(self.flag(argv, "--chat-template-file"), str(self.dir / "template.jinja"))
        self.assertEqual(self.flag(argv, "--reasoning-format"), "deepseek")
        self.assertEqual(self.flag(argv, "--parallel"), "2")
        self.assertEqual(self.flag(argv, "--kv-unified-per-slot"), "65536")
        self.assertEqual(self.flag(argv, "-ngl"), "99")
        self.assertEqual(self.flag(argv, "--cache-type-k"), "f16")
        self.assertIn("--no-webui", argv)
        self.assertIn("--no-slots", argv)
        self.assertNotIn("--api-key", argv)
        self.assertNotIn("--sleep-idle-seconds", argv)
        self.assertNotIn(self.KEY, json.dumps(seen), "the key must be on no argv and in no environment")
        self.assertNotIn(self.KEY, result.stderr)

    def test_optional_settings(self):
        result = self.run_it(LILITH_LLAMA_SLEEP_IDLE_SECONDS="900", LILITH_LLAMA_KV_TYPE="q8_0",
                             LILITH_LLAMA_EXTRA_ARGS="-ub 1024", LILITH_MODEL_NAME="lilith-test")
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = json.loads(self.record.read_text())["argv"]
        self.assertEqual(self.flag(argv, "--sleep-idle-seconds"), "900")
        self.assertEqual(self.flag(argv, "--cache-type-v"), "q8_0")
        self.assertEqual(self.flag(argv, "-ub"), "1024")
        self.assertEqual(self.flag(argv, "--alias"), "lilith-test")

    def test_refusals_name_the_fix(self):
        cases = [
            ({"LILITH_LLAMA_HOST": "0.0.0.0"}, "lilith::heph::bind", 0o600),
            ({"LILITH_LLAMA_HOST": "::"}, "lilith::heph::bind", 0o600),
            ({"LILITH_LLAMA_PARALLEL": "two"}, "lilith::heph::env", 0o600),
            ({"LILITH_LLAMA_CTX_PER_SLOT": "0"}, "lilith::heph::env", 0o600),
            ({"LILITH_GGUF": str(self.dir / "missing.gguf")}, "lilith::heph::model", 0o600),
            ({"LILITH_CHAT_TEMPLATE": str(self.dir / "missing.jinja")}, "lilith::heph::template", 0o600),
            ({"LLAMA_SERVER": str(self.dir / "missing")}, "lilith::heph::server", 0o600),
            ({}, "lilith::heph::key", 0o644),           # a key readable by others
            ({"LILITH_API_KEY_FILE": str(self.dir / "nokey")}, "lilith::heph::key", 0o600),
        ]
        for env, code, mode in cases:
            with self.subTest(env=env):
                (self.dir / "api-key").chmod(mode)
                self.record.unlink(missing_ok=True)
                result = self.run_it(**env)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"error[{code}]", result.stderr)
                self.assertIn("help:", result.stderr)
                self.assertFalse(self.record.exists(), "the server must not start on a refusal")
        result = subprocess.run([BASH, str(HERE / "run_server.sh")], env={"PATH": os.environ["PATH"]},
                                capture_output=True, text=True)
        self.assertIn("error[lilith::heph::env]", result.stderr)


class ServiceFileTests(unittest.TestCase):
    def test_the_unit_keeps_the_key_off_every_command_line_and_environment(self):
        unit = (HERE / "lilith-serve.service").read_text()
        directives = [line for line in unit.splitlines() if line and not line.startswith(("#", "["))]
        keyed = dict(line.split("=", 1) for line in directives)
        self.assertEqual(keyed["ExecStart"], "/opt/lilith-serve/bin/run_server.sh")
        self.assertEqual(keyed["EnvironmentFile"], "/etc/lilith-serve/server.env")
        self.assertEqual(keyed["User"], "lilith-serve")
        self.assertIn("render", keyed["SupplementaryGroups"])
        self.assertNotIn("api-key", keyed["ExecStart"])
        self.assertFalse([d for d in directives if d.startswith(("Environment=", "PassEnvironment="))])
        for line in unit.splitlines():
            self.assertNotRegex(line, r"(?i)(secret|token|password)=", line)
        self.assertEqual(keyed["NoNewPrivileges"], "true")
        self.assertEqual(keyed["ProtectSystem"], "strict")
        self.assertNotIn("PrivateDevices", keyed, "ROCm needs /dev/kfd and /dev/dri")

    def test_the_env_example_binds_the_lan_and_names_a_key_file_not_a_key(self):
        env = dict(line.split("=", 1) for line in (HERE / "server.env.example").read_text().splitlines()
                   if line and not line.startswith("#"))
        self.assertEqual(env["LILITH_LLAMA_HOST"], "192.168.1.114")
        self.assertEqual(env["LILITH_API_KEY_FILE"], "/etc/lilith-serve/api-key")
        self.assertFalse([k for k in env if "KEY" in k and k != "LILITH_API_KEY_FILE"])
        for name in ("LLAMA_SERVER", "LILITH_GGUF", "LILITH_CHAT_TEMPLATE", "LILITH_API_KEY_FILE", "LILITH_LLAMA_HOST",
                     "LILITH_LLAMA_PORT", "LILITH_LLAMA_PARALLEL", "LILITH_LLAMA_CTX_PER_SLOT"):
            self.assertIn(name, env, f"{name} is required by run_server.sh but not in the example")
            self.assertIn(f"{name}", (HERE / "run_server.sh").read_text())

    def test_the_install_script_never_puts_the_key_on_a_command_line(self):
        script = (HERE / "install_service.sh").read_text()
        generation = [line for line in script.splitlines() if "token_urlsafe" in line]
        self.assertEqual(len(generation), 1)
        self.assertIn('| as_root tee "$KEY_FILE.new"', generation[0], "the key goes through a pipe into a 0600 file")
        self.assertIn("umask 077", script)
        self.assertNotIn("--api-key ", script)
        for line in script.splitlines():
            if "$KEY_FILE" in line and not line.lstrip().startswith("#"):
                self.assertNotRegex(line, r"\b(echo|printf)\b.*\$\(.*KEY_FILE", "the key is never echoed")


class FakeServerCase(unittest.TestCase):
    """Common temp layout for the script-level tests with LILITH_NO_SUDO."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)
        self.env = {**os.environ, "LILITH_NO_SUDO": "1", "LILITH_SERVE_PREFIX": str(self.dir / "opt"),
                    "LILITH_SERVE_STATE": str(self.dir / "state"), "LILITH_SERVE_ETC": str(self.dir / "etc"),
                    "LILITH_WORK_DIR": str(self.dir / "work"), "LILITH_NFS_ACCESS": "direct",
                    "LILITH_NFS_MOUNT": str(self.dir / "nfs"), "LILITH_MIN_FREE_GIB": "0"}

    def tearDown(self):
        self.temp.cleanup()

    def run_script(self, name, *args, **env):
        return subprocess.run([BASH, str(HERE / name), *args], env={**self.env, **env}, capture_output=True, text=True)


class BuildServerTests(FakeServerCase):
    REQUIRED = ["--api-key-file", "--jinja", "--chat-template-file", "--reasoning-format", "--kv-unified-per-slot",
                "--parallel", "--no-slots", "--no-webui", "--alias", "--cache-ram", "--flash-attn"]

    def install_fake(self, flags):
        pins = tools.load_pins()
        dest = self.dir / "opt/llama.cpp" / f"{pins['llama_cpp']['commit'][:12]}-hip"
        (dest / "bin").mkdir(parents=True)
        server = dest / "bin/llama-server"
        server.write_text("#!/bin/sh\necho 'options:'\n" + "".join(f"echo '  {flag} X'\n" for flag in flags))
        server.chmod(0o755)
        (dest / "BUILD.json").write_text(json.dumps({"commit": pins["llama_cpp"]["commit"], "backend": "hip"}))
        return dest

    def test_check_accepts_a_complete_build_and_points_current_at_it(self):
        dest = self.install_fake(self.REQUIRED)
        result = self.run_script("build_server.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already installed", result.stderr)
        self.assertEqual((self.dir / "opt/current").resolve(), dest.resolve())

    def test_check_only_never_changes_anything(self):
        self.install_fake(self.REQUIRED)
        result = self.run_script("build_server.sh", "--check")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse((self.dir / "opt/current").exists())

    def test_a_build_missing_a_needed_flag_is_refused(self):
        self.install_fake([f for f in self.REQUIRED if f != "--kv-unified-per-slot"])
        result = self.run_script("build_server.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::flags]", result.stderr)
        self.assertIn("--kv-unified-per-slot", result.stderr)

    def test_check_without_a_build_says_how_to_build(self):
        result = self.run_script("build_server.sh", "--check")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::not_built]", result.stderr)

    def test_bad_arguments(self):
        for args, code in ((("--backend", "cuda"), "usage"), (("--frobnicate",), "usage")):
            result = self.run_script("build_server.sh", *args)
            self.assertEqual(result.returncode, 1)
            self.assertIn(f"error[lilith::heph::{code}]", result.stderr)


class BuildToolchainTests(FakeServerCase):
    """build_server.sh toolchain handling on Ubuntu 22.04: ROCm PATH and version floor, libstdc++-dev,
    and the pinned Vulkan SDK. Fakes only: PATH is a fake bin dir plus /usr/bin:/bin."""

    SDK_VERSION = "9.9.9.0"

    def setUp(self):
        super().setUp()
        self.bin = self.dir / "fakebin"
        self.bin.mkdir()
        self.apt_log = self.dir / "apt.log"
        self.fake("apt-get", f'echo "$@" >> {self.apt_log}')
        self.fake("cmake", "exit 0")
        self.fake("c++", "exit 0")
        # ROCm: bin/ is NOT on PATH, like hephaistos
        self.rocm = self.dir / "rocm"
        (self.rocm / "bin").mkdir(parents=True)
        (self.rocm / ".info").mkdir()
        (self.rocm / "include/hipblas").mkdir(parents=True)
        (self.rocm / "include/hipblas/hipblas.h").write_text("")
        (self.rocm / "bin/hipconfig").write_text(f'#!/bin/sh\ncase "$1" in -R) echo {self.rocm};; -l) echo {self.rocm}/llvm/bin;; esac\n')
        (self.rocm / "bin/rocminfo").write_text("#!/bin/sh\necho 'Name: gfx1100'\n")
        for name in ("hipconfig", "rocminfo"):
            (self.rocm / "bin" / name).chmod(0o755)
        self.set_rocm_version("6.4.1-1")
        # GCC 11 and 12 installed; only 11's libstdc++ headers
        self.gcc = self.dir / "gcc"
        self.cxx = self.dir / "cxx"
        for major in ("9", "10", "12"):
            (self.gcc / major).mkdir(parents=True)
        (self.gcc / "plugin").mkdir()
        (self.cxx / "10").mkdir(parents=True)
        (self.cxx / "10/cmath").write_text("")
        self.vk_include = self.dir / "vkinc"
        self.vk_include.mkdir()
        self.env.update({"PATH": f"{self.bin}:/usr/bin:/bin", "LILITH_ROCM_DIR": str(self.rocm),
                         "LILITH_GCC_LIB_DIR": str(self.gcc), "LILITH_CXX_INCLUDE_DIR": str(self.cxx),
                         "LILITH_VULKAN_INCLUDE_DIR": str(self.vk_include), "LILITH_HOST_ARCH": "x86_64"})

    def fake(self, name, body):
        path = self.bin / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)

    def set_rocm_version(self, version):
        (self.rocm / ".info/version").write_text(version + "\n")

    def set_vulkan_headers(self, patch):
        (self.vk_include / "vulkan_core.h").write_text(f"#define VK_HEADER_VERSION {patch}\n")

    def apt_args(self):
        return self.apt_log.read_text().splitlines() if self.apt_log.exists() else []

    # ---- HIP -------------------------------------------------------------------------------------

    def test_hip_adds_rocm_bin_to_path_itself_and_refuses_the_known_bad_rocm(self):
        self.set_rocm_version("6.3.4-76")
        result = self.run_script("build_server.sh", "--toolchain-only")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("error[lilith::heph::rocm_version]", result.stderr, "hipconfig was found via /opt/rocm/bin")
        self.assertNotIn("error[lilith::heph::tool]", result.stderr)
        self.assertIn("6.3.4", result.stderr)
        self.assertIn("--backend vulkan", result.stderr)
        self.assertIn("upgrade ROCm", result.stderr)

    def test_an_older_rocm_is_refused_too_and_the_override_builds_anyway(self):
        self.set_rocm_version("6.2.4")
        (self.cxx / "12").mkdir()
        (self.cxx / "12/cmath").write_text("")
        self.assertIn("rocm_version", self.run_script("build_server.sh", "--toolchain-only").stderr)
        forced = self.run_script("build_server.sh", "--toolchain-only", LILITH_ALLOW_OLD_ROCM="1")
        self.assertEqual(forced.returncode, 0, forced.stderr)
        self.assertIn("WARNING: ROCm 6.2.4 is known bad", forced.stderr)

    def test_a_newer_rocm_passes_the_floor_then_needs_the_libstdcxx_of_the_highest_gcc(self):
        result = self.run_script("build_server.sh", "--toolchain-only")
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("error[lilith::heph::libstdcxx]", result.stderr)
        self.assertIn("libstdc++-12-dev", result.stderr, "structurally the highest GCC (12), not 10 or 9")
        (self.cxx / "12").mkdir()
        (self.cxx / "12/cmath").write_text("")
        ok = self.run_script("build_server.sh", "--toolchain-only")
        self.assertEqual(ok.returncode, 0, ok.stderr)
        self.assertIn("toolchain ok (hip)", ok.stderr)
        self.assertIn("rocm=6.4.1-1", ok.stderr)

    def test_hip_install_deps_installs_libstdcxx_dev_for_the_highest_gcc_numerically(self):
        self.run_script("build_server.sh", "--install-deps", "--toolchain-only")
        self.assertIn("install -y libstdc++-12-dev", self.apt_args(), "12 beats 9 and 10 (numeric sort)")
        self.assertFalse([line for line in self.apt_args() if "glslc" in line or "vulkan" in line])

    def test_hip_install_deps_follows_a_newer_gcc_without_a_hardcoded_version(self):
        (self.gcc / "13").mkdir()
        self.run_script("build_server.sh", "--install-deps", "--toolchain-only")
        self.assertIn("install -y libstdc++-13-dev", self.apt_args())

    def test_the_rocm_version_is_unreadable_then_it_warns_but_does_not_block(self):
        (self.rocm / ".info/version").unlink()
        (self.cxx / "12").mkdir()
        (self.cxx / "12/cmath").write_text("")
        result = self.run_script("build_server.sh", "--toolchain-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("could not read the ROCm version", result.stderr)

    # ---- Vulkan ----------------------------------------------------------------------------------

    def make_sdk_pins(self, *, sha=None):
        """Point the pins' vulkan_sdk at a local fake tarball (file:// URL) holding <ver>/setup-env.sh."""
        stage = self.dir / "sdkstage"
        sdk = stage / self.SDK_VERSION
        (sdk / "x86_64/bin").mkdir(parents=True)
        glslc = sdk / "x86_64/bin/glslc"
        glslc.write_text("#!/bin/sh\necho glslc fake\n")
        glslc.chmod(0o755)
        (sdk / "setup-env.sh").write_text(
            f'export VULKAN_SDK="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)/x86_64"\n'
            'export PATH="$VULKAN_SDK/bin:$PATH"\necho "$UNSET_IN_INTERACTIVE_SHELLS" >/dev/null\n')
        tarball = self.dir / "vulkan_sdk.tar.xz"
        subprocess.run(["tar", "-cJf", str(tarball), "-C", str(stage), self.SDK_VERSION], check=True)
        pins = tools.load_pins()
        pins["vulkan_sdk"].update({"version": self.SDK_VERSION, "url": tarball.as_uri(),
                                   "sha256": sha or hashlib.sha256(tarball.read_bytes()).hexdigest(),
                                   "size": tarball.stat().st_size})
        pins_file = self.dir / "pins-sdk.json"
        pins_file.write_text(json.dumps(pins))
        self.env["LILITH_PINS_FILE"] = str(pins_file)
        return pins

    def test_vulkan_install_deps_apt_installs_only_the_headers_and_drivers_not_glslc(self):
        self.make_sdk_pins()
        self.set_vulkan_headers(204)
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps", "--toolchain-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        vulkan_lines = [line for line in self.apt_args() if "vulkan" in line]
        self.assertEqual(vulkan_lines, ["install -y libvulkan-dev mesa-vulkan-drivers"])
        self.assertFalse([line for line in self.apt_args() if "glslc" in line or "libstdc++" in line])

    def test_old_system_headers_and_no_glslc_download_the_pinned_sdk_verify_and_use_it(self):
        self.make_sdk_pins()
        self.set_vulkan_headers(204)   # jammy
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps", "--toolchain-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"pinned Vulkan SDK {self.SDK_VERSION}", result.stderr)
        self.assertIn(f"vulkan_sdk={self.SDK_VERSION}", result.stderr)
        root = self.dir / "work/vulkan-sdk"
        self.assertTrue((root / self.SDK_VERSION / "setup-env.sh").is_file())
        self.assertTrue((root / f"{self.SDK_VERSION}.sha256").is_file())
        self.assertFalse(list(root.glob("*.tar.xz*")), "the tarball is not kept")
        # rerun: reused, no second download
        again = self.run_script("build_server.sh", "--backend", "vulkan", "--toolchain-only")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertNotIn("downloading", again.stderr)
        self.assertIn(f"vulkan_sdk={self.SDK_VERSION}", again.stderr)

    def test_a_download_that_does_not_match_its_pinned_sha256_is_refused_and_not_unpacked(self):
        self.make_sdk_pins(sha="0" * 64)
        self.set_vulkan_headers(204)
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps", "--toolchain-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::vulkan_sdk]", result.stderr)
        self.assertIn("pinned sha256", result.stderr)
        self.assertFalse((self.dir / f"work/vulkan-sdk/{self.SDK_VERSION}").exists())

    def test_without_install_deps_nothing_is_downloaded_and_the_help_names_the_flag(self):
        self.make_sdk_pins()
        self.set_vulkan_headers(204)
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--toolchain-only")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::vulkan_sdk]", result.stderr)
        self.assertIn("--backend vulkan --install-deps", result.stderr)
        self.assertFalse((self.dir / "work/vulkan-sdk").exists())

    def test_a_sufficient_system_toolchain_is_used_without_the_sdk(self):
        self.make_sdk_pins()
        self.fake("glslc", "echo glslc")
        self.set_vulkan_headers(275)
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps", "--toolchain-only")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("using the system Vulkan toolchain", result.stderr)
        self.assertIn("vulkan_sdk=system", result.stderr)
        self.assertFalse((self.dir / "work/vulkan-sdk").exists())

    def test_glslc_with_too_old_headers_still_needs_the_sdk(self):
        self.make_sdk_pins()
        self.fake("glslc", "echo glslc")
        self.set_vulkan_headers(204)
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps", "--toolchain-only")
        self.assertIn(f"vulkan_sdk={self.SDK_VERSION}", result.stderr)

    def test_a_full_vulkan_build_records_the_sdk_in_build_json_and_a_toolchain_change_cleans_the_build_dir(self):
        self.make_sdk_pins()
        self.set_vulkan_headers(204)
        # a local llama.cpp "repo" at a known commit, so ensure_llama_source can fetch it by SHA
        repo = self.dir / "llama-repo"
        repo.mkdir()
        git = lambda *a: subprocess.run(["git", "-C", str(repo), *a], check=True, capture_output=True, text=True,
                                        env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                                             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"})
        git("init", "-q")
        (repo / "CMakeLists.txt").write_text("\n")
        git("add", ".")
        git("commit", "-q", "-m", "x")
        commit = git("rev-parse", "HEAD").stdout.strip()
        pins = json.loads(Path(self.env["LILITH_PINS_FILE"]).read_text())
        pins["llama_cpp"].update({"repo": str(repo), "commit": commit})
        Path(self.env["LILITH_PINS_FILE"]).write_text(json.dumps(pins))
        fake_server = self.dir / "fake-llama-server"
        fake_server.write_text("#!/bin/sh\necho options:\n" + "".join(f"echo '  {flag} X'\n" for flag in BuildServerTests.REQUIRED))
        fake_server.chmod(0o755)
        self.fake("cmake", f"""if [ "$1" = --build ]; then
  mkdir -p "$2/bin"
  cp "{fake_server}" "$2/bin/llama-server"
  cp "{fake_server}" "$2/bin/llama-bench"
else
  printf '%s\\n' "$@" > "{self.dir}/cmake-args"
fi""")
        (self.dir / "work/build-vulkan").mkdir(parents=True)
        (self.dir / "work/build-vulkan/stale-cache").write_text("old toolchain")
        result = self.run_script("build_server.sh", "--backend", "vulkan", "--install-deps")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("toolchain changed", result.stderr)
        self.assertFalse((self.dir / "work/build-vulkan/stale-cache").exists(), "clean build dir")
        self.assertIn("-DGGML_VULKAN=ON", (self.dir / "cmake-args").read_text())
        build = json.loads((self.dir / f"opt/llama.cpp/{commit[:12]}-vulkan/BUILD.json").read_text())
        self.assertEqual(build["backend"], "vulkan")
        self.assertEqual(build["commit"], commit)
        self.assertEqual(build["vulkan_sdk"], self.SDK_VERSION)
        self.assertEqual(build["vulkan_sdk_sha256"], pins["vulkan_sdk"]["sha256"])
        self.assertIsNone(build["rocm"])
        self.assertIsNone(build["system_vulkan_header_patch"])


class ConvertParentTests(FakeServerCase):
    """convert_parent.sh: the already-installed and reuse-from-the-share paths, and the pin refusal."""

    def setUp(self):
        super().setUp()
        self.pins = fake_pins(PARENT)
        self.pins_file = self.dir / "pins.json"
        self.pins_file.write_text(json.dumps(self.pins))
        self.env["LILITH_PINS_FILE"] = str(self.pins_file)
        self.parent = self.dir / "nfs/runs/parent"
        self.parent.mkdir(parents=True)
        for name, data in PARENT.items():
            (self.parent / name).write_bytes(data)
        self.name = f"lilith-parent-5926caa4-q8_0-llamacpp-{'a' * 12}"
        self.published = self.dir / "nfs/runs/models" / self.name

    def publish(self, **kw):
        out = self.published
        out.mkdir(parents=True, exist_ok=True)
        gguf = out / "lilith-parent-q8_0.gguf"
        gguf.write_bytes(b"GGUF" + b"\0" * 2048)
        (out / "chat_template.jinja").write_bytes(PARENT["chat_template.jinja"])
        tools.write_provenance(out / "PROVENANCE.json", pins=self.pins, gguf=gguf,
                               receipt=tools.verify_dir(self.parent, self.pins), llama_commit="a" * 40,
                               convert_command="convert", pip_freeze_sha256="f" * 64, template=out / "chat_template.jinja")
        return out

    def test_a_converted_model_on_the_share_is_reused_without_converting(self):
        self.publish()
        result = self.run_script("convert_parent.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        installed = self.dir / "state/models" / self.name
        self.assertTrue((installed / "lilith-parent-q8_0.gguf").is_file())
        self.assertEqual((self.dir / "state/models/current").resolve(), installed.resolve())
        self.assertNotIn("converting", result.stderr)
        self.assertFalse((self.dir / "work/convert-venv").exists(), "no venv, no llama.cpp checkout when reusing")
        self.assertEqual(tools.check_model(installed, self.pins)["output"]["quant"], "q8_0")
        # second run: already installed, nothing to do
        again = self.run_script("convert_parent.sh")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("already installed and verified", again.stderr)

    def test_a_corrupt_model_on_the_share_is_not_installed(self):
        out = self.publish()
        (out / "lilith-parent-q8_0.gguf").write_bytes(b"GGUF" + b"\1" * 2048)  # same size, other bytes
        # the corrupt copy fails verification; conversion is attempted next, which needs the
        # llama.cpp checkout (the fake repo URL does not exist), so the run must fail, and install nothing.
        result = self.run_script("convert_parent.sh", "--no-publish")
        self.assertEqual(result.returncode, 1)
        self.assertIn("failed verification", result.stderr)
        self.assertFalse((self.dir / "state/models/current").exists())

    def test_a_parent_that_differs_from_its_pin_is_never_converted(self):
        (self.parent / "config.json").write_bytes(b'{"a": 2}\n')  # same size as the pin, other bytes
        result = self.run_script("convert_parent.sh", "--no-publish")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::stage_mismatch]", result.stderr)
        self.assertFalse((self.dir / "work/convert-venv").exists())
        self.assertFalse((self.dir / "state/models/current").exists())
        self.assertFalse(list((self.dir / "work").glob("parent-*/config.json")), "the bad file must not be staged")

    def test_the_parent_directory_missing_on_the_share_says_where_to_look(self):
        shutil.rmtree(self.parent)
        result = self.run_script("convert_parent.sh", "--no-publish")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::parent_missing]", result.stderr)

    def test_the_disk_check_names_the_fix(self):
        result = self.run_script("convert_parent.sh", LILITH_MIN_FREE_GIB="99999999")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::disk]", result.stderr)
        self.assertIn("LILITH_WORK_DIR", result.stderr)


class InstallServiceTests(FakeServerCase):
    """install_service.sh without systemd: refusals and the files it lays down."""

    def setUp(self):
        super().setUp()
        self.env.update({"LILITH_NO_SYSTEMD": "1", "LILITH_UNIT_DIR": str(self.dir / "units"),
                         "LILITH_SERVE_USER": os.environ.get("USER", "root")})
        self.pins = tools.load_pins()

    def fake_build_and_model(self):
        (self.dir / "opt/current/bin").mkdir(parents=True)
        (self.dir / "opt/current/bin/llama-server").write_text("#!/bin/sh\n")
        (self.dir / "opt/current/bin/llama-server").chmod(0o755)

    def test_without_a_build_or_a_model_it_refuses_and_says_what_to_run(self):
        result = self.run_script("install_service.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::not_built]", result.stderr)
        self.fake_build_and_model()
        result = self.run_script("install_service.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::no_model]", result.stderr)
        self.assertIn("convert_parent.sh", result.stderr)

    def test_a_wide_bind_address_is_refused(self):
        self.fake_build_and_model()
        # The real pins cannot be met here; point the scripts at fake pins, a fake parent and a model.
        parent_files = PARENT
        pins = fake_pins(parent_files)
        pins_file = self.dir / "pins.json"
        pins_file.write_text(json.dumps(pins))
        parent = self.dir / "parent"
        parent.mkdir()
        for name, data in parent_files.items():
            (parent / name).write_bytes(data)
        model = self.dir / "state/models/m"
        model.mkdir(parents=True)
        (model / "lilith-parent-q8_0.gguf").write_bytes(b"GGUF")
        (model / "chat_template.jinja").write_bytes(parent_files["chat_template.jinja"])
        tools.write_provenance(model / "PROVENANCE.json", pins=pins, gguf=model / "lilith-parent-q8_0.gguf",
                               receipt=tools.verify_dir(parent, pins), llama_commit="a" * 40, convert_command="c",
                               pip_freeze_sha256="f" * 64, template=model / "chat_template.jinja")
        os.symlink(model, self.dir / "state/models/current")
        self.env["LILITH_PINS_FILE"] = str(pins_file)
        (self.dir / "etc").mkdir(parents=True)
        (self.dir / "etc/server.env").write_text("LILITH_LLAMA_HOST=0.0.0.0\nLILITH_LLAMA_PORT=8080\n")
        result = self.run_script("install_service.sh", "--no-start")
        self.assertEqual(result.returncode, 1)
        self.assertIn("error[lilith::heph::bind]", result.stderr)
        # a good address: everything is laid down, the key is generated 0600 and never printed
        (self.dir / "etc/server.env").write_text("LILITH_LLAMA_HOST=192.168.1.114\nLILITH_LLAMA_PORT=8080\n")
        result = self.run_script("install_service.sh", "--no-start")
        self.assertEqual(result.returncode, 0, result.stderr)
        key = self.dir / "etc/api-key"
        self.assertEqual(stat.S_IMODE(key.stat().st_mode), 0o600)
        self.assertGreaterEqual(len(key.read_text().strip()), 48)
        self.assertNotIn(key.read_text().strip(), result.stdout + result.stderr)
        before = key.read_text()
        self.assertEqual(self.run_script("install_service.sh", "--no-start").returncode, 0)
        self.assertEqual(key.read_text(), before, "the key is created once")
        self.assertEqual(self.run_script("install_service.sh", "--no-start", "--rotate-key").returncode, 0)
        self.assertNotEqual(key.read_text(), before)
        unit = (self.dir / "units/lilith-serve.service").read_text()
        self.assertIn(f"ExecStart={self.dir}/opt/bin/run_server.sh", unit)
        self.assertIn(f"EnvironmentFile={self.dir}/etc/server.env", unit)
        self.assertTrue(os.access(self.dir / "opt/bin/run_server.sh", os.X_OK))


class MockLlamaServer:
    """The subset of llama-server that verify.py uses, with switchable misbehaviour."""

    KEY = "verify-test-key-" + "q" * 32

    def __init__(self, mode: str = "good"):
        self.mode = mode
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def authorized(self):
                if outer.mode == "open":
                    return True
                return self.headers.get("Authorization") == f"Bearer {outer.KEY}"

            def send(self, status, body):
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/health":
                    return self.send(200, {"status": "ok"})
                if not self.authorized():
                    return self.send(401, {"error": {"message": "Invalid API Key", "code": 401}})
                if self.path == "/v1/models":
                    return self.send(200, {"data": [{"id": "lilith"}]})
                if self.path == "/props":
                    n_ctx = 4096 if outer.mode == "small_ctx" else 65536
                    return self.send(200, {"default_generation_settings": {"n_ctx": n_ctx}, "total_slots": 2,
                                           "build_info": "b-test"})
                self.send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if not self.authorized():
                    return self.send(401, {"error": {"message": "Invalid API Key", "code": 401}})
                if self.path == "/tokenize":
                    return self.send(200, {"tokens": [0] * (len(body.get("content", "")) // 4)})
                if self.path == "/v1/chat/completions":
                    return self.send(200, outer.completion(body))
                self.send(404, {"error": "not found"})

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def completion(self, body):
        messages = body.get("messages", [])
        has_tool_result = any(m.get("role") == "tool" for m in messages)
        prompt_tokens = len(json.dumps(messages)) // 4
        message: dict = {"role": "assistant", "content": ""}
        finish = "stop"
        if body.get("tools") and not has_tool_result and "get_time" in (messages[-1].get("content") or ""):
            if self.mode == "plaintext":
                message["content"] = "It is probably about noon."
            elif self.mode == "unparsed":
                message["content"] = "<|tool_call>call:get_time{timezone:<|\"|>UTC<|\"|>}<tool_call|>"
            elif self.mode == "badargs":
                message["tool_calls"] = [{"id": "c1", "type": "function",
                                          "function": {"name": "get_time", "arguments": "not json"}}]
                finish = "tool_calls"
            else:
                message["tool_calls"] = [{"id": "call_1", "type": "function",
                                          "function": {"name": "get_time", "arguments": "{\"timezone\": \"UTC\"}"}}]
                finish = "tool_calls"
        elif has_tool_result:
            message["content"] = "It is 08:15 UTC."
        else:
            thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
            message["content"] = "17 times 23 is 391."
            if self.mode == "leak":
                message["content"] = "<|channel>thought\nlet me think<channel|>17 times 23 is 391."
            if thinking and self.mode != "noreason":
                message["reasoning_content"] = "17*23 = 17*20 + 17*3 = 391"
            if body.get("max_tokens") == 128:  # the throughput probe
                message["content"] = " ".join(str(i) for i in range(1, 101))
        return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 100, "total_tokens": prompt_tokens + 100},
                "timings": {"prompt_n": prompt_tokens, "prompt_per_second": 812.5, "predicted_n": 100,
                            "predicted_per_second": 41.25}}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"


class VerifyTests(unittest.TestCase):
    def run_verify(self, mode="good", *extra, key=MockLlamaServer.KEY):
        with tempfile.TemporaryDirectory() as temp, MockLlamaServer(mode) as server:
            key_file = Path(temp) / "key"
            key_file.write_text(key + "\n")
            out = Path(temp) / "out.json"
            stderr = io.StringIO()
            real = sys.stderr
            sys.stderr = stderr
            try:
                code = verify.main(["--base-url", server.url, "--key-file", str(key_file), "--sizes", "600",
                                    "--json", str(out), *extra])
            finally:
                sys.stderr = real
            record = json.loads(out.read_text()) if out.exists() else None
            return code, record, stderr.getvalue()

    def statuses(self, record):
        return {c["name"]: c["status"] for c in record["checks"]}

    def test_a_correct_server_passes_every_check(self):
        code, record, log = self.run_verify()
        self.assertEqual(code, 0, log)
        statuses = self.statuses(record)
        self.assertEqual(statuses, {"health": "PASS", "auth": "PASS", "models": "PASS", "context": "PASS",
                                    "tool_call": "PASS", "tool_roundtrip": "PASS", "reasoning_leak": "PASS",
                                    "throughput_600": "PASS"})
        throughput = next(c for c in record["checks"] if c["name"] == "throughput_600")
        self.assertEqual(throughput["data"]["prefill_tps"], 812.5)
        self.assertEqual(throughput["data"]["decode_tps"], 41.25)
        self.assertIn("tok/s", throughput["detail"])
        self.assertNotIn(MockLlamaServer.KEY, json.dumps(record) + log)

    def test_the_prompt_is_calibrated_to_the_requested_size(self):
        with MockLlamaServer() as server:
            client = verify.Client(server.url, MockLlamaServer.KEY, "lilith")
            for target in (2048, 8192, 32768):
                _, tokens = verify.calibrate(client, target)
                self.assertLessEqual(tokens, target)
                self.assertGreaterEqual(tokens, 0.95 * target, target)

    def test_a_server_that_answers_without_a_key_fails(self):
        code, record, _ = self.run_verify("open")
        self.assertEqual(code, 1)
        self.assertEqual(self.statuses(record)["auth"], "FAIL")

    def test_plain_text_instead_of_a_tool_call_fails(self):
        code, record, _ = self.run_verify("plaintext")
        self.assertEqual(code, 1)
        statuses = self.statuses(record)
        self.assertEqual(statuses["tool_call"], "FAIL")
        self.assertEqual(statuses["tool_roundtrip"], "WARN")

    def test_native_syntax_left_in_content_fails_with_a_clear_reason(self):
        code, record, _ = self.run_verify("unparsed")
        self.assertEqual(code, 1)
        detail = next(c for c in record["checks"] if c["name"] == "tool_call")["detail"]
        self.assertIn("unparsed", detail)

    def test_malformed_arguments_fail(self):
        code, record, _ = self.run_verify("badargs")
        self.assertEqual(code, 1)
        self.assertIn("not a JSON object", next(c for c in record["checks"] if c["name"] == "tool_call")["detail"])

    def test_a_reasoning_channel_in_content_fails(self):
        code, record, _ = self.run_verify("leak")
        self.assertEqual(code, 1)
        leak = next(c for c in record["checks"] if c["name"] == "reasoning_leak")
        self.assertEqual(leak["status"], "FAIL")
        self.assertIn("<|channel>", leak["detail"])

    def test_no_reasoning_content_in_thinking_mode_is_a_warning_not_a_pass(self):
        code, record, _ = self.run_verify("noreason")
        self.assertEqual(code, 0)
        self.assertEqual(self.statuses(record)["reasoning_leak"], "WARN")

    def test_a_small_context_fails(self):
        code, record, _ = self.run_verify("small_ctx")
        self.assertEqual(code, 1)
        self.assertEqual(self.statuses(record)["context"], "FAIL")

    def test_throughput_floors_only_warn(self):
        code, record, _ = self.run_verify("good", "--min-decode-tps", "100")
        self.assertEqual(code, 0)
        self.assertEqual(self.statuses(record)["throughput_600"], "WARN")

    def test_an_unreachable_server_exits_2_and_a_missing_key_exits_2(self):
        with tempfile.TemporaryDirectory() as temp:
            key_file = Path(temp) / "key"
            key_file.write_text("k\n")
            real = sys.stderr
            sys.stderr = io.StringIO()
            try:
                self.assertEqual(verify.main(["--base-url", "http://127.0.0.1:9", "--key-file", str(key_file),
                                              "--sizes", ""]), 2)
                with unittest.mock.patch.dict(os.environ, clear=False):
                    os.environ.pop("LILITH_SERVE_API_KEY", None)
                    self.assertEqual(verify.main(["--base-url", "http://127.0.0.1:9", "--sizes", ""]), 2)
            finally:
                sys.stderr = real

    def test_the_key_comes_from_the_environment_or_a_file_never_argv(self):
        with MockLlamaServer() as server:
            env = dict(os.environ, LILITH_SERVE_API_KEY=MockLlamaServer.KEY)
            result = subprocess.run([sys.executable, str(HERE / "verify.py"), "--base-url", server.url, "--sizes", ""],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn(MockLlamaServer.KEY, result.stdout + result.stderr)
        self.assertNotIn('"--api-key"', (HERE / "verify.py").read_text(), "there is no key flag: files and env only")

    @unittest.skipIf(sys.version_info < (3, 11), "lilith_serve.py needs Python 3.11+")
    def test_the_tool_probe_agrees_with_lilith_serve_on_every_shape(self):
        import datetime as dt
        serve = load("lilith_serve_for_test", REPO / "scripts/lilith/serve/lilith_serve.py")
        self.assertEqual(verify.PROBE_TOOL, serve.PROBE_TOOL)
        request = serve.probe_request()
        request.pop("model")
        self.assertEqual(verify.probe_request(), request)
        good = {"choices": [{"message": {"tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "get_time", "arguments": "{}"}}]}, "finish_reason": "tool_calls"}]}
        plain = {"choices": [{"message": {"content": "noon"}, "finish_reason": "stop"}]}
        native = {"choices": [{"message": {"content": "<|tool_call>call:get_time{}<tool_call|>"}, "finish_reason": "stop"}]}
        bad = {"choices": [{"message": {"tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "get_time", "arguments": "[1]"}}]}, "finish_reason": "tool_calls"}]}
        now = dt.datetime.now(dt.timezone.utc)
        for response in (good, plain, native, bad):
            try:
                theirs = serve.check_tool_calls(200, response, now).status  # "verified" | "unverified"
            except serve.ToolCallingBroken:
                theirs = "broken"
            ours = verify.judge_tool_call(200, response).status
            expected = {"verified": verify.PASS, "unverified": verify.FAIL, "broken": verify.FAIL}[theirs]
            self.assertEqual(ours, expected, response)
        self.assertEqual(verify.judge_tool_call(500, {"error": "x"}).status, verify.FAIL)


if __name__ == "__main__":
    unittest.main()
