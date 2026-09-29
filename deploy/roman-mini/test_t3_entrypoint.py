"""T3 entrypoint: role resolution and Lilith's on-demand serve-pod start (ADR 116, Lilith C2)."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

SPEC = importlib.util.spec_from_file_location("t3_entrypoint", Path(__file__).resolve().parent / "t3_entrypoint.py")
ep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ep)


class Response:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class Opener:
    """A urlopen stand-in: records requests, answers `status` or raises `error`."""

    def __init__(self, status=200, error=None):
        self.status, self.error, self.requests, self.timeouts = status, error, [], []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        self.timeouts.append(timeout)
        if self.error:
            raise self.error
        return Response(self.status)


class Popen:
    """A subprocess.Popen stand-in recording every spawn."""
    calls = []

    def __init__(self, argv, **kwargs):
        self.pid = 424242
        Popen.calls.append((argv, kwargs))


class RoleTests(unittest.TestCase):
    def test_romans_are_the_fallback(self):
        for role in ("janus", "minerva", "vesta"):
            self.assertEqual(ep.resolve_role(f"/x/bin/roman-{role}", {}), role)
        self.assertIsNone(ep.resolve_role("/x/bin/roman-lilith", {}))

    def test_rendered_akhs_extend_the_roles(self):
        settings = {"akhs": ["janus", "minerva", "vesta", "lilith"]}
        self.assertEqual(ep.resolve_role("/x/bin/roman-lilith", settings), "lilith")
        self.assertEqual(ep.resolve_role("roman-janus", settings), "janus")

    def test_unknown_roles_are_refused(self):
        settings = {"akhs": ["janus", "lilith"]}
        for name in ("roman-osiris", "roman-", "roman-minerva", "grok"):
            with self.subTest(name=name):
                self.assertIsNone(ep.resolve_role(f"/x/bin/{name}", settings))

    def test_malformed_akhs_fall_back_to_the_romans(self):
        for bad in ([], "lilith", [1], None):
            with self.subTest(bad=bad):
                self.assertEqual(ep.allowed_roles({"akhs": bad}), ep.ROMANS)

    def test_the_models_label_for_lilith(self):
        self.assertEqual(ep.MODEL_LABELS["lilith"], "Gemma 4 12B (Lilith) via vLLM on Runpod, in-process in akhomed")


class ScriptTests(unittest.TestCase):
    """The installed copy, run the way T3 runs it (basename = role)."""

    def run_as(self, name, akhs, *args):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for d in ("bin", "config", "secrets"):
                (root / d).mkdir()
            (root / "config/t3-acp.json").write_text(json.dumps({"akhs": akhs}))
            binary = root / "bin" / name
            binary.write_text((Path(__file__).resolve().parent / "t3_entrypoint.py").read_text())
            binary.chmod(0o700)
            return subprocess.run([sys.executable, str(binary), *args], capture_output=True, text=True)

    def test_lilith_runs_when_rendered(self):
        result = self.run_as("roman-lilith", ["janus", "lilith"], "--version")
        self.assertEqual((result.returncode, result.stdout.strip()), (0, "Roman Akh ACP bridge 0.1.0"))

    def test_lilith_is_refused_when_not_rendered(self):
        result = self.run_as("roman-lilith", ["janus", "minerva", "vesta"], "--version")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown Roman Akh", result.stderr)

    def test_unknown_role_is_refused(self):
        result = self.run_as("roman-osiris", ["janus", "lilith"], "--version")
        self.assertIn("unknown Roman Akh", result.stderr)


class ConfigureT3Tests(unittest.TestCase):
    def run_configure(self, with_lilith):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name).resolve() / "roman"
        for d in ("bin", "config", "secrets", "evidence"):
            (root / d).mkdir(parents=True)
        (root / "config/sessions.json").write_text(json.dumps({"principals": []}))
        if with_lilith:
            (root / "config/lilith.json").write_text("{}")
        settings = Path(temp.name) / "settings.json"
        settings.write_text(json.dumps({"providerInstances": {"codex": {"driver": "codex"}}}))
        subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "configure_t3.py"), str(root),
                        str(settings)], check=True, capture_output=True)
        return root, json.loads(settings.read_text())["providerInstances"], json.loads(
            (root / "config/sessions.json").read_text())["principals"]

    def test_lilith_provider_when_configured(self):
        root, providers, principals = self.run_configure(True)
        self.assertEqual(providers["roman-lilith"]["displayName"], "Lilith · Chief of Staff")
        self.assertEqual(providers["roman-lilith"]["config"]["binaryPath"], str(root / "bin/roman-lilith"))
        self.assertIn("codex", providers)
        self.assertEqual((root / "secrets/t3-lilith.key").stat().st_mode & 0o777, 0o600)
        # akhomed's host config carries her principal, not the akhsessd session config.
        self.assertNotIn("t3-lilith", [p["id"] for p in principals])
        self.assertEqual([p["id"] for p in principals], ["t3-janus", "t3-minerva", "t3-vesta"])

    def test_romans_only_without_her_config(self):
        root, providers, _ = self.run_configure(False)
        self.assertEqual(sorted(k for k in providers if k.startswith("roman-")),
                         ["roman-janus", "roman-minerva", "roman-vesta"])
        self.assertFalse((root / "bin/roman-lilith").exists())


class LilithServeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for d in ("secrets", "logs"):
            (self.root / d).mkdir()
        (self.root / "secrets/lilith-base-url").write_text("https://pod-8000.proxy.runpod.net/v1\n")
        self.checkout = self.root / "checkout"
        script = self.checkout / "scripts/lilith/serve/lilith_serve.py"
        script.parent.mkdir(parents=True)
        script.write_text("")
        self.settings = {"akhs": ["janus", "lilith"], "lilith_serve": str(self.checkout)}
        Popen.calls = []

    def tearDown(self):
        self.temp.cleanup()

    def ensure(self, opener, settings=None):
        return ep.ensure_lilith_serving(self.root, self.settings if settings is None else settings,
                                        opener=opener, popen=Popen)

    def log(self):
        path = self.root / "logs/lilith-serve-start.log"
        return path.read_text() if path.exists() else ""

    def test_health_check_shape(self):
        opener = Opener(200)
        self.assertTrue(ep.lilith_healthy(self.root, opener))
        request = opener.requests[0]
        self.assertEqual(request.full_url, "https://pod-8000.proxy.runpod.net/health")
        self.assertEqual(request.get_header("User-agent"), "curl/8.7.1")
        self.assertLessEqual(opener.timeouts[0], 3)

    def test_down_endpoint_spawns_once_detached(self):
        for down in (Opener(502), Opener(error=OSError("timed out")), Opener(403)):
            with self.subTest(down=down.status):
                Popen.calls = []
                lock = self.root / "state/lilith/t3-serve-start.pid"
                if lock.exists():
                    lock.unlink()
                self.assertEqual(self.ensure(down), "spawned")
                (argv, kwargs), = Popen.calls
                self.assertEqual(argv, ["/opt/homebrew/bin/python3",
                                        str(self.checkout / "scripts/lilith/serve/lilith_serve.py"), "start"])
                self.assertTrue(kwargs["start_new_session"])
                self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
                self.assertEqual(kwargs["stderr"], subprocess.STDOUT)
                self.assertEqual(Path(kwargs["stdout"].name), self.root / "logs/lilith-serve-start.log")
                self.assertEqual(lock.read_text(), "424242")

    def test_healthy_endpoint_does_not_spawn(self):
        self.assertEqual(self.ensure(Opener(200)), "up")
        self.assertEqual(Popen.calls, [])
        self.assertFalse((self.root / "state/lilith/t3-serve-start.pid").exists())

    def test_lock_prevents_a_double_spawn(self):
        self.assertEqual(self.ensure(Opener(502)), "spawned")
        # A second T3 session while the first start is loading the model.
        lock = self.root / "state/lilith/t3-serve-start.pid"
        lock.write_text(str(os.getpid()))
        old = time.time() - 600
        os.utime(lock, (old, old))
        self.assertEqual(self.ensure(Opener(502)), "locked")
        self.assertEqual(len(Popen.calls), 1)

    def test_recent_lock_blocks_a_retry_even_if_the_start_died(self):
        self.assertEqual(self.ensure(Opener(502)), "spawned")
        (self.root / "state/lilith/t3-serve-start.pid").write_text("2147483646")
        self.assertEqual(self.ensure(Opener(502)), "locked")
        self.assertEqual(len(Popen.calls), 1)

    def test_stale_lock_is_replaced(self):
        self.assertEqual(self.ensure(Opener(502)), "spawned")
        lock = self.root / "state/lilith/t3-serve-start.pid"
        lock.write_text("2147483646")  # not a live pid
        old = time.time() - 3600
        os.utime(lock, (old, old))
        self.assertEqual(self.ensure(Opener(502)), "spawned")
        self.assertEqual(len(Popen.calls), 2)

    def test_expired_lock_with_a_live_pid_is_replaced(self):
        lock = self.root / "state/lilith"
        lock.mkdir(parents=True)
        lock = lock / "t3-serve-start.pid"
        lock.write_text(str(os.getpid()))
        old = time.time() - 3 * 3600
        os.utime(lock, (old, old))
        self.assertEqual(self.ensure(Opener(502)), "spawned")

    def test_missing_checkout_is_logged_not_fatal(self):
        for settings in ({"akhs": ["lilith"]}, {"lilith_serve": str(self.root / "nowhere")}):
            with self.subTest(settings=settings):
                self.assertEqual(self.ensure(Opener(502), settings), "skipped")
        self.assertEqual(Popen.calls, [])
        self.assertIn("no lilith_serve.py to start", self.log())

    def test_a_spawn_failure_is_logged_not_fatal(self):
        def broken(argv, **kwargs):
            raise FileNotFoundError("/opt/homebrew/bin/python3")
        self.assertEqual(ep.ensure_lilith_serving(self.root, self.settings, opener=Opener(502), popen=broken),
                         "skipped")
        self.assertIn("could not start the serve pod: FileNotFoundError", self.log())

    def test_missing_base_url_file_counts_as_down(self):
        (self.root / "secrets/lilith-base-url").unlink()
        self.assertEqual(self.ensure(Opener(200)), "spawned")

    def test_a_budget_refusal_ends_up_in_the_log(self):
        # The child writes its stderr into the same log file it was handed.
        real = subprocess.Popen

        def fake_serve(argv, **kwargs):
            proc = real(["/bin/sh", "-c", "echo 'lilith::serve::budget weekly limit reached' >&2; exit 3"],
                        **{k: v for k, v in kwargs.items() if k != "env"})
            proc.wait()
            return proc
        self.assertEqual(ep.ensure_lilith_serving(self.root, self.settings, opener=Opener(502), popen=fake_serve),
                         "spawned")
        self.assertIn("lilith::serve::budget weekly limit reached", self.log())


if __name__ == "__main__":
    unittest.main()
