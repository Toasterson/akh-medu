"""Recorder lifecycle tests; no real GPU or model is used."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ResidentRecorderTests(unittest.TestCase):
    def run_probe(self, body, timeout=10, probe_args=()):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        binary = root / "probe"
        binary.write_text(f"#!{sys.executable}\nimport time\n{body}\n")
        binary.chmod(0o700)
        model = root / "model"
        model.write_bytes(b"fixture")
        gpu = root / "gpu"
        monitor = gpu / "hwmon" / "hwmon0"
        monitor.mkdir(parents=True)
        for path, value in [(gpu / "gpu_busy_percent", "0"),
                            (gpu / "mem_info_vram_used", "1234"),
                            (monitor / "power1_average", "10000000"),
                            (monitor / "fan1_input", "0")]:
            path.write_text(value)
        output = root / "output"
        command = [sys.executable, str(Path(__file__).with_name("resident-probe.py")),
                   "--device", "vulkan", "--model", str(model), "--tokenizer", str(model),
                   "--binary", str(binary), "--gpu", str(gpu), "--output", str(output),
                   "--timeout", str(timeout)]
        # `--probe-arg=VALUE` is required for values that start with a dash.
        command += [f"--probe-arg={extra}" for extra in probe_args]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return completed, output, command

    def test_records_identity_output_and_telemetry(self):
        completed, output, command = self.run_probe('print("fixture", flush=True)\ntime.sleep(0.1)')
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual((output / "stdout.jsonl").read_text().strip(), "fixture")
        self.assertEqual(json.loads((output / "result.json").read_text())["returncode"], 0)
        sample = json.loads((output / "telemetry.jsonl").read_text().splitlines()[0])
        self.assertEqual(sample["vram_bytes"], 1234)
        identity = json.loads((output / "identity.json").read_text())
        self.assertEqual(len(identity["binary_sha256"]), 64)
        # Never overwrite existing evidence, even on a repeated invocation.
        repeated = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(repeated.returncode, 0)
        self.assertEqual((output / "stdout.jsonl").read_text().strip(), "fixture")

    def test_extra_probe_arguments_reach_the_child_and_identity(self):
        completed, output, _ = self.run_probe(
            'import sys\nprint(" ".join(sys.argv[1:]), flush=True)',
            probe_args=["--keyed-benchmark", "--keyed-sessions", "2"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        forwarded = (output / "stdout.jsonl").read_text().strip().split()
        self.assertEqual(forwarded[-3:], ["--keyed-benchmark", "--keyed-sessions", "2"])
        identity = json.loads((output / "identity.json").read_text())
        self.assertEqual(identity["command"][-3:], ["--keyed-benchmark", "--keyed-sessions", "2"])

    def test_propagates_failed_probe(self):
        completed, output, _ = self.run_probe("raise SystemExit(7)")
        self.assertNotEqual(completed.returncode, 0)
        self.assertEqual(json.loads((output / "result.json").read_text())["returncode"], 7)

    def test_timeout_terminates_its_child(self):
        completed, output, _ = self.run_probe("time.sleep(60)", timeout=1)
        self.assertNotEqual(completed.returncode, 0)
        result = json.loads((output / "result.json").read_text())
        self.assertTrue(result["timed_out"])
        self.assertIsNotNone(result["returncode"])
        self.assertLess(result["seconds"], 10)


if __name__ == "__main__":
    unittest.main()
