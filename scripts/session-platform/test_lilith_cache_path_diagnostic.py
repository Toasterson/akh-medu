"""CPU tests for the cache-path diagnostic and the recorded numerics configuration."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).parent))
import lilith_cache_path_diagnostic as diagnostic  # noqa: E402
import lilith_numerics as numerics  # noqa: E402
import lilith_training_cache_control as control  # noqa: E402
from test_lilith_training_cache import tiny_engine  # noqa: E402


def _comparison(max_abs):
    return {"max_abs": max_abs, "within_tolerance": max_abs <= diagnostic.TOLERANCE,
            "logit_max_abs": {"7": max_abs}, "loss_abs": 0.0, "gradient_max_abs": {}}


def _entry(max_abs, repeat=None, configuration_only=True):
    entry = {"resident_vs_offload": _comparison(max_abs),
             "resident_vs_resident_repeat": None if repeat is None else _comparison(repeat),
             "configuration_only": configuration_only}
    entry["restores_equivalence"] = diagnostic.restores(entry)
    return entry


def _mode(base, resident_repeat, offload_repeat=None, **variations):
    return {"baseline": {"resident_vs_offload": _comparison(base),
                         "resident_vs_resident_repeat": _comparison(resident_repeat),
                         "offload_vs_offload_repeat": (None if offload_repeat is None
                                                       else _comparison(offload_repeat)),
                         "layout_differences": []},
            "variations": variations}


class NumericsTests(unittest.TestCase):
    def test_default_is_the_untouched_qualified_behaviour(self):
        config = numerics.NumericsConfig()
        self.assertTrue(config.is_default)
        self.assertEqual(config.label(), "default")
        before = numerics.observed_state()
        with config.applied():
            self.assertEqual(numerics.observed_state(), before)
        manifest = config.manifest()
        self.assertEqual(manifest["schema"], numerics.SCHEMA)
        self.assertEqual(manifest["requested"]["sdpa_backend"], "default")
        self.assertIn("matmul_allow_tf32", manifest["observed"])

    def test_cli_defaults_select_the_default_config(self):
        parser = argparse.ArgumentParser()
        numerics.add_arguments(parser)
        self.assertEqual(numerics.NumericsConfig.from_args(parser.parse_args([])), numerics.NumericsConfig())
        args = parser.parse_args(["--numerics-sdpa-backend", "efficient", "--numerics-sdpa-math-fallback",
                                  "--numerics-tf32", "off", "--numerics-deterministic",
                                  "--numerics-bf16-reduced-precision-reduction", "off"])
        config = numerics.NumericsConfig.from_args(args)
        self.assertEqual(config.sdpa_backend, numerics.SdpaBackend.EFFICIENT)
        self.assertTrue(config.sdpa_math_fallback and config.deterministic)
        self.assertEqual(config.tf32, numerics.Toggle.OFF)
        self.assertEqual(diagnostic.numerics_flags(config), [
            "--numerics-sdpa-backend", "efficient", "--numerics-sdpa-math-fallback", "--numerics-tf32", "off",
            "--numerics-bf16-reduced-precision-reduction", "off", "--numerics-deterministic"])

    def test_knobs_apply_inside_the_block_and_restore_after(self):
        config = numerics.NumericsConfig(tf32=numerics.Toggle.OFF,
                                         bf16_reduced_precision_reduction=numerics.Toggle.OFF,
                                         deterministic=True, sdpa_backend=numerics.SdpaBackend.MATH)
        before = numerics.observed_state()
        with config.applied():
            state = numerics.observed_state()
            self.assertFalse(state["matmul_allow_tf32"])
            self.assertFalse(state["cudnn_allow_tf32"])
            self.assertFalse(state["matmul_allow_bf16_reduced_precision_reduction"])
            self.assertTrue(state["deterministic_algorithms"])
            self.assertTrue(state["cudnn_deterministic"])
            self.assertTrue(state["sdpa_enabled"]["math"])
            self.assertFalse(state["sdpa_enabled"]["flash"])
            self.assertIn("deterministic", config.label())
        self.assertEqual(numerics.observed_state(), before)

    def test_knobs_restore_on_error(self):
        before = numerics.observed_state()
        with self.assertRaises(KeyError):
            with numerics.NumericsConfig(deterministic=True, tf32=numerics.Toggle.OFF).applied():
                raise KeyError("boom")
        self.assertEqual(numerics.observed_state(), before)

    def test_invalid_configurations_are_rejected_with_a_reason(self):
        with self.assertRaises(numerics.NumericsError):
            numerics.NumericsConfig.from_mapping({"sdpa_backend": "turbo"})
        with self.assertRaises(numerics.NumericsError):
            numerics.NumericsConfig.from_mapping({"sdpa_math_fallback": True})
        with self.assertRaises(numerics.NumericsError):
            numerics.NumericsConfig.from_mapping({"deterministic_warn_only": True})

    def test_deterministic_requires_the_cublas_workspace(self):
        with mock.patch.dict("os.environ", {"CUBLAS_WORKSPACE_CONFIG": ""}):
            with self.assertRaises(numerics.NumericsError):
                with numerics.NumericsConfig(deterministic=True).applied():
                    pass
        self.assertFalse(torch.are_deterministic_algorithms_enabled())

    def test_cache_control_applies_and_records_numerics(self):
        seen = {}

        def fake(engine, output, positions, manifest):
            seen["manifest"] = manifest
            seen["tf32_inside"] = torch.backends.cuda.matmul.allow_tf32
            return {"status": "passed", "numerics": manifest}

        before = torch.backends.cuda.matmul.allow_tf32
        with mock.patch.object(control, "_run_control", side_effect=fake):
            control.run_control(object(), "unused", (7,), numerics.NumericsConfig(tf32=numerics.Toggle.OFF))
            self.assertFalse(seen["tf32_inside"])
            self.assertEqual(seen["manifest"]["requested"]["tf32"], "off")
            self.assertFalse(seen["manifest"]["observed"]["matmul_allow_tf32"])
            control.run_control(object(), "unused", (7,))
            self.assertTrue(seen["manifest"]["is_default"])
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, before)


class VerdictTests(unittest.TestCase):
    variations = diagnostic.VARIATIONS

    def test_repeat_divergence_is_run_to_run_nondeterminism(self):
        modes = {"chunked": _mode(0.5, 0.4, deterministic=_entry(0.0, 0.0)),
                 "tokenwise": _mode(3.8, 2.0, deterministic=_entry(0.0))}
        result = diagnostic.verdict(modes, self.variations)
        self.assertEqual(result["cause"], "run_to_run_nondeterminism")
        self.assertEqual(result["restoring_variations_confirmed_tokenwise"], ["deterministic"])
        self.assertEqual(result["recommended_numerics_flags"], ["--numerics-deterministic"])
        self.assertFalse(result["fix_requires_engine_change"])

    def test_repeatable_paths_that_differ_are_offload_specific(self):
        modes = {"tokenwise": _mode(0.9, 0.0, tf32_off=_entry(0.3), sdpa_math=_entry(0.0))}
        result = diagnostic.verdict(modes, self.variations)
        self.assertEqual(result["cause"], "offload_path_specific")
        self.assertEqual(result["recommended_numerics"]["sdpa_backend"], "math")

    def test_contiguity_only_needs_an_engine_change(self):
        modes = {"tokenwise": _mode(0.9, 0.0, contiguous=_entry(0.0, configuration_only=False))}
        result = diagnostic.verdict(modes, self.variations)
        self.assertIsNone(result["recommended_numerics"])
        self.assertTrue(result["fix_requires_engine_change"])
        self.assertIn("engine change", result["next_step"])

    def test_chunked_only_restoration_is_not_a_recommendation(self):
        modes = {"chunked": _mode(0.5, 0.0, tf32_off=_entry(0.0, 0.0)), "tokenwise": _mode(0.9, 0.0)}
        result = diagnostic.verdict(modes, self.variations)
        self.assertEqual(result["restoring_variations_chunked_only"], ["tf32_off"])
        self.assertIsNone(result["recommended_numerics"])
        self.assertIn("--budget-seconds", result["next_step"])

    def test_unrepeatable_restoration_does_not_count(self):
        self.assertFalse(diagnostic.restores(_entry(0.0, repeat=0.5)))
        self.assertTrue(diagnostic.restores(_entry(0.0, repeat=0.0)))


class DiagnosticRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def run_diagnostic(self, **kwargs):
        engine = tiny_engine()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        output = Path(directory.name) / "diagnostic.json"
        report = diagnostic.run_diagnostic(engine, output, positions=(7, 8, 9), tail_tokens=2, **kwargs)
        self.assertEqual(json.loads(output.read_text())["status"], report["status"])
        return engine, report

    def test_full_sweep_on_cpu_matches_and_records_every_variation(self):
        engine, report = self.run_diagnostic(budget_seconds=600)
        self.assertEqual(report["status"], "finished")
        self.assertTrue(report["initial_policy_identity_restored"])
        self.assertEqual(report["chunked_prefix_tokens"], 4)
        chunked = report["modes_summary"]["chunked"]
        for variation in diagnostic.VARIATIONS:
            name = variation.name
            if name in chunked["variations"] and chunked["variations"][name]["status"]["resident"] == "ok":
                continue
            # A variation that cannot run (no kernel for a forced SDPA backend) has a fallback.
            fallback = diagnostic.fallback_for(variation)
            self.assertIsNotNone(fallback, name)
            self.assertIn(fallback.name, chunked["variations"], name)
        baseline = chunked["baseline"]
        self.assertTrue(baseline["resident_vs_offload"]["within_tolerance"])
        self.assertTrue(baseline["resident_vs_resident_repeat"]["logits_bitwise_equal"])
        self.assertTrue(any("device" in json.dumps(layer) for layer in [report["passes"]["chunked/baseline/offload"]]))
        self.assertIn("tokenwise/baseline/resident#2", report["passes"])
        self.assertEqual(report["verdict"]["cause"], "not_reproduced")
        self.assertIsNone(report["verdict"]["recommended_numerics"])   # nothing to fix
        self.assertEqual(report["passes"]["chunked/contiguous/resident"]["cache_kind"], "resident_contiguous")
        offload_layout = report["passes"]["chunked/baseline/offload"]["layout"]
        self.assertTrue(all(layer["keys"]["device"] == "cpu" for layer in offload_layout["layers"]))
        self.assertFalse(torch.are_deterministic_algorithms_enabled())
        self.assertEqual(engine.cache_offloading, False)

    def test_sliding_view_layout_difference_is_reported(self):
        contiguous = next(v for v in diagnostic.VARIATIONS if v.name == "contiguous")
        _, report = self.run_diagnostic(modes=(diagnostic.PrefixMode.TOKENWISE,), variations=(contiguous,),
                                        budget_seconds=600)
        passes = report["passes"]
        # Past the sliding window the resident layer stores a strided view into the
        # concatenated K/V (nonzero storage offset); the contiguous variant stores a dense copy.
        # (On CPU the offload copy is a same-device no-op, so it keeps the view.)
        differences = diagnostic.layout_differences(passes["tokenwise/baseline/resident"]["layout"],
                                                    passes["tokenwise/contiguous/resident"]["layout"])
        sliding = [item for item in differences if item["is_sliding"]]
        self.assertTrue(sliding)
        self.assertTrue(all("storage_offset" in item["fields"] for item in sliding))
        self.assertFalse([item for item in differences if not item["is_sliding"]])
        entry = report["modes_summary"]["tokenwise"]["variations"]["contiguous"]
        self.assertTrue(entry["resident_vs_offload"]["logits_bitwise_equal"])
        self.assertTrue(entry["resident_vs_baseline_resident"]["within_tolerance"])

    def test_zero_budget_records_skips_and_an_inconclusive_verdict(self):
        _, report = self.run_diagnostic(budget_seconds=0)
        self.assertTrue(report["passes"])
        self.assertEqual({entry["status"] for entry in report["passes"].values()}, {"skipped_budget"})
        self.assertEqual(report["verdict"]["cause"], "inconclusive")

    def test_invalid_positions_are_refused_before_writing(self):
        engine = tiny_engine()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "d.json"
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic.run_diagnostic(engine, output, positions=(9, 7))
            with self.assertRaises(diagnostic.DiagnosticError):
                diagnostic.run_diagnostic(engine, output, positions=(64,))
            self.assertFalse(output.exists())

    def test_failed_forced_backend_schedules_its_math_fallback(self):
        planner = diagnostic.Planner([diagnostic.PrefixMode.CHUNKED], diagnostic.VARIATIONS, 1e9)
        flash = next(v for v in diagnostic.VARIATIONS if v.name == "sdpa_flash")
        failed = diagnostic.PassResult(case=diagnostic.Case(diagnostic.PrefixMode.CHUNKED, flash, "resident"),
                                       status="error", error="RuntimeError: No available kernel")
        planner.record(failed)
        self.assertEqual(planner.queue[0].variation.name, "sdpa_flash_math_fallback")
        self.assertTrue(planner.queue[0].variation.numerics.sdpa_math_fallback)
        self.assertFalse(any(case.variation.name == "sdpa_flash" for case in planner.queue))


if __name__ == "__main__":
    unittest.main()
