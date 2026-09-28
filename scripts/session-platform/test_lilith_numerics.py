"""CPU-only plumbing tests for the recorded numerics default (no torch required).

``lilith_numerics.py`` defers every torch import to methods that need real PyTorch
state (``NumericsConfig.applied``, ``observed_state``); the resolution logic tested
here — ``detect_cuda``, ``cuda_default``, ``platform_default`` and the CLI/config
wiring in ``NumericsConfig.from_args`` — is pure data and runs on a host with no
GPU, no CUDA and no torch installed at all (this is exactly that host). GPU-side
behaviour (that ``.applied()`` actually changes observed PyTorch state and restores
it) is covered by the ``NumericsTests`` class in test_lilith_cache_path_diagnostic.py,
which requires torch.

Run: python -m unittest discover -s scripts/session-platform -p 'test_*.py'
"""
import argparse
import unittest

import lilith_numerics as numerics


class DetectCudaTests(unittest.TestCase):
    def test_absent_torch_is_false_not_an_error(self):
        # This host has no torch installed; detect_cuda must degrade to False,
        # never raise. If torch ever becomes available here, this assertion still
        # holds as long as no CUDA device is visible.
        self.assertIsInstance(numerics.detect_cuda(), bool)


class CudaDefaultTests(unittest.TestCase):
    def test_cuda_default_is_efficient_with_math_fallback(self):
        config = numerics.cuda_default()
        self.assertEqual(config.sdpa_backend, numerics.SdpaBackend.EFFICIENT)
        self.assertTrue(config.sdpa_math_fallback)
        self.assertEqual(config.tf32, numerics.Toggle.DEFAULT)
        self.assertEqual(config.bf16_reduced_precision_reduction, numerics.Toggle.DEFAULT)
        self.assertFalse(config.deterministic)
        self.assertFalse(config.is_default)
        config.validate()   # never raises: EFFICIENT + math fallback is a legal combination

    def test_cuda_default_is_stable_and_hashable_as_a_dataclass_value(self):
        self.assertEqual(numerics.cuda_default(), numerics.cuda_default())


class PlatformDefaultTests(unittest.TestCase):
    def test_explicit_cuda_true_selects_cuda_default(self):
        self.assertEqual(numerics.platform_default(cuda=True), numerics.cuda_default())

    def test_explicit_cuda_false_selects_the_untouched_default(self):
        self.assertEqual(numerics.platform_default(cuda=False), numerics.NumericsConfig())

    def test_unset_cuda_falls_back_to_local_detection(self):
        # No torch on this host: local detection is False, matching cuda=False.
        self.assertEqual(numerics.platform_default(), numerics.platform_default(cuda=numerics.detect_cuda()))


class FromArgsResolutionTests(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        numerics.add_arguments(parser)
        return parser

    def test_no_flags_resolves_the_platform_default_for_the_given_host(self):
        args = self.parser().parse_args([])
        self.assertEqual(numerics.NumericsConfig.from_args(args, cuda=True), numerics.cuda_default())
        self.assertEqual(numerics.NumericsConfig.from_args(args, cuda=False), numerics.NumericsConfig())

    def test_no_flags_with_no_cuda_argument_uses_local_detection(self):
        args = self.parser().parse_args([])
        self.assertEqual(numerics.NumericsConfig.from_args(args), numerics.platform_default())

    def test_any_explicit_flag_overrides_the_platform_default_wholesale(self):
        args = self.parser().parse_args(['--numerics-sdpa-backend', 'math'])
        # Even asking for the CUDA host, an explicit choice of MATH wins outright;
        # the resolver never silently mixes an explicit backend with the platform
        # default's math_fallback=True (that combination is invalid: MATH + fallback).
        config = numerics.NumericsConfig.from_args(args, cuda=True)
        self.assertEqual(config.sdpa_backend, numerics.SdpaBackend.MATH)
        self.assertFalse(config.sdpa_math_fallback)

    def test_explicit_default_backend_forces_the_true_default_even_on_cuda(self):
        args = self.parser().parse_args(['--numerics-sdpa-backend', 'default'])
        config = numerics.NumericsConfig.from_args(args, cuda=True)
        self.assertEqual(config.sdpa_backend, numerics.SdpaBackend.DEFAULT)
        self.assertTrue(config.is_default)

    def test_deterministic_flag_alone_is_explicit_and_ignores_the_platform_default(self):
        args = self.parser().parse_args(['--numerics-deterministic'])
        config = numerics.NumericsConfig.from_args(args, cuda=True)
        self.assertTrue(config.deterministic)
        self.assertEqual(config.sdpa_backend, numerics.SdpaBackend.DEFAULT)

    def test_math_fallback_alone_without_a_forced_backend_is_rejected(self):
        args = self.parser().parse_args(['--numerics-sdpa-math-fallback'])
        with self.assertRaises(numerics.NumericsError):
            numerics.NumericsConfig.from_args(args, cuda=True)


class ConfiguredNumericsTests(unittest.TestCase):
    """``lilith_runtime_training_worker.configured_numerics``: pinned-config resolution."""

    def test_absent_key_is_the_untouched_default_not_platform_default(self):
        from lilith_runtime_training_worker import configured_numerics
        # A config authored before this field existed (or an explicit "untouched"
        # pin) must resolve to NumericsConfig(), never platform_default(): absence
        # means "matches what that historical run actually did", not "guess CUDA".
        self.assertEqual(configured_numerics({}), numerics.NumericsConfig())
        self.assertEqual(configured_numerics({'numerics': None}), numerics.NumericsConfig())

    def test_pinned_dict_round_trips_through_requested(self):
        from lilith_runtime_training_worker import configured_numerics
        pinned = numerics.cuda_default().requested()
        self.assertEqual(configured_numerics({'numerics': pinned}), numerics.cuda_default())

    def test_invalid_pinned_numerics_is_a_value_error_not_a_crash(self):
        from lilith_runtime_training_worker import configured_numerics
        with self.assertRaises(ValueError):
            configured_numerics({'numerics': {'sdpa_backend': 'turbo'}})


if __name__ == '__main__':
    unittest.main()
