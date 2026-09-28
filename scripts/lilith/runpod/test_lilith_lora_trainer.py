"""CPU tests for the C6 full-depth LoRA trainer on a tiny random Gemma-4-like model.

Run: python -m unittest discover -s scripts/lilith/runpod -p 'test_*.py'
Requires torch, transformers (>= 5.10 with Gemma4Unified) and peft; skipped otherwise.
"""
import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import torch
    import transformers  # noqa: F401
    import peft  # noqa: F401
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    HAVE_STACK = True
except ImportError:  # pragma: no cover - environment without the training stack
    HAVE_STACK = False

import lilith_lora_trainer as trainer  # noqa: E402
import reference_microrun as microrun  # noqa: E402

TINY_TEXT = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                 num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                 global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
                 sliding_window=8, layer_types=['sliding_attention'] * 3 + ['full_attention'],
                 final_logit_softcapping=30.0, max_position_embeddings=4096)


def tiny_config():
    return Gemma4UnifiedConfig(text_config=dict(TINY_TEXT),
                               vision_config=dict(mm_embed_dim=32, output_proj_dims=64),
                               audio_config=dict(audio_embed_dim=32))


def tiny_base(seed=0, implementation=None):
    torch.manual_seed(seed)
    config = tiny_config()
    config._attn_implementation = 'eager'
    model = Gemma4UnifiedForConditionalGeneration(config).float()
    if implementation is not None:
        model.set_attn_implementation(implementation)
    return model


def fp32_config(**overrides):
    values = dict(rank=16, base_precision='fp32', ce_chunk_tokens=5, attention_query_chunk=3, seed=11)
    values.update(overrides)
    return trainer.TrainerConfig(**values)


def perturb_lora_b(model, seed=5):
    """PEFT initializes lora_B to zero; make both factors live so every gradient is informative."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if '.lora_B.' in name:
                parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.05)


def lora_model(seed=0, **overrides):
    config = fp32_config(**overrides)
    trainer.register_chunked_attention(config.attention_query_chunk)
    base = tiny_base(seed, trainer.ATTENTION_IMPLEMENTATION)
    model, surface = trainer.prepare_lora_model(base, config)
    perturb_lora_b(model)
    return model, config, surface


def lora_grads(model):
    return {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters()
            if parameter.requires_grad}


def random_example(seed, length=21, weight=1.0, prompt=7):
    generator = torch.Generator().manual_seed(seed)
    ids = torch.randint(3, 500, (length,), generator=generator).tolist()
    return trainer.build_sft_example(ids[:prompt], ids[prompt:], weight=weight, source='test')


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class ChunkedCrossEntropyTests(unittest.TestCase):
    def check(self, softcap):
        torch.manual_seed(1)
        hidden = torch.randn(37, 16, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(300, 16, dtype=torch.float64)
        targets = torch.randint(0, 300, (37,))
        upstream = torch.rand(37, dtype=torch.float64)
        chunked = trainer.chunked_token_nll(hidden, weight, targets, softcap=softcap, chunk=8)
        (grad_chunked,) = torch.autograd.grad((chunked * upstream).sum(), hidden)
        full = trainer.full_token_nll(hidden, weight, targets, softcap=softcap)
        (grad_full,) = torch.autograd.grad((full * upstream).sum(), hidden)
        self.assertEqual(chunked.shape, (37,))
        torch.testing.assert_close(chunked, full.float(), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(grad_chunked, grad_full, rtol=1e-6, atol=1e-6)

    def test_chunked_equals_full_with_softcap(self):
        self.check(30.0)

    def test_chunked_equals_full_without_softcap(self):
        self.check(None)

    def test_matches_torch_cross_entropy(self):
        torch.manual_seed(2)
        hidden = torch.randn(19, 8)
        weight = torch.randn(64, 8)
        targets = torch.randint(0, 64, (19,))
        chunked = trainer.chunked_token_nll(hidden, weight, targets, softcap=None, chunk=4)
        reference = torch.nn.functional.cross_entropy(hidden @ weight.t(), targets, reduction='none')
        torch.testing.assert_close(chunked, reference, rtol=1e-5, atol=1e-5)


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class ChunkedAttentionTests(unittest.TestCase):
    def compare(self, heads, kv_heads, q_len, k_len, dim, window, chunk):
        torch.manual_seed(q_len * 7 + k_len)
        query = torch.randn(2, heads, q_len, dim, requires_grad=True)
        key = torch.randn(2, kv_heads, k_len, dim, requires_grad=True)
        value = torch.randn(2, kv_heads, k_len, dim, requires_grad=True)
        upstream = torch.randn(2, heads, q_len, dim)
        scaling = dim ** -0.5
        out = trainer.chunked_attention(query, key, value, scaling=scaling, window=window, chunk=chunk)
        grads = torch.autograd.grad((out * upstream).sum(), (query, key, value))
        expected = trainer.reference_attention(query, key, value, scaling=scaling, window=window)
        expected_grads = torch.autograd.grad((expected * upstream).sum(), (query, key, value))
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
        for observed, reference in zip(grads, expected_grads):
            torch.testing.assert_close(observed, reference, rtol=1e-4, atol=1e-5)

    def test_global_layer_shape_gqa_single_kv_head(self):
        # Global layers: large head_dim, one KV head shared by every query head.
        self.compare(heads=4, kv_heads=1, q_len=23, k_len=23, dim=32, window=None, chunk=5)

    def test_sliding_window_layer(self):
        self.compare(heads=4, kv_heads=2, q_len=29, k_len=29, dim=16, window=6, chunk=4)

    def test_cached_prefix_offset(self):
        # Queries after a cached prefix see all earlier keys (causal offset Tk - Tq).
        self.compare(heads=4, kv_heads=1, q_len=7, k_len=23, dim=32, window=None, chunk=3)
        self.compare(heads=4, kv_heads=2, q_len=5, k_len=12, dim=16, window=6, chunk=2)

    def test_model_level_equivalence_with_eager_forward_and_lora_grads(self):
        config = fp32_config()
        trainer.register_chunked_attention(config.attention_query_chunk)
        eager = tiny_base(3)
        chunked = copy.deepcopy(eager)
        chunked.set_attn_implementation(trainer.ATTENTION_IMPLEMENTATION)
        eager_model, _ = trainer.prepare_lora_model(eager, config)
        chunked_model, _ = trainer.prepare_lora_model(chunked, config)
        perturb_lora_b(eager_model)
        perturb_lora_b(chunked_model)
        ids = torch.randint(3, 500, (1, 21), generator=torch.Generator().manual_seed(4))
        eager_model.eval(); chunked_model.eval()
        with torch.no_grad():
            torch.testing.assert_close(chunked_model(input_ids=ids).logits, eager_model(input_ids=ids).logits,
                                       rtol=1e-5, atol=1e-5)
        example = random_example(4)
        for model in (eager_model, chunked_model):
            model.train()
            loss, _ = trainer.weighted_sft_loss(model, [example], config)
            loss.backward()
        eager_grads, chunked_grads = lora_grads(eager_model), lora_grads(chunked_model)
        self.assertEqual(eager_grads.keys(), chunked_grads.keys())
        for name in eager_grads:
            torch.testing.assert_close(chunked_grads[name], eager_grads[name], rtol=1e-3, atol=1e-5)

    def test_explicit_mask_is_refused(self):
        module = type('M', (), {'head_dim': 16, 'is_sliding': False})()
        tensor = torch.zeros(1, 2, 3, 16)
        with self.assertRaises(trainer.LilithTrainerError):
            trainer._hf_chunked_attention(module, tensor, tensor, tensor, torch.ones(1, 1, 3, 3))


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class LoraSurfaceTests(unittest.TestCase):
    def test_targets_are_exactly_language_model_attention_and_mlp(self):
        base = tiny_base(0)
        targets = trainer.lora_target_modules(base)
        # 3 sliding layers x (q, k, v, o, gate, up, down) + 1 global (K = V: no v_proj) x 6
        self.assertEqual(len(targets), 3 * 7 + 6)
        self.assertTrue(all(name.startswith('model.language_model.layers.') for name in targets))
        self.assertNotIn('model.language_model.layers.3.self_attn.v_proj', targets)
        linear_names = [name for name, module in base.named_modules() if isinstance(module, torch.nn.Linear)]
        excluded = sorted(set(linear_names) - set(targets))
        self.assertEqual(excluded, ['lm_head', 'model.embed_audio.embedding_projection',
                                    'model.embed_vision.multimodal_embedder.embedding_projection',
                                    'model.embed_vision.patch_dense'])

    def test_only_lora_factors_train_and_count_matches_analytic(self):
        model, config, surface = lora_model()
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        self.assertTrue(all('.lora_A.' in name or '.lora_B.' in name for name in trainable))
        self.assertFalse(any('embed_vision' in name or 'embed_audio' in name or 'lm_head' in name
                             for name in trainable))
        text_config = model.get_base_model().config.get_text_config()
        self.assertEqual(surface['trainable_parameters'], trainer.expected_lora_parameters(text_config, 16))
        self.assertEqual(surface['target_modules'], 27)

    def test_pinned_12b_surface_is_66m_parameters_at_rank_16(self):
        from transformers.models.gemma4_unified.configuration_gemma4_unified import Gemma4UnifiedTextConfig
        text = Gemma4UnifiedTextConfig(
            vocab_size=262144, hidden_size=3840, intermediate_size=15360, num_hidden_layers=48,
            num_attention_heads=16, num_key_value_heads=8, head_dim=256, global_head_dim=512,
            num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=1024,
            layer_types=(['sliding_attention'] * 5 + ['full_attention']) * 8)
        self.assertEqual(trainer.expected_lora_parameters(text, 16), 65_568_768)
        self.assertEqual(trainer.expected_lora_parameters(text, 32), 2 * 65_568_768)

    def test_rank_and_weight_validation(self):
        with self.assertRaises(trainer.LilithTrainerError):
            trainer.TrainerConfig(rank=8)
        trainer.TrainerConfig(rank=32)
        for weight in (-0.1, 1.5, math.nan):
            with self.assertRaises(trainer.LilithTrainerError):
                trainer.build_sft_example([1, 2], [3], weight=weight)
            with self.assertRaises(trainer.LilithTrainerError):
                trainer.PreferencePair((1,), (2,), (3,), weight)

    def test_quantization_spec_only_for_nf4(self):
        self.assertIsNone(trainer.quantization_spec(trainer.TrainerConfig()))
        spec = trainer.quantization_spec(trainer.TrainerConfig(base_precision='nf4'))
        self.assertEqual(spec['bnb_4bit_quant_type'], 'nf4')
        self.assertIn('lm_head', spec['llm_int8_skip_modules'])
        self.assertIn('embed_vision', spec['llm_int8_skip_modules'])


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class Nf4PrecisionTests(unittest.TestCase):
    """Plan §2 NF4 budget: nothing in the frozen base (embeddings, tied head, activations) leaves bf16."""

    def bf16_base(self):
        return tiny_base(0, trainer.register_chunked_attention(3)).to(torch.bfloat16)

    def test_nf4_preparation_keeps_embeddings_head_and_activations_bf16(self):
        from unittest import mock
        import importlib
        config = trainer.TrainerConfig(base_precision='nf4', ce_chunk_tokens=5, attention_query_chunk=3)
        with mock.patch.object(importlib.import_module('peft'), 'prepare_model_for_kbit_training',
                               side_effect=AssertionError('k-bit helper upcasts to fp32')):
            model, surface = trainer.prepare_lora_model(self.bf16_base(), config)
        base = model.get_base_model()
        self.assertEqual(base.model.language_model.embed_tokens.weight.dtype, torch.bfloat16)
        self.assertEqual(base.lm_head.weight.dtype, torch.bfloat16)
        self.assertEqual(set(surface['base_dtypes']), {'torch.bfloat16'})
        activations = []
        hook = base.model.language_model.layers[3].register_forward_hook(
            lambda module, inputs, output: activations.append(output.dtype))
        model.train()
        loss, _ = trainer.weighted_sft_loss(model, [random_example(2)], config)
        loss.backward()
        hook.remove()
        self.assertTrue(activations)
        self.assertTrue(all(dtype == torch.bfloat16 for dtype in activations))

    def test_dtype_assertion_rejects_an_upcast_base_but_allows_fp32_norms(self):
        config = trainer.TrainerConfig(base_precision='nf4')
        base = self.bf16_base()
        base.model.language_model.layers[0].input_layernorm.float()
        trainer.assert_base_dtypes(base, config)                       # fp32 norms are fine
        base.model.language_model.embed_tokens.float()
        with self.assertRaises(trainer.LilithTrainerError) as raised:
            trainer.assert_base_dtypes(base, config)
        self.assertEqual(raised.exception.code, 'lilith::trainer::dtype')

    def test_parent_loader_quantizes_only_for_nf4_and_cold_reload_uses_it(self):
        from unittest import mock
        import importlib
        # Patch the live module: the lazy `transformers` bound at import time is not
        # the sys.modules entry the trainer resolves later.
        live = importlib.import_module('transformers')
        calls = []

        def fake_from_pretrained(parent, **kwargs):
            calls.append(kwargs)
            return self.bf16_base()

        fake_model = mock.Mock(from_pretrained=fake_from_pretrained)
        with mock.patch.object(live, 'Gemma4UnifiedForConditionalGeneration', fake_model), \
                mock.patch.object(live, 'BitsAndBytesConfig', lambda **spec: ('bnb', spec)):
            trainer.load_parent_base(Path('/parent'), trainer.TrainerConfig(base_precision='nf4'), device='cpu')
            trainer.load_parent_base(Path('/parent'), trainer.TrainerConfig(base_precision='bf16'), device='cpu')
        nf4, bf16 = calls
        self.assertEqual(nf4['quantization_config'][1]['bnb_4bit_quant_type'], 'nf4')
        self.assertEqual(nf4['quantization_config'][1]['bnb_4bit_compute_dtype'], torch.bfloat16)
        self.assertEqual(nf4['dtype'], torch.bfloat16)
        self.assertNotIn('quantization_config', bf16)
        source = Path(microrun.__file__).read_text()
        cold = source[source.index('def run_cold_reload'):source.index('def parse_args')]
        self.assertIn('trainer.load_parent_base(args.parent, config', cold)   # adapter's own precision


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class LossTests(unittest.TestCase):
    def test_weight_zero_contributes_zero_gradient(self):
        model, config, _ = lora_model()
        model.train()
        loss, _ = trainer.weighted_sft_loss(model, [random_example(1, weight=0.0)], config)
        loss.backward()
        self.assertTrue(all(float(g.abs().max()) == 0.0 for g in lora_grads(model).values()))

    def test_weight_zero_sibling_leaves_the_other_gradient_unchanged_up_to_normalization(self):
        live, silent = random_example(1, weight=1.0), random_example(2, weight=0.0)
        model, config, _ = lora_model()
        model.train()
        loss, _ = trainer.weighted_sft_loss(model, [live], config)
        loss.backward()
        alone = lora_grads(model)
        model.zero_grad(set_to_none=True)
        loss, total = trainer.weighted_sft_loss(model, [live, silent], config)
        loss.backward()
        mixed = lora_grads(model)
        scale = sum(live.target_mask) / total   # normalization is by all supervised tokens
        for name in alone:
            torch.testing.assert_close(mixed[name], alone[name] * scale, rtol=1e-5, atol=1e-8)

    def test_weights_scale_linearly(self):
        model, config, _ = lora_model()
        model.train()
        full, _ = trainer.weighted_sft_loss(model, [random_example(1, weight=1.0)], config)
        half, _ = trainer.weighted_sft_loss(model, [random_example(1, weight=0.5)], config)
        torch.testing.assert_close(half, full * 0.5)

    def test_preference_loss_sign_shape_and_weight(self):
        zero = torch.zeros(3, requires_grad=True)
        at_reference = trainer.preference_loss(zero, zero, zero.detach(), zero.detach(), beta=0.1)
        self.assertEqual(at_reference.shape, (3,))
        torch.testing.assert_close(at_reference, torch.full((3,), math.log(2.0)))
        chosen = torch.tensor([2.0], requires_grad=True)
        rejected = torch.tensor([-1.0], requires_grad=True)
        loss = trainer.preference_loss(chosen, rejected, torch.zeros(1), torch.zeros(1), beta=0.1)
        self.assertLess(float(loss.detach()), math.log(2.0))
        loss.sum().backward()
        self.assertLess(float(chosen.grad), 0.0)    # descending raises the corrected sibling
        self.assertGreater(float(rejected.grad), 0.0)
        worse = trainer.preference_loss(torch.tensor([-2.0]), torch.tensor([1.0]), torch.zeros(1),
                                        torch.zeros(1), beta=0.1)
        self.assertGreater(float(worse), math.log(2.0))
        silent = trainer.preference_loss(chosen, rejected, torch.zeros(1), torch.zeros(1), beta=0.1, weight=0.0)
        self.assertEqual(float(silent.detach()), 0.0)
        with self.assertRaises(trainer.LilithTrainerError):
            trainer.preference_loss(chosen, rejected, torch.zeros(1), torch.zeros(1), beta=0.1, weight=-1.0)

    def test_steps_update_the_adapter_and_preference_starts_at_log_two(self):
        config = fp32_config()
        trainer.register_chunked_attention(config.attention_query_chunk)
        model, _ = trainer.prepare_lora_model(tiny_base(0, trainer.ATTENTION_IMPLEMENTATION), config)
        optimizer = trainer.make_optimizer(model, config)
        before = trainer.adapter_tensor_sha256(model)
        pair = trainer.PreferencePair(tuple(range(10, 17)), tuple(range(20, 30)), tuple(range(40, 45)))
        # Fresh LoRA (B = 0) equals the reference, so the first preference loss is log 2.
        receipt = trainer.preference_step(model, optimizer, [pair], config, 0)
        self.assertAlmostEqual(receipt.loss, math.log(2.0), places=5)
        self.assertEqual(receipt.extra['margins'], [0.0])
        self.assertNotEqual(trainer.adapter_tensor_sha256(model), before)
        receipt = trainer.sft_step(model, optimizer, [random_example(3)], config, 1)
        self.assertTrue(math.isfinite(receipt.loss) and receipt.grad_norm > 0)
        with torch.no_grad(), model.disable_adapter():
            reference = trainer.sequence_logp(model, pair.chosen(), config)
        self.assertNotEqual(float(trainer.sequence_logp(model, pair.chosen(), config)), float(reference))

    def test_gradient_checkpointing_matches_plain_backward(self):
        grads = []
        for checkpointing in (False, True):
            model, config, _ = lora_model(gradient_checkpointing=checkpointing)
            model.train()
            loss, _ = trainer.weighted_sft_loss(model, [random_example(8)], config)
            loss.backward()
            grads.append(lora_grads(model))
        for name in grads[0]:
            torch.testing.assert_close(grads[1][name], grads[0][name], rtol=1e-5, atol=1e-7)


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class AdapterFileTests(unittest.TestCase):
    def test_save_and_cold_reload_identity(self):
        config = fp32_config()
        trainer.register_chunked_attention(config.attention_query_chunk)
        base = tiny_base(0, trainer.ATTENTION_IMPLEMENTATION)
        pristine = copy.deepcopy(base)
        model, _ = trainer.prepare_lora_model(base, config)
        optimizer = trainer.make_optimizer(model, config)
        for step in range(2):
            trainer.sft_step(model, optimizer, [random_example(step)], config, step)
        probe = list(random_example(9).input_ids)
        expected = microrun.probe_logits(model, probe)
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory) / 'adapter'
            receipt = trainer.save_adapter(model, adapter, config)
            self.assertIn('adapter_model.safetensors', receipt.files_sha256)
            self.assertEqual(receipt.tensor_sha256, trainer.adapter_tensor_sha256(model))
            with self.assertRaises(trainer.LilithTrainerError):
                trainer.save_adapter(model, adapter, config)   # write-once
            reloaded, tensor_sha = trainer.reload_adapter(copy.deepcopy(pristine), adapter, receipt)
            self.assertEqual(tensor_sha, receipt.tensor_sha256)
            self.assertTrue(torch.equal(microrun.probe_logits(reloaded, probe), expected))
            with (adapter / 'adapter_model.safetensors').open('ab') as stream:
                stream.write(b'\0')
            with self.assertRaises(trainer.LilithTrainerError):
                trainer.reload_adapter(copy.deepcopy(pristine), adapter, receipt)


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class MicrorunEndToEndTests(unittest.TestCase):
    """Drive the real micro-run and cold-reload entry points on CPU with a tiny saved parent."""

    def test_microrun_then_cold_reload_in_a_fresh_model(self):
        from unittest import mock
        import json
        original = microrun.synthetic_records

        def tiny_records(seed):
            return original(seed, count=4, prompt=5, answer=7, low=3, high=500)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(trainer, 'verify_parent', lambda parent: {'pins': 'skipped-in-cpu-test'}), \
                mock.patch.object(microrun, 'synthetic_records', tiny_records):
            root = Path(directory)
            tiny_base(0).save_pretrained(root / 'parent')
            common = ['--parent', str(root / 'parent'), '--device', 'cpu', '--base-precision', 'fp32',
                      '--ce-chunk-tokens', '7', '--attention-query-chunk', '5']
            code = microrun.main(['--mode', 'microrun', *common, '--synthetic-data', '--contexts', '24', '40',
                                  '--steps', '3', '--adapter-root', str(root / 'adapters'),
                                  '--generation-backend', 'none', '--output', str(root / 'microrun.json')])
            self.assertEqual(code, 0)
            report = json.loads((root / 'microrun.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertFalse(report['is_reference'])            # synthetic, fp32, tiny contexts
            self.assertIn('data', report['deviations'])
            for context in ('24', '40'):
                entry = report['contexts'][context]
                self.assertEqual(len(entry['loss_curve']), 3)
                self.assertEqual(len(entry['preference']), 1)
                self.assertTrue(all(math.isfinite(row['loss']) for row in entry['loss_curve']))
                self.assertNotEqual(entry['adapter']['tensor_sha256'], report['initial_adapter_sha256'])
            # Both contexts start from the same seeded adapter, so the first loss is shared up to packing.
            with self.assertRaises(FileExistsError):              # evidence files are exclusive
                microrun.main(['--mode', 'microrun', *common, '--synthetic-data', '--contexts', '24',
                               '--steps', '1', '--adapter-root', str(root / 'other'),
                               '--generation-backend', 'none', '--output', str(root / 'microrun.json')])
            code = microrun.main(['--mode', 'cold-reload', *common, '--adapter-dir', str(root / 'adapters/ctx-24'),
                                  '--probe', str(root / 'adapters/ctx-24.probe.json'),
                                  '--output', str(root / 'cold-reload.json')])
            self.assertEqual(code, 0)
            reload = json.loads((root / 'cold-reload.json').read_text())
            self.assertEqual(reload['status'], 'passed')
            self.assertEqual(reload['base_precision'], 'fp32')    # the adapter's own training precision
            self.assertTrue(reload['probe_sha256_identical'])
            self.assertEqual(reload['tensor_sha256'], report['contexts']['24']['adapter']['tensor_sha256'])


class MicrorunDataTests(unittest.TestCase):
    records = [((1, 2, 3), (4, 5)), ((6, 7), (8, 9, 10)), ((11,), (12,))]

    def test_pack_is_exact_length_and_deterministic(self):
        first = microrun.pack_example(self.records, 12, 0, source='t')
        again = microrun.pack_example(self.records, 12, 0, source='t')
        self.assertEqual(first, again)
        self.assertEqual(len(first.input_ids), 12)
        self.assertEqual(first.input_ids, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        self.assertEqual(first.target_mask, (False, False, False, True, True, False, False, True, True, True,
                                             False, True))

    def test_truncated_tail_is_context_only_and_start_rotates(self):
        packed = microrun.pack_example(self.records, 9, 1, source='t')
        self.assertEqual(packed.input_ids, (6, 7, 8, 9, 10, 11, 12, 1, 2))
        self.assertEqual(packed.target_mask[-2:], (False, False))
        self.assertTrue(packed.source.endswith('start=1'))

    def test_mechanics_pair_contrasts_the_next_answer(self):
        pair = microrun.mechanics_pair(self.records, 2, source='t')
        self.assertEqual((pair.prompt_ids, pair.chosen_ids, pair.rejected_ids), ((11,), (12,), (4, 5)))
        self.assertIn('mismatched-answer-contrast', pair.source)

    def test_synthetic_records_are_seeded(self):
        self.assertEqual(microrun.synthetic_records(73), microrun.synthetic_records(73))
        self.assertNotEqual(microrun.synthetic_records(73), microrun.synthetic_records(74))

    def test_reference_arguments_have_no_deviation(self):
        args = microrun.parse_args(['--mode', 'microrun', '--parent', '/p', '--output', '/o.json',
                                    '--adapter-root', '/a', '--data', '/d.jsonl'])
        self.assertEqual(microrun.deviations(args, 'sft-subset-v1'), {})
        args = microrun.parse_args(['--mode', 'microrun', '--parent', '/p', '--output', '/o.json',
                                    '--adapter-root', '/a', '--synthetic-data', '--base-precision', 'nf4'])
        found = microrun.deviations(args, 'synthetic')
        self.assertEqual(set(found), {'base_precision', 'data'})

    def test_memory_probe_verdict_names_oom_contexts(self):
        fits = {'status': 'fits'}
        self.assertEqual(microrun.memory_probe_verdict({'8192': fits, '16384': fits}), ('completed', []))
        self.assertEqual(microrun.memory_probe_verdict({'8192': fits, '32768': {'status': 'oom'},
                                                        '16384': {'status': 'oom'}}), ('oom', [16384, 32768]))

    def test_memory_probe_defaults_to_three_contexts(self):
        args = microrun.parse_args(['--mode', 'memory-probe', '--parent', '/p', '--output', '/o.json'])
        self.assertEqual(args.contexts, [8192, 16384, 32768])


if __name__ == '__main__':
    unittest.main()
