from pathlib import Path
import sys
import unittest

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Gemma4UnifiedConfig,
    Gemma4UnifiedForConditionalGeneration,
    Gemma4UnifiedTextConfig,
)
from transformers.cache_utils import DynamicSlidingWindowLayer

sys.path.insert(0, str(Path(__file__).parent))
from lilith_training_engine import GemmaTrainingEngine, assert_detached_kv


TARGET = "model.language_model.layers.1.self_attn.q_proj"


def tiny_engine(*, cache_offloading=False):
    torch.manual_seed(73)
    text_config = Gemma4UnifiedTextConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        vocab_size=128,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=8,
        use_bidirectional_attention="vision",
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
        },
    )
    config = Gemma4UnifiedConfig(text_config=text_config)
    config._attn_implementation = "sdpa"
    model = get_peft_model(
        Gemma4UnifiedForConditionalGeneration(config),
        LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0, target_modules=[TARGET]),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.uniform_(-0.1, 0.1)
    return GemmaTrainingEngine(
        model, None, "parent", "tokenizer", TARGET,
        context_tokens=32, max_new_tokens=8, cache_offloading=cache_offloading,
    )


class GemmaTrainingCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_cache_uses_hybrid_layers_and_bounds_sliding_storage(self):
        engine = tiny_engine()
        cache = engine._cache()

        self.assertEqual(len(cache.layers), 2)
        self.assertIsInstance(cache.layers[0], DynamicSlidingWindowLayer)
        for position, token in enumerate([2] + list(range(10, 25))):
            engine._forward([token], cache, position, grad=False)

        self.assertEqual(cache.layers[0].keys.shape[-2], 7)
        self.assertEqual(cache.layers[1].keys.shape[-2], 16)

    def test_cached_logits_and_final_query_gradients_match_full_forward(self):
        engine = tiny_engine(cache_offloading=True)
        model = engine.model
        all_tokens = [2] + list(range(10, 19))

        for length in (7, 8, 9):
            with self.subTest(length=length):
                tokens = all_tokens[:length]
                full_ids = torch.tensor([tokens])
                positions = torch.arange(length)[None, :]
                model.zero_grad(set_to_none=True)
                full_logits = model(
                    input_ids=full_ids,
                    use_cache=False,
                    position_ids=positions,
                    logits_to_keep=1,
                ).logits.float()[0, -1]
                (-torch.log_softmax(full_logits, dim=-1)[19]).backward()
                full_gradients = {
                    name: parameter.grad.detach().clone()
                    for name, parameter in model.named_parameters()
                    if parameter.requires_grad
                }
                self.assertEqual(len(full_gradients), 2)
                self.assertTrue(all(gradient.count_nonzero() for gradient in full_gradients.values()))

                model.zero_grad(set_to_none=True)
                cache = engine._cache()
                engine._prefill_no_grad(tokens[:-1], cache)
                cached_logits = engine._forward([tokens[-1]], cache, length - 1, grad=True)
                self.assertGreaterEqual(assert_detached_kv(cache), 4)
                (-torch.log_softmax(cached_logits, dim=-1)[19]).backward()

                torch.testing.assert_close(cached_logits, full_logits, rtol=1e-5, atol=1e-5)
                for name, parameter in model.named_parameters():
                    if parameter.requires_grad:
                        torch.testing.assert_close(parameter.grad, full_gradients[name], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
