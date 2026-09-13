"""Verify critical-token weighting without supervising prompts or padding tails."""
import json
import unittest

from choice_loss import balanced_loss, choice_span


class CharacterTokenizer:
    def decode(self, tokens):
        return ''.join(chr(t) for t in tokens)


class ChoiceSpanTests(unittest.TestCase):
    def test_span_selects_envelope_name_not_names_in_arguments(self):
        target = '<tool_call>\n'+json.dumps(dict(name='kg_query', arguments=dict(symbol='remember', note='ação')))+'\n</tool_call>'
        text = 'PROMPT'+target
        tokens = list(map(ord, text))
        start, end = choice_span(CharacterTokenizer(), tokens, 6, len(tokens))
        self.assertEqual(text[start:end], 'kg_query')
        with self.assertRaises(ValueError):
            choice_span(CharacterTokenizer(), list(map(ord, 'PROMPT'+target[:-8])), 6, len(tokens)-8)

    def test_prompt_and_post_length_padding_have_no_gradient(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest('Run this gradient check with the MLX Python environment')
        # Supervised target indices 3..8; function name at index 4. Index 9 is padding tail.
        batch = mx.array([[1]*10])
        lengths = mx.array([[3,8,4,5]])
        logits = mx.zeros((1,9,4))
        gradient = mx.grad(lambda x: balanced_loss(lambda _:x, batch, lengths)[0])(logits)
        mx.eval(gradient)
        magnitude = mx.abs(gradient).sum(axis=-1)[0].tolist()
        self.assertEqual(magnitude[:2], [0.0,0.0])
        self.assertEqual(magnitude[8], 0.0)
        self.assertGreater(magnitude[3], 4 * magnitude[4])
        self.assertAlmostEqual(magnitude[3], sum(magnitude)-magnitude[3], places=5)


if __name__ == '__main__':
    unittest.main()
