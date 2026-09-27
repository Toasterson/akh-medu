import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

P = Path(__file__).with_name('adam_kaine_gemma_train.py')


def load_module():
    spec = importlib.util.spec_from_file_location('adam_kaine_gemma_train', P)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TrainingProtocolTests(unittest.TestCase):
    def test_transcript_selection_is_train_only_deterministic_and_never_rewarded(self):
        m = load_module()
        rows = [{'record_id': f'r{i}', 'split': 'validation' if i >= 8 else 'train',
                 'schema':'akh/lilith-transcript-sft@1',
                 'messages':[{'role':'user','content':'x'},{'role':'assistant','content':'y'}]}
                for i in range(16)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rows.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            selected = m.transcript_examples(path, m.digest(path), 'train')
            self.assertEqual(len(selected), 8)
            self.assertTrue(all(r['split'] == 'train' for r in selected))
            self.assertEqual(selected, m.transcript_examples(path, m.digest(path), 'train'))

    def test_training_protocol_has_fresh_banks_and_no_holdout_selection(self):
        source = P.read_text()
        self.assertIn("for step in range(1, 4)", source)
        self.assertIn("bank_arm = f'{arm}-bank{step}'", source)
        self.assertIn("tie_priority = {'adapter_blend_0p5':2, 'klpo_online':1, 'sft_distill':0}", source)
        self.assertIn("release_selected=False", source)
        self.assertIn("breeding_performed=True", source)
        self.assertIn("blend_states(candidate_states['sft_distill'], candidate_states['klpo_online'], 0.5)", source)
        self.assertIn("independent_tasks = [task for task in task_list if task.split == 'independent_eval']", source)
        self.assertIn("'lilith-rl/action/' + rng_namespace", source)
        self.assertIn("enable_thinking=True", source)
        self.assertIn("'max_new_tokens': 320", source)
        self.assertIn("all(r['completion']['status'] == 'truncated' for r in base_records)", source)
        self.assertIn("memory_control(model, 1024)", source)
        self.assertIn("cfg['max_new_tokens'] = 800", source)
        self.assertIn("answer = reasoned_answer(text)", source)
        self.assertIn("malformed Gemma thought channel", source)
        self.assertIn("records = collect(arm + '-development', development_tasks", source)
        self.assertIn("progress('selection_frozen_before_independent_eval')", source)
        self.assertIn("'independent-eval-pair-v1'", source)
        self.assertIn("base_eval = collect('base-independent-eval'", source)
        self.assertNotIn("collect('base-final', task_list", source)
        self.assertIsNotNone(ast.parse(source))

    def test_pinned_gemma_reasoning_frame_parser(self):
        m = load_module()
        self.assertEqual(m.reasoned_answer('<|channel>thought\nwork\n<channel|>\ndef f(a,b): return a'),
                         'def f(a,b): return a')
        self.assertEqual(m.reasoned_answer('def f(a,b): return b'), 'def f(a,b): return b')
        rejected = [
            '<|channel>thought\nunfinished',
            '<channel|>def f(a,b): return a',
            'prefix<|channel>thought\nx\n<channel|>answer',
            '<|channel>thought\nx\n<channel|><|channel>thought\ny\n<channel|>answer',
            '<|channel>thought\nx <channel|> fake\n<channel|>answer',
        ]
        for text in rejected:
            with self.assertRaises(ValueError):
                m.reasoned_answer(text)

    def test_frozen_task_manifest_runtime_projection(self):
        import sys
        sys.path.insert(0, str(P.parent))
        import adam_kaine_code_rl_tasks as tasks
        manifest = json.loads((P.parents[2] / 'training/lilith/adam-kaine-code-env-20260922/rl-tasks.json').read_text())
        pinned = {row['id']: row for row in manifest['tasks']}
        for task in tasks.tasks():
            observed = {'id':task.id, 'family':task.family, 'split':task.split,
                'prompt':task.prompt,
                'prompt_sha256':__import__('hashlib').sha256(task.prompt.encode()).hexdigest(),
                'cases_sha256':tasks.digest(task.cases)}
            self.assertTrue(all(pinned[task.id][key] == value for key,value in observed.items()))


if __name__ == '__main__':
    unittest.main()
