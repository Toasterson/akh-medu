import importlib.util
import json
from pathlib import Path
import unittest

P = Path(__file__).with_name('lilith_training_engine.py')
spec = importlib.util.spec_from_file_location('lilith_training_engine', P)
M = importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name] = M
spec.loader.exec_module(M)


class FakeTensor:
    def __init__(self, requires_grad=False, grad_fn=None):
        self.requires_grad = requires_grad
        self.grad_fn = grad_fn
        self.shape = (1,)


class FakeParameter:
    requires_grad = True


class FakeTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is True
        tools = kwargs.get('tools') or []
        return json.dumps({'messages':messages, 'tools':tools}, sort_keys=True)

    def encode(self, text, add_special_tokens=False):
        return list(text.encode())


class FakeModel:
    def __init__(self):
        self.disabled_checkpoint = False
        self.disabled_input_grads = False
        self.eval_called = False

    def named_parameters(self):
        return [('layer47.q_proj.lora_A.default.weight', FakeParameter())]

    def gradient_checkpointing_disable(self): self.disabled_checkpoint = True
    def disable_input_require_grads(self): self.disabled_input_grads = True
    def eval(self): self.eval_called = True


class EngineInterfaceTests(unittest.TestCase):
    def test_trajectory_is_json_serializable_and_masks_exact_response(self):
        row = M.Trajectory(id='id',prompt_token_ids=[1,2],token_ids=[1,2,3],policy_mask=[False,False,True],
                           sampler_logps=[None,None,-1.0],mc_token_ids=[[],[],[4,5]],completion_status='completed',policy_sha256='policy',raw_text='text',token_roles=['prompt','prompt','assistant'],terminator_token_id=106)
        row.validate(32)
        self.assertEqual(json.loads(json.dumps(row.to_dict()))['terminator_token_id'],106)
        with self.assertRaises(ValueError):
            bad = M.Trajectory(id='id',prompt_token_ids=[1],token_ids=[1,2],policy_mask=[True,True],sampler_logps=[None,-1],mc_token_ids=[[],[]],completion_status='completed',policy_sha256='policy',raw_text='text')
            bad.validate(32)

        mismatched_mc = M.Trajectory(
            id='id', prompt_token_ids=[1], token_ids=[1,2],
            policy_mask=[False,True], sampler_logps=[None,-1.0],
            mc_token_ids=[[],[3,4]], mc_sampler_logps=[[],[-2.0]],
            completion_status='completed', policy_sha256='policy', raw_text='text')
        with self.assertRaisesRegex(ValueError, 'MC token IDs'):
            mismatched_mc.validate(32)

    def test_cache_invariant_rejects_any_graph(self):
        self.assertEqual(M.assert_detached_kv({'k':FakeTensor(),'v':FakeTensor()}),2)
        with self.assertRaises(RuntimeError):
            M.assert_detached_kv([FakeTensor(requires_grad=True)])
        with self.assertRaises(RuntimeError):
            M.assert_detached_kv([FakeTensor(grad_fn=object())])

    def test_worker_interface_defaults_and_thinking_tool_render(self):
        model=FakeModel(); tokenizer=FakeTokenizer()
        engine=M.GemmaTrainingEngine(model,tokenizer,'a'*64,'b'*64,'q_proj')
        self.assertEqual((engine.context_tokens,engine.max_new_tokens),(32768,16384))
        rendered, ids=engine.render_messages([{'role':'user','content':'x'}],tools=[{'name':'t'}])
        self.assertTrue(ids)
        self.assertIn('"name": "t"',rendered)
        self.assertTrue(model.disabled_checkpoint and model.disabled_input_grads and model.eval_called)
        contract=engine.contract()
        self.assertTrue(contract['thinking'])
        self.assertEqual(contract['update'],'one optimizer step after full eligible bank')

    def test_no_standalone_entrypoint(self):
        source=P.read_text()
        self.assertNotIn("if __name__ == '__main__'",source)
        self.assertNotIn('argparse',source)
        self.assertIn('return SynchronousOffloadedCache(layers=layers)',source)
        self.assertIn('action_index - 1',source)


if __name__ == '__main__':
    unittest.main()
