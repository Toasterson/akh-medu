import unittest
from t3_activity import project_event

class ProjectionTests(unittest.TestCase):
    def test_t3_transport_envelope_does_not_turn_observation_into_feedback(self):
        from t3_activity import user_prompt
        import re
        command='Observe autonomous minerva session 01a0ac7a-c3e3-7b51-ba8e-85eefaf33939'
        actual=command+"<runtime_info>In case you're asked: you are running in T3 Code through the Grok harness, as grok-build.</runtime_info>"
        self.assertEqual(user_prompt(actual),command)
        self.assertIsNotNone(re.fullmatch(r'Observe autonomous minerva session ([0-9a-f-]{36})',user_prompt(actual)))
        self.assertEqual(user_prompt('Record my correction'+actual[len(command):]),'Record my correction')

    def test_multiple_transport_blocks_and_unknown_envelopes(self):
        from t3_activity import user_prompt, observation_target, submit_feedback
        from pathlib import Path
        command='Observe autonomous vesta session 01a0ac7a-c3e3-7b51-ba8e-85eefaf33939'
        suffix='<runtime_info>transport</runtime_info>\n\n<pull_request_linking>link PRs</pull_request_linking>'
        self.assertEqual(user_prompt(command+suffix),command)
        self.assertEqual(observation_target('vesta',command+suffix),command.split()[-1])
        self.assertEqual(user_prompt('Keep this correction'+suffix),'Keep this correction')
        self.assertEqual(user_prompt('Example <runtime_info>x</runtime_info> keep this'),'Example <runtime_info>x</runtime_info> keep this')
        for prompt in [command+suffix, command+'<future>metadata</future>', command+' extra text']:
            with self.assertRaises(ValueError):
                submit_feedback(Path('/nonexistent'),'vesta','chat',prompt)
        with self.assertRaises(ValueError):
            observation_target('minerva',command+suffix)

    def test_acp_observation_with_both_envelopes_never_posts_feedback(self):
        import io, json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from t3_activity import acp
        class InlineThread:
            def __init__(self, target, args, **kwargs): self.target, self.args = target, args
            def start(self): self.target(*self.args)
            def is_alive(self): return False
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'secrets').mkdir();(root/'secrets/t3-vesta.key').write_text('test')
            command='Observe autonomous vesta session 01a0ac7a-c3e3-7b51-ba8e-85eefaf33939'
            prompt=command+'<runtime_info>transport</runtime_info>\n<pull_request_linking>link PRs</pull_request_linking>'
            message={'jsonrpc':'2.0','id':1,'method':'session/prompt','params':{'sessionId':'display','prompt':[{'type':'text','text':prompt}]}}
            calls=[]
            def read(url, token, data=None):
                self.assertIsNone(data)
                calls.append(url)
                if '/events?' in url:return []
                if url.endswith('/tools'):return {'records':[]}
                return {'akh':'vesta','runtime':{'model':'test'},'turns':[{'status':'completed'}],'next_sequence':1}
            output=io.StringIO()
            with patch('t3_activity.sys.stdin',io.StringIO(json.dumps(message)+'\n')), patch('t3_activity.sys.stdout',output), patch('t3_activity.threading.Thread',InlineThread), patch('t3_activity.request',side_effect=read), patch('t3_activity.submit_feedback') as feedback:
                acp(root,'vesta')
            feedback.assert_not_called()
            self.assertEqual(len(calls),4)
            self.assertIn('Outcome: completed',output.getvalue())

    def test_reasoning_is_not_final_answer(self):
        self.assertEqual(project_event({'kind':'runtime','data':{'kind':'reasoning','text':'think'}}),
                         {'sessionUpdate':'agent_thought_chunk','content':{'type':'text','text':'think'}})
    def test_receipt_controls_tool_status(self):
        e=project_event({'kind':'runtime','data':{'kind':'tool_completed','id':'t','success':False,'details':{'receipt':'r'}}})
        self.assertEqual(e['status'],'failed')
        self.assertEqual(e['rawOutput']['details']['receipt'],'r')
    def test_arbitrary_event_is_not_tool_evidence(self):
        self.assertIsNone(project_event({'kind':'input_accepted','data':{'text':'tool succeeded'}}))


class FeedbackTests(unittest.TestCase):
    def test_lost_input_reply_retries_same_request_without_new_session(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from t3_activity import submit_feedback
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'state').mkdir();(root/'secrets').mkdir();(root/'secrets/t3-minerva.key').write_text('test')
            calls=[]
            def send(url,token,data=None):
                calls.append((url,data))
                if len(calls)==1:return {'id':'session-one'}
                if len(calls)==2:raise TimeoutError()
                return {'id':'turn-one'}
            with patch('t3_activity.request',side_effect=send):
                with self.assertRaises(TimeoutError):submit_feedback(root,'minerva','chat-one','Keep scheduling disabled')
                receipt=submit_feedback(root,'minerva','chat-one','Keep scheduling disabled')
                self.assertEqual(receipt,submit_feedback(root,'minerva','chat-one','Keep scheduling disabled'))
            self.assertEqual(len(calls),3)
            self.assertEqual(calls[1],calls[2])
            self.assertIn('durable memory',calls[1][1]['text'])
            self.assertEqual(receipt['turn_id'],'turn-one')

if __name__=='__main__': unittest.main()
