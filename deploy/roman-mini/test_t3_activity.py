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
