import unittest
from t3_activity import project_event

class ProjectionTests(unittest.TestCase):
    def test_reasoning_is_not_final_answer(self):
        self.assertEqual(project_event({'kind':'runtime','data':{'kind':'reasoning','text':'think'}}),
                         {'sessionUpdate':'agent_thought_chunk','content':{'type':'text','text':'think'}})
    def test_receipt_controls_tool_status(self):
        e=project_event({'kind':'runtime','data':{'kind':'tool_completed','id':'t','success':False,'details':{'receipt':'r'}}})
        self.assertEqual(e['status'],'failed')
        self.assertEqual(e['rawOutput']['details']['receipt'],'r')
    def test_arbitrary_event_is_not_tool_evidence(self):
        self.assertIsNone(project_event({'kind':'input_accepted','data':{'text':'tool succeeded'}}))

if __name__=='__main__': unittest.main()
