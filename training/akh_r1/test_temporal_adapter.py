import json,unittest
from temporal_adapter import TemporalEpisode,TemporalGuard
class Graph:
    def __init__(self):self.calls=[];self.state={'success':True,'holding':False}
    def call(self,tool,arguments=None):
        self.calls.append((tool,arguments))
        if tool=='exam_export':return dict(success=True,triples=[])
        if tool=='exam_temporal_state':return self.state
        return dict(success=True)
class Tests(unittest.TestCase):
    def setup_episode(self,event=None):
        self.graph=Graph();self.fact=dict(subject='operator',predicate='permission-event',object=json.dumps(event or dict(id='grant',timestamp=10,granted=True)))
        return TemporalEpisode(self.graph,{}, {'source':{'facts':[self.fact]}})
    def request(self,p,tool,**arguments):return p.call(dict(tool=tool,arguments=arguments))
    def test_exact_observed_source_required(self):
        p=self.setup_episode();r=self.request(p,'remember',source='invented',fact=self.fact);self.assertFalse(r['success']);self.assertNotIn('exam_temporal_event',[c[0] for c in self.graph.calls])
        receipt=self.request(p,'read_document',id='source')['evidence_id'];r=self.request(p,'remember',source=receipt,fact=self.fact);self.assertTrue(r['success']);self.assertEqual(sum(t=='exam_temporal_event' for t,_ in self.graph.calls),1)
    def test_source_cannot_be_reused_in_new_session(self):
        p=self.setup_episode();receipt=self.request(p,'read_document',id='source')['evidence_id'];q=TemporalEpisode(self.graph,{},{});self.assertFalse(self.request(q,'remember',source=receipt,fact=self.fact)['success'])
    def test_event_types_rejected_before_native_write(self):
        for event in [dict(id='x',timestamp=True,granted=True),dict(id='x',timestamp=-1,granted=True),dict(id='x',timestamp=1,granted='true'),dict(id='x',timestamp=1,granted=True,expected=True)]:
            p=self.setup_episode(event);receipt=self.request(p,'read_document',id='source')['evidence_id'];self.assertFalse(self.request(p,'remember',source=receipt,fact=self.fact)['success']);self.assertNotIn('exam_temporal_event',[c[0] for c in self.graph.calls])
    def test_native_operations_are_not_public(self):
        p=self.setup_episode();self.assertFalse(self.request(p,'exam_temporal_event',id='grant',timestamp=1,granted=True)['success'])
    def test_guard_fails_closed_and_allows_current(self):
        graph=Graph();effects=[];stages=[['sana','apply','nodealpha']];guard=TemporalGuard(graph,stages,lambda s: effects.append(s) or dict(success=True))
        for state in [{},{'success':False,'holding':True},{'success':True,'holding':1},{'success':True,'holding':'true'},{'success':True,'holding':False}]:
            graph.state=state;self.assertFalse(guard(stages)['success'])
        self.assertEqual(effects,[]);graph.state=dict(success=True,holding=True);self.assertTrue(guard(stages)['success']);self.assertEqual(len(effects),1)
    def test_guard_keeps_observation_available(self):
        graph=Graph();guard=TemporalGuard(graph,[['sana','apply','nodealpha']],lambda s:dict(success=True));self.assertTrue(guard([['sana','inspect','nodealpha']])['success']);self.assertEqual(graph.calls,[])
if __name__=='__main__':unittest.main()
