import json
from pathlib import Path
import tempfile
import unittest
from janus_escalation import Broker

class Fake:
 def __init__(self):self.receipts={};self.fail=True;self.turns=0
 def call(self,path,body=None):
  if body is None:return {'threads':[{'id':next(iter(self.receipts.values()))['threadId'],'session':{'status':'ready'},'messages':[{'role':'assistant','text':'Verified repair'}]}]}
  key=body['commandId']
  if key not in self.receipts:
   self.receipts[key]=body
   if body['type']=='thread.turn.start':self.turns+=1
  if body['type']=='thread.turn.start' and self.fail:self.fail=False;raise TimeoutError('reply lost after admission')
  return {'ok':True}

class Tests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
  for name in ['config','state']:(self.root/name).mkdir()
  (self.root/'config/janus-escalation.json').write_text(json.dumps({'projects':{'anima':'project'}}));self.fake=Fake();self.b=Broker(self.root,self.fake)
  self.args={'incident_key':'WI-403-owner','project':'anima','summary':'Closed turn retains owner','evidence':['Exact session IDs and observation'],'attempted':['Read canonical state twice'],'requested_action':'Investigate supported reconciliation'}
 def tearDown(self):self.temp.cleanup()
 def test_lost_reply_and_restart_reuse_exact_commands(self):
  a=self.b.submit(self.args);self.b.drain();saved=dict(self.fake.receipts)
  b=Broker(self.root,self.fake);self.assertTrue(b.submit(self.args)['duplicate']);b.drain()
  self.assertEqual(self.fake.turns,1);self.assertEqual(saved,self.fake.receipts);self.assertEqual(b.status(self.args['incident_key'])['state'],'awaiting_execution');self.assertEqual(a['thread_id'],b.status(self.args['incident_key'])['thread_id']);self.assertEqual(b.status(self.args['incident_key'])['responses'][0]['text'],'Verified repair')
 def test_validation_and_storm_limit(self):
  with self.assertRaises(ValueError):self.b.submit(dict(self.args,project='/etc'))
  with self.assertRaises(ValueError):self.b.submit(dict(self.args,incident_key='bad key'))
  for i in range(4):self.b.submit(dict(self.args,incident_key='incident-'+str(i)))
  self.assertTrue(self.b.submit(dict(self.args,incident_key='incident-0'))['duplicate'])
  with self.assertRaises(ValueError):self.b.submit(dict(self.args,incident_key='incident-4'))
 def test_offline_admission_does_not_need_t3(self):
  self.b.submit(self.args)
  self.assertEqual(self.b.status(self.args['incident_key'])['state'],'queued');self.assertEqual(self.fake.receipts,{})

 def test_provider_block_is_not_successful_submission_and_never_restarts(self):
  from unittest.mock import patch
  thread=self.b.submit(self.args)['thread_id'];self.b.drain();self.b.drain()
  for error,state in [('Your usage limit was reached','blocked_quota'),
                      ('Your refresh token was revoked','blocked_authentication'),
                      ('HTTP 429','blocked_rate_limit'),('connection lost','failed')]:
   snapshot={'threads':[{'id':thread,'session':{'status':'stopped','lastError':error},'latestTurn':{'state':'error'}}]}
   with patch.object(self.fake,'call',return_value=snapshot) as call:
    result=self.b.status(self.args['incident_key'])
    self.assertEqual(result['state'],state);self.assertEqual(result['next_owner'],'Toasty/operator')
    call.assert_called_once_with('/api/orchestration/snapshot')
  self.assertEqual(self.fake.turns,1)

 def test_new_running_or_completed_turn_wins_over_retained_error(self):
  from unittest.mock import patch
  thread=self.b.submit(self.args)['thread_id'];self.b.drain();self.b.drain()
  for turn,expected in [('running','running'),('completed','turn_completed')]:
   snapshot={'threads':[{'id':thread,'session':{'status':'ready','lastError':'usage limit','updatedAt':'2026-09-21T10:00:00Z'},'latestTurn':{'state':turn,'completedAt':'2026-09-21T10:01:00Z'}}]}
   with patch.object(self.fake,'call',return_value=snapshot):self.assertEqual(self.b.status(self.args['incident_key'])['state'],expected)

 def test_new_error_is_not_hidden_by_an_older_completed_turn(self):
  from unittest.mock import patch
  thread=self.b.submit(self.args)['thread_id'];self.b.drain();self.b.drain()
  snapshot={'threads':[{'id':thread,'session':{'status':'error','lastError':'usage limit','updatedAt':'2026-09-21T10:01:00Z'},'latestTurn':{'state':'completed','completedAt':'2026-09-21T10:00:00Z'}}]}
  with patch.object(self.fake,'call',return_value=snapshot):self.assertEqual(self.b.status(self.args['incident_key'])['state'],'blocked_quota')

 def test_status_outage_retains_receipt_without_redelivery(self):
  from unittest.mock import patch
  thread=self.b.submit(self.args)['thread_id'];self.b.drain();self.b.drain()
  with patch.object(self.fake,'call',side_effect=TimeoutError('offline')):
   result=self.b.status(self.args['incident_key'])
  self.assertEqual(result['state'],'status_unavailable');self.assertEqual(result['thread_id'],thread)
  self.assertEqual(self.fake.turns,1)

if __name__=='__main__':unittest.main()
