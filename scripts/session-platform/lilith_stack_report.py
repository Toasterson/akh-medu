#!/usr/bin/env python3
"""Replay deployment comparison evidence without executing generated code."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import lilith_rl_tasks_v2 as tasks

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def report(path,manifest):
 path=Path(path);protocol=json.loads((path/'protocol.json').read_text())
 frozen=json.loads(Path(manifest).read_text())
 if protocol['tasks_sha256']!=digest(manifest) or frozen['verifier_sha256']!=digest(tasks.__file__) or frozen['interpreter_sha256']!=digest(Path(tasks.__file__).with_name('lilith_rl_tasks.py')):
  raise ValueError('Frozen verifier provenance mismatch')
 pack={t.id:t for t in tasks.tasks()};seen=set();scores={}
 records=json.loads((path/'records.json').read_text())
 for row in records:
  role=row['role'];key=(role,row['task_id'])
  if role not in ('lilith','ptah') or key in seen:raise ValueError('Unexpected/duplicate observation')
  seen.add(key);task=pack[row['task_id']]
  evidence=tasks.verify(row['source'],task)
  if row['status']!='completed':evidence.update(passed=False,completion_gate=row['status'])
  if evidence!=row['evidence']:raise ValueError('Verifier evidence drift')
  score=scores.setdefault(role,{'passed':0,'observed':0,'failures':Counter(),'tasks':[]})
  score['passed']+=evidence['passed'];score['observed']+=1
  if not evidence['passed']:score['failures'][evidence['category']]+=1
  score['tasks'].append({'id':task.id,'passed':evidence['passed'],'category':evidence['category'],
   'counterexample':evidence['counterexample']})
 receipt=json.loads((path/'result.json').read_text())
 complete=receipt['status']=='completed' and seen=={(r,t) for r in ('lilith','ptah') for t in pack}
 return dict(schema='lilith/deployment-replay@1',complete=complete,scores=scores,
  public_single_sample=True,release_parity_established=False,reference_identity=protocol['reference_identity'],
  limitation='Artifact contract includes syntax limits; grammar rejection does not establish algorithmic error.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('manifest');a=p.parse_args()
 print(json.dumps(report(a.run,a.manifest),indent=2))
