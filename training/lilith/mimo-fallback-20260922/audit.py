import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts/session-platform'))
import lilith_rl_tasks_v2 as tasks
RUN=Path(__file__).with_name('full8-thinking8192')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=json.loads((RUN/'protocol.json').read_text());records=json.loads((RUN/'records.json').read_text())
pack={x.id:x for x in tasks.tasks()}
assert protocol['verifier_sha256']==sha(tasks.__file__)
assert protocol['interpreter_sha256']==sha(Path(tasks.__file__).with_name('lilith_rl_tasks.py'))
seen=set();checks=[]
for row in records:
 assert row['task_id'] not in seen;seen.add(row['task_id'])
 assert row['completion']=='answer_present' and row['eos_reached'] is True
 fresh=tasks.verify(row['content'],pack[row['task_id']])
 assert fresh==row['evidence']
 checks.append({'task_id':row['task_id'],'passed':fresh['passed'],'category':fresh['category'],
                'counterexample':fresh['counterexample'],'content_sha256':tasks.digest(row['content'])})
assert seen==set(pack)
receipt={'schema':'lilith/mimo-independent-audit@1','complete':True,'records':len(records),
 'passed':sum(x['passed'] for x in checks),'failed':sum(not x['passed'] for x in checks),
 'checks':checks,'records_sha256':sha(RUN/'records.json'),'protocol_sha256':sha(RUN/'protocol.json'),
 'verifier_sha256':sha(tasks.__file__),'interpreter_sha256':sha(Path(tasks.__file__).with_name('lilith_rl_tasks.py')),
 'limitations':['Finite public cases are not proofs.','MiMo 8192-token BF16 Transformers results are not cap/backend/template comparable to 512-token quantized Gemma/Ptah.','One seeded sample per task; no promotion claim.']}
(RUN/'independent-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'passed':receipt['passed'],'failed':receipt['failed']}))
