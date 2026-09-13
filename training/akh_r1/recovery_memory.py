"""Public receipt-backed recovery memory plus examiner-only fixture snapshots."""
import hashlib,json,os
from pathlib import Path
from recovery_fixture import RecoveryFixture
KINDS={'dependency_document':'dependency','command_document':'commands'}
def documents(observations):
 return {e['source']:dict(facts=[dict(subject='recovery-skill-v1',predicate=KINDS[e['kind']],object=json.dumps(e['content'],sort_keys=True))]) for e in observations if e['kind'] in KINDS}
def request(port,tool,**arguments):
 r=port.call(dict(tool=tool,arguments=arguments))
 if not r.get('success'):raise ValueError('Public memory request failed')
 return r

def acquire(port):
 for name in request(port,'list_documents')['documents']:
  observed=request(port,'read_document',id=name)
  for fact in observed['document']['facts']:request(port,'remember',source=observed['evidence_id'],fact=fact)
 return recall(port)
def recall(port):
 facts=request(port,'query',subject='recovery-skill-v1')['facts'];result={}
 for predicate in KINDS.values():
  values={f['object'] for f in facts if f['predicate']==predicate}
  if len(values)!=1:raise ValueError('Missing or conflicting retained skill')
  result[predicate]=json.loads(next(iter(values)))
 return result

def save_fixture(path,episode,fixture):
 path=Path(path);blob=json.dumps(dict(schema='akh-recovery-snapshot-v1',episode_sha256=hashlib.sha256(json.dumps(episode,sort_keys=True).encode()).hexdigest(),state=fixture.state),sort_keys=True).encode()
 with path.open('xb') as f:os.chmod(path,0o600);f.write(blob);f.flush();os.fsync(f.fileno())
 return hashlib.sha256(blob).hexdigest()
def restore_fixture(path,episode,expected_digest):
 blob=Path(path).read_bytes()
 if hashlib.sha256(blob).hexdigest()!=expected_digest:raise ValueError('Snapshot fingerprint mismatch')
 d=json.loads(blob)
 if set(d)!={'schema','episode_sha256','state'} or d['schema']!='akh-recovery-snapshot-v1' or d['episode_sha256']!=hashlib.sha256(json.dumps(episode,sort_keys=True).encode()).hexdigest():raise ValueError('Snapshot episode identity mismatch')
 if d['state']['stage']!='disconnected':raise ValueError('Snapshot not interrupted')
 return RecoveryFixture(episode,d['state']).restart()
