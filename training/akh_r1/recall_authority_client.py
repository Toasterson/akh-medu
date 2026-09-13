#!/usr/bin/python3
import json,sys

def select(records,goal,witnesses):
    current=[w for w in witnesses if w['effect']==goal['effect'] and w['scope']==goal['scope']]
    if len(current)!=1:return None
    witness=current[0]
    candidates=[r for r in records if r['effect']==goal['effect'] and r['scope']==goal['scope'] and r['id']==witness['id'] and r['version']==witness['version']]
    if len(candidates)!=1:return None
    return candidates[0]['action']

def call(tool,**arguments):
    print(json.dumps(dict(tool=tool,arguments=arguments)),flush=True);return json.loads(sys.stdin.readline())
def finish(status,answer):call('finish',status=status,answer=answer)
def main():
    g=json.loads(sys.stdin.readline())['goal']
    for name in call('list_documents')['documents']:
        r=call('read_document',id=name)
        for f in r['document']['facts']:
            if not call('remember',source=r['evidence_id'],fact=f)['success']:return finish('blocked','Source rejected')
    if g['phase']=='learn':return finish('complete','Knowledge retained')
    facts=call('query',subject='skills')['facts'];records=[json.loads(f['object']) for f in facts if f['predicate']=='skill']
    witnesses=[json.loads(f['object']) for f in facts if f['predicate']=='current_skill']
    action=select(records,g,witnesses)
    if action is None:return finish('blocked','No unique remembered skill')
    if not call('terminal',stages=[['sana','choose',action]])['success']:return finish('blocked','Action rejected')
    r=call('terminal',stages=[['sana','state']]);ok=r['success'] and json.loads(r['stdout'])['resolved']
    finish('complete' if ok else 'blocked',dict(action=action,verified=bool(ok)))
if __name__=='__main__':main()
