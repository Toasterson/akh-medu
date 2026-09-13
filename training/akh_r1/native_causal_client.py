#!/usr/bin/python3
import json,sys

def call(tool,**arguments):
    print(json.dumps(dict(tool=tool,arguments=arguments)),flush=True);return json.loads(sys.stdin.readline())
def finish(status,answer):call('finish',status=status,answer=answer)
def main():
    g=json.loads(sys.stdin.readline())['goal']
    for name in call('list_documents')['documents']:
        r=call('read_document',id=name)
        for f in r['document']['facts']:
            if not call('revise',source=r['evidence_id'],fact=f)['success']:return finish('blocked','Source rejected')
    predicted=call('interact',action='predict',value='repair')
    if not predicted['success'] or not predicted.get('applicable'):return finish('blocked','No applicable sourced action')
    token=predicted['prediction']
    before=call('interact',action='verify',value=token)
    if before.get('verified'):return finish('blocked','Effect already present')
    if not call('terminal',stages=[['sana','apply',g['target']]])['success']:return finish('blocked','Action rejected')
    if not call('terminal',stages=[['sana','inspect',g['target']]])['success']:return finish('blocked','Observation failed')
    result=call('interact',action='verify',value=token)
    finish('complete' if result.get('verified') else 'blocked',dict(prediction_verified=bool(result.get('verified'))))
if __name__=='__main__':main()
