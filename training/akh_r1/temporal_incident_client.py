#!/usr/bin/python3
import json,sys
def call(tool,**arguments):
    print(json.dumps(dict(tool=tool,arguments=arguments)),flush=True)
    return json.loads(sys.stdin.readline())
def main():
    goal=json.loads(sys.stdin.readline())['goal']
    for name in call('list_documents')['documents']:
        r=call('read_document',id=name)
        for fact in r['document']['facts']:
            if not call('revise',source=r['evidence_id'],fact=fact)['success']:
                return call('finish',status='blocked',answer='Source rejected')
    if goal['phase']=='acquire':return call('finish',status='complete',answer='Recorded permission')
    state=call('interact',action='permission',value='repair')
    if not state.get('holding') and goal['phase']!='stale-attempt':
        return call('finish',status='blocked',answer='Permission not current')
    result=call('terminal',stages=[['sana','apply','nodealpha']])
    if not result['success']:return call('finish',status='blocked',answer='Authority guard rejected')
    observed=call('terminal',stages=[['sana','inspect','nodealpha']])
    good=observed['success'] and json.loads(observed['stdout'])['healthy']
    call('finish',status='complete' if good else 'blocked',answer='Fresh effect checked')
if __name__=='__main__':main()
