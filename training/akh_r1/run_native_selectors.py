"""Four retained host/native selectors; private oracle never enters host input."""
import json
import hashlib
from pathlib import Path
from environment import Graph
from selector_owned_host import SelectorHost
from sourced_native_rule_contract import digest
from native_selector_contract import token

class BoundedGraph(Graph):
    launched=0
    def __init__(self,*args):
        if type(self).launched>=8:raise RuntimeError('Session cap')
        type(self).launched+=1;self.calls=0
        super().__init__(*args)
    def call(self,*args,**kw):
        if self.calls>=16:raise RuntimeError('Phase call cap')
        self.calls+=1
        return super().call(*args,**kw)

def check_targets(graph,expected):
    result=graph.call('exam_owned_snapshot',{})
    if result.get('success') is not True:raise RuntimeError('Snapshot failure')
    actual=sorted({r['labels']['subject'] for r in result['rows'] if r['labels']['predicate']=='selected' and r['labels']['object']=='yes'})
    if actual!=sorted(token(x) for x in expected):raise AssertionError('Independent native target mismatch')
    return actual

def run(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    for p,h in manifest['fingerprints'].items():
        if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise ValueError('Changed dependency '+p)
    output=root/'execution';output.mkdir(exist_ok=False);results=[]
    for case in json.loads((root/'inputs.json').read_text()):
        directory=output/case['id'];directory.mkdir();phases=[]
        for phase in ['acquire','correction']:
            graph=BoundedGraph(manifest['binary'],directory/'graph',directory/'native.log');host=None
            try:
                host=SelectorHost(graph,directory/'host',case['id'],case['family'],case['split'],case['goal'])
                current=host.state['current'];previous=None if current is None else dict(receipt=current['receipt'],sha256=current['sha256'])
                if phase=='correction':
                    if current is None:raise AssertionError('Missing persisted host state')
                    check_targets(graph,case['expected']['initial'])
                doc=case['documents'][phase];receipt=case['id']+'-'+phase
                host.admit(receipt,doc);witness=case['witnesses'][phase];args={}
                if witness is not None:
                    wr=receipt+'-witness';host.admit(wr,witness);args=dict(witness_receipt=wr,witness_hash=digest(witness))
                if previous:
                    count=graph.calls
                    try:host.update(receipt,digest(doc),dict(receipt=previous['receipt'],sha256='wrong'),**args)
                    except ValueError:pass
                    else:raise AssertionError('Stale group witness admitted')
                    if graph.calls!=count:raise AssertionError('Stale request performed native I/O')
                answer=host.update(receipt,digest(doc),previous,**args)
                expected=case['expected']['initial' if phase=='acquire' else 'later']
                if answer['selected']!=expected or answer['status']!=('unique' if len(expected)==1 else 'blocked'):raise AssertionError('Host selection mismatch')
                native=check_targets(graph,expected)
                phases.append(dict(phase=phase,pid=graph.process.pid,calls=graph.calls,answer=answer,native_targets=native))
            finally:
                if host:host.close()
                graph.close()
        if phases[0]['pid']==phases[1]['pid']:raise AssertionError('No cold native boundary')
        results.append(dict(id=case['id'],phases=phases));(directory/'audit.json').write_text(json.dumps(results[-1],indent=2)+'\n')
    (output/'audit.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':
    import sys
    run(sys.argv[1])
