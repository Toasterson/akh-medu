"""Observed field-group ownership and fresh witnesses for native selectors."""
import copy
from native_selector_contract import compile_selector,exact
from owned_revision_host import Host
from sourced_native_rule_contract import digest

class SelectorContract:
    def __init__(self):self.observed={}
    def observe(self,receipt,envelope):
        exact(envelope,['family','split','groups','goal','witness','source_owners','witness_owner'])
        if receipt in self.observed:raise ValueError('Receipt replay')
        compile_selector(envelope['family'],envelope['split'],envelope['groups'],envelope['goal'],envelope['witness'])
        self.observed[receipt]=copy.deepcopy(envelope)
    def request(self,receipt,sha):
        e=self.observed[receipt]
        if digest(e)!=sha:raise ValueError('Envelope hash mismatch')
        p=compile_selector(e['family'],e['split'],e['groups'],e['goal'],e['witness'])
        p['observed_facts']+=p['witness_facts']
        # Exact duplicate triples are one owned native edge.
        p['observed_facts']=list({tuple(sorted(f.items())):f for f in p['observed_facts']}.values())
        if len(p['observed_facts'])>32:raise ValueError('Owned native source bound exceeded')
        return p

class SelectorHost(Host):
    def __init__(self,graph,directory,identity,family,split,goal):
        super().__init__(graph,directory,identity,SelectorContract)
        context=dict(family=family,split=split,goal=goal)
        if self.state.get('selector_context',context)!=context:
            self.close();raise ValueError('Selector context mismatch')
        self.state['selector_context']=copy.deepcopy(context)
        self.state.setdefault('selector_observations',{})
        self.state.setdefault('used_witnesses',[])
        self.save()
    def admit(self,receipt,value):
        if not isinstance(receipt,str) or not receipt or receipt in self.state['selector_observations']:raise ValueError('Unique observed receipt required')
        self.state['selector_observations'][receipt]=copy.deepcopy(value);self.save()
    def observed(self,receipt,sha):
        if receipt not in self.state['selector_observations']:raise ValueError('Unobserved source')
        v=self.state['selector_observations'][receipt]
        if digest(v)!=sha:raise ValueError('Observed hash mismatch')
        return copy.deepcopy(v)
    def update(self,source_receipt,source_hash,previous,*,witness_receipt=None,witness_hash=None):
        if self.state['pending']:raise RuntimeError('Pending native update')
        current=self.state['current']
        expected=None if current is None else dict(receipt=current['receipt'],sha256=current['sha256'])
        if previous!=expected:raise ValueError('Stale group ownership witness')
        delta=self.observed(source_receipt,source_hash)
        if not isinstance(delta,dict):raise ValueError('Field groups required')
        context=self.state['selector_context']
        old=None if current is None else self.state['observations'][current['receipt']]
        groups={} if old is None else copy.deepcopy(old['groups'])
        owners={} if old is None else copy.deepcopy(old['source_owners'])
        if old and source_receipt in [o['receipt'] for o in owners.values()]:raise ValueError('Source receipt replay')
        # Only event updates may retain an omitted inventory group. Other
        # families require their complete record group on every replacement.
        if context['family']!='acquisition' or context['split']!='validation':groups={};owners={}
        for field,value in delta.items():
            groups[field]=value;owners[field]=dict(receipt=source_receipt,sha256=source_hash)
        witness=None;witness_owner=None
        if witness_receipt is not None:
            if witness_receipt in self.state['used_witnesses']:raise ValueError('Stale witness receipt')
            witness=self.observed(witness_receipt,witness_hash);witness_owner=dict(receipt=witness_receipt,sha256=witness_hash)
        elif witness_hash is not None:raise ValueError('Witness receipt required')
        envelope=dict(**context,groups=groups,source_owners=owners,witness=witness,witness_owner=witness_owner)
        receipt='selector-'+digest(envelope)
        self.observe(receipt,envelope)
        # Consume before native I/O; failed operations must not reuse witnesses.
        if witness_receipt:self.state['used_witnesses'].append(witness_receipt);self.save()
        result=self.apply(receipt,digest(envelope),previous)
        return self.result(result,envelope)

    def handle(self):
        current=self.state['current']
        return None if current is None else dict(receipt=current['receipt'],sha256=current['sha256'])

    def reuse(self,handle,*,witness_receipt=None,witness_hash=None):
        if self.state['pending']:raise RuntimeError('Pending native update')
        if handle is None or handle!=self.handle():raise ValueError('Stale retained source handle')
        current=self.state['current']
        envelope=copy.deepcopy(self.state['observations'][current['receipt']])
        envelope['witness']=None;envelope['witness_owner']=None
        if witness_receipt is not None:
            if witness_receipt in self.state['used_witnesses']:raise ValueError('Stale witness receipt')
            envelope['witness']=self.observed(witness_receipt,witness_hash)
            envelope['witness_owner']=dict(receipt=witness_receipt,sha256=witness_hash)
        elif witness_hash is not None:raise ValueError('Witness receipt required')
        # A new computation receipt is not a new source observation. The
        # envelope preserves original group owners and observation history.
        import uuid
        receipt='reuse-'+uuid.uuid4().hex
        self.observe(receipt,envelope)
        if witness_receipt:self.state['used_witnesses'].append(witness_receipt);self.save()
        result=self.apply(receipt,digest(envelope),handle)
        return self.result(result,envelope)

    def result(self,result,envelope):
        rows=self.snapshot()
        plan=self.contract.request(result['receipt'],digest(envelope))
        owned={tuple(r['labels'][x] for x in ('subject','predicate','object')) for r in result['derived']}
        selected={k[0] for k in rows if k[1:]==('selected','yes')}
        trusted=all((x,'selected','yes') in owned and x in plan['symbols'] for x in selected)
        return dict(status='unique' if trusted and len(selected)==1 else 'blocked',
                    selected=[plan['symbols'][x] for x in selected] if trusted and len(selected)==1 else [],
                    current=dict(receipt=result['receipt'],sha256=result['sha256']))
