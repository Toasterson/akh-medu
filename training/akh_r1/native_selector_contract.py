"""Compile observed selectors to native joins. Never evaluates a selected answer."""
import copy
import hashlib
from sourced_native_rule_contract import digest


def exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError('Unsupported or private fields')


def token(value):
    if not isinstance(value, str) or not value or len(value)>256:
        raise ValueError('Bounded string required')
    return 'v'+hashlib.sha256(value.encode()).hexdigest()[:32]


def version(value):
    if type(value) is not int or not 0<=value<=100000:
        raise ValueError('Bounded version required')
    return 'version-'+str(value)


def compile_selector(family, split, document, goal, witness=None):
    facts=[];live=[];rules=[];symbols={}
    def val(x):
        t=token(x);symbols[t]=x;return t
    def add(s,p,o):facts.append(dict(subject=s,predicate=p,object=o))
    def rule(a,c):
        if not 1<=len(a)<=8:raise ValueError('Native antecedent bound')
        rules.append(dict(name='selector-'+str(len(rules)),antecedents=a,consequents=[c]))
    def rows(items):
        if not isinstance(items,list) or len(items)>16:raise ValueError('Record bound')
        return items
    if (family,split)==('acquisition','development'):
        exact(document,['records']);exact(goal,['task','constraints'])
        constraints=goal['constraints']
        if not isinstance(constraints,dict) or not constraints or not set(constraints)<= {'site','rack','role','name'}:raise ValueError('Constraint schema')
        for i,r in enumerate(rows(document['records'])):
            exact(r,['entity','name','site','rack','role']);node='row-'+str(i)
            add(node,'entity',val(r['entity']))
            for field in ['name','site','rack','role']:add(node,field,val(r[field]))
        rule(['(?r entity ?x)']+['(?r '+k+' '+val(v)+')' for k,v in sorted(constraints.items())],'(?x selected yes)')
    elif (family,split)==('persistent_recall','development'):
        exact(document,['records']);exact(goal,['task','required'])
        required=goal['required']
        if not isinstance(required,list) or not 1<=len(required)<=7 or len(set(required))!=len(required):raise ValueError('Capability bound')
        for i,r in enumerate(rows(document['records'])):
            exact(r,['id','action','capabilities']);node='row-'+str(i);add(node,'skill',val(r['id']));add(node,'action',val(r['action']))
            for cap in rows(r['capabilities']):add(node,'capability',val(cap))
        rule(['(?r skill ?x)']+['(?r capability '+val(c)+')' for c in required],'(?x selected yes)')
    elif (family,split)==('persistent_recall','validation'):
        exact(document,['records']);exact(goal,['task','scope'])
        for i,r in enumerate(rows(document['records'])):
            exact(r,['id','action','scope','version']);node='row-'+str(i)
            for p,v in [('skill',val(r['id'])),('action',val(r['action'])),('scope',val(r['scope'])),('version',version(r['version']))]:add(node,p,v)
        if witness is not None:
            exact(witness,['scope','version'])
            if witness['scope']!=goal['scope']:raise ValueError('Witness scope mismatch')
            live=[dict(subject='current',predicate='scope',object=val(witness['scope'])),dict(subject='current',predicate='version',object=version(witness['version']))]
        rule(['(?r skill ?x)','(?r scope '+val(goal['scope'])+')','(?r version ?v)','(current scope '+val(goal['scope'])+')','(current version ?v)'],'(?x selected yes)')
    elif (family,split)==('acquisition','validation'):
        if not isinstance(document,dict) or not set(document)<= {'inventory','events'} or 'events' not in document:raise ValueError('Event schema')
        exact(goal,['task','identity'])
        for r in rows(document.get('inventory',[])):
            exact(r,['identity','name']);identity=val(r['identity']);add(identity,'known','yes');add('event-'+identity+'-0','reachable','yes')
        seen=set();counts={}
        for i,r in enumerate(rows(document['events'])):
            exact(r,['identity','location','version','predecessor']);identity=val(r['identity']);v=r['version'];prev=r['predecessor'];version(v);version(prev)
            if not 0<=prev<v:raise ValueError('Non-advancing event')
            if (identity,v) in seen:raise ValueError('Conflicting event identity/version')
            seen.add((identity,v));counts[identity]=counts.get(identity,0)+1
            if counts[identity]>3:raise ValueError('Four-iteration native chain bound')
            node='event-'+identity+'-'+str(v)
            for p,o in [('identity',identity),('location',val(r['location'])),('version',version(v)),('predecessor','event-'+identity+'-'+str(prev))]:add(node,p,o)
        if witness is not None:
            exact(witness,['identity','version'])
            if witness['identity']!=goal['identity']:raise ValueError('Witness identity mismatch')
            live=[dict(subject='current',predicate='identity',object=val(witness['identity'])),dict(subject='current',predicate='version',object=version(witness['version']))]
        rule(['(?e predecessor ?p)','(?p reachable yes)'],'(?e reachable yes)')
        rule(['(?e reachable yes)','(?e identity '+val(goal['identity'])+')','(?e version ?v)','(?e location ?x)','(current identity '+val(goal['identity'])+')','(current version ?v)'],'(?x selected yes)')
    else:raise ValueError('Unsupported family/split')
    if family=='persistent_recall':
        # Qualify the exact record, not any other record sharing the skill ID.
        rules[0]['consequents'].append('(?r selector-qualified yes)')
        rule(['(?r selector-qualified yes)','(?r skill ?x)','(?r action ?a)'],'(?x selected-action ?a)')
    if witness is not None and split=='development':raise ValueError('Unexpected witness')
    return dict(observed_facts=facts,witness_facts=live,native_rules=rules,symbols=symbols,
                result_predicate='selected',required_unique_results=1,execution_ready=False,
                assistance='Authored schema/rule compilation; native joins; explicit unique-result guard',
                source_update_policy='Replace supplied field groups; preserve omitted inventory; never retain old witness',
                blockers=['Bind field-group ownership and current witness to native host','Execute native selectors and audit missing/conflicting outcomes'])


class Binding:
    def __init__(self):self.observed={}
    def observe(self,receipt,value):
        if not isinstance(receipt,str) or not receipt or receipt in self.observed:raise ValueError('Unique receipt')
        self.observed[receipt]=copy.deepcopy(value)
    def request(self,family,split,source_receipt,source_hash,goal,witness_receipt=None,witness_hash=None):
        def read(receipt,h):
            if receipt not in self.observed or digest(self.observed[receipt])!=h:raise ValueError('Receipt/hash mismatch')
            return self.observed[receipt]
        document=read(source_receipt,source_hash)
        witness=None if witness_receipt is None else read(witness_receipt,witness_hash)
        return dict(source_receipt=source_receipt,source_hash=source_hash,witness_receipt=witness_receipt,
                    witness_hash=witness_hash,**compile_selector(family,split,document,goal,witness))


class TwoSourceAdmission:
    """Host-owned admission; no effect authority and no legacy route changes.

    The transport must call observe_witness only for actual reads in this
    session. Retained snapshots may carry old source receipts, never witnesses.
    A graph consumer must call commit only after installing the complete batch.
    """
    def __init__(self, retained=None):
        self.sources = copy.deepcopy(retained or {})
        self.witnesses = {}
        self.pending = None
        self.admitted = None
        self.generation = 0

    def invalidate(self):
        self.generation += 1
        self.pending = None
        self.admitted = None

    def observe_source(self, role, receipt, envelope):
        self.invalidate()
        if role not in ('inventory', 'movement'):
            raise ValueError('Unsupported source role')
        token(receipt)
        exact(envelope, ['source_id', 'version', 'records'])
        token(envelope['source_id']); version(envelope['version'])
        self.sources[role] = dict(receipt=receipt, value=copy.deepcopy(envelope))
        self.witnesses.pop(role, None)

    def observe_witness(self, role, receipt, witness):
        self.invalidate()
        if role not in ('inventory', 'movement'):
            raise ValueError('Unsupported witness role')
        self.witnesses.pop(role, None)
        token(receipt)
        exact(witness, ['source_id', 'version', 'content_sha256'])
        self.witnesses[role] = dict(receipt=receipt, value=copy.deepcopy(witness))

    def prepare(self, goal):
        self.invalidate()
        exact(goal, ['task', 'identity'])
        token(goal['task']); token(goal['identity'])
        lineage = {}; documents = {}; receipts = set(); ids = set()
        for role in ('inventory', 'movement'):
            source = self.sources.get(role); witness = self.witnesses.get(role)
            if source is None or witness is None:
                raise ValueError('Both sources and session witnesses required')
            exact(source, ['receipt', 'value']); exact(witness, ['receipt', 'value'])
            envelope = source['value']; live = witness['value']
            exact(envelope, ['source_id', 'version', 'records'])
            exact(live, ['source_id', 'version', 'content_sha256'])
            token(envelope['source_id']); version(envelope['version'])
            version(live['version'])
            if envelope['source_id'] in ids:
                raise ValueError('Independent source identities required')
            ids.add(envelope['source_id'])
            h = digest(envelope)
            if live != dict(source_id=envelope['source_id'],
                            version=envelope['version'], content_sha256=h):
                raise ValueError('Stale or mismatched source witness')
            for receipt in (source['receipt'], witness['receipt']):
                token(receipt)
                if receipt in receipts:
                    raise ValueError('Distinct observation receipts required')
                receipts.add(receipt)
            rows = envelope['records']
            if not isinstance(rows, list) or not 1 <= len(rows) <= 16:
                raise ValueError('Bounded nonempty records required')
            seen = set()
            for row in rows:
                exact(row, ['identity', 'name'] if role == 'inventory'
                      else ['identity', 'location'])
                for value in row.values(): token(value)
                if row['identity'] in seen:
                    raise ValueError('Duplicate or conflicting identity')
                seen.add(row['identity'])
            if goal['identity'] not in seen:
                raise ValueError('Requested identity lacks support')
            documents[role] = rows
            lineage[role] = dict(source_receipt=source['receipt'], source_hash=h,
                                 witness_receipt=witness['receipt'],
                                 witness_hash=digest(live))
        # Compile an identity join, not an examiner-supplied answer.
        facts = []; symbols = {}
        def val(value):
            symbol = token(value); symbols[symbol] = value; return symbol
        for row in documents['inventory']:
            facts.append(dict(subject=val(row['identity']), predicate='inventory-known', object='yes'))
        for row in documents['movement']:
            facts.append(dict(subject=val(row['identity']), predicate='movement-location', object=val(row['location'])))
        identity = val(goal['identity'])
        batch = dict(generation=self.generation, lineage=lineage,
                     observed_facts=facts, symbols=symbols,
                     native_rules=[dict(name='two-source-identity',
                       antecedents=['('+identity+' inventory-known yes)',
                                    '('+identity+' movement-location ?x)'],
                       consequents=['(?x selected yes)'])],
                     required_unique_results=1, result_predicate='selected',
                     permission_granted=False, execution_ready=False)
        self.pending = copy.deepcopy(batch)
        return copy.deepcopy(batch)

    def commit(self, generation):
        if self.pending is None or type(generation) is not int or generation != self.generation:
            self.invalidate()
            raise ValueError('Admission invalidated')
        self.admitted = self.pending
        self.pending = None
        return copy.deepcopy(self.admitted)

    def graph_update_failed(self):
        self.invalidate()

    def retained_sources(self):
        return copy.deepcopy(self.sources)
