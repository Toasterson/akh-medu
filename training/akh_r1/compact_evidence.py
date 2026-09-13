"""Lossless receipt deduplication plus an explicitly assisted relation index.

Only returned tool receipts enter this view. There is no world, goal, target
selection, recommended action, inferred repair status or cross-session cache.
"""
import copy
import csv
import io
import json


class CompactEvidence:
    def __init__(self):
        self.actions={};self.responses={};self.receipts={};self.timeline=[];self.relations={}
        self.action_ids={};self.response_ids={}

    @staticmethod
    def intern(value,values,identities,prefix):
        key=json.dumps(value,sort_keys=True)
        if key not in identities:
            alias=f'{prefix}{len(values)}';identities[key]=alias;values[alias]=copy.deepcopy(value)
        return identities[key]

    def observe(self,request,response):
        if not isinstance(response.get('evidence_id'),str):raise ValueError('Receipt identity required')
        alias=f'r{len(self.timeline)}'
        self.receipts[alias]=response['evidence_id']
        action=self.intern(request,self.actions,self.action_ids,'a')
        payload={k:v for k,v in response.items() if k!='evidence_id'}
        value=self.intern(payload,self.responses,self.response_ids,'v')
        self.timeline.append(dict(receipt=alias,action=action,response=value))
        if not response.get('success'):return
        stdout=response.get('stdout','')
        try:
            obj=json.loads(stdout);rows=[obj] if isinstance(obj,dict) else []
        except ValueError:
            rows=[]
            for delimiter in ['\t',',',':']:
                reader=csv.DictReader(io.StringIO(stdout),delimiter=delimiter)
                if reader.fieldnames and {'job','worker'}<=set(reader.fieldnames):
                    rows=list(reader);break
        for row in rows:
            if not isinstance(row.get('job'),str) and not isinstance(row.get('worker'),str):continue
            # Preserve literal values/types. Timestamps remain in the exact raw
            # receipt; identical relation rows may have several supporting reads.
            fields={k:v for k,v in row.items() if k!='observed_at'}
            key=json.dumps(fields,sort_keys=True)
            if key not in self.relations:self.relations[key]=dict(fields=copy.deepcopy(fields),receipts=[])
            self.relations[key]['receipts'].append(alias)

    def view(self):
        return copy.deepcopy(dict(schema='akh-observed-evidence-v1',
            interpretation='Receipt order is chronological. Relation rows are literal observations; conflicting or superseded rows remain visible with their sources. Select and verify the next action yourself.',
            relation_rows=list(self.relations.values()),actions=self.actions,responses=self.responses,
            receipt_ids=self.receipts,timeline=self.timeline))

    @staticmethod
    def reconstruct(view):
        return [dict(request=copy.deepcopy(view['actions'][t['action']]),
                     response=dict(copy.deepcopy(view['responses'][t['response']]),evidence_id=view['receipt_ids'][t['receipt']]))
                for t in view['timeline']]


def compact_messages(messages,state,menu):
    # Preserve every operator message verbatim, including corrections. The tool
    # conversation is represented by the fully reconstructible receipt view.
    instructions=[copy.deepcopy(m) for m in messages if m['role'] in {'system','user'}]
    return instructions+[dict(role='user',content='Observed tool evidence, deduplicated with a relation index:\n'+json.dumps(state.view())),menu]
