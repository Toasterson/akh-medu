"""Candidate-facing tool boundary. Private expectations never enter responses.

The caller owns world setup/scoring. Candidates receive only start() and call()
results over JSONL. Structured-document facts are an explicit assistance level.
"""
import copy
import json
import uuid
from safe_pipeline import validate_pipeline


class PublicEpisode:
    def __init__(self, graph, goal, documents, terminal=None, max_calls=32, interaction=None):
        self.graph=graph;self.goal=copy.deepcopy(goal);self.documents=copy.deepcopy(documents)
        self.terminal=terminal;self.max_calls=max_calls;self.calls=0;self.events=[];self.sources={}
        self.interaction=interaction
        self.session=uuid.uuid4().hex;self.finished=None

    def start(self):
        return dict(schema='akh-public-episode-v1',goal=self.goal,
                    tools=['list_documents','read_document','remember','revise','query','terminal','finish']+(['interact'] if self.interaction else []),
                    max_calls=self.max_calls)

    def _facts(self, subject):
        value=self.graph.call('exam_export')
        if not value.get('success'):raise RuntimeError('Graph unavailable')
        return [dict(subject=t['subject_label'],predicate=t['predicate_label'],object=t['object_label'])
                for t in value['triples'] if t['subject_label']==subject and not t['predicate_label'].startswith('r0_')]

    def call(self, request):
        if self.finished is not None:raise ValueError('Episode already finished')
        self.calls+=1
        if self.calls>self.max_calls:raise ValueError('Tool budget exceeded')
        identity='public-receipt-'+self.session+'-'+str(self.calls)
        try:
            if not isinstance(request,dict) or set(request)!={'tool','arguments'}:raise ValueError('Expected tool and arguments only')
            name,args=request['tool'],request['arguments']
            if not isinstance(args,dict):raise ValueError('Arguments must be an object')
            if name=='list_documents':
                self.keys(args,set());response=dict(documents=sorted(self.documents))
            elif name=='read_document':
                self.keys(args,{'id'})
                if args['id'] not in self.documents:raise ValueError('Document unavailable')
                response=dict(document=copy.deepcopy(self.documents[args['id']]))
                self.sources[identity]=copy.deepcopy(response['document'].get('facts',[]))
            elif name=='query':
                self.keys(args,{'subject'});response=dict(facts=self._facts(args['subject']))
            elif name in ('remember','revise'):
                self.keys(args,{'source','fact'})
                fact=args['fact'];source=args['source']
                if source not in self.sources or fact not in self.sources[source]:raise ValueError('Fact lacks an exact previously observed structured source')
                if set(fact)!={'subject','predicate','object'} or not all(isinstance(v,str) for v in fact.values()):raise ValueError('Invalid fact')
                if fact['predicate'].startswith('r0_'):raise ValueError('Reserved predicate')
                self.validate_fact_write(fact)
                if name=='revise':
                    for old in self._facts(fact['subject']):
                        if old['predicate']==fact['predicate'] and old!=fact:
                            history=dict(subject=fact['subject'],predicate='r0_revision_history',object=json.dumps(dict(old=old,new=fact,source=source),sort_keys=True))
                            self.checked_write([history],source)
                            result=self.graph.call('kg_delete',old)
                            if not result.get('success'):raise RuntimeError('Graph retraction failed')
                self.checked_write([fact],source);response=dict(stored=True)
            elif name=='terminal':
                self.keys(args,{'stages'})
                if self.terminal is None:raise ValueError('Terminal unavailable in this world')
                stages=args['stages']
                validate_pipeline(stages,self.goal.get('command'),32 if self.goal.get('command_profile')=='conditional-v1' else 4)
                response=self.terminal(args['stages'])
                # Only actual successful help output may supply command facts.
                if response.get('success') and len(args['stages'])==1 and args['stages'][0][1:]==['--help']:
                    value=json.loads(response['stdout']);self.sources[identity]=value.get('facts',[])
                    if self.goal.get('command_profile')=='conditional-v1':
                        facts=[dict(subject='sana-skill',predicate='document',object=json.dumps(value,sort_keys=True))]
                        self.sources[identity]=facts;response['observed_facts']=facts
            elif name=='interact':
                if self.interaction is None:raise ValueError('Interaction unavailable')
                self.keys(args,{'action','value'})
                response=self.interaction(args['action'],args['value'])
                if self.goal.get('interaction_profile')=='ownership-native-v1' and response.get('success') and args['action'] in ('inquire_rights','inquire','inspect_feasibility'):
                    subject='ownership-'+(str(args['value']) if args['action']=='inquire' else args['action'])
                    observed={k:v for k,v in response.items() if k in ('epoch','source','holders','options','participant')}
                    facts=[dict(subject=subject,predicate='observation',object=json.dumps(observed,sort_keys=True))]
                    self.sources[identity]=facts;response['observed_facts']=facts
            elif name=='finish':
                self.keys(args,{'status','answer'})
                if args['status'] not in ('complete','blocked'):raise ValueError('Invalid terminal status')
                self.finished=copy.deepcopy(args);response=dict(finished=True)
            else:raise ValueError('Tool not exposed')
            response=dict(success=True,**response) if 'success' not in response else response
        except (ValueError,KeyError,TypeError) as error:
            response=dict(success=False,error=str(error))
        response['evidence_id']=identity
        self.events.append(dict(request=copy.deepcopy(request),response=copy.deepcopy(response)))
        self.checked_write([dict(subject=identity,predicate='r0_public_receipt',object=json.dumps(self.events[-1],sort_keys=True))],identity)
        return response

    def validate_fact_write(self,fact):
        pass

    @staticmethod
    def keys(args, expected):
        if set(args)!=expected:raise ValueError('Unexpected or missing arguments')

    def checked_write(self,facts,source):
        r=self.graph.call('memory_write',dict(summary='R0 public evidence',facts=facts,source=source))
        if not r.get('success'):raise RuntimeError('Graph write failed')
