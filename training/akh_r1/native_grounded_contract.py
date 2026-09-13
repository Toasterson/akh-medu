"""Observed graph/query compiler; no traversal, answer selection or oracle access."""
import copy,re
from sourced_native_rule_contract import digest,label
from owned_revision_host import Host
from public_owned_revision_effect import RevisionEffectEpisode
VARIABLE=re.compile(r'\?[A-Za-z][A-Za-z0-9_]{0,31}\Z')
PREDICATES={'edge','active','route','approved'}
def compile_graph(document):
 if not isinstance(document,dict) or set(document)!={'facts','clauses'}:raise ValueError('Only observed facts and public clauses accepted')
 facts=document['facts'];clauses=document['clauses']
 if not isinstance(facts,list) or not 1<=len(facts)<=32 or not isinstance(clauses,list) or not 1<=len(clauses)<=8:raise ValueError('Native graph/query bounds')
 triples=[];keys=set()
 for fact in facts:
  if not isinstance(fact,list) or len(fact)!=3:raise ValueError('Triple required')
  s,p,o=map(label,fact)
  if p not in PREDICATES:raise ValueError('Unsupported source predicate')
  if (s,p,o) in keys:raise ValueError('Duplicate source edge')
  keys.add((s,p,o));triples.append(dict(subject=s,predicate=p,object=o))
 patterns=[];variables=set()
 for clause in clauses:
  if not isinstance(clause,list) or len(clause)!=3:raise ValueError('Clause triple required')
  s,p,o=clause
  if p not in PREDICATES:raise ValueError('Unsupported query predicate')
  for value in (s,o):
   if not isinstance(value,str):raise ValueError('Query token required')
   if value.startswith('?'):
    if not VARIABLE.fullmatch(value):raise ValueError('Unsafe variable')
    variables.add(value)
   else:label(value)
  patterns.append('('+s+' '+p+' '+o+')')
 if '?answer' not in variables:raise ValueError('Unbound answer head')
 return dict(observed_facts=triples,native_rules=[dict(name='observed-grounded-query',antecedents=patterns,consequents=['(?answer supported true)'])],assistance='Authored safe query compiler; native conjunction performs selection')

class GroundedContract:
 def __init__(self):self.receipts={}
 def observe(self,receipt,document):
  if not isinstance(receipt,str) or not receipt or receipt in self.receipts:raise ValueError('Unique observed receipt required')
  plan=compile_graph(document);self.receipts[receipt]=(copy.deepcopy(document),plan)
 def request(self,receipt,sha):
  if receipt not in self.receipts:raise ValueError('Unobserved source')
  document,plan=self.receipts[receipt]
  if digest(document)!=sha:raise ValueError('Source/query fingerprint mismatch')
  return copy.deepcopy(plan)

class GroundedHost(Host):
 def __init__(self,graph,directory,identity):super().__init__(graph,directory,identity,GroundedContract)

class GroundedEffectEpisode(RevisionEffectEpisode):
 def __init__(self,host,document,terminal,**kwargs):
  super().__init__(host,document,terminal,document_validator=compile_graph,**kwargs)
 def start(self):
  result=super().start();result['goal']={'task':'Repair only the uniquely supported target of the observed conjunctive query'};result['assistance']='Authored graph/query schema; native conjunctive rule, exact-owned derivation and sourced command contract';return result
