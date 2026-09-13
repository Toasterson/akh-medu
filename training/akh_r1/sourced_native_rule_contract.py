"""Receipt-bound compiler into existing native rules; performs no inference."""
import copy,re,hashlib,json
LABEL=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,63}\Z')
def label(x):
 if not isinstance(x,str) or not LABEL.fullmatch(x):raise ValueError('Unsupported native label')
 return x
class Contract:
 def __init__(self):self.receipts={}
 def observe(self,receipt,document):
  if not isinstance(receipt,str) or not receipt or receipt in self.receipts:raise ValueError('Unique observation receipt required')
  plan=compile_document(document)
  self.receipts[receipt]=(copy.deepcopy(document),plan)
 def request(self,receipt,document_hash):
  if receipt not in self.receipts:raise ValueError('Unknown observed source')
  document,plan=self.receipts[receipt]
  if digest(document)!=document_hash:raise ValueError('Source identity mismatch')
  return dict(source_receipt=receipt,source_sha256=document_hash,**copy.deepcopy(plan))
def digest(document):return hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest()
def compile_document(d):
 if not isinstance(d,dict) or set(d)!={'supports','rules'}:raise ValueError('Only observed supports and rules accepted')
 if not 1<=len(d['supports'])<=16 or not 1<=len(d['rules'])<=16:raise ValueError('Native bounds')
 supports={};facts=[]
 for s in d['supports']:
  if set(s)!={'id','source','active'} or type(s['active']) is not bool:raise ValueError('Malformed support')
  name=label(s['id']);source=label(s['source'])
  if name in supports:raise ValueError('Duplicate support')
  supports[name]=s
  facts.extend([dict(subject=name,predicate='support-active',object='true' if s['active'] else 'false'),dict(subject=name,predicate='support-source',object=source)])
 rules=[]
 for i,r in enumerate(d['rules']):
  if set(r)!={'target','requires'}:raise ValueError('Malformed rule')
  target=label(r['target']);requires=r['requires']
  if not isinstance(requires,list) or not 1<=len(requires)<=8 or len(set(requires))!=len(requires):raise ValueError('Native prerequisite bound')
  if any(s not in supports for s in requires):raise ValueError('Unsupported prerequisite')
  rules.append(dict(name='observed-support-rule-'+str(i),antecedents=['('+label(s)+' support-active true)' for s in requires],consequents=['('+target+' supported true)']))
 return dict(observed_facts=facts,native_rules=rules,semantics='Conjunction within rule; alternative rules per target',execution_ready=False,required_dispatch=['Use receipt-authorized facts only in isolated native graph','Retract changed prior support triples through native remove_triple before applying current facts','Run admitted rules through host wrapper; do not expose examiner entry point','Return source-bound derivation and retraction receipts separately from observations'])
