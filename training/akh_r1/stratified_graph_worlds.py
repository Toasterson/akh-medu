"""Balanced constructed strata; randomized structure, no rejection sampling."""
import random
STRATA={'unique':1,'missing':0,'conflicting':2}
def generate(seed):
 worlds=[]
 for split in ['development','validation']:
  for index,(stratum,k) in enumerate(STRATA.items()):
   rng=random.Random(f'{seed}:{split}:{stratum}');depth=rng.randint(2,4);layers=[['root']]+[[f'l{i}n{j}' for j in range(rng.randint(2,4))] for i in range(depth-1)]+[['answer-a','answer-b']];facts=[]
   for left,right in zip(layers,layers[1:]):
    for i,b in enumerate(right):facts.append([left[i%len(left)],'edge',b])
    for a in left:
     for b in right:
      if rng.random()<0.3 and [a,'edge',b] not in facts:facts.append([a,'edge',b])
   clauses=[['root' if i==0 else f'?x{i}','edge','?answer' if i==depth-1 else f'?x{i+1}'] for i in range(depth)]
   if split=='validation':
    facts += [['policy','route','review'],['review','approved','answer-a'],['review','approved','answer-b']]
    clauses += [['policy','route','?review'],['?review','approved','?answer']]
   order=['answer-a','answer-b'];rng.shuffle(order)
   facts += [[a,'active','true' if i<k else 'false'] for i,a in enumerate(order)]
   clauses.append(['?answer','active','true'])
   initial=sorted(order[:k]);replacement=[order[0],'active','true' if k==0 else 'false'];revised=sorted([order[0]] if k==0 else order[1:k])
   worlds.append(dict(id=split+'-'+stratum,split=split,stratum=stratum,depth=depth,widths=[len(l) for l in layers],facts=facts,clauses=clauses,replacement=replacement,initial=initial,revised=revised))
 return worlds
