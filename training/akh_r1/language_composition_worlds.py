"""Counterbalanced finite mechanisms; lexical variants carry no world credit."""
NODES=['nodealpha','nodebeta','nodegamma']
DESCRIPTORS={'development':{'mechanism':'region membership intersected with explicit exclusion','languages':['en','de'],'required_sources':['region','exclusion in instruction']},'validation':{'mechanism':'team to service to host join intersected with readiness','languages':['fr','pt'],'required_sources':['team-service','service-host','ready']}}
def fact(s,p,o):return dict(subject=s,predicate=p,object=o)
def worlds():
    rows=[]
    for split,descriptor in DESCRIPTORS.items():
        for lang in descriptor['languages']:
            for target in NODES[:2]:
                other=NODES[1] if target==NODES[0] else NODES[0]
                for control in ['complete','missing-route','missing-filter']:
                    if split=='development':
                        instruction={'en':f'Repair a node in east except {other}.','de':f'Repariere einen Knoten in east außer {other}.'}[lang]
                        facts=[fact(target,'region','east'),fact(other,'region','east'),fact('nodegamma','region','west')]
                        if control=='missing-route':facts=[]
                        if control=='missing-filter':instruction=instruction.replace(other,'unknown')
                    else:
                        instruction={'fr':'Répare un nœud prêt du service de blue.','pt':'Repare um nó pronto do serviço de blue.'}[lang]
                        facts=[fact('blue','service','api'),fact('api','host',target),fact('api','host',other),fact(target,'ready','true'),fact('nodegamma','ready','true')]
                        if control=='missing-route':facts=[f for f in facts if f['predicate']!='service']
                        if control=='missing-filter':facts=[f for f in facts if f['predicate']!='ready']
                    rows.append(dict(id=f'{split}-{lang}-{target}-{control}',split=split,language=lang,control=control,goal=dict(command='sana',instruction=instruction),documents={'observations':dict(facts=facts)},expected_target=target,expected_effect=control=='complete'))
    return rows

LANGUAGES=('en','de','fr','pt')
def generate_language_descriptor(split,seed):
 """One structural descriptor with correlated languages and answerability controls."""
 import random
 if split not in ('development','validation'):raise ValueError('Unknown split')
 rng=random.Random(seed);nodes=['node'+str(i) for i in range(rng.randint(4,6))];rng.shuffle(nodes);first,later,excluded,extra=nodes[:4]
 if split=='development':
  texts={'en':f'Repair exactly one ready node in east except {excluded}.','de':f'Repariere genau einen bereiten Knoten in east außer {excluded}.','fr':f'Répare exactement un nœud prêt dans east sauf {excluded}.','pt':f'Repare exatamente um nó pronto em east exceto {excluded}.'}
  base=[fact(n,'region','east' if n!=extra else 'west') for n in nodes]
 else:
  texts={'en':f'Repair exactly one ready host of team blue except {excluded}.','de':f'Repariere genau einen bereiten Host des Teams blue außer {excluded}.','fr':f'Répare exactement un hôte prêt de l’équipe blue sauf {excluded}.','pt':f'Repare exatamente um host pronto da equipe blue exceto {excluded}.'}
  base=[fact('blue','service','api')]+[fact('api','host',n) for n in nodes if n!=extra]
 variants=[]
 for stratum in ('unique','missing','conflicting'):
  initial_ready=[first] if stratum=='unique' else [] if stratum=='missing' else [first,later]
  later_ready=[later] if stratum=='unique' else [] if stratum=='missing' else [first,later]
  # Expected sets follow the sampled role construction, independently of the language parser.
  variants.append(dict(stratum=stratum,initial_facts=base+[fact(n,'ready','true') for n in initial_ready+[excluded,extra]],later_facts=base+[fact(n,'ready','true') for n in later_ready+[excluded,extra]],expected_initial=sorted(initial_ready),expected_later=sorted(later_ready)))
 return dict(schema='akh-language-breadth-v1',split=split,seed=seed,mechanism='exclusion_quantifier' if split=='development' else 'relation_exception',public_start=dict(command='sana',instructions=texts),examiner=dict(nodes=nodes,variants=variants,later_trigger='after_initial_verification_or_abstention'),languages=list(LANGUAGES),assistance=['Authored multilingual grammar','Structured fact observations','Exactly-one answer contract'],qualified=False,new_world_credit=0)
