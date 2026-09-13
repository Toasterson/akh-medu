"""Finite authored language controls; translations are not independent worlds."""
import copy
DESCRIPTORS={
 'development':dict(mechanism='set exclusion',instruction='repair every eligible node except the named node; require one remaining target',required=['eligible set','excluded node'],languages=['en','de']),
 'validation':dict(mechanism='relational ownership join',instruction='repair the node owned by the named team; resolve ownership evidence',required=['team','ownership relation'],languages=['fr','pt'])}
PROMPTS={'en':'Repair the eligible node except nodealpha.','de':'Repariere den geeigneten Knoten außer nodealpha.','fr':"Répare le nœud de l’équipe bleue.",'pt':'Repare o nó da equipe azul.'}
def worlds():
    out=[]
    for split,descriptor in DESCRIPTORS.items():
        for language in descriptor['languages']:
            for control in ['complete','missing-evidence','unknown-instruction']:
                facts=([dict(subject=n,predicate='eligible',object='true') for n in ['nodealpha','nodebeta']] if split=='development' else [dict(subject='nodealpha',predicate='owner',object='red'),dict(subject='nodebeta',predicate='owner',object='blue')])
                if control=='missing-evidence':facts=[]
                out.append(dict(id=split+'-'+language+'-'+control,split=split,language=language,control=control,descriptor=copy.deepcopy(descriptor),goal=dict(command='sana',instruction=PROMPTS[language] if control!='unknown-instruction' else '???'),documents={'observations':dict(facts=facts)},expected_target='nodebeta',expected_effect=control=='complete'))
    return out
