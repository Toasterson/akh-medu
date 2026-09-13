"""Recovery goal admission using only initial public source documents.

Does not certify the episode, execute it, or read hidden outcomes.
"""
def validate_candidate(episode):
 docs=[e for e in episode['delivery_events'] if e.get('after')=='initial_goal' and e.get('kind')=='dependency_document']
 if len(docs)!=1 or not docs[0].get('source'):raise ValueError('One sourced initial dependency document required')
 content=docs[0]['content'];nodes=content['operations'];edges=content['dependencies']
 if not isinstance(nodes,list) or not nodes or any(not isinstance(n,str) or not n for n in nodes) or len(set(nodes))!=len(nodes):raise ValueError('Invalid operation identities')
 root=episode['public_start'].get('root_operation')
 if not isinstance(root,str) or root not in nodes:raise ValueError('Undefined public root operation')
 if not isinstance(edges,list) or any(not isinstance(e,(list,tuple)) or len(e)!=2 or any(n not in nodes for n in e) for e in edges):raise ValueError('Undefined dependency operation')
 if any(b==root for a,b in edges):raise ValueError('Root operation has a prerequisite')
 return dict(root_operation=root,source=docs[0]['source'],operations=len(nodes))
