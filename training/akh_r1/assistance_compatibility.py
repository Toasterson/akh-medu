"""Compare frozen assistance inputs, never candidate scores or R1 readiness.

Caller supplies a trusted expected manifest. Paths are local operator metadata,
not candidate inputs. Every referenced artifact is hashed before comparison.
"""
import argparse,hashlib,json
from pathlib import Path
FIELDS={'profile','adapter','native','fixture','pack'}
def fingerprint(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def validate(manifest):
    if not isinstance(manifest,dict) or set(manifest)!={'schema','candidate','artifacts'}:raise ValueError('Manifest fields differ')
    if manifest['schema']!='akh-assistance-comparison-v1':raise ValueError('Unknown schema')
    if not isinstance(manifest['candidate'],str) or not manifest['candidate'].strip():raise ValueError('Candidate identity required')
    artifacts=manifest['artifacts']
    if not isinstance(artifacts,dict) or set(artifacts)!=FIELDS:raise ValueError('All assistance artifacts required')
    verified={}
    for name,item in artifacts.items():
        if not isinstance(item,dict) or set(item)!={'path','sha256'}:raise ValueError('Invalid artifact entry')
        digest=item['sha256']
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Invalid SHA256')
        if not isinstance(item['path'],str) or not Path(item['path']).is_absolute():raise ValueError('Absolute artifact path required')
        if fingerprint(item['path'])!=digest:raise ValueError('Artifact changed: '+name)
        verified[name]=digest
    profile=json.loads(Path(artifacts['profile']['path']).read_text())
    if profile.get('schema')!='akh-assistance-profile-v1' or not isinstance(profile.get('id'),str) or not profile['id']:raise ValueError('Invalid assistance profile')
    return verified

def compare(expected,candidates):
    baseline=validate(expected)
    if not isinstance(candidates,list) or len(candidates)<2:raise ValueError('At least two candidate conditions required')
    names=set();rows=[]
    for candidate in candidates:
        hashes=validate(candidate);name=candidate['candidate']
        if name in names:raise ValueError('Duplicate candidate identity')
        names.add(name)
        different=sorted(k for k in FIELDS if hashes[k]!=baseline[k])
        if different:raise ValueError('Assistance mismatch for '+name+': '+','.join(different))
        rows.append(dict(candidate=name,artifact_hashes=hashes))
    return dict(schema='akh-assistance-compatibility-result-v1',compatible=True,conditions=rows,
                scope='Declared local artifact byte compatibility only; not runtime execution, candidate quality or pack readiness',R1_ready=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--expected',type=Path,required=True);ap.add_argument('--candidate',type=Path,action='append',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result=compare(json.loads(a.expected.read_text()),[json.loads(p.read_text()) for p in a.candidate])
    # Never overwrite prior comparison evidence.
    with a.out.open('x') as output:json.dump(result,output,indent=2)
if __name__=='__main__':main()
