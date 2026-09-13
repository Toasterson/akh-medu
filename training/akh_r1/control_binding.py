"""Bind retained pack preparation to caller-pinned local assistance files.

No execution. Hash identity is not candidate quality or semantic equivalence.
"""
import json
from pathlib import Path
from episode_pack import file_sha
from pack_preparation import prepare
from perception_assistance import verified_reference
from prepared_control_dispatch import LanguageControl,OwnershipControl
ROOT=Path(__file__).parent
COMMON=['control_binding.py','prepared_control_dispatch.py','pack_preparation.py','episode_pack.py','r0_contract.py','public_episode.py','safe_pipeline.py','environment.py','run_public_breadth.py','build_curriculum.py','language_pack.py','control_pack.py','ownership_worlds.py','ownership_score.py','perception_assistance.py','assistance_compatibility.py']
FAMILY={'language-v1':['language_composition_client.py','graph_effect_cli.py'],'ownership-v1':['ownership_client.py']}
def bind(pack,episode_id,adapter,expected):
 if adapter not in FAMILY:raise ValueError('Unsupported control family')
 if set(expected)!={'schema','adapter','episode_id','files'} or expected['schema']!='akh-control-binding-v1' or expected['adapter']!=adapter or expected['episode_id']!=episode_id:raise ValueError('Binding identity mismatch')
 pack=Path(pack).resolve()
 paths={str(p.resolve()) for p in [pack/'manifest.json',pack/'registry.json',*sorted((pack/'examiner').glob('*.json')),*[ROOT/n for n in COMMON+FAMILY[adapter]]]}
 if set(expected['files'])!=paths:raise ValueError('Binding file set mismatch')
 for path,digest in expected['files'].items():verified_reference(dict(path=path,sha256=digest))
 p=prepare(pack,episode_id,adapter)
 (LanguageControl(lambda:None) if adapter=='language-v1' else OwnershipControl()).validate(p)
 p['binding']=dict(schema='akh-control-binding-result-v1',files=expected['files'].copy(),scope='Preparation and declared adapter files only; native binary/image must be pinned separately before execution')
 return p
