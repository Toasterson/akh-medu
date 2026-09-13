"""Declared perception dependency closure layered on existing byte compatibility."""
import json
from pathlib import Path
from assistance_compatibility import compare,fingerprint
ROLES={'ocr_engine','ocr_language_data','spatial_interpreter','ocr_observer','candidate','command_fixture','public_boundary','relation_scorer','effect_scorer','transport','pipeline','jsonl_transport','runner'}

def verified_reference(ref):
 if not isinstance(ref,dict) or set(ref)!={'path','sha256'}:raise ValueError('Invalid dependency reference')
 if not isinstance(ref['path'],str) or not Path(ref['path']).is_absolute():raise ValueError('Dependency path must be absolute')
 if not isinstance(ref['sha256'],str) or len(ref['sha256'])!=64 or any(c not in '0123456789abcdef' for c in ref['sha256']):raise ValueError('Invalid dependency digest')
 if fingerprint(ref['path'])!=ref['sha256']:raise ValueError('Dependency changed')
 return Path(ref['path'])

def verify_profile(manifest):
 path=verified_reference(manifest['artifacts']['profile']);profile=json.loads(path.read_text())
 if profile.get('schema')!='akh-assistance-profile-v1' or profile.get('kind')!='perception-fixed-layout-v1':raise ValueError('Wrong perception profile')
 dependencies=json.loads(verified_reference(profile['dependency_manifest']).read_text())
 if set(dependencies)!={'schema','artifacts'} or dependencies['schema']!='akh-perception-dependencies-v1':raise ValueError('Invalid dependency manifest')
 if set(dependencies['artifacts'])!=ROLES:raise ValueError('Missing or extra dependency roles')
 for ref in dependencies['artifacts'].values():verified_reference(ref)
 return {role:ref['sha256'] for role,ref in dependencies['artifacts'].items()}

def compare_perception(expected,candidates):
 baseline=verify_profile(expected)
 for candidate in candidates:
  if verify_profile(candidate)!=baseline:raise ValueError('Perception dependency mismatch')
 result=compare(expected,candidates)
 result.update(perception_dependencies=baseline,scope='Declared dependency file compatibility only; no runtime execution, shared-library closure, model quality or R1 readiness')
 return result
