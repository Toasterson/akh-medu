"""Local host preparation for retained packs. Does not execute episodes."""
import argparse,copy,json,os
from pathlib import Path
from episode_pack import build,file_sha
from language_pack import load_case
ADAPTERS={'recovery-v1':{'recovery_durability'},'ownership-v1':{'dojo_pressure'},'ocr-v1':{'perception_knowledge'},'language-v1':{'language_transfer'},'graph-query-v1':{'acquisition','grounded_reasoning','knowledge_revision'}}
def read_entry(pack,episode_id):
 pack=Path(pack);m=json.loads((pack/'manifest.json').read_text());registry=json.loads((pack/'registry.json').read_text());episodes=[json.loads(p.read_text()) for p in sorted((pack/'examiner').glob('*.json'))];result=build(episodes,registry)
 if not result['manifest_valid'] or result['rows']!=m['rows']:raise ValueError('Pack validation failed')
 matches=[e for e in episodes if e['id']==episode_id]
 if len(matches)!=1:raise ValueError('Unknown or duplicate episode')
 return matches[0]
def prepare(pack,episode_id,adapter):
 e=read_entry(pack,episode_id)
 if adapter not in ADAPTERS or e['family'] not in ADAPTERS[adapter]:raise ValueError('Incompatible family adapter')
 if adapter=='recovery-v1':
  from recovery_pack import load_entry as load_recovery
  config=load_recovery(e)
 elif adapter in ('ownership-v1','ocr-v1'):
  from control_pack import load_entry
  if e['source_requirements']['kind']!=adapter.removesuffix('-v1'):raise ValueError('Control kind mismatch')
  config=load_entry(e)
 elif adapter=='language-v1':
  config=load_case(pack,episode_id)
 else:
  # This retained family places its qualified pack directly below its run.
  root=Path(pack).parent;plan=json.loads((root/'plan.json').read_text());audit=json.loads((root/'artifact-audit.json').read_text())
  if not audit.get('all_passed'):raise ValueError('Source audit failed')
  for path,digest in plan['source_hashes'].items():
   if file_sha(path)!=digest:raise ValueError('Source hash changed')
  if e['public_start']['goal'].get('kind')!='conjunctive_query':raise ValueError('Unsupported graph goal')
  world=e['examiner']['initial_world']
  if world['clauses']!=e['public_start']['goal']['clauses']:raise ValueError('Clause mismatch')
  config=dict(world=copy.deepcopy(world),events=copy.deepcopy(e['examiner']['events']),expected=copy.deepcopy(e['examiner']['expected_effects']))
 return dict(schema='akh-pack-preparation-v1',adapter=adapter,episode_id=episode_id,assistance=copy.deepcopy(e['assistance']),public_start=copy.deepcopy(e['public_start']),host_config=config,executed=False,R1_ready=False)
def write_preparation(out,prepared):
 out=Path(out);out.mkdir(exist_ok=False)
 with (out/'host.json').open('x') as f:
  os.chmod(out/'host.json',0o600);json.dump(prepared,f,indent=2)
 (out/'public.json').write_text(json.dumps(prepared['public_start'],indent=2)+'\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pack',type=Path,required=True);ap.add_argument('--episode-id',required=True);ap.add_argument('--adapter',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();write_preparation(a.out,prepare(a.pack,a.episode_id,a.adapter))
if __name__=='__main__':main()


def prepare_neutral_lifecycle(entry):
 """Prepare a pinned lifecycle observation entry with the common output schema.

Separate from qualified pack loading: acceptance earns no world credit and does
not imply a runnable neutral candidate. write_preparation keeps host state private.
 """
 from neutral_lifecycle import prepare as prepare_neutral
 return prepare_neutral(entry)


def prepare_retained_envelope(manifest_path, expected_manifest_sha256, family,
                              split, expected_input_sha256, expected_record_sha256):
 """Admit an exact retained envelope; never dispatch or qualify an episode."""
 import hashlib
 allowed={'tool_choice':'retained-tool-effect-neutral-v1',
          'multi_step_work':'inventory-service-neutral-v1',
          'language_transfer':'inventory-language-neutral-v2',
          'dojo_pressure':'inventory-ownership-interaction-v1'}
 def checked_bytes(path,expected):
  if not isinstance(expected,str) or len(expected)!=64 or any(c not in '0123456789abcdef' for c in expected):
   raise ValueError('Invalid caller fingerprint')
  data=Path(path).read_bytes()
  if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Retained artifact changed')
  return data
 manifest=json.loads(checked_bytes(manifest_path,expected_manifest_sha256))
 if family not in allowed or split not in ('development','validation'):raise ValueError('Unsupported retained family/split')
 matches=[r for r in manifest['rows'] if r['family']==family and r['split']==split]
 if len(matches)!=1:raise ValueError('Missing or duplicate retained row')
 row=matches[0]
 if row.get('availability')!='retained_input_located' or row.get('json_pointer')!='':raise ValueError('Not a complete retained envelope')
 if row['input']['sha256']!=expected_input_sha256 or row['record_sha256']!=expected_record_sha256:raise ValueError('Caller identity mismatch')
 entry=json.loads(checked_bytes(row['input']['path'],expected_input_sha256))
 record=hashlib.sha256(json.dumps(entry,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if record!=expected_record_sha256:raise ValueError('Record identity mismatch')
 if entry.get('schema')!='akh-pack-preparation-v1' or entry.get('adapter')!=allowed[family]:raise ValueError('Unsupported retained adapter')
 public=entry.get('public_start');private=entry.get('host_config')
 if not isinstance(public,dict) or not isinstance(private,dict) or not entry.get('assistance'):raise ValueError('Incomplete retained envelope')
 if not {'goal','max_calls'}<=set(public) or not set(public)<={'goal','tools','max_calls'} or ('tools' in public and not isinstance(public['tools'],list)) or type(public['max_calls']) is not int or not 0<public['max_calls']<=48:raise ValueError('Invalid public contract')
 # Preserve the pinned public contract verbatim; never construct it from private state.
 result=copy.deepcopy(entry)
 result.update(executed=False,R1_ready=False)
 return result
