#!/usr/bin/env python3
"""Run two fresh Vulkan probe processes with durable exit/provenance receipts."""
import hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'training/lilith/vulkan-adapter-20260922'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def run():
 binary=ROOT/'target/release/examples/lilith_vulkan_adapter'
 adapter=OUT/'klpo-export'
 receipt=ROOT/'training/lilith/rl-20260922/live/run2/result.json'
 candidate=json.loads((adapter/'manifest.json').read_text())
 experiment=json.loads(receipt.read_text())
 if candidate['source_adapter_sha256']!=experiment['klpo']['adapter_file_sha256']:
  raise ValueError('Candidate/experiment mismatch')
 protocol=OUT/'probe-protocol.json'
 if protocol.exists():raise FileExistsError('Preserve earlier run; choose a new evidence directory')
 write(protocol,dict(schema='lilith/vulkan-cold-probe@1',binary_sha256=sha(binary),
  driver_sha256=sha(__file__),example_sha256=sha(ROOT/'examples/lilith_vulkan_adapter.rs'),
  runtime_sha256=sha(ROOT/'src/nlu/vendored/quantized_gemma4.rs'),
  manifest_sha256=sha(adapter/'manifest.json'),experiment_receipt_sha256=sha(receipt),
  candidate_source_sha256=candidate['source_adapter_sha256'],tokens=32,processes=2,
  quality_claim=False,performance_design='Single adapted then base timing per process; descriptive only'))
 env=dict(os.environ,AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',
  VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json')
 results=[]
 for n in (1,2):
  output=OUT/f'probe-{n}.json';status=OUT/f'probe-{n}-process.json'
  command=[str(binary),'--model',str(Path.home()/'.local/share/akh-medu/models/llm/gemma4-12b.gguf'),
   '--tokenizer',str(Path.home()/'.local/share/akh-medu/models/llm/gemma4-12b.tokenizer.json'),
   '--adapter',str(adapter),'--output',str(output),'--tokens','32']
  state=dict(status='running',command=command);write(status,state);start=time.monotonic()
  try:
   with (OUT/f'probe-{n}.log').open('w') as log:
    p=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
   state.update(status='completed' if p.returncode==0 else 'failed',exit_code=p.returncode)
  except Exception as e:
   state.update(status='failed',error=str(e));raise
  finally:
   state['elapsed_seconds']=time.monotonic()-start;write(status,state)
  if p.returncode:raise RuntimeError(f'probe {n} failed; see durable log')
  results.append(json.loads(output.read_text()))
 keys=('base_logits_sha256','adapted_logits_sha256','base_tokens','adapter_tokens','manifest_sha256')
 equal={key:results[0][key]==results[1][key] for key in keys}
 write(OUT/'cold-reload.json',dict(passed=all(equal.values()) and all(x['passed'] for x in results),
  exact_across_fresh_processes=equal,quality_claim=False))
 if not all(equal.values()):raise RuntimeError('Fresh-process outputs differ')

if __name__=='__main__':run()
