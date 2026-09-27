#!/usr/bin/env python3
"""Bounded replay of the real report prompt, with separate prefill/decode timing."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--binary',type=Path,default=ROOT/'target/release/examples/candle_session_probe');p.add_argument('--timeout',type=int,default=240);p.add_argument('--ttft-limit',type=float,default=30);p.add_argument('--tokens',type=int,default=8);p.add_argument('--prompt-file',type=Path);p.add_argument('--min-decode-tps',type=float,default=15);p.add_argument('--require-complete',action='store_true');p.add_argument('--max-generation-seconds',type=float);a=p.parse_args()
 a.out.mkdir(parents=True,exist_ok=False)
 sessions=json.loads((ROOT/'training/lilith/coordination-20260922/model-run3/sessions.json').read_text())
 s=sessions['lilith_report'];prompt='<|turn>user\n'+s['instructions']+'\n\n'+s['turns'][0]['input']['text']+'<turn|>\n<|turn>model\n'
 if a.prompt_file:prompt=a.prompt_file.read_text()
 (a.out/'prompt.txt').write_text(prompt)
 model=Path.home()/'.local/share/akh-medu/models/llm/gemma4-12b.gguf';tok=model.with_name('gemma4-12b.tokenizer.json')
 env=dict(os.environ,AKH_GPU_MATMUL='0',AKH_GPU_QUANT='0',AKH_VULKAN_PIPELINE_CACHE_ENTRIES='16384',VK_DRIVER_FILES='/usr/share/vulkan/icd.d/radeon_icd.x86_64.json')
 cmd=[str(a.binary),'--model',str(model),'--tokenizer',str(tok),'--device','vulkan','--prompt-file',str(a.out/'prompt.txt'),'--tokens',str(a.tokens)]
 write(a.out/'protocol.json',{'schema':'lilith/latency-replay@2','binary_sha256':sha(a.binary),'driver_sha256':sha(__file__),'model_sha256':sha(model),'tokenizer_sha256':sha(tok),'prompt_sha256':sha(a.out/'prompt.txt'),'command':cmd,'timeout':a.timeout,'ttft_limit_seconds':a.ttft_limit,'min_decode_tps':a.min_decode_tps,'require_complete':a.require_complete,'max_generation_seconds':a.max_generation_seconds,'scope':'Exact tool-free reporter system/user rendering; completion CLI replaces session services. Requested output cap is recorded in command; session transport is outside this probe.','env':{k:env[k] for k in sorted(k for k in env if k.startswith('AKH_VULKAN_') or k in ('AKH_GPU_MATMUL','AKH_GPU_QUANT','VK_DRIVER_FILES'))}})
 result={'status':'running'};write(a.out/'result.json',result);start=time.monotonic()
 try:
  with (a.out/'stdout.jsonl').open('w') as out,(a.out/'stderr.log').open('w') as err:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=out,stderr=err)
   result['pid']=proc.pid
   try:code=proc.wait(timeout=a.timeout)
   except subprocess.TimeoutExpired:proc.kill();proc.wait();result.update(status='timeout',exit_code=proc.returncode);return 1
  records=[json.loads(x) for x in (a.out/'stdout.jsonl').read_text().splitlines() if x.startswith('{')]
  times=[x for x in records if x.get('phase')=='generation']
  passed=code==0 and len(times)==2 and all(x['time_to_first_token_seconds']<=a.ttft_limit and x['decode_steps_per_second']>=a.min_decode_tps and (not a.require_complete or x['finish']=='Stop') and (a.max_generation_seconds is None or x['seconds_including_prefill']<=a.max_generation_seconds) for x in times)
  result.update(status='passed' if passed else 'failed',exit_code=code,generations=times)
  return 0 if passed else 1
 finally:result['elapsed_seconds']=time.monotonic()-start;write(a.out/'result.json',result)
if __name__=='__main__':raise SystemExit(main())
