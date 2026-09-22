import json,sys,time,torch,traceback
from pathlib import Path
sys.path.insert(0,'scripts/session-platform')
from lilith_adapter_mechanics import PARENT_SHA256,PARENT_CONFIG_SHA256,TARGET
from lilith_training_engine import GemmaTrainingEngine,assert_detached_kv
out=Path('training/lilith/runtime-training-engine-20260922/boundary-control-short-20260922.json'); start=time.monotonic(); r={'schema':'lilith/training-engine-boundary-control@2','status':'running','shared_kv_persisted':False};out.write_text(json.dumps(r,indent=2)+'\n')
try:
 e=GemmaTrainingEngine.load(Path.home()/'.local/share/akh-medu/lilith-training/parent-5926caa4',expected_parent_sha256=PARENT_SHA256,expected_config_sha256=PARENT_CONFIG_SHA256,expected_tokenizer_sha256='cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',target_module=TARGET); ts=[2]+[100+i%40 for i in range(1100)];m=e.model;d=m.device
 def full(n,g=False):
  ids=torch.tensor([ts[:n]],device=d);pos=torch.arange(n,device=d)[None,:]
  with (torch.enable_grad() if g else torch.no_grad()):return m(input_ids=ids,use_cache=False,position_ids=pos,cache_position=pos[0],logits_to_keep=1).logits.float()[0,-1]
 fs={n:full(n).detach().cpu() for n in (2,4,8)};c=e._cache();cs={}
 for n in range(1,9):
  v=e._forward([ts[n-1]],c,n-1,False).detach().cpu()
  if n in fs:cs[n]=v
 diff_unused=None;diff={str(n):{'max_abs':float((fs[n]-cs[n]).abs().max()),'mean_abs':float((fs[n]-cs[n]).abs().mean())} for n in cs}
 m.zero_grad(set_to_none=True);a=full(8,True);la=-torch.log_softmax(a,-1)[101];la.backward();ga={n:p.grad.detach().float().cpu().clone() for n,p in m.named_parameters() if p.requires_grad};m.zero_grad(set_to_none=True);c=e._cache();e._prefill_no_grad(ts[:7],c);b=e._forward([ts[7]],c,7,True);lb=-torch.log_softmax(b,-1)[101];lb.backward();gb={n:p.grad.detach().float().cpu().clone() for n,p in m.named_parameters() if p.requires_grad};gd={n:{'max_abs':float((ga[n]-gb[n]).abs().max()),'mean_abs':float((ga[n]-gb[n]).abs().mean()),'full_norm':float(ga[n].norm()),'cached_norm':float(gb[n].norm())} for n in ga};ml=max(v['max_abs'] for v in diff.values());mg=max(v['max_abs'] for v in gd.values());r.update(status='passed' if ml<=1e-4 and mg<=1e-4 else 'failed_equivalence',positions=[2,4,8],tolerances={'logit_max_abs':1e-4,'gradient_max_abs':1e-4},logit_differences=diff,losses={'full':float(la),'cached':float(lb),'abs':abs(float(la)-float(lb))},gradient_differences=gd,cache_tensors_checked=assert_detached_kv(c),peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start)
except BaseException as x:r.update(status='error',error=type(x).__name__+': '+str(x),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-start);out.write_text(json.dumps(r,indent=2)+'\n');raise
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
