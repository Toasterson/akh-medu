import time,hashlib,dataclasses
start=time.monotonic();output=Path('training/lilith/runtime-training-engine-20260922/fixed-boundary-control.json')
r={'schema':'lilith/cache-correctness@3','status':'running','engine_sha256':hashlib.sha256(Path('scripts/session-platform/lilith_training_engine.py').read_bytes()).hexdigest(),'contract':e.contract(),'positions':[1023,1024,1025],'reference':'resident tokenwise; full BF16 measured separately','variants':{}}
def save():output.write_text(json.dumps(r,indent=2)+'\n')
save();ts=[2]+[100+i%40 for i in range(1100)];snap=e.snapshot_adapter()
try:
 torch.manual_seed(73)
 with torch.no_grad():
  for name,p in m.named_parameters():
   if p.requires_grad and 'lora_B' in name:p.uniform_(-0.0001,0.0001)
 ref={};grads={};losses={}
 for kind in ['resident','synchronous']:
  e.cache_offloading=(kind=='synchronous');c=e._cache();m.zero_grad(set_to_none=True);loss_total=0.
  for i in range(1025):
   selected=i+1 in [1023,1024,1025]
   logits=e._forward([ts[i]],c,i,selected)
   if selected:
    ref.setdefault(kind,{})[i+1]=logits.detach().cpu()
    loss=-torch.log_softmax(logits,-1)[101]/3;loss_total+=float(loss.detach().cpu());loss.backward()
   if (i+1)%256==0:print(kind,i+1,flush=True)
  grads[kind]={name:p.grad.detach().float().cpu().clone() for name,p in m.named_parameters() if p.requires_grad}
  losses[kind]=loss_total
  r['variants'][kind]={'loss':loss_total,'cache_tensors':assert_detached_kv(c),'cache_length':c.get_seq_length(),'sliding_stored':c.layers[0].keys.shape[-2],'global_stored':c.layers[5].keys.shape[-2]};save()
  del c,logits,loss;m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
 r['logit_max_abs']={str(n):float((ref['resident'][n]-ref['synchronous'][n]).abs().max()) for n in ref['resident']}
 r['gradient_max_abs']={name:float((grads['resident'][name]-grads['synchronous'][name]).abs().max()) for name in grads['resident']}
 r['gradient_norms']={name:float(v.norm()) for name,v in grads['resident'].items()}
 r['full_bf16_diagnostic']={}
 for n in [1023,1024,1025]:
  with torch.no_grad():f=m(input_ids=torch.tensor([ts[:n]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
  r['full_bf16_diagnostic'][str(n)]={'max_abs':float((f-ref['synchronous'][n]).abs().max()),'mean_abs':float((f-ref['synchronous'][n]).abs().mean())};save()
 passed=max(r['logit_max_abs'].values())<=1e-4 and max(r['gradient_max_abs'].values())<=1e-4 and all(v>0 for v in r['gradient_norms'].values())
 r.update(status='passed' if passed else 'failed',elapsed_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated());save();print(json.dumps(r),flush=True)
finally:
 e.restore_adapter(snap);e.cache_offloading=True;m.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
