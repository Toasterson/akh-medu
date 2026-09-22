from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
ALL_ATTENTION_FUNCTIONS.register('sdpa',original);m.set_attn_implementation('sdpa')
traces={};phase='full';handles=[]
modules=m.base_model.model.model.language_model
for name,module in modules.named_modules():
 if (name.startswith('layers.') and int(name.split('.')[1])<=5) or name=='embed_tokens':
  if True:
   def hook(mod,args,result,label=name):
    v=result[0] if isinstance(result,tuple) else result
    traces.setdefault(phase,{}).setdefault(label,[]).append(v.detach().float().cpu())
   handles.append(module.register_forward_hook(hook))
with torch.no_grad():m(input_ids=torch.tensor([ts[:8]],device=d),use_cache=False,logits_to_keep=1)
phase='cached';c=e._cache()
for i in range(8):e._forward([ts[i]],c,i,False)
for h in handles:h.remove()
res={}
for key,values in traces['full'].items():
 f=values[0];g=torch.cat(traces['cached'][key],dim=1)
 delta=(f-g).abs();res[key]={'max':float(delta.max()),'positions':delta.reshape(8,-1).max(dim=1).values.tolist()}
print(json.dumps({k:v for k,v in res.items() if v['max']}),flush=True)
Path('training/lilith/runtime-training-engine-20260922/layer-trace-mlp.json').write_text(json.dumps(res,indent=2)+'\n')
del c,traces;gc.collect();torch.cuda.empty_cache()
