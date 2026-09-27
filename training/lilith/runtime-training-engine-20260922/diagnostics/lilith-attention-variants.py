from contextlib import nullcontext
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
original=ALL_ATTENTION_FUNCTIONS['sdpa']
result={}
def stable_sdpa(module,query,key,value,attention_mask,**kwargs):
 dtype=query.dtype
 if attention_mask is not None and attention_mask.dtype != torch.bool:attention_mask=attention_mask.float()
 out,weights=original(module,query.float(),key.float(),value.float(),attention_mask,**kwargs)
 return out.to(dtype),weights
for name in ['sdpa-default','sdpa-math','eager','sdpa-fp32']:
 m.set_attn_implementation('eager' if name=='eager' else 'sdpa')
 ALL_ATTENTION_FUNCTIONS.register('sdpa',stable_sdpa if name=='sdpa-fp32' else original)
 with sdpa_kernel(SDPBackend.MATH) if name=='sdpa-math' else nullcontext():
  with torch.no_grad():f=m(input_ids=torch.tensor([ts[:8]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
  c=Cache(layers=[DynamicSlidingWindowLayer(sliding_window=1024) if t=='sliding_attention' else DynamicLayer() for t in m.config.text_config.layer_types],offloading=True,offload_only_non_sliding=False)
  for i in range(8):g=e._forward([ts[i]],c,i,False).detach().cpu()
  result[name]={'max_abs':float((f-g).abs().max()),'mean_abs':float((f-g).abs().mean())}
  del c;gc.collect();torch.cuda.empty_cache()
 print(name,result[name],flush=True)
 Path('training/lilith/runtime-training-engine-20260922/attention-variants.json').write_text(json.dumps(result,indent=2)+'\n')
ALL_ATTENTION_FUNCTIONS.register('sdpa',original);m.set_attn_implementation('sdpa')
