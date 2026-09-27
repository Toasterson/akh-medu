class SynchronousCache(Cache):
 def update(self,key_states,value_states,layer_idx,*args,**kwargs):
  layer=self.layers[layer_idx]
  if layer.is_initialized:
   layer.keys=layer.keys.to(key_states.device)
   layer.values=layer.values.to(value_states.device)
  keys,values=layer.update(key_states,value_states,*args,**kwargs)
  layer.keys=layer.keys.to('cpu')
  layer.values=layer.values.to('cpu')
  return keys,values

def newcache(kind):
 layers=[DynamicSlidingWindowLayer(sliding_window=1024) if t=='sliding_attention' else DynamicLayer() for t in m.config.text_config.layer_types]
 return SynchronousCache(layers=layers) if kind=='synchronous' else Cache(layers=layers,offloading=(kind=='async'),offload_only_non_sliding=False)
results={}
with torch.no_grad():reference=m(input_ids=torch.tensor([ts[:32]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
for kind in ['async','resident','synchronous']:
 outputs=[]
 for repeat in range(2):
  c=newcache(kind)
  for i in range(32):value=e._forward([ts[i]],c,i,False)
  outputs.append(value.detach().cpu());del c;gc.collect();torch.cuda.empty_cache()
 results[kind]={'repeat_max_abs':float((outputs[0]-outputs[1]).abs().max()),'full_max_abs':float((reference-outputs[0]).abs().max())};print(kind,results[kind],flush=True)
Path('training/lilith/runtime-training-engine-20260922/transfer-variants.json').write_text(json.dumps(results,indent=2)+'\n')
