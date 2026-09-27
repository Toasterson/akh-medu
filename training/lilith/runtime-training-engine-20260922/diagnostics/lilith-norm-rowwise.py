for module,original_forward in norm_originals:module.forward=original_forward
for module,original_forward in norm_originals:
 def rowwise(self,hidden_states,fn=original_forward):
  if hidden_states.shape[1] == 1:return fn(hidden_states)
  return torch.cat([fn(hidden_states[:,i:i+1]) for i in range(hidden_states.shape[1])],dim=1)
 module.forward=MethodType(rowwise,module)
results={}
for n in [8,32]:
 with torch.no_grad():f=m(input_ids=torch.tensor([ts[:n]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
 c=e._cache()
 for i in range(n):g=e._forward([ts[i]],c,i,False).detach().cpu()
 results[n]={'max_abs':float((f-g).abs().max()),'mean_abs':float((f-g).abs().mean())};print(n,results[n],flush=True)
 del c;gc.collect();torch.cuda.empty_cache()
Path('training/lilith/runtime-training-engine-20260922/norm-rowwise-variant.json').write_text(json.dumps(results,indent=2)+'\n')
