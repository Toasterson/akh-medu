from types import MethodType
from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedRMSNorm
norm_originals=[]
def accurate_norm(self,hidden_states):
 v=hidden_states.double();v=v*torch.pow(v.square().mean(-1,keepdim=True)+self.eps,-0.5)
 if self.with_scale:v=v*self.weight.double()
 return v.to(hidden_states.dtype)
for module in m.modules():
 if isinstance(module,Gemma4UnifiedRMSNorm):
  norm_originals.append((module,module.forward));module.forward=MethodType(accurate_norm,module)
results={}
for n in [8,32]:
 with torch.no_grad():f=m(input_ids=torch.tensor([ts[:n]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
 c=e._cache()
 for i in range(n):g=e._forward([ts[i]],c,i,False).detach().cpu()
 results[n]={'max_abs':float((f-g).abs().max()),'mean_abs':float((f-g).abs().mean())};print(n,results[n],flush=True)
 del c;gc.collect();torch.cuda.empty_cache()
Path('training/lilith/runtime-training-engine-20260922/norm-fp64-variant.json').write_text(json.dumps(results,indent=2)+'\n')
