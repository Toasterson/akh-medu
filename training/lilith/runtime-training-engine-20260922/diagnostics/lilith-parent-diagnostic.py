import sys,torch,json,time,gc,hashlib
from pathlib import Path
sys.path.insert(0,'scripts/session-platform')
from lilith_adapter_mechanics import PARENT_SHA256,PARENT_CONFIG_SHA256,TARGET
from lilith_training_engine import GemmaTrainingEngine
from transformers import Cache,DynamicCache
from transformers.cache_utils import DynamicLayer,DynamicSlidingWindowLayer
out=Path('training/lilith/runtime-training-engine-20260922/short-diagnostic.json')
r={'status':'running','engine_sha256':hashlib.sha256(Path('scripts/session-platform/lilith_training_engine.py').read_bytes()).hexdigest(),'variants':{}}
def save():out.write_text(json.dumps(r,indent=2)+'\n')
save();torch.set_num_threads(2)
e=GemmaTrainingEngine.load(Path.home()/'.local/share/akh-medu/lilith-training/parent-5926caa4',expected_parent_sha256=PARENT_SHA256,expected_config_sha256=PARENT_CONFIG_SHA256,expected_tokenizer_sha256='cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f',target_module=TARGET)
m=e.model;ts=[2,100,101,102,103,104,105,106];d=m.device; traces={};phase='';handles=[]
for i,l in enumerate(m.base_model.model.model.language_model.layers):
 def hook(mod,args,result,index=i):
  v=result[0] if isinstance(result,tuple) else result
  traces.setdefault(phase,{})[index]=v[0,-1].detach().float().cpu()
 handles.append(l.register_forward_hook(hook))
def full():
 with torch.no_grad():return m(input_ids=torch.tensor([ts],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
phase='full';ref=full()
for name,offload,hybrid,chunk in [('lazy-offload',True,False,1),('lazy-resident',False,False,1),('hybrid-offload',True,True,1),('hybrid-chunk7',True,True,7)]:
 phase=name
 if hybrid:
  c=Cache(layers=[DynamicSlidingWindowLayer(sliding_window=1024) if t=='sliding_attention' else DynamicLayer() for t in m.config.text_config.layer_types],offloading=offload,offload_only_non_sliding=False)
 else:c=DynamicCache(config=m.config,offloading=offload,offload_only_non_sliding=False)
 for start in range(0,len(ts),chunk):v=e._forward(ts[start:start+chunk],c,start,False).detach().cpu()
 r['variants'][name]={'max_abs':float((ref-v).abs().max()),'mean_abs':float((ref-v).abs().mean()),'layers':[float((traces['full'][i]-traces[name][i]).abs().max()) for i in range(48)]};save()
 del c;gc.collect();torch.cuda.empty_cache()
phase='full-repeat';v=full();r['repeat_max_abs']=float((ref-v).abs().max());r['status']='completed';save()
