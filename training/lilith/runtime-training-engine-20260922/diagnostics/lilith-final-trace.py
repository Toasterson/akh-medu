phase='full';states={};hs=[]
for label,module in [('norm',m.base_model.model.model.language_model.norm),('lm_head',m.base_model.model.lm_head)]:
 def capture(mod,args,output,label=label):states.setdefault(phase,{})[label]={'input':args[0].detach().cpu().clone(),'output':output.detach().cpu().clone(),'shape':list(args[0].shape),'stride':list(args[0].stride()),'offset':args[0].storage_offset()}
 hs.append(module.register_forward_hook(capture))
with torch.no_grad():f=m(input_ids=torch.tensor([ts[:32]],device=d),use_cache=False,logits_to_keep=1).logits.float()[0,-1].cpu()
phase='cached';c=e._cache()
for i in range(32):g=e._forward([ts[i]],c,i,False).detach().cpu()
for h in hs:h.remove()
result={}
for k in ['norm','lm_head']:
 a=states['full'][k];b=states['cached'][k];result[k]={'input_diff':float((a['input'][:,-1].float()-b['input'][:,-1].float()).abs().max()),'output_diff':float((a['output'][:,-1].float()-b['output'][:,-1].float()).abs().max()),'full_layout':{kk:a[kk] for kk in ['shape','stride','offset']},'cached_layout':{kk:b[kk] for kk in ['shape','stride','offset']}}
result['logits']=float((f-g).abs().max());print(json.dumps(result),flush=True)
Path('training/lilith/runtime-training-engine-20260922/final-trace.json').write_text(json.dumps(result,indent=2)+'\n')
del c,states;gc.collect();torch.cuda.empty_cache()
