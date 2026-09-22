import time,hashlib
import lilith_training_engine as current_engine
# Fresh engine wrapper over the already pinned, loaded parent; no weight reload.
e=current_engine.GemmaTrainingEngine(m,e.tokenizer,e.parent_sha256,e.tokenizer_sha256,e.target_module)
e.prefill_chunk_tokens=32
out=Path('training/lilith/runtime-training-engine-20260922/fixed-context-memory-control.json')
r={'schema':'lilith/cache-memory@1','status':'running','tokens':32768,'contract':e.contract(),'mode':'memory-only chunked forward; canonical sampler/replay remains tokenwise','engine_sha256':hashlib.sha256(Path(current_engine.__file__).read_bytes()).hexdigest(),'optimizer_steps':0}
out.write_text(json.dumps(r,indent=2)+'\n');start=time.monotonic()
try:
 torch.cuda.reset_peak_memory_stats();c=e._cache();tokens=[2]+[100+i%40 for i in range(32767)]
 for offset in range(0,len(tokens),32):
  value=e._forward(tokens[offset:offset+32],c,offset,False)
  if (offset+32)%1024==0:print('memory',offset+32,flush=True)
 finite=bool(torch.isfinite(value).all());r.update(status='passed' if finite else 'failed',finite_logits=finite,cache_seq_length=c.get_seq_length(),detached_tensors=assert_detached_kv(c),sliding_stored=c.layers[0].keys.shape[-2],global_stored=c.layers[5].keys.shape[-2],peak_gpu_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start)
except BaseException as error:
 r.update(status='error',error=type(error).__name__+': '+str(error));raise
finally:
 out.write_text(json.dumps(r,indent=2)+'\n');e.prefill_chunk_tokens=1
 if 'c' in globals():del c
 gc.collect();torch.cuda.empty_cache()
print(json.dumps(r),flush=True)
