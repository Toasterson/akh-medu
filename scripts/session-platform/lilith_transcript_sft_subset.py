#!/usr/bin/env python3
"""Freeze a bounded Gemma SFT subset without inspecting quality outcomes."""
from __future__ import annotations
import argparse, hashlib, json, os, re
from pathlib import Path
from transformers import AutoTokenizer

ACK = re.compile(r"(?is)^\s*(?:ok(?:ay)?|done|thanks?|got it|sure|yes|no|understood|sounds good)[.!\s]*$")
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha(x): return hashlib.sha256(x if isinstance(x,bytes) else canonical(x)).hexdigest()

def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--tokenizer',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--private-manifest',type=Path,required=True); p.add_argument('--train-cap',type=int,default=8); p.add_argument('--validation-cap',type=int,default=8); p.add_argument('--max-total',type=int,default=512); p.add_argument('--max-assistant',type=int,default=480); a=p.parse_args()
 if a.output.exists() or a.private_manifest.exists(): raise SystemExit('refusing to overwrite subset')
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True); accepted={'train':[],'validation':[]}; rejected={}; lengths=[]
 def reject(k): rejected.__setitem__(k,rejected.get(k,0)+1)
 for row in map(json.loads,a.input.open()):
  if row.get('training_kind')!='supervised_imitation': reject('not_imitation'); continue
  prompt,answer=row.get('input','').strip(),row.get('output','').strip()
  if len(prompt)<8: reject('prompt_too_short'); continue
  if len(answer)<40 or ACK.fullmatch(answer): reject('empty_or_acknowledgement'); continue
  prompt_tokens=len(tok.encode(prompt,add_special_tokens=False)); answer_tokens=len(tok.encode(answer,add_special_tokens=False))
  if prompt_tokens<4 or answer_tokens<16: reject('insufficient_learning_content'); continue
  if answer_tokens>a.max_assistant: reject('assistant_over_budget_no_truncation'); continue
  messages=[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
  rendered=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=False)
  total=len(tok.encode(rendered,add_special_tokens=False))
  if total>a.max_total: reject('total_over_budget_no_truncation'); continue
  split=row['split']; accepted[split].append((sha({'record_id':row['record_id'],'policy':'lilith-sft-subset-v1'}),row,total,prompt_tokens,answer_tokens))
 eligible={split:len(rows) for split,rows in accepted.items()}
 for split in accepted: accepted[split].sort(key=lambda x:x[0]); accepted[split]=accepted[split][:(a.train_cap if split=='train' else a.validation_cap)]
 out=[]
 for split in ('train','validation'):
  for _,row,total,pt,at in accepted[split]:
   out.append({'schema':'akh/lilith-transcript-sft@1','record_id':row['record_id'],'split':split,'messages':[{'role':'user','content':row['input']},{'role':'assistant','content':row['output']}],'token_counts':{'prompt':pt,'assistant':at,'chat_template_total':total},'provenance':row['provenance']}); lengths.append(total)
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in out)); os.chmod(a.output,0o600)
 manifest={'schema':'akh/lilith-transcript-sft-subset@1','records':len(out),'splits':{s:sum(x['split']==s for x in out) for s in ('train','validation')},'eligible_before_cap':eligible,'excluded_by_cap':{s:eligible[s]-sum(x['split']==s for x in out) for s in eligible},'caps':{'train':a.train_cap,'validation':a.validation_cap,'max_total_tokens':a.max_total,'max_assistant_tokens':a.max_assistant,'min_prompt_tokens':4,'min_assistant_tokens':16},'selection':'sha256(record_id, fixed policy), ascending; structural filters before cap','rejected':dict(sorted(rejected.items())),'token_lengths':{'min':min(lengths),'max':max(lengths),'mean':round(sum(lengths)/len(lengths),2)},'record_ids_sha256':sha([x['record_id'] for x in out]),'private_subset_sha256':hashlib.sha256(a.output.read_bytes()).hexdigest(),'tokenizer_json_sha256':hashlib.sha256((a.tokenizer/'tokenizer.json').read_bytes()).hexdigest(),'tokenizer_config_sha256':hashlib.sha256((a.tokenizer/'tokenizer_config.json').read_bytes()).hexdigest(),'chat_template_sha256':sha(tok.chat_template),'source_dataset_sha256':'18d221763bd9acb6b0e7b90945eee2bb45383ad323dab6f02ab00606802a97e9','limitations':['structural selection does not assert response correctness','validation uses the pre-existing held-out session split','over-budget examples are rejected, never truncated']}
 a.private_manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n'); os.chmod(a.private_manifest,0o600); print(json.dumps({k:manifest[k] for k in ('records','splits','rejected','token_lengths','private_subset_sha256')},sort_keys=True))
if __name__=='__main__': main()
