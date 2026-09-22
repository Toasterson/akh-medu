#!/usr/bin/env python3
"""Pinned MiMo text-only coding trial on the frozen Lilith v2 task pack."""
import argparse, hashlib, json, os, platform, signal, time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

import lilith_rl_tasks_v2 as tasks

REVISION = '2367e865d009c13ac81713a2878291d33ab28177'
REPO_ID = 'XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B'
SEED = 20260922
EXPECTED_SHARDS = {
    'model-00001-of-00004.safetensors':'aab052180118aee34abc3029b54eaa49096aac606b97d703866b420dceb703c3',
    'model-00002-of-00004.safetensors':'7a0486565f06d25ac4628e9dba470dc3f604353471d240d5a0bf7128f64df396',
    'model-00003-of-00004.safetensors':'6c73207563d1879bfd6c143a028cc70be68edff56280458c71a43df4240300f4',
    'model-00004-of-00004.safetensors':'1379a7cf8c0b8555a39ab65a47e830e0eb45e776b46045734da3afd73c09eea2',
}

def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):value.update(block)
    return value.hexdigest()

def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')

def split_reasoning(text):
    if '</think>' in text:reasoning,content=text.rsplit('</think>',1)
    else:reasoning,content='',text
    content=content.lstrip()
    for marker in ('<|im_end|>','<|endoftext|>'):
        content=content.removesuffix(marker).rstrip()
    return reasoning.removeprefix('<think>'),content

def deadline(_signum,_frame):raise TimeoutError('per-task generation deadline')

class TimingStreamer:
    def __init__(self,tokenizer,started):
        self.tokenizer,self.started=tokenizer,started
        self.prompt_seen=False;self.generated=[];self.first_at=None;self.reasoning_end_at=None;self.ended_at=None
        self.close_ids=tokenizer.encode('</think>',add_special_tokens=False)
    def put(self,value):
        values=value.tolist()
        if values and isinstance(values[0],list):values=values[0]
        if not self.prompt_seen:self.prompt_seen=True;return
        now=time.monotonic()
        if self.first_at is None:self.first_at=now
        self.generated.extend(values)
        if self.reasoning_end_at is None and self.generated[-len(self.close_ids):]==self.close_ids:
            self.reasoning_end_at=now
    def end(self):self.ended_at=time.monotonic()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--max-tasks',type=int,choices=range(1,9),default=8)
    parser.add_argument('--max-new-tokens',type=int,choices=(512,2048,4096,8192),default=512)
    parser.add_argument('--task-timeout',type=int,default=180)
    parser.add_argument('--disable-thinking',action='store_true')
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    model=args.model.resolve(strict=True);manifest=args.manifest.resolve(strict=True)
    required={**EXPECTED_SHARDS,
        'config.json':None,'generation_config.json':None,'chat_template.jinja':None,
        'tokenizer.json':'06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523',
        'tokenizer_config.json':None,'processor_config.json':None,'model.safetensors.index.json':None}
    pins={}
    for name,expected in required.items():
        path=model/name
        if not path.is_file():raise FileNotFoundError(path)
        actual=sha(path);pins[name]=actual
        if expected and actual!=expected:raise ValueError(name+' differs from official LFS SHA-256')
    frozen=json.loads(manifest.read_text())
    source=Path(tasks.__file__);interpreter=source.with_name('lilith_rl_tasks.py')
    if frozen['verifier_sha256']!=sha(source) or frozen['interpreter_sha256']!=sha(interpreter):
        raise ValueError('frozen task verifier source mismatch')
    full_pack=list(tasks.tasks());task_pins={item['id']:item for item in frozen['tasks']}
    for task in full_pack:
        if task_pins[task.id]['prompt_sha256']!=tasks.digest(task.prompt):raise ValueError('prompt drift')
    pack=full_pack[:args.max_tasks]
    if args.check:
        print(json.dumps({'status':'prepared','gpu_used':False,'revision':REVISION,'files':pins}));return
    if os.environ.get('LILITH_MIMO_GPU_HANDOFF')!='1':
        raise RuntimeError('refusing GPU use without LILITH_MIMO_GPU_HANDOFF=1')
    args.output.mkdir(parents=True,exist_ok=False)
    protocol={'schema':'lilith/mimo-coding-trial@1','repo_id':REPO_ID,'revision':REVISION,
        'files_sha256':pins,'manifest':str(manifest),'manifest_sha256':sha(manifest),
        'verifier_sha256':sha(source),'interpreter_sha256':sha(interpreter),
        'driver_sha256':sha(__file__),'template':'exact checkpoint chat_template.jinja',
        'enable_thinking':not args.disable_thinking,'reasoning_split':'last literal </think> marker',
        'generation':{'do_sample':True,'temperature':0.6,'top_k':20,'top_p':0.95,
                      'max_new_tokens':args.max_new_tokens,'seed':SEED,
                      'per_task_wall_seconds':args.task_timeout},
        'selected_tasks':[task.id for task in pack],
        'comparison':{'task_pack':'same frozen eight v2 tasks as Gemma/Ptah deployment comparison',
            'output_cap_equal':args.max_new_tokens==512,'backend_precision_equal':False,'prompt_template_equal':False,
            'notes':['MiMo uses its required template/reasoning contract; Gemma/Ptah used session-service rendering.',
                     'MiMo is official BF16 Transformers on HIP; Gemma was quantized Vulkan and Ptah quantized HIP.',
                     'One sampled observation per public development task is not a promotion gate.']}}
    write(args.output/'protocol.json',protocol)
    result={'status':'running','services_stopped':True,'quality_claim':False,'records':0}
    write(args.output/'result.json',result)
    started=time.monotonic();records=[]
    try:
        tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
        model_object=AutoModelForImageTextToText.from_pretrained(model,local_files_only=True,
            dtype=torch.bfloat16,device_map={'':0},low_cpu_mem_usage=True,attn_implementation='eager')
        model_object.eval();load_seconds=time.monotonic()-started
        environment={'python':platform.python_version(),'torch':torch.__version__,
            'transformers':__import__('transformers').__version__,'device':torch.cuda.get_device_name(0),
            'load_seconds':load_seconds,'allocated_bytes':torch.cuda.memory_allocated(0),
            'reserved_bytes':torch.cuda.memory_reserved(0)}
        write(args.output/'environment.json',environment)
        for index,task in enumerate(pack):
            messages=[{'role':'user','content':task.prompt}]
            encoded=tokenizer.apply_chat_template(messages,add_generation_prompt=True,
                tokenize=True,return_dict=True,return_tensors='pt',
                enable_thinking=not args.disable_thinking)
            encoded={key:value.to('cuda:0') for key,value in encoded.items() if torch.is_tensor(value)}
            torch.manual_seed(SEED+index);torch.cuda.manual_seed_all(SEED+index)
            torch.cuda.synchronize();task_started=time.monotonic()
            timing=TimingStreamer(tokenizer,task_started)
            old_handler=signal.signal(signal.SIGALRM,deadline)
            signal.setitimer(signal.ITIMER_REAL,args.task_timeout)
            try:
                with torch.inference_mode():
                    generated=model_object.generate(**encoded,do_sample=True,temperature=0.6,top_k=20,
                        top_p=0.95,max_new_tokens=args.max_new_tokens,use_cache=True,streamer=timing)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
            torch.cuda.synchronize();elapsed=time.monotonic()-task_started
            input_tokens=encoded['input_ids'].shape[-1]
            output_ids=generated[0,input_tokens:]
            raw=tokenizer.decode(output_ids,skip_special_tokens=False)
            reasoning,content=split_reasoning(raw)
            answer_present=args.disable_thinking or '</think>' in raw
            evidence=tasks.verify(content,task) if answer_present else None
            completion=('answer_present' if answer_present else
                        'completion_budget_exhausted' if output_ids.shape[-1]>=args.max_new_tokens else
                        'reasoning_incomplete')
            eos=model_object.generation_config.eos_token_id
            eos_ids={eos} if isinstance(eos,int) else set(eos or [])
            record={'task_id':task.id,'family':task.family,'split':task.split,
                'prompt_sha256':tasks.digest(task.prompt),'input_tokens':input_tokens,
                'output_tokens':int(output_ids.shape[-1]),'elapsed_seconds':elapsed,
                'raw_output':raw,'reasoning':reasoning,'content':content,'completion':completion,
                'eos_reached':int(output_ids[-1]) in eos_ids,
                'time_to_first_token_seconds':None if timing.first_at is None else timing.first_at-task_started,
                'reasoning_seconds':None if timing.reasoning_end_at is None or timing.first_at is None else timing.reasoning_end_at-timing.first_at,
                'answer_seconds':None if timing.reasoning_end_at is None else time.monotonic()-timing.reasoning_end_at,
                'evidence':evidence,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(0)}
            records.append(record);write(args.output/'records.json',records)
            print(json.dumps({'task':task.id,'completion':completion,
                              'passed':None if evidence is None else evidence['passed'],'seconds':elapsed,
                              'output_tokens':record['output_tokens']}),flush=True)
        scores={split:{'passed':sum(r['evidence']['passed'] for r in records if r['split']==split and r['evidence'] is not None),
                       'evaluated':sum(r['split']==split and r['evidence'] is not None for r in records),
                       'completion_budget_exhausted':sum(r['split']==split and r['completion']=='completion_budget_exhausted' for r in records),
                       'total_tasks':sum(r['split']==split for r in records)}
                for split in ('train','development_holdout')}
        result.update(status='completed',records=len(records),scores=scores,
                      total_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(0))
    except Exception as error:
        result.update(status='interrupted',records=len(records),error=type(error).__name__+': '+str(error),
                      failed_task=task.id if 'task' in locals() else None)
        raise
    finally:
        write(args.output/'result.json',result)

if __name__=='__main__':main()
