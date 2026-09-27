#!/usr/bin/env python3
"""Frozen no-update retention evaluation plus exact KLPO cold reload."""
import argparse, hashlib, json, os, time
from pathlib import Path
import lilith_rl
import lilith_rl_tasks as old_tasks
import lilith_rl_tasks_v2 as v2
import lilith_rl_probe as probe

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2)+'\n'); os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--parent',type=Path,required=True); p.add_argument('--experiment',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--retention-manifest',type=Path,required=True); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False); started=time.monotonic()
    frozen=json.loads(a.retention_manifest.read_text())
    observed={'retention_probe_sha256':digest(__file__),'protocol_sha256':digest(a.protocol),
              'v1_tasks_sha256':digest(old_tasks.__file__),'v2_artifact_contract_sha256':digest(v2.__file__),
              'trajectory_core_sha256':digest(lilith_rl.__file__),'runner_core_sha256':digest(probe.__file__)}
    if any(frozen[k]!=value for k,value in observed.items()): raise RuntimeError('retention provenance drift')
    report={'schema':'lilith/rl-retention@2','status':'running','pid':os.getpid(),**observed,
            'retention_manifest_sha256':digest(a.retention_manifest),'experiment_result_sha256':digest(a.experiment/'result.json'),'deployed_weights_changed':False}
    write(a.output/'result.json',report)
    global torch
    import torch
    probe.torch=torch
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    model,tokenizer=probe.load(a)
    initial=probe.snapshot(model); previous=json.loads((a.experiment/'result.json').read_text())
    task_ids=json.loads(a.protocol.read_text())['retention']['task_ids']; index={t.id:t for t in old_tasks.tasks()}; selected=[index[x] for x in task_ids]
    def verify(text,task):
        artifact,presentation=v2.extract_artifact(text); result=old_tasks.verify(artifact,task)
        result.update(generated_text_sha256=lilith_rl.digest(text),artifact_sha256=lilith_rl.digest(artifact),presentation=presentation,formatting_compliant=presentation=='raw_source')
        return result
    prompts={}
    for task in selected:
        rendered=tokenizer.apply_chat_template([{'role':'user','content':task.prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prompts[task.id]=(rendered,tokenizer.encode(rendered,add_special_tokens=False))
    cfg=json.loads((a.experiment/'config.json').read_text()); verifier_hash=digest(__file__); tokenizer_hash=digest(a.parent/'tokenizer.json'); arms={}
    for arm in ('base','sft','klpo'):
        probe.restore(model,initial)
        if arm!='base': set_peft_model_state_dict(model,load_file(str(a.experiment/f'{arm}-adapter/adapter_model.safetensors')))
        pol=probe.policy(model,tokenizer_hash); pol['version']=arm+'-retention-'+pol['adapter_sha256'][:16]
        rows=[]
        for task in selected:
            rendered,ids=prompts[task.id]
            row=probe.rollout(model,tokenizer,task,rendered,ids,pol,arm+'-retention',cfg,verify,lilith_rl.seal_record,verifier_hash,lilith_rl.sample_policy_tokens)
            rows.append(row); write(a.output/f'{arm}-trajectories.json',rows)
        arms[arm]={'passed':sum(x['reward']['value']==1 for x in rows),'completed':sum(x['completion']['status']=='completed' for x in rows),'total':len(rows),'adapter_sha256':pol['adapter_sha256']}
    candidate=a.experiment/previous['cold_reload_candidate']/'adapter_model.safetensors'; reference=load_file(str(a.experiment/'cold-reference.safetensors'))
    probe.restore(model,initial); set_peft_model_state_dict(model,load_file(str(candidate)))
    model.eval()
    with torch.no_grad(): actual=probe.scores(model,reference['input_ids'].tolist(),[len(reference['input_ids'])-1]).cpu()
    report.update(status='passed',arms=arms,cold_reload_arm=previous['cold_reload_arm'],cold_reload_candidate_sha256=digest(candidate),cold_reload_exact=bool(torch.equal(actual,reference['logps'])),previous_pid=previous['pid'],elapsed_seconds=time.monotonic()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated())
    if not report['cold_reload_exact']: raise RuntimeError('cold reload mismatch')
    write(a.output/'result.json',report)

if __name__=='__main__': main()
