#!/usr/bin/env python3
"""Independent CPU replay of v2 collected verifier receipts, not model grading."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import lilith_rl as core
import lilith_rl_tasks_v2 as exercises


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run, manifest):
    run,manifest=Path(run),Path(manifest)
    c=json.loads((run/'config.json').read_text());result=json.loads((run/'result.json').read_text())
    frozen=json.loads(manifest.read_text())
    if (file_hash(manifest)!=c['task_manifest_sha256'] or
        file_hash(exercises.__file__)!=c['task_module_sha256'] or
        file_hash(core.__file__)!=c['trajectory_module_sha256'] or
        file_hash(Path(exercises.__file__).with_name('lilith_rl_tasks.py'))!=frozen['interpreter_sha256']):
        raise ValueError('Frozen verifier/task/interpreter provenance mismatch')
    tasks={x.id:x for x in exercises.tasks()};arms={};outcomes=[]
    for arm in ('base','sft','klpo'):
        p=run/(arm+'-trajectories.json');records=json.loads(p.read_text()) if p.exists() else []
        seen=set();counts={s:dict(observed=0,expected=sum(t.split==s for t in tasks.values()),passed=0,
            completed=0,failures=Counter(),presentation=Counter()) for s in ('train','development_holdout')}
        for row in records:
            task=tasks[row['task']['id']]
            if task.id in seen:raise ValueError('Duplicate task in arm')
            seen.add(task.id)
            expected=dict(id=task.id,family=task.family,split=task.split,prompt_sha256=core.digest(task.prompt))
            policy=dict(result['initial_policy'])
            if arm!='base':
                policy['adapter_sha256']=result[arm]['adapter_sha256'];policy['version']=arm+'-'+policy['adapter_sha256'][:16]
            core.validate_record(row,expected_policy=policy,expected_verifier_sha256=c['task_module_sha256'],
                                 for_update=False,expected_task=expected)
            source=row['completion']['generated_text'];evidence=exercises.verify(source,task)
            evidence['generated_text_sha256']=core.digest(source)
            status=row['completion']['status']
            if status!='completed':
                evidence['artifact_verifier_passed_before_completion_gate']=evidence['passed']
                evidence['passed']=False;evidence['completion_gate']=status
            if evidence!=row['reward']['evidence']:raise ValueError('Verifier receipt mismatch: '+task.id)
            count=counts[task.split];count['observed']+=1;count['passed']+=evidence['passed']
            count['completed']+=status=='completed';count['presentation'][evidence['presentation']]+=1
            category=evidence['category'] if status=='completed' else status
            if not evidence['passed']:count['failures'][category]+=1
            outcomes.append(dict(arm=arm,task=task.id,split=task.split,passed=evidence['passed'],
                category=category,counterexample=evidence['counterexample'],record_sha256=row['record_sha256']))
        arms[arm]=counts
    return dict(schema='lilith/rl-independent-replay@2',runner_status=result['status'],arms=arms,outcomes=outcomes,
        complete=result['status']=='passed' and all(c['observed']==c['expected'] for a in arms.values() for c in a.values()),
        quality_improvement_established=False,native_illumos=False,
        note='Public finite development pack; single samples; formatting accepted by the frozen v2 contract, not retroactive rescoring.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('manifest',type=Path)
    a=p.parse_args();print(json.dumps(audit(a.run,a.manifest),indent=2))
