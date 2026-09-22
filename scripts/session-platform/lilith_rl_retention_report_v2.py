#!/usr/bin/env python3
"""Independent CPU audit of retention trajectories and adapter provenance."""
import argparse, hashlib, json
from pathlib import Path
import lilith_rl as core
import lilith_rl_tasks as old
import lilith_rl_tasks_v2 as v2

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit(run,experiment,protocol,manifest,probe):
    run,experiment,protocol,manifest,probe=map(Path,(run,experiment,protocol,manifest,probe))
    receipt=json.loads((run/'result.json').read_text()); primary=json.loads((experiment/'result.json').read_text()); frozen=json.loads(manifest.read_text()); plan=json.loads(protocol.read_text())
    observed={'retention_probe_sha256':digest(probe),'protocol_sha256':digest(protocol),'v1_tasks_sha256':digest(old.__file__),
              'v2_artifact_contract_sha256':digest(v2.__file__),'trajectory_core_sha256':digest(core.__file__)}
    if any(receipt[k]!=v or frozen[k]!=v for k,v in observed.items()): raise ValueError('retention provenance drift')
    if receipt['experiment_result_sha256']!=digest(experiment/'result.json') or receipt['retention_manifest_sha256']!=digest(manifest): raise ValueError('input artifact drift')
    if receipt['status']!='passed' or receipt['previous_pid']!=primary['pid'] or receipt['pid']==primary['pid']: raise ValueError('cold process provenance invalid')
    if receipt['cold_reload_arm']!='klpo' or receipt['cold_reload_exact'] is not True: raise ValueError('exact cold reload not established')
    candidate=experiment/'klpo-adapter/adapter_model.safetensors'
    if digest(candidate)!=primary['klpo']['adapter_file_sha256'] or receipt['cold_reload_candidate_sha256']!=digest(candidate): raise ValueError('candidate file drift')
    expected_hashes={'base':primary['initial_policy']['adapter_sha256'],'sft':primary['sft']['adapter_sha256'],'klpo':primary['klpo']['adapter_sha256']}
    tasks={t.id:t for t in old.tasks()}; ids=plan['retention']['task_ids']; arms={}
    for arm in ('base','sft','klpo'):
        if receipt['arms'][arm]['adapter_sha256']!=expected_hashes[arm]: raise ValueError('adapter tensor fingerprint drift')
        rows=json.loads((run/f'{arm}-trajectories.json').read_text()); seen=set(); passed=completed=0
        for row in rows:
            task=tasks[row['task']['id']]
            if task.id in seen or task.id not in ids: raise ValueError('duplicate or unexpected retention task')
            seen.add(task.id); policy=dict(primary['initial_policy']); policy['adapter_sha256']=expected_hashes[arm]; policy['version']=arm+'-retention-'+expected_hashes[arm][:16]
            expected={'id':task.id,'family':task.family,'split':task.split,'prompt_sha256':core.digest(task.prompt)}
            core.validate_record(row,policy,digest(probe),for_update=False,expected_task=expected)
            source=row['completion']['generated_text']; artifact,presentation=v2.extract_artifact(source); evidence=old.verify(artifact,task)
            evidence.update(generated_text_sha256=core.digest(source),artifact_sha256=core.digest(artifact),presentation=presentation,formatting_compliant=presentation=='raw_source')
            if row['completion']['status']!='completed': evidence.update(artifact_verifier_passed_before_completion_gate=evidence['passed'],passed=False,completion_gate=row['completion']['status'])
            if evidence!=row['reward']['evidence']: raise ValueError('retention reward receipt mismatch')
            passed+=evidence['passed']; completed+=row['completion']['status']=='completed'
        if seen!=set(ids): raise ValueError('incomplete retention arm')
        arms[arm]={'passed':passed,'completed':completed,'total':len(rows),'adapter_sha256':expected_hashes[arm]}
        if receipt['arms'][arm]!=arms[arm]: raise ValueError('retention aggregate mismatch')
    return {'schema':'lilith/rl-retention-independent-replay@2','complete':True,'arms':arms,'cold_reload_exact':True,
            'candidate_sha256':digest(candidate),'quality_improvement_established':False,'retention_regression_observed':False}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('run'); p.add_argument('experiment'); p.add_argument('protocol'); p.add_argument('manifest'); p.add_argument('probe'); a=p.parse_args()
    print(json.dumps(audit(a.run,a.experiment,a.protocol,a.manifest,a.probe),indent=2))
