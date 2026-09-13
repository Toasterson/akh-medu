#!/usr/bin/env python3
"""Prepare and supervise exactly two bounded local training conditions."""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from bounded_process import ExperimentStop, run_bounded
from build_curriculum import digest, now
from environment import world


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def unchanged(files):
    return all(Path(p).is_file() and digest(p)==h for p,h in files.items())


def prepare(args):
    root = args.out.resolve()
    inputs = []
    corpora = []
    for name in ['base', 'repair']:
        corpus = getattr(args, name+'_corpus').resolve()
        preflight = getattr(args, name+'_preflight').resolve()
        check = read(preflight)
        if not check['passed'] or not check['mask_prompt'] or check['max_seq_length'] != 4096:
            raise ValueError('Both conditions need passing 4096-token masked preflights')
        if check['splits']['train']['decisions'] > 396:
            raise ValueError('396 updates must cover every training row')
        if not all(digest(corpus/p)==h for p,h in check['corpus_files'].items()):
            raise ValueError('Corpus differs from preflight')
        corpora.append((name,corpus,preflight))
        inputs += list(corpus.glob('*.json*')) + [preflight]
    if any(digest(corpora[0][1]/name) != digest(corpora[1][1]/name)
           for name in ['valid.jsonl','test.jsonl']):
        raise ValueError('Comparisons must preserve the same held-out files')
    # Predeclare a second development world without selecting by model outcome.
    ids = set()
    for _,corpus,_ in corpora:
        rows=[json.loads(s) for s in (corpus/'train.jsonl').read_text().splitlines()]
        ids.update(re.findall(r'\b(?:job|unit)-\d+\b',json.dumps([x['messages'] for x in rows])))
    for seed in [83,107]:
        state,_=world(seed)
        if ids & set(re.findall(r'\b(?:job|unit)-\d+\b',json.dumps(state))):
            raise ValueError('Development world identifiers overlap training')
    root.mkdir(parents=True, exist_ok=False)
    source = root/'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path,source/path.name)
    shutil.copy2(args.bridge,source/'akh_episode_bridge')
    model = args.model.resolve()
    inputs += [p for p in model.iterdir() if p.is_file()] + list(source.iterdir())
    image = subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    steps=[]
    for name,corpus,preflight in corpora:
        pilot=root/name/'pilot'
        steps.append(dict(id=name+'_train',kind='train',output=str(pilot),timeout=3660,
            argv=[sys.executable,str(source/'train_pilot.py'),'--model',str(model),
                  '--corpus',str(corpus),'--preflight',str(preflight),'--out',str(pilot),
                  '--objective','tool_name_balanced','--iters','396','--timeout-seconds','3600','--full-coverage']))
        probe=root/name/'teacher-context'
        steps.append(dict(id=name+'_probe',kind='probe',output=str(probe),timeout=240,
            argv=[sys.executable,str(source/'probe_choices.py'),'--model',str(model),
                  '--adapter',str(pilot/'adapter'),'--corpus',str(corpora[0][1]),'--out',str(probe)]))
        for label,seed,guard in [('heldout83',83,False),('fresh107',107,False),('sources83',83,True)]:
            output=root/name/label
            argv=[sys.executable,str(source/'run_exam.py'),'--model',str(model),'--adapter',str(pilot/'adapter'),
                  '--bridge',str(source/'akh_episode_bridge'),'--out',str(output),'--image',image,
                  '--protocol','flat','--contract-family','subject_first_colon','--seed',str(seed),
                  '--sampling-seed','83','--phases','acquire,cold_graph,cold_empty','--max-turns','24',
                  '--receipt-namespace','adapter-heldout']
            if guard:
                argv.append('--validate-sources')
            steps.append(dict(id=name+'_'+label,kind='exam',output=str(output),timeout=600,argv=argv))
    plan=dict(schema='akh-r1-exposure-plan-v1',created_at=now(),root=str(root),
              lock_file=str(args.lock_file.resolve()),files={str(p):digest(p) for p in inputs},
              image_id=image,steps=steps,total_timeout_seconds=9000,min_free_bytes=5*2**30,
              source_validation_is_separate_condition=True,new_runpod_gpu_spend_usd=0,
              release_certification=False,automatic_promotion=False,
              limits='Two 396-update training runs only; one at a time; stop on process/input/numerical failure. Negative capability results are retained and do not trigger retries.')
    write(root/'plan.json',plan)
    (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now(),plan_sha256=digest(root/'plan.json')))
    print(json.dumps(dict(stage='prepared',plan=str(root/'plan.json'),steps=len(steps))))


def check_result(step):
    output=Path(step['output'])
    if step['kind']=='train':
        m=read(output/'manifest.json')
        log=(output/'training.log').read_text(errors='replace')
        if m.get('exit_code')!=0 or not m.get('input_hashes_unchanged'):
            raise RuntimeError('Training failed or its inputs changed')
        updates = step.get('updates', 396)
        if not re.search(rf'^Iter {updates}: Train loss ',log,re.M) or re.search(r'(?:Train|Val) loss (?:nan|inf)',log,re.I):
            raise RuntimeError('Training did not finish the budget with finite reported losses')
        if not m.get('adapter_files') or not unchanged(m['adapter_files']):
            raise RuntimeError('Training adapter receipt is missing or changed')
        result=dict(updates=updates,adapter_files=m['adapter_files'],finished_at=m['finished_at'])
        if m.get('audit_initialization'):
            initial=read(output/'initialization.json');cost=read(output/'training-cost.json')
            if digest(initial['path'])!=initial['sha256'] or not cost['completed'] or cost['iterations']!=updates:
                raise RuntimeError('Initialization/cost receipt mismatch')
            if cost['initialization_sha256']!=initial['sha256']:raise RuntimeError('Cost initialization mismatch')
            result.update(initialization=initial,cost=cost)
        if m.get('paired_row_order'):
            order=(output/'training-row-order.jsonl').read_text().splitlines()
            visits=[json.loads(line) for line in order]
            if len(visits)!=updates or any(v['epoch']!=0 for v in visits) or sorted(v['index'] for v in visits)!=list(range(updates)):
                raise RuntimeError('Paired training did not consume each row exactly once')
            result['row_order']=[v['index'] for v in visits]
        return result
    result=read(output/'results.json')
    if step['kind']=='registry':
        if not result.get('input_hashes_unchanged') or read(output/'status.json')['stage']!='completed':
            raise RuntimeError('Registry cycle failed or its inputs changed')
        return dict(scores=result['scores'],certification=False)
    if step['kind']=='binding':
        summary=read(output/'summary.json')
        if not result.get('input_hashes_unchanged') or not summary['all_checks_passed']:
            raise RuntimeError('Binding evaluation audit failed')
        return summary
    if step['kind']=='exam':
        if not result.get('artifact_hashes_unchanged'):
            raise RuntimeError('Exam input hashes changed')
        return dict(scores=result['scores'],certification=False)
    if not read(output/'manifest.json').get('files_unchanged') or not result.get('all_tokenization_checks_pass'):
        raise RuntimeError('Probe inputs or tokenization checks failed')
    return result


def execute(path):
    path=path.resolve()
    root=path.parent
    if digest(path)!=(root/'plan.sha256').read_text().strip():
        raise ValueError('Prepared plan changed')
    plan=read(path)
    lock_path=Path(plan['lock_file'])
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if read(root/'status.json')['stage']!='prepared':
            raise ValueError('Refusing replay of an already started experiment')
        status=dict(stage='running',pid=os.getpid(),started_at=now(),updated_at=now(),
                    plan_sha256=digest(path),completed_steps=[],release_certification=False)
        write(root/'status.json',status)
        deadline=time.monotonic()+plan['total_timeout_seconds']
        def stop(signum,frame):
            raise ExperimentStop(f'Received signal {signum}')
        handlers={sig:signal.signal(sig,stop) for sig in [signal.SIGTERM,signal.SIGINT]}
        try:
            for step in plan['steps']:
                if not unchanged(plan['files']):
                    raise RuntimeError('Frozen experiment inputs changed')
                status.update(active_step=step['id'],kind=step['kind'],output=step['output'])
                def heartbeat(pid):
                    if time.monotonic()>=deadline:
                        raise ExperimentStop('Total experiment wall-time budget exhausted')
                    free=shutil.disk_usage(root).free
                    if free<plan['min_free_bytes']:
                        raise ExperimentStop('Desktop disk below five GiB')
                    status.update(child_pid=pid,updated_at=now(),free_bytes=free)
                    if step['kind']=='train':
                        logfile=Path(step['output'])/'training.log'
                        if logfile.exists():
                            contents=logfile.read_text(errors='replace')
                            lines=[s for s in contents.splitlines() if s.startswith('Iter ')]
                            status['training_progress']=lines[-1] if lines else 'loading'
                            if re.search(r'(?:Train|Val) loss (?:nan|inf)',contents,re.I):raise ExperimentStop('Nonfinite reported training loss')
                            peaks=re.findall(r'Peak mem ([0-9.]+) GB',contents)
                            if peaks and plan.get('max_reported_training_peak_mlx_bytes'):
                                status['reported_peak_mlx_bytes']=float(peaks[-1])*1e9
                                if status['reported_peak_mlx_bytes']>plan['max_reported_training_peak_mlx_bytes']:raise ExperimentStop('Reported MLX memory peak exceeded the declared training guard')
                        manifest=Path(step['output'])/'manifest.json'
                        if manifest.exists():
                            try:status['training_pid']=read(manifest).get('pid')
                            except json.JSONDecodeError:pass
                    write(root/'status.json',status)
                code=run_bounded(step['argv'],root/(step['id']+'.log'),
                                 min(step['timeout'],max(.1,deadline-time.monotonic())),heartbeat)
                if code:
                    raise RuntimeError(f'{step["id"]} exited {code}; inspect its log')
                result=check_result(step)
                status['completed_steps'].append(dict(id=step['id'],result=result,finished_at=now()))
                status.pop('child_pid',None)
                status.pop('training_pid',None)
                status.pop('training_progress',None)
                write(root/'status.json',status)
            if not unchanged(plan['files']):
                raise RuntimeError('Frozen inputs changed at completion')
            write(root/'results.json',dict(completed_at=now(),steps=status['completed_steps'],
                                          input_hashes_unchanged=True,certification=False,automatic_promotion=False))
            status.update(stage='completed',active_step=None,updated_at=now())
            write(root/'status.json',status)
        except BaseException as error:
            status.update(stage='stopped' if isinstance(error,(ExperimentStop,TimeoutError,KeyboardInterrupt)) else 'failed',
                          reason=str(error),updated_at=now())
            write(root/'status.json',status)
            raise
        finally:
            for sig,handler in handlers.items():
                signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare')
    for name in ['out','model','base-corpus','base-preflight','repair-corpus','repair-preflight','bridge','lock-file']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image',default='akh-medu/native-ocr-offline:recovery-20260910')
    e=sub.add_parser('execute')
    e.add_argument('plan',type=Path)
    args=parser.parse_args()
    prepare(args) if args.action=='prepare' else execute(args.plan)


if __name__=='__main__':
    main()
