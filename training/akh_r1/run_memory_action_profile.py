#!/usr/bin/env python3
"""Two-step longest-row MLX profiles; scratch adapters never become candidates."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from build_curriculum import digest,now
from run_binding_kata import load,write
from run_exposure import execute,unchanged


def prepare(args):
    from mlx_lm.tuner.datasets import ChatDataset
    from transformers import AutoTokenizer
    lessons=args.lessons.resolve();m=load(lessons/'manifest.json');lp=load(lessons/'plan.json')
    assert load(lessons/'status.json')['stage']=='completed' and load(lessons/'artifact-audit.json')['passed']
    assert unchanged(m['files']) and unchanged(lp['files'])
    model=args.model.resolve();tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    root=args.out.resolve();root.mkdir(exist_ok=False);source=root/'source';source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):shutil.copy2(path,source/path.name)
    selections={};steps=[];files={**m['files'],**m['rehearsal_files'],**lp['files']}
    for arm in ['history','compact']:
        original=lessons/('corpus-'+arm);pre=load(lessons/('preflight-'+arm+'.json'));assert pre['passed']
        corpus=root/arm/'corpus';corpus.mkdir(parents=True);selections[arm]={}
        for split in ['train','valid','test']:
            rows=[json.loads(s) for s in (original/(split+'.jsonl')).read_text().splitlines()]
            dataset=ChatDataset(rows,tokenizer,mask_prompt=True)
            sizes=[len(dataset.process(r)[0]) for r in rows];index=max(range(len(rows)),key=sizes.__getitem__);row=rows[index]
            selected=[row,row] if split=='train' else [row]
            (corpus/(split+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in selected))
            selections[arm][split]=dict(source_row=index,episode=row['episode_id'],decision=row['decision_id'],tokens=sizes[index],repeats=len(selected))
        preflight=root/arm/'preflight.json'
        subprocess.run([sys.executable,str(source/'preflight_curriculum.py'),'--corpus',str(corpus),'--tokenizer',str(model),
                        '--max-seq-length','8192','--out',str(preflight)],check=True)
        pilot=root/arm/'pilot'
        steps.append(dict(id=arm,kind='train',updates=2,output=str(pilot),timeout=660,
            argv=[sys.executable,str(source/'train_pilot.py'),'--model',str(model),'--corpus',str(corpus),'--preflight',str(preflight),
                  '--out',str(pilot),'--iters','2','--timeout-seconds','600','--objective','token_mean','--audit-initialization']))
        files[str(lessons/('preflight-'+arm+'.json'))]=digest(lessons/('preflight-'+arm+'.json'))
    write(root/'selection.json',selections)
    inputs=[p for p in root.rglob('*') if p.is_file()]+[p for p in model.iterdir() if p.is_file()]
    files.update({str(p):digest(p) for p in inputs})
    plan=dict(schema='akh-r1-memory-action-profile-v1',created_at=now(),root=str(root),lessons=str(lessons),model=str(model),files=files,steps=steps,
        lock_file=str(args.lock_file.resolve()),total_timeout_seconds=1320,min_free_bytes=5*2**30,
        max_accepted_peak_mlx_bytes=11*2**30,training_updates=4,release_certification=False,automatic_promotion=False,new_runpod_gpu_spend_usd=0,
        scope='Two scratch optimizer steps on each longest training row plus longest validation rows. Scratch weights are retained as profiling evidence only, never resumed by candidate training.')
    write(root/'plan.json',plan);(root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json',dict(stage='prepared',updated_at=now()));print(json.dumps(dict(prepared=True,selection=selections)),flush=True)


def summarize(root):
    p=load(root/'plan.json');assert load(root/'status.json')['stage']=='completed';assert unchanged(p['files'])
    results=load(root/'results.json');arms={x['id']:x['result'] for x in results['steps']}
    hashes=[v['initialization']['sha256'] for v in arms.values()]
    passed=len(arms)==2 and len(set(hashes))==1 and all(v['cost']['completed'] and v['cost']['peak_mlx_bytes']<=p['max_accepted_peak_mlx_bytes'] for v in arms.values())
    summary=dict(checked_at=now(),passed=passed,arms=arms,initialization_identical=len(set(hashes))==1,initialization_sha256=hashes[0],
                 scratch_updates=4,candidate_training_started=False,max_accepted_peak_mlx_bytes=p['max_accepted_peak_mlx_bytes'],certification=False)
    write(root/'summary.json',summary);print(json.dumps(summary,indent=2))
    if not passed:raise ValueError('Profile acceptance failed; do not start paired training')


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('mode',choices=['prepare','execute','summarize']);a.add_argument('--out',type=Path,required=True)
    for name in ['lessons','model','lock-file']:a.add_argument('--'+name,type=Path)
    args=a.parse_args()
    if args.mode=='prepare':prepare(args)
    elif args.mode=='execute':execute(args.out/'plan.json');summarize(args.out)
    else:summarize(args.out)
