#!/usr/bin/env python3
"""Bounded local adapter diagnostic with frozen inputs and an audited stop limit."""
import argparse
import importlib.metadata
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

from build_curriculum import digest, now


class TrainingStop(Exception):
    def __init__(self, signum):
        self.signum = signum


def validate_budget(iters, timeout, full_coverage, examples):
    max_iters, max_seconds = (512, 3600) if full_coverage else (128, 1800)
    if not 1 <= iters <= max_iters or not 1 <= timeout <= max_seconds:
        raise ValueError(f'Pilot limit: 1-{max_iters} updates, 1-{max_seconds} seconds')
    if full_coverage and iters < examples:
        raise ValueError('Full-coverage budget must include at least one pass over training rows')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--iters', type=int, default=80)
    parser.add_argument('--timeout-seconds', type=int, default=900)
    parser.add_argument('--objective', choices=['token_mean', 'tool_name_balanced'], default='token_mean')
    parser.add_argument('--audit-initialization', action='store_true')
    parser.add_argument('--expected-initialization-sha256')
    parser.add_argument('--paired-row-order', action='store_true')
    parser.add_argument('--full-coverage', action='store_true',
                        help='Explicit larger diagnostic budget: at most 512 updates and 3600 seconds')
    args = parser.parse_args()
    if args.expected_initialization_sha256 and not args.audit_initialization:
        parser.error('Expected initialization requires initialization audit')
    if args.audit_initialization and args.objective!='token_mean':
        parser.error('Audited entrypoint currently requires token_mean')
    if args.paired_row_order and not args.audit_initialization:parser.error('Paired order requires audited entrypoint')
    for key in ['model', 'corpus', 'preflight', 'out']:
        setattr(args, key, getattr(args, key).resolve())
    preflight = json.loads(args.preflight.read_text())
    try:
        validate_budget(args.iters, args.timeout_seconds, args.full_coverage,
                        preflight['splits']['train']['decisions'])
    except ValueError as error:
        parser.error(str(error))
    if not preflight['passed'] or not preflight['mask_prompt']:
        raise ValueError('A passing prompt-mask and token-length preflight is required')
    for package, version in preflight['packages'].items():
        if importlib.metadata.version(package) != version:
            raise ValueError('Training packages changed after preflight')
    for name, checksum in preflight['corpus_files'].items():
        if digest(args.corpus / name) != checksum:
            raise ValueError('Corpus changed after preflight')
    for name, checksum in preflight['tokenizer_files'].items():
        if digest(args.model / name) != checksum:
            raise ValueError('Tokenizer does not match the preflight')
    args.out.mkdir(parents=True, exist_ok=False)
    config = dict(model=str(args.model), data=str(args.corpus), adapter_path=str(args.out / 'adapter'),
                  train=True, test=False, seed=83, num_layers=16, batch_size=1, iters=args.iters,
                  learning_rate=0.00005, steps_per_report=10, steps_per_eval=40, val_batches=16,
                  save_every=40, max_seq_length=preflight['max_seq_length'], grad_checkpoint=True,
                  mask_prompt=True, lora_parameters=dict(rank=8, dropout=0.0, scale=16.0))
    if args.expected_initialization_sha256:config['expected_initialization_sha256']=args.expected_initialization_sha256
    if args.paired_row_order:config['paired_row_order']=True
    config_path = args.out / 'config.json'
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    source = args.out / 'source'
    source.mkdir()
    for filename in ['train_pilot.py', 'choice_loss.py', 'train_choice_entry.py', 'train_audited_entry.py','paired_batches.py']:
        shutil.copy2(Path(__file__).with_name(filename), source / filename)
    files = ([p for p in args.model.iterdir() if p.is_file()]
             + list(args.corpus.glob('*.json*')) + [args.preflight, config_path] + list(source.iterdir()))
    manifest = dict(started_at=now(), scope='local external diagnostic adapter; no release foundation selection',
                    files={str(p):digest(p) for p in files}, new_runpod_gpu_spend_usd=0,
                    timeout_seconds=args.timeout_seconds, release_certification=False,
                    objective=args.objective,
                    audit_initialization=args.audit_initialization,
                    paired_row_order=args.paired_row_order,
                    full_coverage_budget=args.full_coverage,
                    packages={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers']})
    manifest_path = args.out / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    entry = (['-m', 'mlx_lm', 'lora'] if args.objective == 'token_mean'
             else [str(source / 'train_choice_entry.py')])
    if args.audit_initialization:entry=[str(source/'train_audited_entry.py')]
    with (args.out / 'training.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', *entry, '--config',
                                    str(config_path), '--train'], stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        manifest['pid'] = process.pid
        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
        print(json.dumps(dict(stage='training', pid=process.pid, out=str(args.out))), flush=True)
        def stop(signum, frame):
            raise TrainingStop(signum)
        old_handler = signal.signal(signal.SIGTERM, stop)
        try:
            code = process.wait(timeout=args.timeout_seconds)
        except (subprocess.TimeoutExpired, KeyboardInterrupt, TrainingStop) as error:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            code = (124 if isinstance(error, subprocess.TimeoutExpired)
                    else 128 + error.signum if isinstance(error, TrainingStop) else 130)
            manifest['stop_reason'] = type(error).__name__
        finally:
            signal.signal(signal.SIGTERM, old_handler)
    manifest.update(finished_at=now(), exit_code=code,
                    input_hashes_unchanged=all(digest(Path(p))==h for p,h in manifest['files'].items()),
                    adapter_files={str(p):digest(p) for p in (args.out / 'adapter').glob('*') if p.is_file()})
    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:manifest[k] for k in ['exit_code','input_hashes_unchanged','finished_at']}), flush=True)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
