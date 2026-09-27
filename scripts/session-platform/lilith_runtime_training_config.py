#!/usr/bin/env python3
"""Freeze operator-owned inputs for a managed Lilith learning experiment."""
import argparse
import json
from pathlib import Path

from adam_kaine_code_rl_tasks import verifier_identity
from lilith_adapter_mechanics import PARENT_SHA256, PARENT_CONFIG_SHA256, TARGET
from lilith_runtime_training_worker import TRAINING_SOURCES, digest, write
from lilith_training_transport import digest_file

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack-output', type=Path, required=True)
    parser.add_argument('--config-output', type=Path, required=True)
    args = parser.parse_args()
    parent = Path.home() / '.local/share/akh-medu/lilith-training/parent-5926caa4'
    subset = Path.home() / '.local/share/akh-medu/lilith-training/transcripts-20260922/sft-subset-v1.jsonl'
    manifest_path = ROOT / 'training/lilith/transcripts-20260922/sft-subset-v1-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if digest_file(subset) != manifest['private_subset_sha256']:
        raise ValueError('frozen harvested transcript subset changed')
    profile_path = Path.home() / '.local/share/akh-medu/lilith-dev-vulkan-20260921/akhsessd.json'
    profile = json.loads(profile_path.read_text())['akhs']['lilith']
    task_manifest = ROOT / 'training/lilith/adam-kaine-code-env-20260922/rl-tasks.json'
    sources = [SCRIPTS / name for name in TRAINING_SOURCES]
    sources += [parent / 'chat_template.jinja', parent / 'generation_config.json', manifest_path, task_manifest]
    config = {'schema':'lilith/runtime-training-worker@1', 'mode':'train',
        'output':str(args.stack_output.resolve() / 'worker'), 'parent':str(parent),
        'parent_pins':{'expected_parent_sha256':PARENT_SHA256,
                       'expected_config_sha256':PARENT_CONFIG_SHA256,
                       'expected_tokenizer_sha256':digest_file(parent / 'tokenizer.json'),
                       'target_module':TARGET},
        'tools_sha256':digest(profile['tools']), 'approval_policy':profile['approval_policy'],
        'source_pins':{str(path):digest_file(path) for path in sources},
        'verifier_sha256':verifier_identity(), 'transcript_dataset':str(subset),
        'task_manifest':str(task_manifest), 'task_manifest_sha256':digest_file(task_manifest),
        'transcript_dataset_sha256':manifest['private_subset_sha256'],
        'artifact_path':'clamp_u8.py', 'max_context_tokens':32768, 'max_output_tokens':16384,
        'max_iterations':16, 'online_banks':3, 'learning_rate':1e-4, 'beta':0.1,
        'reasoning':{'enabled':True, 'efficiency_experiment':False,
                     'truncation_is_resource_limit':True, 'context_discarding':False},
        'arms':['base', 'sft_distill', 'klpo_online', 'adapter_blend_0p5'],
        'selection':{'split':'development_holdout', 'metric':'summed external effect reward',
                     'ties':['sft_distill', 'klpo_online', 'adapter_blend_0p5'],
                     'sealed_eval':'paired base and selected only, after durable selection',
                     'release_selected':False},
        'retention':'earlier v1 train pack, paired base and selected after selection',
        'reward':'actual model write bound to independent broker readback and pinned finite-case verifier',
        'eligible_negative':'complete invalid tool plans receive zero reward; incomplete generations are excluded',
        'transcripts':'eight frozen train records for supervised imitation only; validation sessions never fit',
        'required_runtime':'akhrtd-managed worker through unchanged Lilith inference tool catalog'}
    if args.config_output.exists():
        raise ValueError('refuse to overwrite frozen worker configuration')
    args.config_output.parent.mkdir(parents=True, exist_ok=True)
    write(args.config_output, config)
    print(json.dumps({'config':str(args.config_output), 'sha256':digest_file(args.config_output),
                      'model_update_performed':False}))


if __name__ == '__main__':
    main()
