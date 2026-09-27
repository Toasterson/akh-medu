#!/usr/bin/env python3
"""Export the bounded last-query PEFT adapter for an explicit Vulkan probe.

No model merging, requantization, registry metadata or deployment activation.
"""
import argparse
import hashlib
import json
import tempfile
import shutil
from pathlib import Path

BASE_SHA256 = '43fec98c5102b1c446b4ddd0a9439f1db3a2e1f2e0b8cd143ce1ea619a9403d6'
PARENT_SHA256 = '5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d'
TARGET = 'model.language_model.layers.47.self_attn.q_proj'

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def export(adapter, output, expected_sha, receipt):
    import torch
    from safetensors.torch import load_file, save_file
    adapter, output = Path(adapter), Path(output)
    if receipt.get('status') != 'passed':
        raise ValueError('Experiment mechanics have not passed')
    if receipt.get('initial_policy', {}).get('model_sha256') != PARENT_SHA256:
        raise ValueError('Experiment receipt does not identify the pinned training parent')
    if not any(receipt.get(arm, {}).get('adapter_file_sha256') == expected_sha for arm in ('sft', 'klpo')):
        raise ValueError('Experiment receipt does not identify this candidate')
    source = adapter / 'adapter_model.safetensors'
    if digest(source) != expected_sha:
        raise ValueError('Adapter does not match the experiment receipt')
    c = json.loads((adapter / 'adapter_config.json').read_text())
    if (c.get('peft_type') != 'LORA' or c.get('target_modules') != [TARGET]
        or c.get('r') != 2 or c.get('lora_alpha') != 2 or c.get('bias') != 'none'
        or any(c.get(k) for k in ('use_dora', 'use_rslora', 'fan_in_fan_out', 'lora_bias',
                                  'modules_to_save', 'rank_pattern', 'alpha_pattern',
                                  'layer_replication', 'use_qalora', 'alora_invocation_tokens'))):
        raise ValueError('Unsupported adapter contract; refuse approximate conversion')
    source_tensors = load_file(str(source), device='cpu')
    prefix = 'base_model.model.' + TARGET
    names = [prefix + '.lora_A.weight', prefix + '.lora_B.weight']
    if set(source_tensors) != set(names):
        raise ValueError('Expected exactly the last-query A/B tensors')
    if any(source_tensors[n].dtype not in (torch.float16, torch.bfloat16, torch.float32) for n in names):
        raise ValueError('Only exactly F32-representable floating weights are supported')
    a, b = [source_tensors[n].float().contiguous() for n in names]
    if tuple(a.shape) != (2, 3840) or tuple(b.shape) != (8192, 2):
        raise ValueError('Adapter geometry does not match pinned Gemma 4 parent')
    if not all(torch.isfinite(t).all().item() for t in (a, b)):
        raise ValueError('Non-finite adapter values')
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=output.parent))
    try:
        target = staging / 'query-lora.safetensors'
        save_file({'lora_a': a, 'lora_b': b}, str(target))
        loaded = load_file(str(target), device='cpu')
        if not torch.equal(a, loaded['lora_a']) or not torch.equal(b, loaded['lora_b']):
            raise ValueError('Export reload changed tensor values')
        manifest = dict(schema='lilith/vulkan-query-adapter@1', candidate_only=True,
                        quality_certified=False, base_sha256=BASE_SHA256,
                        training_parent_sha256=PARENT_SHA256,
                        source_adapter_sha256=expected_sha, config_sha256=digest(adapter/'adapter_config.json'),
                        adapter_sha256=digest(target), target=TARGET, scale=1.0,
                        export_dtype='F32', exact_tensor_roundtrip=True,
                        limitation='Trained on BF16 parent; quantized-base transfer requires separate quality evaluation.')
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--receipt', required=True, type=Path)
    a = p.parse_args()
    print(json.dumps(export(a.adapter, a.output, a.expected_sha256, json.loads(a.receipt.read_text())), indent=2))
