#!/usr/bin/env python3
"""Read-only, CPU-only adapter readiness audit. Never trains or promotes weights."""
import argparse
import hashlib
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path

PACKAGES = ('torch', 'transformers', 'peft', 'safetensors', 'datasets')
SOURCES = ('scripts/train_lora.py', 'src/training/lora.rs',
           'src/nlu/candle_backend.rs', 'src/nlu/vendored/quantized_gemma4.rs')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_corpus(manifest_path, corpus_path):
    """Validate export integrity without admitting data or changing its eligibility."""
    errors = []
    manifest = json.loads(manifest_path.read_text())
    rows = [json.loads(line) for line in corpus_path.read_text().splitlines() if line.strip()]
    digest = sha256(corpus_path)
    if digest != manifest.get('corpus_sha256'):
        errors.append('corpus_hash_mismatch')
    if type(manifest.get('examples')) is not int or len(rows) != manifest['examples']:
        errors.append('example_count_mismatch')
    if not rows:
        errors.append('empty_corpus')
    if manifest.get('split') != 'development_candidate':
        errors.append('not_development_candidate')
    if manifest.get('eligible_for_training') is not True:
        errors.append('corpus_not_admitted_for_training')
    return {'sha256': digest, 'examples': len(rows), 'errors': errors,
            'manifest': manifest, 'scope': 'integrity only; no quality or leakage certification'}


def audit(repo, model, manifest, corpus, hash_model=False):
    findings = []
    packages = {name: importlib.util.find_spec(name) is not None for name in PACKAGES}
    if not all(packages.values()):
        findings.append({'id': 'training_dependencies_missing', 'detail':
                         'Missing in this interpreter: ' + ', '.join(k for k, v in packages.items() if not v)})
    model_info = {'path': str(model), 'exists': model.is_file()}
    if model.is_file():
        with model.open('rb') as handle:
            model_info['gguf_magic'] = handle.read(4) == b'GGUF'
        model_info['bytes'] = model.stat().st_size
        if hash_model:
            model_info['sha256'] = sha256(model)
    else:
        findings.append({'id': 'model_missing', 'detail': 'No model at supplied path.'})
    if model_info.get('gguf_magic'):
        findings.append({'id': 'training_parent_not_certified', 'detail':
                         'Local inference GGUF alone does not certify a trainable HF parent or conversion lineage; separately recorded candidate lineage is not a completed training/export round trip.'})
    corpus_info = inspect_corpus(manifest, corpus)
    findings.extend({'id': error, 'detail': 'See corpus audit.'} for error in corpus_info['errors'])
    sources = {name: sha256(repo / name) for name in SOURCES}
    trainer = (repo / SOURCES[0]).read_text()
    if 'Copying {bin_path.name} as safetensors (no format conversion)' in trainer:
        findings.append({'id': 'legacy_checkpoint_conversion_placeholder', 'detail':
                         'Existing llama.cpp fallback renames/copies bytes; this is not safetensors conversion.'})
    if 'llama.cpp LoRA training is disabled:' in trainer:
        findings.append({'id': 'legacy_llamacpp_backend_disabled', 'detail':
                         'Unsafe binary relabeling is disabled; no validated converter exists.'})
    # These are explicit certification requirements, not inferred from package presence.
    # This command intentionally cannot promote a candidate on a dependency-only check.
    findings.extend([
        {'id': 'training_smoke_not_certified', 'detail':
         'This inventory audit does not ingest or certify mechanics receipts; inspect the separate exact-parent forward/backward/save-reload evidence. Vulkan inference alone is not training evidence.'},
        {'id': 'deployment_adapter_not_certified', 'detail':
         'No demonstrated adapter application in the deployed quantized Gemma path or merged/requantized artifact parity; raw registry byte loading is insufficient.'},
        {'id': 'quality_gates_not_certified', 'detail':
         'Require diagnosed failure targets, retention replay, independent family holdouts and exact deployed-artifact evaluation before promotion.'},
    ])
    return {'schema': 'lilith/adapter-readiness@v1', 'status': 'blocked',
            'ready_to_train': False, 'ready_to_deploy': False, 'weights_updated': False,
            'audit_scope': 'Interpreter/artifact inventory only; separate mechanics receipts are not consumed',
            'interpreter': sys.executable, 'platform': platform.platform(),
            'packages_discoverable_without_import': packages,
            'llama_finetune_on_path': shutil.which('llama-finetune'),
            'model': model_info, 'corpus': corpus_info, 'source_sha256': sources,
            'blockers': findings,
            'next_action': 'Establish an exact trainable parent and a pinned isolated AMD training environment, then prove a tiny non-production forward/backward/save/reload/export round trip before using admitted lessons.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--model', type=Path, default=Path.home() / '.local/share/akh-medu/models/llm/gemma4-12b.gguf')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--hash-model', action='store_true', help='Read entire model to pin its content (CPU/I/O only).')
    args = parser.parse_args()
    try:
        report = audit(args.repo, args.model, args.manifest, args.corpus, args.hash_model)
    except (OSError, ValueError, TypeError) as exc:
        print(json.dumps({'schema': 'lilith/adapter-readiness@v1', 'status': 'invalid_input',
                          'ready_to_train': False, 'ready_to_deploy': False, 'error': str(exc)}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 2  # Readiness gates unmet, even when integrity checks succeed.


if __name__ == '__main__':
    raise SystemExit(main())
