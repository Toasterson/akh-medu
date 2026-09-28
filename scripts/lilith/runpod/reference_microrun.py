#!/usr/bin/env python3
"""Frozen Lilith reference micro-run and memory probe (C6; qualification plan §4).

Modes (each writes one exclusively created JSON file and refuses an existing one):

* ``microrun``     fixed seed / data slice / config: per context, 200 weighted-SFT
                   steps + 1 sibling-preference step from the same seeded adapter;
                   records loss curve, adapter hashes, peak memory, train tok/s,
                   wall clock, then measures generation tok/s in a fresh process.
* ``memory-probe`` per context (8K/16K/32K by default) one warm-up optimizer step
                   (``first_step_including_warmup``) and one steady-state step; records
                   peak allocated/reserved memory and steady tok/s, or the OOM. Exits 3
                   and lists ``oom_contexts`` if any requested context does not fit.
* ``generate``     generation tok/s for a saved adapter (vLLM if importable, else HF).
* ``cold-reload``  fresh process: reload a saved adapter onto a re-verified parent and
                   prove file hashes, tensor identity and probe-logit identity.

The micro-run is a mechanics/throughput reference for comparing devices, not a
quality claim. Any argument that differs from ``REFERENCE`` is recorded as a
deviation so that non-reference runs cannot pass as the reference.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lilith_lora_trainer as trainer  # noqa: E402
import lilith_numerics  # noqa: E402  (on sys.path via lilith_lora_trainer's session-platform insert)

REPOSITORY = Path(__file__).resolve().parents[3]
SUBSET_MANIFEST = REPOSITORY / 'training/lilith/transcripts-20260922/sft-subset-v1-manifest.json'
END_OF_TURN_TOKEN = 106  # same terminator the runtime worker appends to transcript answers
GENERATION_PROMPT = ('Explain, step by step, how you would verify that an illumos SMF service '
                     'restarted cleanly after a configuration change.')

REFERENCE: dict[str, Any] = {
    'schema': 'lilith/reference-microrun@1',
    'seed': 73,
    'sft_steps': 200,
    'preference_steps': 1,
    'contexts': [8192, 16384],
    'optional_contexts': [32768],
    'memory_probe_contexts': [8192, 16384, 32768],
    'trainer': {'rank': 16, 'base_precision': 'bf16', 'learning_rate': 1e-4, 'max_grad_norm': 1.0,
                'preference_beta': 0.1, 'ce_chunk_tokens': 1024, 'attention_query_chunk': 512,
                'gradient_checkpointing': True, 'offload_activations': False},
    'data': {'kind': 'sft-subset-v1', 'split': 'train', 'records': 8},
    'packing': 'step k starts at record k mod n; whole records; a truncated tail is context only',
    'preference': 'record k answer (chosen) vs record k+1 answer (rejected): mechanics contrast, not a label',
    'generation': {'prompt': GENERATION_PROMPT, 'max_new_tokens': 256, 'greedy': True, 'streams': 1},
    'probe_tokens': 64,
}


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


REFERENCE_SHA256 = hashlib.sha256(canonical(REFERENCE)).hexdigest()


# ------------------------------------------------------------------------------ data

Record = tuple[tuple[int, ...], tuple[int, ...]]   # (prompt ids, answer ids)


def synthetic_records(seed: int, count: int = 8, prompt: int = 200, answer: int = 300,
                      low: int = 1000, high: int = 200_000) -> list[Record]:
    """Deterministic token records; memory/throughput only, loss curves are not comparable to real data."""
    generator = random.Random(seed)
    return [(tuple(generator.randrange(low, high) for _ in range(prompt)),
             tuple(generator.randrange(low, high) for _ in range(answer))) for _ in range(count)]


def pack_example(records: Sequence[Record], context: int, start: int, *, source: str) -> trainer.SftExample:
    """Exactly ``context`` tokens: whole records cycled from ``start``; a partial tail is unsupervised."""
    if not records:
        raise trainer.LilithTrainerError('lilith::microrun::data', 'no training records',
                                         'check the data slice')
    ids: list[int] = []
    mask: list[bool] = []
    index = start % len(records)
    while len(ids) < context:
        prompt, answer = records[index]
        room = context - len(ids)
        if len(prompt) + len(answer) <= room:
            ids += list(prompt) + list(answer)
            mask += [False] * len(prompt) + [True] * len(answer)
        else:
            tail = (list(prompt) + list(answer))[:room]
            ids += tail
            mask += [False] * len(tail)
        index = (index + 1) % len(records)
    return trainer.SftExample(tuple(ids), tuple(mask), 1.0, f'{source}:start={start % len(records)}')


def mechanics_pair(records: Sequence[Record], k: int, *, source: str) -> trainer.PreferencePair:
    prompt, chosen = records[k % len(records)]
    rejected = records[(k + 1) % len(records)][1]
    return trainer.PreferencePair(prompt, chosen, rejected, 1.0, source + ':mismatched-answer-contrast')


def load_subset_records(path: Path, tokenizer, manifest_path: Path = SUBSET_MANIFEST) -> tuple[list[Record], dict]:
    manifest = json.loads(Path(manifest_path).read_text())
    observed = trainer.sha256_file(path)
    if observed != manifest['private_subset_sha256']:
        raise trainer.LilithTrainerError(
            'lilith::microrun::data_pin', f'{path} sha256 {observed} differs from the frozen subset manifest',
            'transfer the admitted sft-subset-v1.jsonl unchanged, or pass --synthetic-data (recorded)')
    records: list[Record] = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row['split'] != 'train':
            continue   # validation sessions never fit
        user, assistant = row['messages']
        rendered = tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True)
        prompt = tokenizer.encode(rendered, add_special_tokens=False)
        answer = tokenizer.encode(assistant['content'], add_special_tokens=False) + [END_OF_TURN_TOKEN]
        records.append((tuple(prompt), tuple(answer)))
    if len(records) != manifest['splits']['train']:
        raise trainer.LilithTrainerError('lilith::microrun::data_pin', 'train record count changed',
                                         'the frozen subset has exactly 8 train records')
    return records, {'kind': 'sft-subset-v1', 'path': str(path), 'sha256': observed,
                     'manifest_sha256': trainer.sha256_file(manifest_path), 'records': len(records)}


def check_chat_template(parent: Path, manifest_path: Path = SUBSET_MANIFEST) -> dict[str, Any]:
    """Match the parent's chat template against the pin the subset was admitted under.

    The admission manifest (``sft-subset-v1-manifest.json``) recorded the template as
    ``sha256(json.dumps(template_text))``, the digest of the JSON-encoded string, while
    the parent receipt pins file bytes. Both conventions identify the same template, so
    either may match. The one that matched is recorded; nothing else is accepted.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    template = parent / 'chat_template.jinja'
    file_digest = trainer.sha256_file(template)
    text_digest = hashlib.sha256(json.dumps(template.read_text(encoding='utf-8')).encode()).hexdigest()
    pinned = manifest['chat_template_sha256']
    convention = {file_digest: 'file_bytes', text_digest: 'json_string'}.get(pinned)
    if convention is None:
        raise trainer.LilithTrainerError(
            'lilith::microrun::template_pin', 'chat template differs from the one the subset was admitted under',
            'use the pinned parent revision; the rendered prompts would otherwise drift')
    return {'chat_template_sha256': file_digest, 'manifest_pin': pinned, 'pin_convention': convention}


# ------------------------------------------------------------------------- recording

class Report:
    """Exclusively created, progressively persisted JSON evidence."""

    def __init__(self, path: Path, initial: dict[str, Any]):
        self.path = Path(path)
        self.data = dict(initial)
        self.started = time.monotonic()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('x') as stream:
            stream.write(json.dumps(self.data, indent=2) + '\n')

    def persist(self) -> None:
        self.data['wall_clock_seconds'] = time.monotonic() - self.started
        temporary = self.path.with_suffix(self.path.suffix + '.tmp')
        temporary.write_text(json.dumps(self.data, indent=2, default=str) + '\n')
        temporary.replace(self.path)


def env_bin_first_on_path() -> str:
    """Put this interpreter's bin dir first on PATH and return it.

    vLLM/flashinfer JIT-compile kernels by running ``ninja`` from PATH; the round-2
    smoke fell back to HF because the env's ninja was not on PATH. run_smoke.sh
    exports it too; this keeps the generate subprocess right however it is launched."""
    env_bin = str(Path(sys.executable).parent)
    rest = [entry for entry in os.environ.get('PATH', '').split(os.pathsep) if entry and entry != env_bin]
    os.environ['PATH'] = os.pathsep.join([env_bin, *rest])
    return env_bin


def environment_record() -> dict[str, Any]:
    versions = {}
    for package in ('torch', 'transformers', 'peft', 'trl', 'bitsandbytes', 'vllm', 'accelerate',
                    'safetensors', 'tokenizers'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    record: dict[str, Any] = {'python': sys.version, 'platform': platform.platform(), 'versions': versions,
                              'argv': sys.argv, 'reference_sha256': REFERENCE_SHA256,
                              'ninja': shutil.which('ninja'),
                              'source_sha256': {name: trainer.sha256_file(Path(__file__).parent / name)
                                                for name in ('lilith_lora_trainer.py', 'reference_microrun.py')}}
    import torch
    record['torch_cuda'] = torch.version.cuda
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        record['device'] = {'name': torch.cuda.get_device_name(0), 'total_bytes': total, 'free_bytes_at_start': free,
                            'capability': list(torch.cuda.get_device_capability(0))}
    return record


def resolve_numerics(args) -> lilith_numerics.NumericsConfig:
    """This process runs on the host being measured, so local CUDA detection is correct
    (unlike ``lilith_runtime_training_config.py``, which authors a config for a
    possibly different host and must pass ``cuda`` explicitly)."""
    try:
        return lilith_numerics.NumericsConfig.from_args(args)
    except lilith_numerics.NumericsError as error:
        raise trainer.LilithTrainerError('lilith::microrun::numerics', str(error),
                                         'pass a valid --numerics-* combination or leave them all unset')


def numerics_manifest_under(numerics: lilith_numerics.NumericsConfig) -> dict[str, Any]:
    """The requested/observed manifest, captured while ``numerics`` is actually applied."""
    with numerics.applied():
        return numerics.manifest()


def trainer_config(args) -> trainer.TrainerConfig:
    return trainer.TrainerConfig(rank=args.rank, base_precision=args.base_precision,
                                 learning_rate=REFERENCE['trainer']['learning_rate'],
                                 max_grad_norm=REFERENCE['trainer']['max_grad_norm'],
                                 preference_beta=REFERENCE['trainer']['preference_beta'],
                                 ce_chunk_tokens=args.ce_chunk_tokens,
                                 attention_query_chunk=args.attention_query_chunk,
                                 offload_activations=args.offload_activations, seed=args.seed)


def deviations(args, data_kind: str) -> dict[str, Any]:
    observed = {'seed': args.seed, 'sft_steps': args.steps, 'contexts': list(args.contexts),
                'rank': args.rank, 'base_precision': args.base_precision,
                'ce_chunk_tokens': args.ce_chunk_tokens, 'attention_query_chunk': args.attention_query_chunk,
                'offload_activations': args.offload_activations, 'data': data_kind}
    expected = {'seed': REFERENCE['seed'], 'sft_steps': REFERENCE['sft_steps'], 'contexts': REFERENCE['contexts'],
                'rank': REFERENCE['trainer']['rank'], 'base_precision': REFERENCE['trainer']['base_precision'],
                'ce_chunk_tokens': REFERENCE['trainer']['ce_chunk_tokens'],
                'attention_query_chunk': REFERENCE['trainer']['attention_query_chunk'],
                'offload_activations': REFERENCE['trainer']['offload_activations'],
                'data': REFERENCE['data']['kind']}
    result = {key: {'reference': expected[key], 'observed': value}
              for key, value in observed.items() if value != expected[key]}
    extra = [c for c in observed['contexts'] if c not in REFERENCE['contexts'] + REFERENCE['optional_contexts']]
    if extra:
        result['undeclared_contexts'] = extra
    return result


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def probe_logits(model, token_ids: Sequence[int]):
    """Final-position logits (last 16 rows) through the trainer's own attention/CE path."""
    import torch
    base = trainer._unwrap(model)
    device = next(base.parameters()).device
    ids = torch.tensor([list(token_ids)], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        hidden = base.model(input_ids=ids, position_ids=torch.arange(ids.shape[1], device=device)[None, :],
                            use_cache=False).last_hidden_state[0, -16:]
        logits = torch.matmul(hidden, base.lm_head.weight.t()).float()
        softcap = base.config.get_text_config().final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
    return logits.cpu()


def logits_record(logits) -> dict[str, Any]:
    return {'sha256': hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest(),
            'shape': list(logits.shape), 'argmax': logits.argmax(-1).tolist(),
            'sample': logits[:, :8].tolist()}


# ------------------------------------------------------------------------------ modes

def _data(args, parent: Path) -> tuple[list[Record], dict[str, Any]]:
    if args.synthetic_data:
        return synthetic_records(args.seed), {'kind': 'synthetic', 'seed': args.seed,
                                              'comparable_loss_curve': False}
    if args.data is None:
        raise trainer.LilithTrainerError('lilith::microrun::data', 'no --data given',
                                         'pass the admitted subset path or --synthetic-data')
    info = check_chat_template(parent)
    records, record = load_subset_records(args.data, trainer.load_tokenizer(parent))
    return records, {**record, **info}


def run_microrun(args) -> int:
    import torch
    config = trainer_config(args)
    numerics = resolve_numerics(args)
    report = Report(args.output, {'schema': 'lilith/reference-microrun-result@1', 'status': 'starting',
                                  'reference': REFERENCE, 'reference_sha256': REFERENCE_SHA256,
                                  'quality_claim': False, 'release_selected': False})
    try:
        report.data['environment'] = environment_record()
        report.data['trainer_config'] = config.to_record()
        report.data['numerics'] = numerics_manifest_under(numerics)
        report.data['phase'] = 'loading'
        report.persist()
        seed_everything(args.seed)
        loaded = time.monotonic()
        model, load_receipt = trainer.load_training_model(args.parent, config, device=args.device)
        report.data['load'] = {**load_receipt, 'seconds': time.monotonic() - loaded}
        records, data = _data(args, args.parent)
        report.data['data'] = data
        report.data['deviations'] = deviations(args, data['kind'])
        report.data['is_reference'] = not report.data['deviations']
        initial = {name: tensor.clone() for name, tensor in trainer.adapter_tensor_state(model).items()}
        report.data['initial_adapter_sha256'] = trainer.adapter_tensor_sha256(model)
        report.data['contexts'] = {}
        deadline = None if args.max_wall_seconds is None else time.monotonic() + args.max_wall_seconds
        args.adapter_root.mkdir(parents=True, exist_ok=True)
        for context in args.contexts:
            key = str(context)
            entry: dict[str, Any] = {'status': 'running', 'loss_curve': []}
            report.data['contexts'][key] = entry
            report.data['phase'] = f'context-{context}'
            report.persist()
            with torch.no_grad():   # identical seeded starting adapter for every context
                for name, parameter in trainer.adapter_tensor_state(model).items():
                    parameter.copy_(initial[name])
            if trainer.adapter_tensor_sha256(model) != report.data['initial_adapter_sha256']:
                raise trainer.LilithTrainerError('lilith::microrun::reset', 'adapter reset failed',
                                                 'contexts would not start from the same policy')
            seed_everything(args.seed)
            optimizer = trainer.make_optimizer(model, config)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            for step in range(args.steps):
                if deadline is not None and time.monotonic() > deadline:
                    entry['status'] = 'budget_exhausted'
                    break
                example = pack_example(records, context, step, source=data['kind'])
                receipt = trainer.sft_step(model, optimizer, [example], config, step, numerics=numerics)
                entry['loss_curve'].append(asdict(receipt))
                if step % 10 == 0 or step == args.steps - 1:
                    report.persist()
            if entry['status'] == 'running':
                for step in range(REFERENCE['preference_steps']):
                    pair = mechanics_pair(records, step, source=data['kind'])
                    entry.setdefault('preference', []).append(
                        asdict(trainer.preference_step(model, optimizer, [pair], config, args.steps + step,
                                                        numerics=numerics)))
                entry['status'] = 'completed'
            timed = entry['loss_curve'][1:] or entry['loss_curve']
            entry['train_tokens_per_second'] = (sum(r['tokens'] for r in timed) / sum(r['seconds'] for r in timed)
                                                if timed else None)
            entry['train_tokens_per_second_includes_warmup'] = len(entry['loss_curve']) < 2
            entry['seconds'] = time.monotonic() - started
            if torch.cuda.is_available():
                entry['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                entry['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
            adapter_dir = args.adapter_root / f'ctx-{context}'
            entry['adapter'] = trainer.save_adapter(model, adapter_dir, config).to_record()
            probe_ids = list(pack_example(records, context, 0, source=data['kind'])
                             .input_ids[:REFERENCE['probe_tokens']])
            entry['probe'] = {'input_ids': probe_ids, **logits_record(probe_logits(model, probe_ids))}
            (args.adapter_root / f'ctx-{context}.probe.json').write_text(
                json.dumps({'adapter_dir': str(adapter_dir), 'adapter': entry['adapter'],
                            'probe': entry['probe'], 'trainer_config': config.to_record()}, indent=2) + '\n')
            del optimizer
            report.persist()
            if entry['status'] != 'completed':
                break
        report.data['phase'] = 'generation'
        report.persist()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        completed = [c for c in args.contexts if report.data['contexts'].get(str(c), {}).get('status') == 'completed']
        if args.generation_backend != 'none' and completed:
            generation_output = args.output.with_name(args.output.stem + '.generation.json')
            command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'generate',
                       '--parent', str(args.parent), '--adapter-dir', str(args.adapter_root / f'ctx-{completed[0]}'),
                       '--rank', str(args.rank), '--generation-backend', args.generation_backend,
                       '--output', str(generation_output), '--device', args.device]
            process = subprocess.run(command, check=False)
            report.data['generation'] = (json.loads(generation_output.read_text())
                                         if generation_output.exists() else None)
            report.data['generation_exit_code'] = process.returncode
        statuses = {entry['status'] for entry in report.data['contexts'].values()}
        report.data['status'] = 'completed' if statuses == {'completed'} else 'incomplete'
        report.data['phase'] = 'finished'
        report.persist()
        return 0 if report.data['status'] == 'completed' else 2
    except BaseException as error:
        report.data.update(status='failed', error=f'{type(error).__name__}: {error}',
                           traceback=traceback.format_exc())
        report.persist()
        raise


def memory_probe_verdict(contexts: dict[str, dict[str, Any]]) -> tuple[str, list[int]]:
    """'completed' only if every requested context fits; otherwise name the OOM contexts."""
    oom = sorted(int(context) for context, entry in contexts.items() if entry.get('status') != 'fits')
    return ('completed' if not oom else 'oom'), oom


def run_memory_probe(args) -> int:
    import torch
    if not torch.cuda.is_available():
        raise trainer.LilithTrainerError('lilith::microrun::no_cuda', 'the memory probe measures CUDA memory',
                                         'run it on the GPU pod; CPU tests cover the micro-run path instead')
    config = trainer_config(args)
    numerics = resolve_numerics(args)
    report = Report(args.output, {'schema': 'lilith/memory-probe@1', 'status': 'starting',
                                  'reference_sha256': REFERENCE_SHA256, 'contexts': {}})
    try:
        report.data['environment'] = environment_record()
        report.data['trainer_config'] = config.to_record()
        report.data['numerics'] = numerics_manifest_under(numerics)
        seed_everything(args.seed)
        model, load_receipt = trainer.load_training_model(args.parent, config, device=args.device)
        report.data['load'] = load_receipt
        records = synthetic_records(args.seed)
        report.data['data'] = {'kind': 'synthetic', 'note': 'token content does not change memory'}
        optimizer = trainer.make_optimizer(model, config)
        for context in args.contexts:
            entry: dict[str, Any] = {}
            report.data['contexts'][str(context)] = entry
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            example = pack_example(records, context, 0, source='synthetic')
            try:
                # Step 1 includes allocator/kernel warm-up and first Adam state allocation;
                # step 2 is the steady-state measurement.
                first = trainer.sft_step(model, optimizer, [example], config, 0, numerics=numerics)
                entry['first_step_including_warmup'] = {
                    'seconds': first.seconds, 'tokens_per_second': first.tokens / first.seconds}
                steady = trainer.sft_step(model, optimizer, [example], config, 1, numerics=numerics)
                total = torch.cuda.mem_get_info()[1]
                entry.update(status='fits', steady_state_seconds=steady.seconds, loss=steady.loss,
                             steady_state_tokens_per_second=steady.tokens / steady.seconds,
                             peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                             peak_reserved_bytes=torch.cuda.max_memory_reserved(), device_total_bytes=total)
                entry['headroom_fraction'] = 1 - entry['peak_reserved_bytes'] / total
                entry['meets_10pct_headroom'] = entry['headroom_fraction'] >= 0.10
            except torch.cuda.OutOfMemoryError as error:
                optimizer.zero_grad(set_to_none=True)
                entry.update(status='oom', error=str(error)[:2000],
                             peak_allocated_bytes=torch.cuda.max_memory_allocated())
            report.persist()
        status, oom = memory_probe_verdict(report.data['contexts'])
        report.data.update(status=status, oom_contexts=oom)
        report.persist()
        return 0 if status == 'completed' else 3
    except BaseException as error:
        report.data.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        report.persist()
        raise


def _generation_prompt_ids(tokenizer) -> list[int]:
    rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': GENERATION_PROMPT}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=True)
    return tokenizer.encode(rendered, add_special_tokens=False)


def _generate_vllm(args, prompt_ids: list[int], max_new: int) -> dict[str, Any]:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=str(args.parent), tokenizer=str(args.parent), dtype='bfloat16', enable_lora=True,
              max_lora_rank=args.rank, max_loras=1, max_model_len=8192, gpu_memory_utilization=0.85,
              seed=args.seed, limit_mm_per_prompt={'image': 0, 'audio': 0})
    lora = LoRARequest('lilith-microrun', 1, str(args.adapter_dir))
    warm = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate([{'prompt_token_ids': prompt_ids}], warm, lora_request=lora)
    params = SamplingParams(temperature=0.0, max_tokens=max_new, ignore_eos=True)
    started = time.perf_counter()
    outputs = llm.generate([{'prompt_token_ids': prompt_ids}], params, lora_request=lora)
    seconds = time.perf_counter() - started
    generated = len(outputs[0].outputs[0].token_ids)
    return {'backend': 'vllm', 'generated_tokens': generated, 'seconds': seconds,
            'tokens_per_second': generated / seconds, 'lora_applied': True}


def _generate_hf(args, prompt_ids: list[int], max_new: int,
                 numerics: lilith_numerics.NumericsConfig) -> dict[str, Any]:
    """The HF fallback path: real ``attn_implementation='sdpa'``, so ``numerics`` (unlike
    for the custom chunked-attention training path) can change its generated logits,
    not just its numeric noise floor."""
    import torch
    from peft import PeftModel
    from transformers import Gemma4UnifiedForConditionalGeneration
    with numerics.applied():
        base = Gemma4UnifiedForConditionalGeneration.from_pretrained(
            args.parent, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa',
            device_map={'': args.device})
        model = PeftModel.from_pretrained(base, args.adapter_dir, is_trainable=False).eval()
        ids = torch.tensor([prompt_ids], device=args.device)
        with torch.no_grad():
            model.generate(input_ids=ids, max_new_tokens=8, min_new_tokens=8, do_sample=False)
            trainer._synchronize()
            started = time.perf_counter()
            output = model.generate(input_ids=ids, max_new_tokens=max_new, min_new_tokens=max_new, do_sample=False)
            trainer._synchronize()
    seconds = time.perf_counter() - started
    generated = output.shape[1] - ids.shape[1]
    return {'backend': 'hf', 'generated_tokens': generated, 'seconds': seconds,
            'tokens_per_second': generated / seconds, 'lora_applied': True}


def run_generate(args) -> int:
    from transformers import AutoTokenizer
    numerics = resolve_numerics(args)
    report = Report(args.output, {'schema': 'lilith/generation-throughput@1', 'status': 'starting',
                                  'adapter_dir': str(args.adapter_dir), 'streams': 1})
    try:
        report.data['environment'] = environment_record()
        report.data['numerics'] = numerics_manifest_under(numerics)
        tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True)
        prompt_ids = _generation_prompt_ids(tokenizer)
        max_new = REFERENCE['generation']['max_new_tokens']
        report.data['prompt_tokens'] = len(prompt_ids)
        result = None
        if args.generation_backend in ('auto', 'vllm'):
            try:
                # vLLM manages its own kernels (flashinfer); the recorded numerics knobs
                # target torch's own SDPA dispatch and do not apply to this backend.
                result = _generate_vllm(args, prompt_ids, max_new)
            except Exception as error:   # recorded, never hidden
                report.data['vllm_error'] = f'{type(error).__name__}: {error}'[:4000]
                report.persist()
                if args.generation_backend == 'vllm':
                    raise
        if result is None:
            result = _generate_hf(args, prompt_ids, max_new, numerics)
        report.data.update(result)
        report.data['status'] = 'completed'
        report.persist()
        return 0
    except BaseException as error:
        report.data.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        report.persist()
        raise


def run_cold_reload(args) -> int:
    import torch
    numerics = resolve_numerics(args)
    probe_file = json.loads(Path(args.probe).read_text())
    config = trainer.TrainerConfig(**{key: value for key, value in probe_file['trainer_config'].items()
                                      if key not in ('lora_alpha', 'target_modules')})
    report = Report(args.output, {'schema': 'lilith/cold-reload@1', 'status': 'starting',
                                  'adapter_dir': str(args.adapter_dir), 'probe': str(args.probe)})
    try:
        report.data['environment'] = environment_record()
        report.data['numerics'] = numerics_manifest_under(numerics)
        report.data['parent_pins'] = trainer.verify_parent(args.parent)
        # Like with like: the adapter's own training precision (bf16, or an NF4 base).
        report.data['base_precision'] = config.base_precision
        base, _ = trainer.load_parent_base(args.parent, config, device=args.device)
        model, tensor_sha = trainer.reload_adapter(base, args.adapter_dir, probe_file['adapter'])
        with numerics.applied():
            observed = probe_logits(model, probe_file['probe']['input_ids'])
        recorded = logits_record(observed)
        expected_sample = torch.tensor(probe_file['probe']['sample'])
        max_abs = float((observed[:, :8] - expected_sample).abs().max())
        identical = recorded['sha256'] == probe_file['probe']['sha256']
        report.data.update(tensor_sha256=tensor_sha, files_verified=True,
                           probe_sha256_identical=identical, probe_sample_max_abs=max_abs,
                           argmax_identical=recorded['argmax'] == probe_file['probe']['argmax'],
                           optimizer_steps=0)
        passed = identical or (max_abs <= args.logit_tolerance and report.data['argmax_identical'])
        report.data['status'] = 'passed' if passed else 'failed_identity'
        report.persist()
        return 0 if passed else 1
    except BaseException as error:
        report.data.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        report.persist()
        raise


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('microrun', 'memory-probe', 'generate', 'cold-reload'), required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='admitted sft-subset-v1.jsonl (sha pinned by the manifest)')
    parser.add_argument('--synthetic-data', action='store_true', help='recorded deviation: seeded token records')
    parser.add_argument('--contexts', type=int, nargs='+')
    parser.add_argument('--steps', type=int, default=REFERENCE['sft_steps'])
    parser.add_argument('--rank', type=int, default=REFERENCE['trainer']['rank'], choices=trainer.ALLOWED_RANKS)
    parser.add_argument('--base-precision', default=REFERENCE['trainer']['base_precision'],
                        choices=('bf16', 'nf4', 'fp32'), help='fp32 is for CPU tests only (recorded deviation)')
    parser.add_argument('--ce-chunk-tokens', type=int, default=REFERENCE['trainer']['ce_chunk_tokens'])
    parser.add_argument('--attention-query-chunk', type=int, default=REFERENCE['trainer']['attention_query_chunk'])
    parser.add_argument('--offload-activations', action='store_true')
    parser.add_argument('--seed', type=int, default=REFERENCE['seed'])
    parser.add_argument('--adapter-root', type=Path)
    parser.add_argument('--adapter-dir', type=Path)
    parser.add_argument('--probe', type=Path)
    parser.add_argument('--generation-backend', choices=('auto', 'vllm', 'hf', 'none'), default='auto')
    parser.add_argument('--max-wall-seconds', type=float)
    parser.add_argument('--logit-tolerance', type=float, default=1e-3)
    parser.add_argument('--device', default='cuda:0')
    lilith_numerics.add_arguments(parser)
    args = parser.parse_args(argv)
    if args.contexts is None:
        args.contexts = list(REFERENCE['memory_probe_contexts'] if args.mode == 'memory-probe'
                             else REFERENCE['contexts'])
    if args.mode == 'microrun' and args.adapter_root is None:
        parser.error('--adapter-root is required for microrun')
    if args.mode in ('generate', 'cold-reload') and args.adapter_dir is None:
        parser.error('--adapter-dir is required')
    if args.mode == 'cold-reload' and args.probe is None:
        parser.error('--probe is required for cold-reload')
    if args.steps <= 0 or any(c <= 0 for c in args.contexts):
        parser.error('steps and contexts must be positive')
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    env_bin_first_on_path()
    mode = {'microrun': run_microrun, 'memory-probe': run_memory_probe,
            'generate': run_generate, 'cold-reload': run_cold_reload}[args.mode]
    return mode(args)


if __name__ == '__main__':
    raise SystemExit(main())
