#!/usr/bin/env python3
"""Same-parent Gemma distill and three-step on-policy KLPO training."""
import argparse
import copy
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from lilith_adapter_mechanics import PARENT_REVISION, PARENT_SHA256, PARENT_CONFIG_SHA256, TARGET


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(4 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def seed(text):
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], 'little') % (2**63 - 1)


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode())
        h.update(str(list(tensor.shape)).encode())
        h.update(str(tensor.dtype).encode())
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def snapshot(model):
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def blend_states(left, right, alpha=0.5):
    if set(left) != set(right):
        raise RuntimeError('Adapter tensor keys differ')
    blended = {}
    for name in sorted(left):
        if left[name].shape != right[name].shape or left[name].dtype != right[name].dtype:
            raise RuntimeError('Adapter tensor layout differs: ' + name)
        blended[name] = left[name].mul(alpha).add(right[name], alpha=1.0 - alpha)
    return blended


def restore(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(state[n].to(p.device))
    model.zero_grad(set_to_none=True)
    if state_hash(snapshot(model)) != state_hash(state):
        raise RuntimeError('Initial adapter reset mismatch')


def load(args):
    if digest(args.parent / 'model.safetensors') != PARENT_SHA256:
        raise RuntimeError('Parent weight hash mismatch')
    if digest(args.parent / 'config.json') != PARENT_CONFIG_SHA256:
        raise RuntimeError('Parent config hash mismatch')
    from transformers import Gemma4UnifiedForConditionalGeneration, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(args.parent, local_files_only=True)
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError('Require actual ROCm device')
    torch.manual_seed(73)
    torch.set_num_threads(2)
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
        args.parent, local_files_only=True, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        low_cpu_mem_usage=True, attn_implementation='eager')
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=2, lora_alpha=2, lora_dropout=0.0,
                                            target_modules=[TARGET], bias='none'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    if not all('lora_' in n and TARGET in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError('Unexpected trainable tensor')
    return model, tokenizer


def logits_for(model, token_ids, positions):
    # Every scoring chunk uses a complete prefix; only logits are sliced.
    inputs = torch.tensor([token_ids], dtype=torch.long, device='cuda')
    positions = torch.tensor(positions, dtype=torch.long, device='cuda')
    logits = model(input_ids=inputs, use_cache=False, logits_to_keep=positions).logits.float()[0]
    if not torch.isfinite(logits).all():
        raise RuntimeError('Nonfinite logits')
    return logits


def scores(model, token_ids, positions):
    return torch.log_softmax(logits_for(model, token_ids, positions), dim=-1)  # Trainer FP32, T=1.


def memory_control(model, count):
    model.train()
    ids = [2] + [100 + i % 40 for i in range(count - 1)]
    values = scores(model, ids, list(range(count - 8, count)))
    loss = -values[:, 106].sum()
    loss.backward()
    norms = {n: p.grad.float().norm().item() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    if not norms or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise RuntimeError('Memory-control gradients invalid')
    model.zero_grad(set_to_none=True)
    del values, loss
    torch.cuda.empty_cache()
    return {'tokens': count, 'scored_positions': 8, 'finite_gradients': True,
            'gradient_norms': norms, 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}


def policy(model, tokenizer_hash):
    adapter_hash = state_hash(snapshot(model))
    return {'model_sha256': PARENT_SHA256, 'tokenizer_sha256': tokenizer_hash,
            'adapter_sha256': adapter_hash, 'version': 'initial-' + adapter_hash[:16]}


def reasoned_answer(text):
    """Return answer after one complete pinned Gemma thought channel."""
    opener = "<|channel>thought\n"
    closer = "<channel|>"
    channel_tokens = ("<|channel>", closer, "<|think|>")
    if not any(token in text for token in channel_tokens):
        return text.lstrip()  # Explicit policy: a direct bare answer is admissible.
    if text.count(opener) != 1 or text.count(closer) != 1 or not text.startswith(opener):
        raise ValueError("malformed Gemma thought channel")
    thought, answer = text[len(opener):].split(closer, 1)
    if any(token in thought for token in channel_tokens) or any(token in answer for token in channel_tokens):
        raise ValueError("nested or stray Gemma channel token")
    return answer.lstrip()


def rollout(model, tokenizer, task, prompt, prompt_ids, pol, arm, cfg, verify, seal_record, verifier_hash, sample_policy_tokens, rng_pair_id=None):
    rng_namespace = rng_pair_id or arm
    action_id = 'lilith-rl/action/' + rng_namespace + '/' + task.id
    aux_id = 'lilith-rl/aux/' + rng_namespace + '/' + task.id
    action_rng = torch.Generator(device='cpu').manual_seed(seed(action_id))
    aux_rng = torch.Generator(device='cpu').manual_seed(seed(aux_id))
    ids, logs, draws, draw_logs = list(prompt_ids), [None] * len(prompt_ids), [[] for _ in prompt_ids], [[] for _ in prompt_ids]
    model.eval()
    status, error, terminator = 'truncated', None, None
    start = time.monotonic()
    try:
        with torch.no_grad():
            for _ in range(cfg['max_new_tokens']):
                if len(ids) >= cfg['max_context_tokens']:
                    break
                raw_logits = logits_for(model, ids, [len(ids) - 1])[0].cpu()
                draw = sample_policy_tokens(raw_logits, action_rng, aux_rng, cfg['mc_samples'], 1.0)
                action = draw['token_id']
                ids.append(action)
                logs.append(draw['sampler_logp'])
                draws.append(draw['mc_token_ids'])
                draw_logs.append(draw['mc_sampler_logps'])
                if action in cfg['eos_ids']:
                    status, terminator = 'completed', action
                    break
    except Exception as exc:
        status, error = 'failed', f'{type(exc).__name__}: {exc}'
    generated = ids[len(prompt_ids):]
    content_tokens = generated[:-1] if terminator is not None else generated
    text = tokenizer.decode(content_tokens, skip_special_tokens=False)
    try:
        answer = reasoned_answer(text)
        reasoning_gate = 'complete'
    except ValueError as exc:
        answer = ''
        reasoning_gate = str(exc)
    evidence = verify(answer, task)
    evidence['reasoning_gate'] = reasoning_gate
    evidence['reasoned_answer_sha256'] = hashlib.sha256(answer.encode()).hexdigest()
    evidence['generated_text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
    if status != 'completed':
        evidence['artifact_verifier_passed_before_completion_gate'] = evidence['passed']
        evidence['passed'] = False
        evidence['completion_gate'] = status
    record = {'schema': 'lilith/trajectory@1', 'id': f'{arm}/{task.id}',
              'task': {'id': task.id, 'family': task.family, 'split': task.split,
                       'prompt_sha256': hashlib.sha256(task.prompt.encode()).hexdigest()},
              'policy': pol,
              'sampler': {'kind': 'categorical', 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0,
                          'full_support': True, 'action_rng_id': action_id, 'aux_rng_id': aux_id,
                          'with_replacement': True, 'aux_independent': True,
                          'termination_token_ids': cfg['eos_ids']},
              'context': {'prompt': task.prompt, 'rendered_prompt': prompt,
                          'features': [{'chat_template_sha256': cfg['chat_template_sha256'],
                          'enable_thinking': True, 'use_cache': False, 'arm': arm}],
                          'lesson_ids': [], 'prompt_token_ids': prompt_ids},
              'token_ids': ids, 'token_roles': ['prompt'] * len(prompt_ids) + ['assistant'] * len(generated),
              'policy_mask': [False] * len(prompt_ids) + [True] * len(generated),
              'sampler_logps': logs, 'mc_token_ids': draws, 'mc_sampler_logps': draw_logs,
              'completion': {'status': status, 'termination_token_id': terminator, 'generated_text': text,
                             'error': error, 'wall_seconds': time.monotonic() - start},
              'reward': {'value': evidence['klpo_reward'] if status == 'completed' else 0.0,
                         'verifier_sha256': verifier_hash, 'evidence': evidence}}
    return seal_record(record)


def response_chunks(record, chunk_size):
    first = len(record['context']['prompt_token_ids'])
    count = len(record['token_ids'])
    for begin in range(first, count, chunk_size):
        end = min(begin + chunk_size, count)
        # To score token i, forward its prefix ending at i-1.
        yield begin, end, record['token_ids'][:end - 1], list(range(begin - 1, end - 1))


def drift(model, records, chunk_size):
    maximum = 0.0
    model.eval()
    with torch.no_grad():
        for record in records:
            for begin, end, prefix, positions in response_chunks(record, chunk_size):
                lp = scores(model, prefix, positions)
                actions = torch.tensor(record['token_ids'][begin:end], device='cuda')
                action_lp = lp.gather(1, actions[:, None]).squeeze(1).cpu()
                reference = torch.tensor(record['sampler_logps'][begin:end])
                maximum = max(maximum, (action_lp - reference).abs().max().item())
    return maximum


def transcript_examples(path, expected_sha256, split):
    if digest(path) != expected_sha256:
        raise RuntimeError('Transcript SFT subset hash mismatch')
    selected = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get('schema') != 'akh/lilith-transcript-sft@1':
            raise RuntimeError('Unexpected transcript SFT schema')
        if row.get('split') != split:
            continue
        messages = row.get('messages')
        if not isinstance(messages, list) or len(messages) != 2 or [m.get('role') for m in messages] != ['user','assistant']:
            raise RuntimeError('Transcript example must have one user and one assistant message')
        if any(key in row for key in ('reward', 'evidence', 'verified_outcome')):
            raise RuntimeError('Transcript SFT subset cannot carry RL evidence or reward')
        selected.append({'record_id':row['record_id'], 'input':messages[0]['content'],
                         'output':messages[1]['content'], 'split':split})
    if len(selected) != 8:
        raise RuntimeError('Require exactly eight frozen transcript rows for split ' + split)
    return selected


def transcript_validation_nll(model, examples, tokenizer, cfg):
    total, tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for row in examples:
            prompt = tokenizer.apply_chat_template([{'role':'user','content':row['input']}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            target = tokenizer.encode(row['output'], add_special_tokens=False) + [106]
            ids = prompt_ids + target
            if len(ids) > cfg['max_context_tokens']:
                raise RuntimeError('Frozen validation example exceeds token budget: ' + row['record_id'])
            record = {'context': {'prompt_token_ids': prompt_ids}, 'token_ids': ids}
            for begin, end, prefix, positions in response_chunks(record, cfg['scoring_chunk']):
                lp = scores(model, prefix, positions)
                actions = torch.tensor(ids[begin:end], device='cuda')
                total -= lp.gather(1, actions[:, None]).sum().item()
                tokens += end - begin
    return {'sum_nll':total, 'tokens':tokens, 'mean_token_nll':total / tokens,
            'record_ids':[row['record_id'] for row in examples]}


def sft_update(model, examples, tokenizer, cfg):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    total, count = 0.0, 0
    identities = []
    for row in examples:
        prompt = tokenizer.apply_chat_template([{'role':'user','content':row['input']}], tokenize=False,
                                               add_generation_prompt=True, enable_thinking=True)
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target = tokenizer.encode(row['output'], add_special_tokens=False) + [106]
        ids = prompt_ids + target
        if len(ids) > cfg['max_context_tokens']:
            raise RuntimeError('Frozen transcript example exceeds token budget: ' + row['record_id'])
        record = {'context': {'prompt_token_ids': prompt_ids}, 'token_ids': ids}
        for begin, end, prefix, positions in response_chunks(record, cfg['scoring_chunk']):
            lp = scores(model, prefix, positions)
            actions = torch.tensor(ids[begin:end], device='cuda')
            loss = -lp.gather(1, actions[:, None]).sum() / len(examples)
            total += loss.item()
            loss.backward()
            count += end - begin
        identities.append(row['record_id'])
    check_gradients(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'optimizer_steps': 1, 'sequence_sum_nll_mean': total, 'supervised_tokens': count,
            'supervision': 'frozen admitted transcript train records; validation excluded',
            'record_ids': identities}


def check_gradients(model):
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    if not grads or any(g is None or not torch.isfinite(g).all() for g in grads):
        raise RuntimeError('Missing or nonfinite trainable gradients')


def klpo_update(model, records, cfg, output, trajectory_module, expected_policy, verifier_hash, expected_tasks, update_id):
    eligible = []
    for record in records:
        if record['task']['split'] == 'train' and record['completion']['status'] == 'completed':
            trajectory_module.validate_record(record, expected_policy=expected_policy, expected_verifier_sha256=verifier_hash, for_update=True, expected_task=expected_tasks[record['task']['id']])
            eligible.append(record)
    rewards = [record['reward']['value'] for record in eligible]
    if len(set(rewards)) < 2:
        return {'optimizer_steps': 0, 'status': 'uninformative_reward', 'eligible_complete_records': len(eligible), 'rewards': rewards}
    claim = trajectory_module.claim_update(eligible, output / 'consumption.sqlite',
        update_id, expected_policy=expected_policy,
        expected_verifier_sha256=verifier_hash, expected_tasks=expected_tasks)
    if claim['replayed']:
        raise RuntimeError('A durable bank claim already exists; never replay optimizer work')
    write(output / f'{update_id}-consumption.json', claim)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['learning_rate'], weight_decay=0)
    optimizer.zero_grad(set_to_none=True)
    model.train()
    total = 0.0
    for record in eligible:
        for begin, end, prefix, positions in response_chunks(record, cfg['scoring_chunk']):
            lp = scores(model, prefix, positions)
            actions = torch.tensor(record['token_ids'][begin:end], device='cuda')
            auxiliary = torch.tensor(record['mc_token_ids'][begin:end], device='cuda')
            a = lp.gather(1, actions[:, None]).squeeze(1)[None, :]
            q = torch.tensor(record['sampler_logps'][begin:end], device='cuda')[None, :]
            aux = lp.gather(1, auxiliary)[None, :, :]
            reward = torch.tensor([record['reward']['value']], device='cuda')
            mask = torch.ones_like(a, dtype=torch.bool)
            loss = trajectory_module.token_mc_loss(a, q, aux, reward, mask, cfg['beta']) / len(eligible)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite KLPO objective')
            total += loss.item()
            loss.backward()
    check_gradients(model)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'optimizer_steps': 1, 'status': 'updated', 'eligible_complete_records': len(eligible),
            'rewards': rewards, 'loss': total, 'normalization': 'mean complete responses of summed token objective'}


def main():
    if not __debug__:
        raise RuntimeError('Assertions require unoptimized Python')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-manifest', type=Path, default=Path(__file__).resolve().parents[2] / 'training/lilith/adam-kaine-code-env-20260922/rl-tasks.json')
    parser.add_argument('--reload', type=Path, help='Completed experiment directory: perform independent cold reload')
    parser.add_argument('--training-protocol', type=Path, required=True)
    parser.add_argument('--expected-training-protocol-sha256', required=True)
    parser.add_argument('--transcript-dataset', type=Path, required=False,
                        default=Path.home() / '.local/share/akh-medu/lilith-training/transcripts-20260922/sft-subset-v1.jsonl')
    parser.add_argument('--transcript-manifest', type=Path, required=False,
                        default=Path(__file__).resolve().parents[2] / 'training/lilith/transcripts-20260922/sft-subset-v1-manifest.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if digest(args.training_protocol) != args.expected_training_protocol_sha256:
        raise RuntimeError('Training protocol artifact hash mismatch')
    protocol = json.loads(args.training_protocol.read_text())
    if protocol.get('runner', {}).get('sha256') != digest(__file__):
        raise RuntimeError('Training protocol does not bind this runner source')
    report = {'schema': 'lilith/rl-probe@1', 'status': 'running', 'pid': os.getpid(),
              'deployed_weights_changed': False, 'quality_claim': False, 'source_sha256': digest(__file__),
              'training_protocol_sha256': args.expected_training_protocol_sha256}
    started = time.monotonic()
    def progress(phase):
        report['phase'], report['elapsed_seconds'] = phase, time.monotonic() - started
        write(args.output / 'result.json', report)
    try:
        global torch
        import torch
        from safetensors.torch import save_file, load_file
        from peft import set_peft_model_state_dict
        if not args.reload:
            import adam_kaine_code_rl_tasks as task_module
            import lilith_rl as trajectory_module
            task_list = task_module.tasks()
            manifest = json.loads(args.task_manifest.read_text())
            observed_components = task_module.verifier_components()
            if manifest['verifier_components'] != observed_components:
                raise RuntimeError('Verifier component changed since frozen manifest')
            if manifest['verifier_sha256'] != task_module.verifier_identity():
                raise RuntimeError('Composite verifier identity drift')
            pinned = {row['id']: row for row in manifest['tasks']}
            if set(pinned) != {task.id for task in task_list}:
                raise RuntimeError('Task set changed since frozen manifest')
            for task in task_list:
                observed = {'id':task.id, 'family':task.family, 'split':task.split,
                    'prompt':task.prompt, 'prompt_sha256':hashlib.sha256(task.prompt.encode()).hexdigest(),
                    'cases_sha256':task_module.digest(task.cases)}
                if any(pinned[task.id][key] != value for key, value in observed.items()):
                    raise RuntimeError('Task definition drift: ' + task.id)
            report['task_manifest_sha256'] = digest(args.task_manifest)
            if protocol['environment']['rl_tasks_sha256'] != report['task_manifest_sha256']:
                raise RuntimeError('Protocol task manifest hash drift')
            if protocol['environment']['manifest_semantic_sha256'] != manifest['environment_manifest_sha256']:
                raise RuntimeError('Protocol environment identity drift')
        progress('loading_parent')
        model, tokenizer = load(args)
        frozen = {n: p._version for n, p in model.named_parameters() if not p.requires_grad}
        if args.reload:
            previous = json.loads((args.reload / 'result.json').read_text())
            candidate = args.reload / previous['cold_reload_candidate']
            if previous['status'] != 'passed' or previous['pid'] == os.getpid():
                raise RuntimeError('Require completed experiment and distinct cold process')
            selected_arm = previous['cold_reload_arm']
            if digest(candidate / 'adapter_model.safetensors') != previous[selected_arm]['adapter_file_sha256']:
                raise RuntimeError('Candidate artifact hash drift')
            if digest(args.reload / 'cold-reference.safetensors') != previous['cold_reference_sha256']:
                raise RuntimeError('Cold reference artifact drift')
            state = load_file(str(candidate / 'adapter_model.safetensors'))
            set_peft_model_state_dict(model, state)
            if state_hash(snapshot(model)) != previous[selected_arm]['adapter_sha256']:
                raise RuntimeError('Cold adapter tensor hash mismatch')
            reference = load_file(str(args.reload / 'cold-reference.safetensors'))
            model.eval()
            with torch.no_grad():
                actual = scores(model, reference['input_ids'].tolist(), [len(reference['input_ids']) - 1]).cpu()
            if not torch.isfinite(actual).all() or not torch.equal(actual, reference['logps']):
                raise RuntimeError('Cold-process reload log probabilities differ')
            report.update(status='passed', previous_pid=previous['pid'], cold_reload_exact=True,
                          selected_arm=selected_arm,
                          candidate_sha256=digest(candidate / 'adapter_model.safetensors'))
            progress('finished')
            return
        verifier_hash = task_module.verifier_identity()
        tokenizer_hash = digest(args.parent / 'tokenizer.json')
        prompts = {}
        for task in task_list:
            prompt = tokenizer.apply_chat_template([{'role':'user','content':task.prompt}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=True)
            ids = tokenizer.encode(prompt, add_special_tokens=False)
            prompts[task.id] = (prompt, ids)
            if task.split == 'train' and not task_module.verify(task.reference, task)['passed']:
                raise RuntimeError('Canonical reference verifier failed')
        maximum = max(len(ids) for _, ids in prompts.values()) + 128
        context = 512  # Frozen transcript subset admits examples up to this proven envelope.
        if maximum > 512:
            raise RuntimeError('Task prompt exceeds frozen resource budget')
        cfg = {'train_tasks': 4, 'development_tasks': 4, 'independent_eval_tasks': 4, 'max_new_tokens': 320,
               'max_context_tokens': context, 'max_reasoning_completion_tokens': 320, 'mc_samples': 8, 'temperature': 1.0,
               'scoring_chunk': 8, 'drift_chunk_fallback_threshold': 1e-4,
               'beta': 0.1, 'learning_rate': 1e-4, 'optimizer_steps_per_arm': 1,
               'eos_ids': list(model.config.eos_token_id), 'chat_template_sha256': digest(args.parent / 'chat_template.jinja'),
               'task_module_sha256': verifier_hash, 'trajectory_module_sha256': digest(trajectory_module.__file__),
               'prompt_lengths': {task_id: len(ids) for task_id, (_, ids) in prompts.items()},
               'canonical_train_token_lengths': {t.id: len(tokenizer.encode(t.reference, add_special_tokens=False)) + 1 for t in task_list if t.split == 'train'},
               'sampler': 'full-support FP64 CPU categorical from actual logits; trainer FP32; MC8 IID replacement separate generator',
               'uninformative_reward_rule': 'skip KLPO if fewer than two distinct eligible complete rewards',
               'reasoning_mode': 'Gemma template enable_thinking=true; full reasoning tokens retain policy logps; verifier strips only complete <|channel>thought...<channel|> blocks; 512/320 initial with frozen all-truncated 1024/800 escalation'}
        expected_tasks = {t.id: {'id':t.id, 'family':t.family, 'split':t.split,
            'prompt_sha256':hashlib.sha256(t.prompt.encode()).hexdigest()} for t in task_list}
        cfg['authoritative_tasks'] = expected_tasks
        cfg['task_manifest_sha256'] = report['task_manifest_sha256']
        if ({split_name: sum(t.split == split_name for t in task_list) for split_name in ('train','development_holdout','independent_eval')}
                != {'train':4, 'development_holdout':4, 'independent_eval':4}):
            raise RuntimeError('Require exactly four disjoint train, development, and independent-eval tasks')
        write(args.output / 'config.json', cfg)
        initial = snapshot(model)
        progress('memory_control')
        report['memory_control'] = memory_control(model, context)
        restore(model, initial)
        initial_policy = policy(model, tokenizer_hash)
        report['initial_policy'] = initial_policy
        transcript_manifest = json.loads(args.transcript_manifest.read_text())
        transcript_rows = transcript_examples(args.transcript_dataset, transcript_manifest['private_subset_sha256'], 'train')
        transcript_validation = transcript_examples(args.transcript_dataset, transcript_manifest['private_subset_sha256'], 'validation')
        if transcript_manifest['splits'] != {'train': 8, 'validation': 8} or transcript_manifest['caps']['max_total_tokens'] != 512:
            raise RuntimeError('Transcript subset manifest drift')
        cfg['transcript_dataset_sha256'] = transcript_manifest['private_subset_sha256']
        cfg['transcript_selection'] = {
            'rule': transcript_manifest['selection'] + '; no truncation',
            'record_ids': [row['record_id'] for row in transcript_rows],
            'validation_record_ids': [row['record_id'] for row in transcript_validation],
            'validation_records_used_for_fit': 0}
        cfg['optimizer_steps_per_rl_arm'] = 3
        report['base_transcript_validation'] = transcript_validation_nll(model, transcript_validation, tokenizer, cfg)
        write(args.output / 'config.json', cfg)
        all_records = {}

        def collect(arm, selected_tasks, pol, filename, rng_pair_id=None):
            records = []
            for task in selected_tasks:
                progress(f'{arm}_rollout_{task.id}')
                prompt, ids = prompts[task.id]
                record = rollout(model, tokenizer, task, prompt, ids, pol, arm, cfg,
                    task_module.verify, trajectory_module.seal_record, verifier_hash,
                    trajectory_module.sample_policy_tokens, rng_pair_id=rng_pair_id)
                records.append(record)
                write(args.output / filename, records)
                trajectory_module.validate_record(record, for_update=False,
                    expected_task=expected_tasks[task.id], expected_policy=pol,
                    expected_verifier_sha256=verifier_hash)
            return records

        train_tasks = [task for task in task_list if task.split == 'train']
        development_tasks = [task for task in task_list if task.split == 'development_holdout']
        independent_tasks = [task for task in task_list if task.split == 'independent_eval']
        restore(model, initial)
        base_policy = initial_policy
        base_records = collect('base-train-512', train_tasks, base_policy, 'base-train-512-trajectories.json')
        completed = [r for r in base_records if r['completion']['status'] == 'completed']
        if len(base_records) == len(train_tasks) and all(r['completion']['status'] == 'truncated' for r in base_records):
            # Frozen resource-only escalation inspects statuses only: exactly four
            # expected records, all truncated. Failed records never trigger escalation.
            progress('memory_control_1024_after_all_truncated')
            report['memory_control_1024'] = memory_control(model, 1024)
            cfg['max_context_tokens'] = 1024
            cfg['max_new_tokens'] = 800
            cfg['reasoning_envelope_selected'] = '1024-total/800-new after all 512-envelope train rollouts truncated'
            write(args.output / 'config.json', cfg)
            base_records = collect('base-train-1024', train_tasks, base_policy, 'base-train-1024-trajectories.json')
            completed = [r for r in base_records if r['completion']['status'] == 'completed']
            if not completed:
                raise RuntimeError('1024-token escalation produced no complete training rollout; resource-limited stop')
        elif completed:
            cfg['reasoning_envelope_selected'] = '512-total/320-new; at least one train rollout completed'
            write(args.output / 'config.json', cfg)
        else:
            raise RuntimeError('Base reasoning rollouts failed without satisfying exact all-truncated escalation gate')
        all_records['base'] = base_records
        base_train = list(base_records)
        report['initial_logprob_drift_chunk8'] = drift(model, completed, 8) if completed else None
        if completed and report['initial_logprob_drift_chunk8'] > cfg['drift_chunk_fallback_threshold']:
            cfg['scoring_chunk'] = 1
            report['initial_logprob_drift_chunk1'] = drift(model, completed, 1)
            if report['initial_logprob_drift_chunk1'] > cfg['drift_chunk_fallback_threshold']:
                raise RuntimeError('Initial sampler/trainer log probability drift remains above frozen tolerance')

        candidate_states = {}
        for arm in ['sft_distill', 'klpo_online']:
            restore(model, initial)
            updates = []
            if arm == 'sft_distill':
                progress(arm + '_sft_update')
                updates.append({'kind':'sft_distill', **sft_update(model, transcript_rows, tokenizer, cfg)})
            else:
                for step in range(1, 4):
                    bank_arm = f'{arm}-bank{step}'
                    bank_policy = policy(model, tokenizer_hash)
                    bank_policy['version'] = bank_arm + '-' + bank_policy['adapter_sha256'][:16]
                    bank = collect(bank_arm, train_tasks, bank_policy, f'{bank_arm}-trajectories.json')
                    progress(f'{arm}_update_{step}')
                    update = klpo_update(model, bank, cfg, args.output, trajectory_module,
                        bank_policy, verifier_hash, expected_tasks, f'{arm}-update-{step}')
                    updates.append({'kind':'klpo', 'step':step, **update})
                    if update['optimizer_steps'] == 0:
                        break
            if not all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
                raise RuntimeError('Optimizer produced nonfinite adapter weights')
            candidate_states[arm] = snapshot(model)
            pol = policy(model, tokenizer_hash)
            pol['version'] = arm + '-development-' + pol['adapter_sha256'][:16]
            records = collect(arm + '-development', development_tasks, pol, f'{arm}-development-trajectories.json')
            model.save_pretrained(args.output / f'{arm}-adapter', safe_serialization=True)
            report[arm] = {'development_passed': sum(r['reward']['value'] == 1 for r in records),
                'development_total': len(records), 'adapter_sha256': pol['adapter_sha256'], 'updates': updates,
                'adapter_file_sha256': digest(args.output / f'{arm}-adapter' / 'adapter_model.safetensors')}
            if not all(p._version == frozen[n] for n, p in model.named_parameters() if n in frozen):
                raise RuntimeError('Frozen base mutated')

        # The alpha and eligible parents are protocol constants, frozen before results.
        # This is adapter-coordinate interpolation, not a claim that model deltas combine linearly.
        blend = blend_states(candidate_states['sft_distill'], candidate_states['klpo_online'], 0.5)
        restore(model, blend)
        blend_policy = policy(model, tokenizer_hash)
        blend_policy['version'] = 'adapter_blend_0p5-development-' + blend_policy['adapter_sha256'][:16]
        blend_records = collect('adapter_blend_0p5-development', development_tasks, blend_policy,
                                'adapter_blend_0p5-development-trajectories.json')
        model.save_pretrained(args.output / 'adapter_blend_0p5-adapter', safe_serialization=True)
        candidate_states['adapter_blend_0p5'] = snapshot(model)
        report['adapter_blend_0p5'] = {
            'development_passed': sum(r['reward']['value'] == 1 for r in blend_records),
            'development_total': len(blend_records), 'adapter_sha256': blend_policy['adapter_sha256'],
            'adapter_file_sha256': digest(args.output / 'adapter_blend_0p5-adapter' / 'adapter_model.safetensors'),
            'lineage': {'kind':'adapter_coordinate_interpolation', 'alpha_left':0.5,
                'left':'sft_distill', 'right':'klpo_online', 'same_parent_sha256':PARENT_SHA256,
                'target_module':TARGET, 'rank':2}}

        report['base'] = {'passed': sum(r['reward']['value'] == 1 for r in base_records),
            'completed': sum(r['completion']['status'] == 'completed' for r in base_records),
            'total': len(base_records), 'adapter_sha256': initial_policy['adapter_sha256']}
        # Development-only choice, with a predeclared tie order. The selected candidate
        # is frozen before the selected/base paired independent evaluation opens.
        tie_priority = {'adapter_blend_0p5':2, 'klpo_online':1, 'sft_distill':0}
        selected_arm = max(tie_priority, key=lambda name: (report[name]['development_passed'], tie_priority[name]))
        # Persist the development-only decision before either policy sees eval prompts.
        report['selection'] = {'rule':'max development passes; ties blend > KLPO > SFT',
            'selected':selected_arm, 'frozen_before_independent_eval':True,
            'independent_eval_rng_pair_id':'independent-eval-pair-v1',
            'release_automatic':False}
        progress('selection_frozen_before_independent_eval')
        restore(model, candidate_states[selected_arm])
        selected_policy = policy(model, tokenizer_hash)
        selected_policy['version'] = selected_arm + '-independent-eval-' + selected_policy['adapter_sha256'][:16]
        selected_eval = collect(selected_arm + '-independent-eval', independent_tasks, selected_policy,
                                'selected-independent-eval-trajectories.json', 'independent-eval-pair-v1')
        report['selected_transcript_validation'] = transcript_validation_nll(model, transcript_validation, tokenizer, cfg)
        restore(model, initial)
        base_eval = collect('base-independent-eval', independent_tasks, base_policy,
                            'base-independent-eval-trajectories.json', 'independent-eval-pair-v1')
        report['selection'].update(
            selected_passed=sum(r['reward']['value'] == 1 for r in selected_eval),
            base_passed=sum(r['reward']['value'] == 1 for r in base_eval),
            independent_eval_total=len(selected_eval),
            paired_common_random_numbers=True)
        report['retention'] = {'baseline_mean_token_nll':report['base_transcript_validation']['mean_token_nll'],
            'selected_mean_token_nll':report['selected_transcript_validation']['mean_token_nll'],
            'delta_mean_token_nll':report['selected_transcript_validation']['mean_token_nll'] - report['base_transcript_validation']['mean_token_nll']}
        candidate_path = args.output / f'{selected_arm}-adapter' / 'adapter_model.safetensors'
        state = load_file(str(candidate_path))
        set_peft_model_state_dict(model, state)
        if state_hash(snapshot(model)) != report[selected_arm]['adapter_sha256']:
            raise RuntimeError('Selected adapter does not match evaluated arm')
        first_ids = prompts[task_list[0].id][1]
        model.eval()
        with torch.no_grad():
            reference_lp = scores(model, first_ids, [len(first_ids)-1]).cpu()
        save_file({'input_ids':torch.tensor(first_ids), 'logps':reference_lp}, str(args.output / 'cold-reference.safetensors'))
        report.update(status='passed', cold_reload_candidate=selected_arm + '-adapter',
                      cold_reload_arm=selected_arm,
                      cold_reload_selection_basis='development-only frozen rule; paired selected/base independent eval after selection',
                      release_selected=False, breeding_performed=True,
                      breeding_note='0.5 adapter-coordinate interpolation of same-parent/rank/target SFT and independent KLPO candidates; no guaranteed benefit.',
                      cold_reference_sha256=digest(args.output / 'cold-reference.safetensors'),
                      frozen_tensor_versions_unchanged=True, peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                      final_scoring_chunk=cfg['scoring_chunk'])
        progress('finished')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        progress('failed')
        raise


if __name__ == '__main__':
    main()
