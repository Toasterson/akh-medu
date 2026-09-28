#!/usr/bin/env python3
"""akhrtd-owned learning worker. All workspace effects use the runtime broker."""
import contextlib
import hashlib
import json
import sqlite3
import sys
from dataclasses import asdict
from pathlib import Path

from lilith_gemma_tool_frames import parse_response
from lilith_training_transport import RuntimeTransport, digest_file

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
TRAINING_SOURCES = (
    'lilith_runtime_training_worker.py', 'lilith_runtime_training_stack.py',
    'lilith_runtime_training_config.py', 'lilith_training_engine.py', 'lilith_deferred_review.py',
    'lilith_training_transport.py', 'lilith_gemma_tool_frames.py', 'lilith_rl.py',
    'adam_kaine_code_rl_tasks.py', 'adam_kaine_code_environment.py',
    'lilith_rl_tasks_v2.py', 'lilith_rl_tasks.py', 'lilith_adapter_mechanics.py',
    'lilith_paths.py')


def configured_cache_offloading(config):
    """Explicit KV placement from the frozen config; absent means the historical offload default."""
    value = config.get('cache_offloading', True)
    if type(value) is not bool:
        raise ValueError('cache_offloading must be a JSON boolean in the worker configuration')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(canonical(value) + b'\n')
    temporary.chmod(0o600)
    temporary.replace(path)


def checked_read(result):
    data = result.get('data')
    if not result.get('success') or not result.get('receipt') or not isinstance(data, dict):
        raise ValueError('workspace read lacks a successful broker receipt')
    content = data.get('content')
    if data.get('encoding') != 'utf8' or not isinstance(content, str):
        raise ValueError('workspace read is not UTF-8 content')
    if data.get('revision') != hashlib.sha256(content.encode()).hexdigest():
        raise ValueError('workspace revision does not identify returned bytes')
    return content


def checked_write(result, content):
    data = result.get('data')
    if (not result.get('success') or not result.get('receipt') or not isinstance(data, dict)
            or data.get('revision') != hashlib.sha256(content.encode()).hexdigest()
            or data.get('bytes') != len(content.encode())):
        raise ValueError('workspace write lacks a matching broker receipt')


def read_is_missing(result):
    data = result.get('data')
    error = data.get('error') if isinstance(data, dict) else None
    return (not result.get('success') and bool(result.get('receipt'))
            and isinstance(error, dict) and error.get('code') == 'not_found')


def reset_artifact(transport, path, content):
    prior = transport.tool('read_file', {'path':path})
    if prior.get('success'):
        checked_read(prior)
        create = False
        expected_revision = prior['data']['revision']
    elif read_is_missing(prior):
        create = True
        expected_revision = None
    else:
        raise ValueError('workspace initialization read failed without a not_found receipt')
    arguments = {'path':path, 'content':content, 'create':create}
    if expected_revision is not None:
        arguments['expected_revision'] = expected_revision
    result = transport.tool('write_file', arguments)
    checked_write(result, content)


def tool_schema(tools):
    return [{'type':'function', 'function':{'name':t['name'], 'description':t['description'],
            'parameters':t['input_schema']}} for t in tools]


def validate_catalog(tools):
    import jsonschema
    names = set()
    for tool in tools:
        if (not isinstance(tool, dict) or set(tool) != {'name','description','input_schema','read_only','destination'}
                or not isinstance(tool['name'], str) or not tool['name'] or tool['name'] in names
                or not isinstance(tool['description'], str) or type(tool['read_only']) is not bool
                or not isinstance(tool['destination'], dict) or not isinstance(tool['input_schema'], dict)):
            raise ValueError('invalid authoritative tool catalog')
        jsonschema.Draft202012Validator.check_schema(tool['input_schema'])
        names.add(tool['name'])
    if not {'read_file', 'write_file'}.issubset(names):
        raise ValueError('learning workspace requires the inference read/write tools')


def required_pins(config):
    if config['mode'] == 'fixture':
        return {str(SCRIPTS / name) for name in
                ('lilith_runtime_training_worker.py','lilith_training_transport.py','lilith_gemma_tool_frames.py')}
    parent = Path(config['parent'])
    paths = [SCRIPTS / name for name in TRAINING_SOURCES]
    paths += [parent / 'chat_template.jinja', parent / 'generation_config.json',
              ROOT / 'training/lilith/transcripts-20260922/sft-subset-v1-manifest.json',
              ROOT / 'training/lilith/adam-kaine-code-env-20260922/rl-tasks.json']
    return {str(path) for path in paths}


def validate_learning_inputs(config):
    import adam_kaine_code_rl_tasks as task_module
    if task_module.verifier_identity() != config['verifier_sha256']:
        raise ValueError('verifier identity changed before model load')
    manifest_path = Path(config['task_manifest'])
    if digest_file(manifest_path) != config['task_manifest_sha256']:
        raise ValueError('task manifest identity changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['verifier_sha256'] != config['verifier_sha256']:
        raise ValueError('task manifest uses a different verifier')
    pinned = {row['id']:row for row in manifest['tasks']}
    tasks = task_module.tasks()
    if set(pinned) != {task.id for task in tasks}:
        raise ValueError('task membership changed')
    for task in tasks:
        expected = {'family':task.family, 'split':task.split, 'prompt':task.prompt,
                    'cases_sha256':task_module.digest(task.cases)}
        if any(pinned[task.id][key] != value for key,value in expected.items()):
            raise ValueError('task semantics changed')
    if digest_file(config['transcript_dataset']) != config['transcript_dataset_sha256']:
        raise ValueError('transcript subset changed before model load')
    if config['max_output_tokens'] < 8192 or config['max_context_tokens'] <= config['max_output_tokens']:
        raise ValueError('ordinary reasoning requires the frozen generous token budget')


def episode(engine, transport, task, arm, config, output):
    """Collect a whole tool episode; partial plans never authorize effects."""
    import jsonschema
    name = config['artifact_path']
    write(output.parent / 'active-task.json' if output.name in ('selected-eval','base-eval','base-retention','selected-retention')
          else output / 'active-task.json', {'task_id':task.id, 'arm':arm, 'artifact_path':name,
               'split':task.split, 'verifier_sha256':config.get('verifier_sha256'), 'status':'generating'})
    initial = '# Assigned task: ' + task.id + '\n# Implement the function in the task specification.\n'
    reset_artifact(transport, name, initial)
    observed = transport.tool('read_file', {'path':name})
    if checked_read(observed) != initial:
        raise ValueError('workspace initialization changed before observation')
    first_receipt = len(transport.receipts)
    messages = [
        {'role':'system', 'content':transport.request['instructions'] +
         '\nReason normally. For a tool handoff emit only native tool call frames after your thinking block, without prose between calls. '
         'Use only the supplied tools. Acknowledge pending reviews truthfully; never claim an unobserved test passed.'},
        {'role':'user', 'content':task.prompt + '\nThe source-only formatting requirements above apply to the file contents. '
         'Use the native tool protocol to work on the file. Inspect and implement this task in ' + name +
         ' using the workspace tools. The file must contain the requested Python function, with no Markdown. '
         'Finish with a concise account of the actual actions and evidence.'},
    ]
    records, status, failure = [], 'iteration_limit', None
    catalog = {t['name']:t for t in transport.request['tools']}
    for step in range(config['max_iterations']):
        seed = int(digest({'namespace':arm, 'task':task.id, 'step':step})[:15], 16)
        transport.progress(arm + '/' + task.id + '/generate-' + str(step))
        progress_path = output / 'progress.json'
        write(progress_path, {'phase':'prefill', 'task_id':task.id, 'tokens':0})
        def observe_tokens(event):
            count = event['step'] + 1
            if count % 128 == 0:
                write(progress_path, {'phase':'generation', 'task_id':task.id, 'tokens':count})
                transport.progress(arm + '/' + task.id + '/tokens-' + str(count))
        trajectory = engine.generate(messages=messages, tools=tool_schema(transport.request['tools']),
            max_new_tokens=config['max_output_tokens'], seed=seed,
            token_observer=observe_tokens)
        records.append(trajectory)
        write(output / (arm + '-' + task.id.replace('/', '_') + '-trajectories.json'),
              [row.to_dict() for row in records])
        if trajectory.completion_status not in ('completed', 'tool_handoff'):
            status = 'resource_limited'
            break
        try:
            plan = parse_response(trajectory.response_text)
            if bool(plan['calls']) != (trajectory.completion_status == 'tool_handoff'):
                raise ValueError('tool plan does not match generation handoff marker')
            # Validate the entire plan before the first potentially effective call.
            for call in plan['calls']:
                if call['name'] not in catalog:
                    raise ValueError('model requested a tool outside the session catalog')
                jsonschema.validate(call['arguments'], catalog[call['name']]['input_schema'])
        except (ValueError, jsonschema.ValidationError) as error:
            # Framing/schema failure is a model outcome, never a fabricated tool result.
            status, failure = 'invalid_plan', type(error).__name__
            break
        if not plan['calls']:
            status = 'completed'
            break
        assistant = {'role':'assistant', 'content':'', 'reasoning':plan['reasoning'], 'tool_calls':[]}
        for index, call in enumerate(plan['calls']):
            assistant['tool_calls'].append({'id':f'step-{step}-{index}', 'type':'function',
                'function':{'name':call['name'], 'arguments':call['arguments']}})
        messages.append(assistant)
        for index, call in enumerate(plan['calls']):
            result = transport.tool(call['name'], call['arguments'])
            messages.append({'role':'tool', 'tool_call_id':f'step-{step}-{index}', 'name':call['name'],
                             'content':result['text']})
    source_result = transport.tool('read_file', {'path':name})
    source = checked_read(source_result)
    from adam_kaine_code_rl_tasks import verify, verifier_identity
    evidence = verify(source, task)
    model_receipts = transport.receipts[first_receipt:-1]
    writes = [r for r in model_receipts if r['name'] == 'write_file' and r['arguments'].get('path') == name
              and r['result'].get('success') and r['result'].get('receipt')]
    bound_write = bool(writes) and writes[-1]['result'].get('data', {}).get('revision') == source_result['data']['revision']
    if bound_write:
        checked_write(writes[-1]['result'], source)
    reward = evidence['klpo_reward'] if status == 'completed' and bound_write else 0.0
    for row in records:
        row.reward = reward
    receipt = {'schema':'lilith/runtime-learning-episode@1', 'arm':arm, 'task_id':task.id,
        'split':task.split, 'family':task.family, 'status':status, 'failure':failure,
        'reward':reward, 'eligible_for_update':status in ('completed', 'invalid_plan'),
        'verifier_sha256':verifier_identity(), 'verification':evidence,
        'artifact_sha256':source_result['data']['revision'], 'artifact_read_receipt':source_result['receipt'],
        'write_bound_to_readback':bound_write, 'tools_sha256':digest(transport.request['tools']),
        'session_id':transport.request['session_id'], 'job_id':transport.request['id'],
        'workspace':transport.request['workspace'], 'model_tool_receipts':model_receipts,
        'trajectory_ids':[row.id for row in records]}
    write(output / (arm + '-' + task.id.replace('/', '_') + '-episode.json'), receipt)
    return records, receipt


def snapshot(engine):
    return {name:p.detach().cpu().clone() for name,p in engine.model.named_parameters() if p.requires_grad}


def restore(engine, state):
    import torch
    parameters = {name:p for name,p in engine.model.named_parameters() if p.requires_grad}
    if parameters.keys() != state.keys():
        raise ValueError('adapter tensor surface changed')
    with torch.no_grad():
        for name, parameter in parameters.items():
            if parameter.shape != state[name].shape:
                raise ValueError('adapter tensor shape changed')
            parameter.copy_(state[name].to(parameter.device))
    engine.model.zero_grad(set_to_none=True)


def claim_bank(path, records, bank_id, verifier_sha):
    with sqlite3.connect(path) as database:
        database.execute('CREATE TABLE IF NOT EXISTS claims (trajectory TEXT PRIMARY KEY, bank TEXT, policy TEXT, verifier TEXT)')
        database.execute('BEGIN IMMEDIATE')
        for record in records:
            database.execute('INSERT INTO claims VALUES (?,?,?,?)',
                             (record.id, bank_id, record.policy_sha256, verifier_sha))
        database.commit()


def experiment(engine, transport, config, output):
    """Separate SFT and online reward arms, then fixed blend and sealed comparison."""
    import torch
    from adam_kaine_code_rl_tasks import tasks, verifier_identity
    from lilith_rl import token_mc_loss
    task_list = tasks()
    if verifier_identity() != config['verifier_sha256']:
        raise ValueError('frozen verifier identity changed')
    splits = {split:[t for t in task_list if t.split == split]
              for split in ('train', 'development_holdout', 'independent_eval')}
    if any(len(group) != 4 for group in splits.values()):
        raise ValueError('frozen train/development/evaluation task split changed')
    initial = snapshot(engine)
    initial_policy = asdict(engine.policy_identity())
    report = {'schema':'lilith/runtime-training-result@1', 'status':'running',
              'initial_policy':initial_policy, 'updates':[], 'release_selected':False}
    def collect(arm, selected):
        rows, receipts = [], []
        for task in selected:
            segments, receipt = episode(engine, transport, task, arm, config, output)
            rows.extend(segments); receipts.append(receipt)
        return rows, receipts
    base_rows, base_receipts = collect('base-train', splits['train'])
    report['base_train'] = base_receipts
    write(output / 'result.json', report)
    dataset = Path(config['transcript_dataset'])
    if digest_file(dataset) != config['transcript_dataset_sha256']:
        raise ValueError('transcript dataset identity changed')
    examples = []
    for line in dataset.read_text().splitlines():
        row = json.loads(line)
        if row['split'] == 'train':
            user, assistant = row['messages']
            prompt_ids = engine.render_messages([user], thinking=True)[1]
            answer = engine.tokenizer.encode(assistant['content'], add_special_tokens=False) + [106]
            examples.append((prompt_ids, answer))
    if len(examples) != 8:
        raise ValueError('frozen transcript training subset changed')
    transport.progress('transcript-sft')
    report['sft'] = asdict(engine.sft_step(examples, learning_rate=config['learning_rate']))
    states = {'sft_distill':snapshot(engine)}
    engine.model.save_pretrained(output / 'sft_distill', safe_serialization=True)
    restore(engine, initial)
    for bank in range(config['online_banks']):
        bank_id = 'online-bank-' + str(bank)
        rows, receipts = collect(bank_id, splits['train'])
        completed_ids = {identifier for receipt in receipts if receipt['eligible_for_update']
                         for identifier in receipt['trajectory_ids']}
        eligible = [row for row in rows if row.id in completed_ids]
        rewards = {row.reward for row in eligible}
        if len(rewards) < 2:
            report['updates'].append({'bank':bank_id, 'optimizer_steps':0, 'status':'uninformative_reward'})
        else:
            claim_bank(output / 'consumption.sqlite', eligible, bank_id, config['verifier_sha256'])
            count = sum(receipt['eligible_for_update'] for receipt in receipts)
            def loss(action_logp, aux_logp, old_action_logp, old_aux_logps, reward):
                return token_mc_loss(action_logp.reshape(1, 1),
                    torch.tensor([[old_action_logp]], device=action_logp.device),
                    aux_logp.reshape(1, 1, -1), torch.tensor([reward], device=action_logp.device),
                    torch.ones((1, 1), dtype=torch.bool, device=action_logp.device), config['beta']) / count
            update = engine.online_update(eligible, learning_rate=config['learning_rate'], token_loss=loss)
            report['updates'].append({'bank':bank_id, **asdict(update)})
        write(output / 'result.json', report)
    states['klpo_online'] = snapshot(engine)
    engine.model.save_pretrained(output / 'klpo_online', safe_serialization=True)
    states['adapter_blend_0p5'] = {name:states['sft_distill'][name] * 0.5 + states['klpo_online'][name] * 0.5
                                 for name in initial}
    restore(engine, states['adapter_blend_0p5'])
    engine.model.save_pretrained(output / 'adapter_blend_0p5', safe_serialization=True)
    development = {}
    for arm, state in states.items():
        restore(engine, state)
        _, receipts = collect(arm + '-development', splits['development_holdout'])
        development[arm] = receipts
    priority = {'sft_distill':0, 'klpo_online':1, 'adapter_blend_0p5':2}
    selected = max(states, key=lambda arm:(sum(r['reward'] for r in development[arm]), -priority[arm]))
    selection = {'selected':selected, 'development':development, 'frozen_before_eval':True,
                 'candidate_policy':{arm:digest({name:hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                     for name,tensor in state.items()}) for arm,state in states.items()}}
    write(output / 'selection.json', selection)
    # Common RNG namespace, paired after immutable selection. Different filenames.
    restore(engine, states[selected])
    selected_out = output / 'selected-eval'; selected_out.mkdir()
    selected_eval = [episode(engine, transport, task, 'paired-eval', config, selected_out)[1]
                     for task in splits['independent_eval']]
    restore(engine, initial)
    base_out = output / 'base-eval'; base_out.mkdir()
    base_eval = [episode(engine, transport, task, 'paired-eval', config, base_out)[1]
                 for task in splits['independent_eval']]
    from lilith_rl_tasks import tasks as earlier_tasks
    retention = [task for task in earlier_tasks() if task.split == 'train']
    base_retention_out = output / 'base-retention'; base_retention_out.mkdir()
    base_retention = [episode(engine, transport, task, 'paired-retention', config, base_retention_out)[1]
                      for task in retention]
    restore(engine, states[selected])
    selected_retention_out = output / 'selected-retention'; selected_retention_out.mkdir()
    selected_retention = [episode(engine, transport, task, 'paired-retention', config, selected_retention_out)[1]
                          for task in retention]
    report.update(status='completed', selected=selected, selection=selection,
                  selected_eval=selected_eval, base_eval=base_eval,
                  base_retention=base_retention, selected_retention=selected_retention,
                  selected_policy=asdict(engine.policy_identity()), cold_reload='pending',
                  release_selected=False)
    write(output / 'result.json', report)
    return report


def run(transport, config):
    validate_catalog(transport.request['tools'])
    if digest(transport.request['tools']) != config['tools_sha256']:
        raise ValueError('training tool catalog differs from frozen inference catalog')
    if set(config['source_pins']) != required_pins(config):
        raise ValueError('worker source pins do not cover the exact required dependency set')
    for path, expected in config['source_pins'].items():
        if digest_file(path) != expected:
            raise ValueError('training worker source identity changed')
    if config['mode'] != 'fixture':
        validate_learning_inputs(config)
    output = Path(config['output'])
    if not output.is_absolute():
        raise ValueError('training output must be operator-selected absolute path')
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    write(output / 'runtime-request.json', transport.request)
    write(output / 'config.json', config)
    write(output / 'result.json', {'status':'starting', 'mode':config['mode'], 'release_selected':False})
    if config['mode'] == 'fixture':
        content = 'Runtime-owned training tool parity fixture.\n'
        reset_artifact(transport, config['artifact_path'], content)
        reset_artifact(transport, config['artifact_path'], content)
        readback = transport.tool('read_file', {'path':config['artifact_path']})
        if checked_read(readback) != content:
            raise ValueError('fixture readback mismatch')
        report = {'status':'completed', 'kind':'tool-parity-fixture', 'optimizer_steps':0,
                  'tool_receipts':transport.receipts, 'tools_sha256':config['tools_sha256']}
    elif config['mode'] in ('train', 'cold_reload'):
        from lilith_training_engine import GemmaTrainingEngine
        transport.progress('loading-parent')
        engine = GemmaTrainingEngine.load(Path(config['parent']),
            context_tokens=config['max_context_tokens'], max_new_tokens=config['max_output_tokens'],
            **config['parent_pins'])
        # The qualified engine source stays untouched: KV placement is set after load
        # from the pinned configuration (False = GPU-resident, e.g. H100 80GB).
        engine.cache_offloading = configured_cache_offloading(config)
        write(output / 'engine-contract.json', engine.contract())
        if config['mode'] == 'train':
            report = experiment(engine, transport, config, output)
        else:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            prior = Path(config['reload'])
            previous = json.loads((prior / 'result.json').read_text())
            if previous['status'] != 'completed':
                raise ValueError('cold reload requires a completed learning run')
            adapter = prior / previous['selected'] / 'adapter_model.safetensors'
            if digest_file(adapter) != config['reload_adapter_sha256']:
                raise ValueError('cold adapter artifact changed')
            set_peft_model_state_dict(engine.model, load_file(str(adapter)))
            observed = asdict(engine.policy_identity())
            if observed != previous['selected_policy']:
                raise ValueError('cold reload policy identity differs')
            report = {'status':'completed', 'kind':'cold-reload', 'policy_identity':observed,
                      'adapter_sha256':config['reload_adapter_sha256'], 'optimizer_steps':0,
                      'release_selected':False}
    else:
        raise ValueError('unsupported worker mode')
    write(output / 'result.json', report)
    return report


def main():
    transport = RuntimeTransport(sys.stdin.buffer, sys.stdout.buffer)
    config = None
    try:
        config = transport.start()
        # Third-party diagnostics must never become protocol frames.
        with contextlib.redirect_stdout(sys.stderr):
            report = run(transport, config)
        transport.send({'type':'completed', 'output':json.dumps({'status':report['status'],
                        'artifact_directory':config['output'], 'release_selected':False})})
    except Exception as error:
        if config is not None:
            output = Path(config['output'])
            if output.is_dir():
                write(output / 'failure.json', {'status':'failed', 'error_type':type(error).__name__,
                      'error':str(error)[:4096], 'release_selected':False})
        transport.send({'type':'error', 'message':'training worker failed: ' + type(error).__name__})
        raise


if __name__ == '__main__':
    main()
