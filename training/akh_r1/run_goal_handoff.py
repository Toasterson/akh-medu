#!/usr/bin/env python3
"""Neural effect selection followed by native symbolic command/binding derivation."""
import argparse
import json
from pathlib import Path
import shutil

from build_curriculum import digest, now
from command_registry import CommandRegistry
from environment import Graph
from protocol import function, parse_action
from run_binding_kata import fixture_for, score
from run_exposure import read, write, unchanged
from run_symbolic_completion import execute
from symbolic_policy import populate, infer, proof_matches_rules, rules
from typed_registry_choices import TokenChoiceMask

INTENTS = ['observe_worker', 'resume_worker', 'observe_job']
TOOLS = [function('set_goal', 'Identify the requested effect. observe_worker reads the worker mode; '
    'resume_worker makes its work flow; observe_job reads fresh job pending/archive counts. '
    'The symbolic engine will select a command and derive its target from the graph.',
    dict(intent=dict(type='string', enum=INTENTS)), ['intent'])]
REQUESTS = [dict(tool='set_goal', arguments=dict(intent=v)) for v in INTENTS]


def messages(public, evidence='goal_only'):
    result = [dict(role='system', content='Identify the requested effect from the operator goal. '
                 'Choose one supplied set_goal call. The graph/rule engine handles command selection, '
                 'identity binding and execution. Do not infer observations or claim a completed action.'),
            dict(role='user', content=public['goal'])]
    if evidence == 'supplied_observations':
        result.append(dict(role='user', content='Supplied observation receipts:\n' +
                           json.dumps(public['observations'], sort_keys=True)))
    elif evidence != 'goal_only':
        raise ValueError('Unknown evidence condition')
    return result


def prepare(args):
    parent = args.parent.resolve(); pp = read(parent / 'plan.json')
    if read(parent / 'status.json')['stage'] != 'completed' or not read(parent / 'summary.json')['all_checks_passed'] or not unchanged(pp['files']):
        raise ValueError('Completed audited symbolic comparison required')
    training = Path(pp['parent']); training_plan = read(training / 'plan.json')
    model = read(Path(training_plan['profile']) / 'plan.json')['model']
    root = args.out.resolve(); root.mkdir(exist_ok=False); source = root / 'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'): shutil.copy2(p, source / p.name)
    shutil.copy2(pp['bridge'], source / 'akh_symbolic_episode_bridge')
    shutil.copy2(parent / 'public-inputs.json', root / 'public-inputs.json'); write(root / 'rules.json', rules())
    adapters = {a:str(training / a / 'pilot/adapter') for a in ['history', 'compact']}
    files = dict(pp['files'])
    for path in [*source.iterdir(), root / 'public-inputs.json', root / 'rules.json', parent / 'summary.json',
                 *[p for p in Path(model).iterdir() if p.is_file()]]:
        files[str(path)] = digest(path)
    for adapter in adapters.values():
        for name in ['adapters.safetensors', 'adapter_config.json']:
            path = Path(adapter) / name; files[str(path)] = digest(path)
    plan = dict(created_at=now(), parent=str(parent), files=files, cases=pp['cases'], image=pp['image'], model=model,
        adapters=adapters, bridge=str(source / 'akh_symbolic_episode_bridge'), lock_file=str(args.lock_file.resolve()),
        timeout_seconds=900, min_free_bytes=5 * 2**30, decisions=36, max_new_tokens=96,
        evidence=args.evidence,
        training_updates=0, certification=False, automatic_retries=False,
        scope='Each adapter classifies 18 public operator goals in the declared evidence condition; native rules select commands and targets from supplied cold KG/observations. '
              'Three effect choices with greedy constrained syntax. Independent literal authorization guard. Actual one-operation fixture effects. '
              'No new training, no hidden examiner answers in policy/model input, no claim of autonomous acquisition or full episodes.')
    write(root / 'plan.json', plan); (root / 'plan.sha256').write_text(digest(root / 'plan.json') + '\n')
    write(root / 'status.json', dict(stage='prepared', updated_at=now()))


def run(root):
    import mlx.core as mx
    from mlx_lm import load as load_model, generate
    from mlx_lm.sample_utils import make_sampler
    p = read(root / 'plan.json'); public = read(root / 'public-inputs.json'); count = 0
    with (root / 'trace.jsonl').open('x') as trace:
        for arm, adapter in p['adapters'].items():
            model, tokenizer = load_model(p['model'], adapter_path=adapter)
            for case_path in p['cases']:
                case = read(case_path); visible = public[case['id']]; inputs = messages(visible, p.get('evidence', 'goal_only'))
                prompt = tokenizer.apply_chat_template(inputs, tools=TOOLS, tokenize=True, add_generation_prompt=True, enable_thinking=False)
                if len(prompt) > 2048: raise ValueError('Goal prompt exceeds declared 2048-token budget')
                mx.random.seed(83)
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=p['max_new_tokens'], sampler=make_sampler(temp=0),
                               logits_processors=[TokenChoiceMask(tokenizer, REQUESTS)])
                request = parse_action(raw, 'native')
                if request not in REQUESTS: raise ValueError('Intent outside declared choices')
                dest = root / arm / case['id']; dest.mkdir(parents=True)
                graph = Graph(p['bridge'], dest / 'graph', dest / 'engine.log')
                try:
                    first_pid = graph.process.pid
                    facts = populate(graph, visible, supplied_intent=request['arguments']['intent'])
                    write(dest / 'facts.json', facts); before_graph = graph.call('exam_export')
                finally: graph.close()
                graph = Graph(p['bridge'], dest / 'graph', dest / 'engine-cold.log')
                fixture = None
                try:
                    if graph.call('exam_export') != before_graph: raise ValueError('Cold graph changed')
                    derived = infer(graph); write(dest / 'derivation.json', derived)
                    if not proof_matches_rules(derived['proof'], rules()): raise ValueError('Native proof failed independent check')
                    fixture = fixture_for(p['image'], case['before']['contract'], 'handoff')
                    fixture.reset(case['before']); execution = None; stages = []; response = dict(success=False, error='No authorized derivation')
                    if derived['action']:
                        registry = CommandRegistry(graph, [case['before']['contract']['observer'], case['before']['contract']['controller']])
                        execution = registry.invoke(fixture, **derived['action']['arguments']); stages = [execution['argv']]; response = execution['response']
                    after = fixture.state()
                    judged = score(case['before'], after, case['target_job'], case['operation'], response, stages, case['examiner']['expected_argv'])
                    row = dict(arm=arm, case=case['id'], split=case['split'], input=inputs, raw=raw, request=request,
                        prompt_tokens=len(prompt), symbolic_action=derived['action'], proof_path=str(dest / 'derivation.json'),
                        facts=facts, engine_pids=[first_pid, graph.process.pid], cold_graph_verified=True,
                        execution=execution, response=response, stages=stages, after=after, score=judged,
                        model_role='requested effect only', symbolic_role='command selection and identity binding',
                        host_role='literal policy compiler, authorization, version checks and argv execution')
                    trace.write(json.dumps(row) + '\n'); trace.flush(); count += 1
                    write(root / 'progress.json', dict(updated_at=now(), decisions=count, total=p['decisions'], arm=arm))
                finally:
                    graph.close()
                    if fixture: fixture.close()
            del model, tokenizer
            mx.clear_cache()
    write(root / 'results.json', dict(completed_at=now(), decisions=count, input_hashes_unchanged=unchanged(p['files'])))


def audit(root):
    p = read(root / 'plan.json'); rows = [json.loads(s) for s in (root / 'trace.jsonl').read_text().splitlines()]
    public = read(root / 'public-inputs.json'); cases = {read(path)['id']:read(path) for path in p['cases']}
    checks = []; groups = {}
    def check(label, value): checks.append(dict(label=label, passed=bool(value)))
    check('frozen_inputs', unchanged(p['files'])); check('count', len(rows) == p['decisions'])
    check('coverage', {(r['arm'], r['case']) for r in rows} == {(a,c) for a in p['adapters'] for c in cases})
    for i, row in enumerate(rows):
        c = cases[row['case']]; expected_intent = dict(status='observe_worker', resume='resume_worker', watch='observe_job')[c['operation']]
        correct_intent = row['request']['arguments']['intent'] == expected_intent
        judged = score(c['before'], row['after'], c['target_job'], c['operation'], row['response'], row['stages'], c['examiner']['expected_argv'])
        check(f'{i}/input_and_call', row['input'] == messages(public[c['id']], p.get('evidence', 'goal_only')) and parse_action(row['raw'], 'native') == row['request'] and row['request'] in REQUESTS)
        check(f'{i}/proof', proof_matches_rules(read(row['proof_path'])['proof'], read(root / 'rules.json')))
        check(f'{i}/effect', judged == row['score']); check(f'{i}/cold_process', len(set(row['engine_pids'])) == 2)
        if c['operation'] != 'resume': check(f'{i}/readonly', row['after'] == c['before'])
        g = groups.setdefault(row['arm'], dict(total=0, correct_intent=0, effects=0, by_operation={}))
        g['total'] += 1; g['correct_intent'] += correct_intent; g['effects'] += judged['kata_effect_pass']
        op = g['by_operation'].setdefault(c['operation'], dict(total=0, correct_intent=0, effects=0))
        op['total'] += 1; op['correct_intent'] += correct_intent; op['effects'] += judged['kata_effect_pass']
    passed = all(c['passed'] for c in checks)
    write(root / 'artifact-audit.json', dict(passed=passed, checked_at=now(), checks=checks))
    write(root / 'summary.json', dict(completed_at=read(root / 'results.json')['completed_at'], groups=groups,
        evidence=p.get('evidence', 'goal_only'),
        all_checks_passed=passed, audit_checks=len(checks), training_updates=0, full_episode_success=None, certification=False))
    if not passed: raise ValueError('Handoff audit failed')
    print(groups, flush=True)


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__); a.add_argument('mode', choices=['prepare', 'execute', 'run', 'audit'])
    for name in ['out', 'parent', 'lock-file']: a.add_argument('--' + name, type=Path, required=name == 'out')
    a.add_argument('--evidence', choices=['goal_only', 'supplied_observations'], default='goal_only')
    args = a.parse_args()
    if args.mode == 'prepare': prepare(args)
    elif args.mode == 'execute': execute(args.out.resolve(), worker='run_goal_handoff.py', audit_fn=audit)
    elif args.mode == 'run': run(args.out.resolve())
    else: audit(args.out.resolve())
