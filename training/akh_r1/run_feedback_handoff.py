#!/usr/bin/env python3
"""Bounded neural replanning, public goal feedback and native completion controls."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from build_curriculum import digest, now
from command_registry import CommandRegistry
from environment import Graph
from protocol import parse_action
from run_binding_kata import fixture_for, score
from run_exposure import read, write, unchanged
from run_goal_handoff import TOOLS, REQUESTS, messages
from run_symbolic_completion import execute
from symbolic_policy import compile_public, populate, infer, proof_matches_rules, rules, DESCRIPTIONS
from typed_registry_choices import TokenChoiceMask

def public_contract_json(value):
    """Canonical public-only payload. Key checks are not a semantic leak detector."""
    forbidden = {'expected', 'expected_effects', 'examiner', 'private_state',
                 'future_correction', 'answer_key', 'condition', 'phase', 'score_labels'}
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item) or forbidden.intersection(item):
                raise ValueError('Private or invalid public keys')
            for child in item.values(): check(child)
        elif isinstance(item, list):
            for child in item: check(child)
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def validate_public_arguments(value, schema):
    """Only the frozen small schema subset; unsupported keywords fail closed."""
    def check_schema(spec):
        if not isinstance(spec, dict): raise ValueError('Schema must be an object')
        if not spec: return
        kind = spec.get('type')
        allowed = {'type', 'enum'}
        if kind == 'object': allowed |= {'properties', 'required', 'additionalProperties'}
        elif kind == 'array': allowed |= {'items'}
        elif kind != 'string': raise ValueError('Unsupported schema type')
        if set(spec) - allowed: raise ValueError('Unsupported schema keyword')
        if 'enum' in spec and (not isinstance(spec['enum'], list) or not spec['enum']):
            raise ValueError('Invalid enum')
        if kind == 'object':
            props = spec.get('properties'); required = spec.get('required')
            if (not isinstance(props, dict) or not isinstance(required, list)
                    or any(not isinstance(k, str) for k in required)
                    or len(set(required)) != len(required) or set(required) != set(props)
                    or spec.get('additionalProperties') is not False):
                raise ValueError('Exact object schema required')
            for child in props.values(): check_schema(child)
        if kind == 'array':
            if 'items' not in spec: raise ValueError('Array item schema required')
            check_schema(spec['items'])
    def validate(item, spec):
        if not spec: return
        kind = spec['type']
        if 'enum' in spec and item not in spec['enum']: raise ValueError('Value outside enum')
        if kind == 'string' and not isinstance(item, str): raise ValueError('String required')
        if kind == 'object':
            if not isinstance(item, dict) or set(item) != set(spec['properties']):
                raise ValueError('Unexpected or missing arguments')
            for key, child in spec['properties'].items(): validate(item[key], child)
        if kind == 'array':
            if not isinstance(item, list): raise ValueError('Array required')
            for child in item: validate(child, spec['items'])
    check_schema(schema)
    public_contract_json(value)
    validate(value, schema)
    return value


def validate_public_model_request(raw, schemas):
    request = parse_public_model_request(raw, list(schemas))
    validate_public_arguments(request['arguments'], schemas[request['tool']])
    return request


def public_sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Lowercase SHA256 required')
    return value


def admit_public_profile(profile, expected_sha256):
    import hashlib
    public_sha256(expected_sha256)
    actual = hashlib.sha256(public_contract_json(profile).encode()).hexdigest()
    if actual != expected_sha256: raise ValueError('Profile fingerprint mismatch')
    return profile


def validate_public_profile(start, profile, request=None):
    """Pure validation against operator-supplied frozen profile; no runtime access."""
    supported = {'base-public-v1', 'selector-public-v1',
                 'owned-revision-public-v1', 'perception-effect-public-v1'}
    if profile.get('id') not in supported: raise ValueError('Unsupported profile')
    if set(start) != set(profile['start_fields']): raise ValueError('Profile start fields differ')
    if start.get('schema') != 'akh-public-episode-v1': raise ValueError('Unsupported envelope')
    if type(start['max_calls']) is not int or not 1 <= start['max_calls'] <= 32:
        raise ValueError('Call limit outside proposal')
    public_contract_json(start)
    if isinstance(profile['tools'], list) and start['tools'] != profile['tools']:
        raise ValueError('Profile tools differ')
    def handle(value, nullable=False):
        if value is None and nullable: return
        if (not isinstance(value, dict) or set(value) != {'receipt', 'sha256'}
                or not all(isinstance(v, str) and v for v in value.values())):
            raise ValueError('Invalid source handle')
        public_sha256(value['sha256'])
    if 'retained_source_handle' in start: handle(start['retained_source_handle'], True)
    if request is None: return start
    parsed = parse_public_model_request(public_contract_json(request), start['tools'])
    if parsed['tool'] == 'interact':
        args = parsed['arguments']
        if set(args) != {'action', 'value'}: raise ValueError('Exact interaction required')
        actions = profile['actions']
        if not isinstance(actions, dict) or args['action'] not in actions:
            raise ValueError('Undeclared interaction')
        fields = actions[args['action']]; value = args['value']
        if fields is None:
            if value is not None: raise ValueError('Null action value required')
        else:
            if not isinstance(value, dict) or set(value) != set(fields):
                raise ValueError('Action fields differ')
            for key, item in value.items():
                if key in ('previous', 'handle'): handle(item, key == 'previous')
                elif not isinstance(item, str) or not item: raise ValueError('Receipt/hash string required')
                elif key == 'sha256' or key.endswith('_hash'): public_sha256(item)
    return parsed


def public_profile_messages(start, profile, schemas, exchanges, expected_profile_sha256):
    admit_public_profile(profile, expected_profile_sha256)
    validate_public_profile(start, profile)
    if set(schemas) != set(start['tools']): raise ValueError('Schema tool mismatch')
    if len(exchanges) > start['max_calls']: raise ValueError('Decision ceiling exceeded')
    result = [{'role': 'system', 'content': 'Return one declared JSON tool request; finish explicitly. No hidden repair.'},
              {'role': 'user', 'content': public_contract_json({'start': start, 'tool_schemas': schemas})}]
    for index, exchange in enumerate(exchanges):
        if set(exchange) != {'request', 'response'}: raise ValueError('Invalid exchange')
        request = validate_public_profile(start, profile, exchange['request'])
        validate_public_arguments(request['arguments'], schemas[request['tool']])
        if request['tool'] == 'finish' and index != len(exchanges)-1: raise ValueError('Events after finish')
        result.extend([{'role': 'assistant', 'content': public_contract_json(request)},
                       {'role': 'user', 'content': public_contract_json({'tool_response': exchange['response']})}])
    return result


def public_model_messages(start, schemas, exchanges):
    """No hidden state, model calls, inferred completion or context truncation."""
    required = {'schema', 'goal', 'tools', 'max_calls'}
    if not required <= set(start) or set(start) - required - {'retained_source_handle'}:
        raise ValueError('Unexpected start fields')
    if (not isinstance(start['tools'], list) or len(set(start['tools'])) != len(start['tools'])
            or set(start['tools']) != set(schemas) or 'finish' not in schemas):
        raise ValueError('Tool schema mismatch')
    if type(start['max_calls']) is not int or not 1 <= start['max_calls'] <= 32:
        raise ValueError('Proposed call ceiling exceeded')
    if len(exchanges) > start['max_calls']: raise ValueError('Decision ceiling exceeded')
    result = [{'role': 'system', 'content': 'Return exactly one JSON object with tool and arguments. Use only declared tools. Completion requires an explicit finish request.'},
              {'role': 'user', 'content': public_contract_json({'start': start, 'tool_schemas': schemas})}]
    for index, exchange in enumerate(exchanges):
        if set(exchange) != {'request', 'response'}: raise ValueError('Invalid exchange')
        request = validate_public_model_request(public_contract_json(exchange['request']), schemas)
        if request['tool'] == 'finish' and index != len(exchanges)-1: raise ValueError('Events after finish')
        result.append({'role': 'assistant', 'content': public_contract_json(request)})
        result.append({'role': 'user', 'content': public_contract_json({'tool_response': exchange['response']})})
    return result


def parse_public_model_request(raw, tools):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out: raise ValueError('Duplicate JSON key')
            out[key] = value
        return out
    if not isinstance(raw, str) or len(raw) > 65536: raise ValueError('Invalid request length')
    request = json.loads(raw, object_pairs_hook=pairs,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    if (not isinstance(request, dict) or set(request) != {'tool', 'arguments'}
            or request['tool'] not in tools or not isinstance(request['arguments'], dict)):
        raise ValueError('Invalid tool request')
    public_contract_json(request)
    if request['tool'] == 'finish' and request['arguments'].get('status') not in ('complete', 'blocked'):
        raise ValueError('Explicit finish status required')
    return request


CONDITIONS = {
    'withheld': dict(feedback=False, completion=False),
    'feedback': dict(feedback=True, completion=False),
    'withheld_completion': dict(feedback=False, completion=True),
    'feedback_completion': dict(feedback=True, completion=True),
}


def goal_status(public, action=None, response=None):
    """Declared public-language policy, never an examiner-derived outcome label."""
    try:
        facts = compile_public(public)
    except ValueError as error:
        return dict(supported=False, satisfied=False, required_intent=None, reason=str(error))
    wanted = next(f['object'] for f in facts if f['predicate'] == 'wants')
    job = next(f['object'] for f in facts if f['predicate'] == 'target_job')
    worker = next(f['object'] for f in facts if f['predicate'] == 'observed_worker')
    subject = action['arguments']['subject'] if action else None
    capability = DESCRIPTIONS.get(public['records'].get(subject, {}).get('description'))
    try:
        value = json.loads((response or {}).get('stdout', ''))
    except (ValueError, TypeError):
        value = None
    satisfied = False
    if (response or {}).get('success') and capability == wanted and isinstance(value, dict):
        if wanted == 'observe_job':
            satisfied = (value.get('job') == job and value.get('worker') == worker and
                         all(isinstance(value.get(k), (int, float)) for k in ['pending', 'archives', 'observed_at']))
        elif wanted == 'observe_worker':
            satisfied = value.get('worker') == worker and value.get('mode') in ['paused', 'flowing']
        else:
            satisfied = value.get('worker') == worker and value.get('mode') == 'flowing'
    return dict(supported=True, satisfied=bool(satisfied), required_intent=wanted,
                reason='receipt_satisfies_public_goal' if satisfied else 'public_goal_not_yet_satisfied')


def absorb(public, response):
    """Carry actual successful observations forward, retaining their receipts."""
    result = copy.deepcopy(public)
    if not response.get('success'):
        return result
    try:
        value = json.loads(response.get('stdout', ''))
        old_job = json.loads(public['observations']['job']['stdout'])
    except (ValueError, TypeError):
        return result
    if not isinstance(value, dict):
        return result
    if value.get('worker') != old_job.get('worker'):
        return result
    if value.get('job') == old_job.get('job'):
        result['observations']['job'] = copy.deepcopy(response)
    if value.get('mode') in ['paused', 'flowing']:
        result['observations']['worker'] = copy.deepcopy(response)
    return result


def next_messages(inputs, raw, response, derivation, checked, goal, feedback):
    result = copy.deepcopy(inputs)
    result.append(dict(role='assistant', content=raw))
    payload = (dict(execution=response, native_reason=derivation['reason'],
                    native_derived=derivation['proof']['derived'], public_goal_check=checked)
               if feedback else dict(feedback_withheld=True))
    result.append(dict(role='tool', content=json.dumps(payload, sort_keys=True)))
    result.append(dict(role='user', content='Retained operator goal: ' + goal +
                       '\nChoose the requested effect again using the available evidence.'))
    return result


def prepare(args):
    parent = args.parent.resolve(); pp = read(parent / 'plan.json')
    if (read(parent / 'status.json')['stage'] != 'completed' or
        not read(parent / 'summary.json')['all_checks_passed'] or not unchanged(pp['files'])):
        raise ValueError('Completed audited observed-handoff parent required')
    root = args.out.resolve(); root.mkdir(exist_ok=False); source = root / 'source'; source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'): shutil.copy2(path, source / path.name)
    shutil.copy2(pp['bridge'], source / 'akh_symbolic_episode_bridge')
    shutil.copy2(parent / 'public-inputs.json', root / 'public-inputs.json')
    write(root / 'rules.json', rules())
    files = dict(pp['files'])
    for path in [*source.iterdir(), root / 'public-inputs.json', root / 'rules.json', parent / 'summary.json']:
        files[str(path)] = digest(path)
    p = dict(created_at=now(), parent=str(parent), files=files, cases=pp['cases'],
        model=pp['model'], adapters=pp['adapters'], image=pp['image'],
        bridge=str(source / 'akh_symbolic_episode_bridge'), lock_file=str(args.lock_file.resolve()),
        conditions=CONDITIONS, max_decisions_per_episode=3, episodes=144, max_model_decisions=432,
        max_new_tokens=96, max_prompt_tokens=4096, timeout_seconds=1800, min_free_bytes=5 * 2**30,
        seed=83, training_updates=0, automatic_retries=False, certification=False,
        scope='2 adapters x 18 prepared public goals x 2 feedback settings x 2 native completion settings. '
              'Feedback includes actual execution, native derivation and explicit literal-domain goal checking. '
              'Final-decision completion may replace the neural intent with a supported public goal. '
              'Cold native graph per decision, supplied command records, actual retained fixture state per episode. '
              'No hidden examiner outcome enters model, goal checker or runtime decisions; no acquisition or full incident claim.')
    write(root / 'plan.json', p); (root / 'plan.sha256').write_text(digest(root / 'plan.json') + '\n')
    write(root / 'status.json', dict(stage='prepared', updated_at=now()))


def derive(root, bridge, public, intent):
    root.mkdir(parents=True)
    graph = Graph(bridge, root / 'graph', root / 'engine.log')
    try:
        first_pid = graph.process.pid; facts = populate(graph, public, supplied_intent=intent)
        exported = graph.call('exam_export')
    finally:
        graph.close()
    graph = Graph(bridge, root / 'graph', root / 'engine-cold.log')
    try:
        if graph.call('exam_export') != exported: raise ValueError('Cold graph changed')
        derivation = infer(graph)
        if not proof_matches_rules(derivation['proof'], rules()): raise ValueError('Native proof failed')
        write(root / 'derivation.json', derivation)
        return graph, facts, derivation, [first_pid, graph.process.pid]
    except BaseException:
        graph.close(); raise


def episode_effect(case, after, rows):
    stages = [row['execution']['argv'] for row in rows if row['execution']]
    scores = [score(case['before'], after, case['target_job'], case['operation'],
                    row['response'], stages, case['examiner']['expected_argv']) for row in rows]
    return dict(effect_achieved=any(s['kata_effect_pass'] for s in scores),
                receipt_scores=scores, full_episode_success=None, certification=False)


def run(root):
    import mlx.core as mx
    from mlx_lm import load as load_model, generate
    from mlx_lm.sample_utils import make_sampler
    p = read(root / 'plan.json'); public = read(root / 'public-inputs.json'); count = 0; episodes = 0
    with (root / 'trace.jsonl').open('x') as trace, (root / 'episodes.jsonl').open('x') as finished:
        for arm, adapter in p['adapters'].items():
            model, tokenizer = load_model(p['model'], adapter_path=adapter)
            for case_path in p['cases']:
                case = read(case_path)
                for condition, options in p['conditions'].items():
                    visible = copy.deepcopy(public[case['id']])
                    inputs = messages(visible, 'supplied_observations')
                    fixture = fixture_for(p['image'], case['before']['contract'], 'feedback')
                    rows = []; stopped = 'decision_budget_exhausted'
                    try:
                        fixture.reset(case['before'])
                        for index in range(p['max_decisions_per_episode']):
                            prompt = tokenizer.apply_chat_template(inputs, tools=TOOLS, tokenize=True,
                                add_generation_prompt=True, enable_thinking=False)
                            if len(prompt) > p['max_prompt_tokens']: raise ValueError('Prompt budget exceeded')
                            mx.random.seed(p['seed'])
                            raw = generate(model, tokenizer, prompt=prompt, max_tokens=p['max_new_tokens'],
                                sampler=make_sampler(temp=0), logits_processors=[TokenChoiceMask(tokenizer, REQUESTS)])
                            request = parse_action(raw, 'native')
                            if request not in REQUESTS: raise ValueError('Undeclared model intent')
                            proposed = request['arguments']['intent']; checked_before = goal_status(visible)
                            completion_allowed = (options['completion'] and index == p['max_decisions_per_episode'] - 1
                                                  and checked_before['supported'])
                            selected = checked_before['required_intent'] if completion_allowed else proposed
                            dest = root / arm / condition / case['id'] / str(index)
                            graph, facts, derivation, pids = derive(dest, p['bridge'], visible, selected)
                            before = fixture.state()
                            try:
                                execution = None; response = dict(success=False, error='No authorized native derivation')
                                if derivation['action']:
                                    registry = CommandRegistry(graph, [case['before']['contract']['observer'], case['before']['contract']['controller']])
                                    execution = registry.invoke(fixture, **derivation['action']['arguments'])
                                    response = execution['response']
                            finally:
                                graph.close()
                            checked = goal_status(visible, derivation['action'], response)
                            row = dict(arm=arm, case=case['id'], condition=condition, index=index,
                                input=copy.deepcopy(inputs), public_before=visible, raw=raw, request=request,
                                prompt_tokens=len(prompt), proposed_intent=proposed, executed_intent=selected,
                                completion_allowed=completion_allowed, symbolic_override=selected != proposed,
                                goal_before=checked_before, goal_after=checked, facts=facts, engine_pids=pids,
                                proof_path=str(dest / 'derivation.json'), symbolic_action=derivation['action'],
                                execution=execution, response=response, before=before, after=fixture.state())
                            rows.append(row); trace.write(json.dumps(row) + '\n'); trace.flush(); count += 1
                            if checked['satisfied']:
                                stopped = 'public_goal_receipt_satisfied'; break
                            visible = absorb(visible, response)
                            inputs = next_messages(inputs, raw, response, derivation, checked, visible['goal'], options['feedback'])
                        after = fixture.state()
                        record = dict(arm=arm, case=case['id'], condition=condition, decisions=len(rows),
                            stop_reason=stopped, symbolic_overrides=sum(r['symbolic_override'] for r in rows),
                            after=after, score=episode_effect(case, after, rows))
                        finished.write(json.dumps(record) + '\n'); finished.flush(); episodes += 1
                        write(root / 'progress.json', dict(updated_at=now(), episodes=episodes, total=p['episodes'],
                            model_decisions=count, arm=arm, condition=condition))
                    finally:
                        fixture.close()
            del model, tokenizer; mx.clear_cache()
    write(root / 'results.json', dict(completed_at=now(), episodes=episodes, model_decisions=count,
                                    input_hashes_unchanged=unchanged(p['files'])))


def audit(root):
    p = read(root / 'plan.json'); public = read(root / 'public-inputs.json')
    cases = {read(path)['id']:read(path) for path in p['cases']}
    rows = [json.loads(s) for s in (root / 'trace.jsonl').read_text().splitlines()]
    episodes = [json.loads(s) for s in (root / 'episodes.jsonl').read_text().splitlines()]
    checks = []; groups = {}
    def check(label, value): checks.append(dict(label=label, passed=bool(value)))
    check('inputs_unchanged', unchanged(p['files']))
    check('plan_unchanged', digest(root / 'plan.json') == (root / 'plan.sha256').read_text().strip())
    expected = {(a,c,k) for a in p['adapters'] for c in cases for k in p['conditions']}
    check('coverage', len(episodes) == len(expected) == p['episodes'] and
          {(e['arm'],e['case'],e['condition']) for e in episodes} == expected)
    check('model_budget', len(rows) <= p['max_model_decisions'])
    for e in episodes:
        key = (e['arm'], e['case'], e['condition']); case = cases[e['case']]; options = p['conditions'][e['condition']]
        selected_rows = [r for r in rows if (r['arm'],r['case'],r['condition']) == key]
        label = '/'.join(key); visible = copy.deepcopy(public[e['case']]); inputs = messages(visible, 'supplied_observations')
        before = case['before']
        check(label+'/count', 1 <= len(selected_rows) == e['decisions'] <= p['max_decisions_per_episode'])
        for index, r in enumerate(selected_rows):
            step = label+'/'+str(index); derivation = read(r['proof_path'])
            check(step+'/causal_input', r['index'] == index and r['public_before'] == visible and r['input'] == inputs)
            check(step+'/request', parse_action(r['raw'], 'native') == r['request'] and r['request'] in REQUESTS)
            check(step+'/budget', r['prompt_tokens'] <= p['max_prompt_tokens'])
            checked_before = goal_status(visible); allowed = options['completion'] and index == p['max_decisions_per_episode']-1 and checked_before['supported']
            intent = checked_before['required_intent'] if allowed else r['request']['arguments']['intent']
            check(step+'/selection', r['completion_allowed'] == allowed and r['executed_intent'] == intent and
                  r['proposed_intent'] == r['request']['arguments']['intent'] and r['symbolic_override'] == (intent != r['proposed_intent']))
            check(step+'/facts', r['facts'] == compile_public(visible, supplied_intent=intent))
            check(step+'/native', proof_matches_rules(derivation['proof'], read(root/'rules.json')) and
                  derivation['action'] == r['symbolic_action'] and len(set(r['engine_pids'])) == 2)
            check(step+'/execution_receipt', (r['execution'] is None and r['symbolic_action'] is None and not r['response']['success']) or
                  (r['execution'] is not None and r['symbolic_action'] is not None and r['execution']['response'] == r['response']))
            check(step+'/state_chain', before == r['before'])
            if case['operation'] != 'resume': check(step+'/readonly', r['before'] == r['after'])
            checked = goal_status(visible, r['symbolic_action'], r['response'])
            check(step+'/goal_check', r['goal_before'] == checked_before and r['goal_after'] == checked)
            if checked['satisfied']: check(step+'/stop', index == len(selected_rows)-1)
            before = r['after']; visible = absorb(visible, r['response'])
            inputs = next_messages(inputs, r['raw'], r['response'], derivation, checked, visible['goal'], options['feedback'])
        check(label+'/final_state', before == e['after'])
        check(label+'/override_count', e['symbolic_overrides'] == sum(r['symbolic_override'] for r in selected_rows))
        judged = episode_effect(case, e['after'], selected_rows); check(label+'/effect', judged == e['score'])
        check(label+'/stop_reason', (e['stop_reason'] == 'public_goal_receipt_satisfied') == selected_rows[-1]['goal_after']['satisfied'])
        group = groups.setdefault(e['arm']+'/'+e['condition'], dict(episodes=0, effects=0, stopped_on_goal=0, overrides=0, decisions=0, by_operation={}))
        group['episodes'] += 1; group['effects'] += judged['effect_achieved']; group['stopped_on_goal'] += e['stop_reason'] == 'public_goal_receipt_satisfied'
        group['overrides'] += e['symbolic_overrides']; group['decisions'] += len(selected_rows)
        operation = group['by_operation'].setdefault(case['operation'], dict(total=0, effects=0))
        operation['total'] += 1; operation['effects'] += judged['effect_achieved']
    check('all_rows_accounted', sum(e['decisions'] for e in episodes) == len(rows))
    passed = all(c['passed'] for c in checks)
    write(root / 'artifact-audit.json', dict(passed=passed, checked_at=now(), checks=checks))
    write(root / 'summary.json', dict(completed_at=read(root/'results.json')['completed_at'], groups=groups,
        episodes=len(episodes), model_decisions=len(rows), all_checks_passed=passed, audit_checks=len(checks),
        training_updates=0, full_incident_success=None, certification=False))
    if not passed: raise ValueError('Feedback audit failed')
    print(groups, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'execute', 'run', 'audit'])
    for name in ['out', 'parent', 'lock-file']: parser.add_argument('--'+name, type=Path, required=name=='out')
    args = parser.parse_args()
    if args.mode == 'prepare': prepare(args)
    elif args.mode == 'execute': execute(args.out.resolve(), worker='run_feedback_handoff.py', audit_fn=audit)
    elif args.mode == 'run': run(args.out.resolve())
    else: audit(args.out.resolve())
