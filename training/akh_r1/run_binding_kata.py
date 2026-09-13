#!/usr/bin/env python3
"""Assisted, single-action kata: 2 representations x 2 executors, actual effects.

Oracle-prepared observations and cold KG records are supplied to the candidate.
This is not autonomous knowledge acquisition, a full episode, or certification.
"""
import argparse
import copy
import fcntl
import json
import os
import random
import shlex
import shutil
import signal
import subprocess
import time
from pathlib import Path

from build_curriculum import FAMILIES, argv, digest, now
from command_templates import from_help, invoke, read, single_binding, store
from environment import Fixture, Graph, world
from flat_protocol import parse_pipeline
from protocol import STRING, function, parse_action

ARMS = ['prose_raw', 'structured_raw', 'prose_template', 'structured_template']


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def fixture_for(image, contract, suffix):
    return Fixture(image, f'akh-r1-binding-{os.getpid()}-{suffix}',
                   commands=[contract['observer'], contract['controller']],
                   cli=Path(__file__).with_name('contract_cli.py'))


def score(before, after, job, operation, response, stages=None, expected_argv=None):
    worker = before['jobs'][job]['worker']
    try:
        observed = json.loads(response.get('stdout', ''))
    except (ValueError, TypeError):
        observed = {}
    if not isinstance(observed, dict):
        observed = {}
    if operation == 'resume':
        target = before['jobs'][job]
        expected = dict(worker=worker, pending=0, archives=target['archives']+target['pending'])
        achieved = (after['jobs'][job] == expected and observed.get('worker') == worker
                    and observed.get('mode') == 'flowing' and after['workers'][worker]['mode'] == 'flowing')
        bounded = (all(after['jobs'][j] == value for j, value in before['jobs'].items() if j != job)
                   and all(after['workers'][w] == value for w, value in before['workers'].items() if w != worker)
                   and after['workers'][worker]['restarts'] == before['workers'][worker]['restarts']
                   and len(after['audit']) == len(before['audit'])+1
                   and after['audit'][-1]['worker'] == worker)
    else:
        # Plausible JSON alone is not evidence that a fresh observation was read.
        bounded = before == after and expected_argv is not None and expected_argv in (stages or [])
        achieved = (observed.get('worker') == worker and
                    (observed.get('mode') == before['workers'][worker]['mode'] if operation == 'status'
                     else observed.get('job') == job and
                     all(observed.get(k) == v for k, v in before['jobs'][job].items()) and
                     isinstance(observed.get('observed_at'), (int, float))))
    return dict(kata_effect_pass=bool(response.get('success') and achieved and bounded),
                desired_observation_or_effect=bool(achieved), bounded=bool(bounded),
                full_episode_success=None, certification=False)


def prepare(args):
    root = args.out.resolve()
    root.mkdir(exist_ok=False)
    source = root/'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, source/path.name)
    shutil.copy2(args.bridge, source/'akh_episode_bridge')
    image = subprocess.check_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'], text=True).strip()
    (root/'cases').mkdir()
    cases = []
    for family, seed in [('named_flags_tsv', 131), ('positional_csv', 137)]:
        spec = FAMILIES[family]
        assert spec['split'] == 'train'
        contract = dict({k:copy.deepcopy(v) for k,v in spec.items() if k != 'split'}, version=1)
        for operation in ['status', 'resume', 'watch']:
            identity = f'{family}-{seed}-{operation}'
            dest = root/'cases'/identity
            dest.mkdir()
            initial, job = world(seed)
            initial['contract'] = contract
            fixture = fixture_for(image, contract, 'prepare')
            graph = None
            try:
                fixture.reset(initial)
                graph = Graph(source/'akh_episode_bridge', dest/'graph', dest/'engine.log')
                receipts, records = {}, {}
                operations = ['catalog', 'watch', 'status', 'resume']
                random.Random(seed).shuffle(operations)
                for name in [contract['observer'], contract['controller']]:
                    for flag in ['--help', '--version']:
                        result = fixture.pipeline([[name, flag]])
                        if not result['success']:
                            raise RuntimeError('Oracle documentation call failed')
                        receipts[name+flag] = dict(result, evidence_id=f'kata:{identity}/{name}{flag}')
                for index, op in enumerate(operations):
                    executable = contract['observer' if op in ['catalog', 'watch'] else 'controller']
                    help_receipt = receipts[executable+'--help']
                    literal = ' '.join([executable, *contract['argv'][op]])
                    line = help_receipt['stdout'].splitlines().index(literal)+1
                    record = from_help(help_receipt, receipts[executable+'--version'], line,
                                       {contract['observer'], contract['controller']})
                    subject = f'command-{index+1}'
                    if not store(graph, subject, record)['success']:
                        raise RuntimeError('Oracle graph write failed')
                    records[subject] = record
                graph.close()
                graph = Graph(source/'akh_episode_bridge', dest/'graph', dest/'engine-cold.log')
                if {subject:read(graph, subject) for subject in records} != records:
                    raise RuntimeError('Cold graph records differ')
                mapping = fixture.pipeline([argv(contract, 'watch', job=job)])
                if not mapping['success']:
                    raise RuntimeError('Oracle mapping observation failed')
                worker = json.loads(mapping['stdout'])['worker']
                worker_status = fixture.pipeline([argv(contract, 'status', worker=worker)])
                if not worker_status['success']:
                    raise RuntimeError('Oracle worker observation failed')
                preparation = []
                if operation == 'watch':
                    preparation.append(fixture.pipeline([argv(contract, 'resume', worker=worker)]))
                    if not preparation[-1]['success']:
                        raise RuntimeError('Oracle watch precondition failed')
                before = fixture.state()
                before['docs_available'] = False
                fixture.reset(before)
                bindings = {'job':job} if operation == 'watch' else {'worker':worker}
                expected_argv = argv(contract, operation, **bindings)
                selected = next(s for s,r in records.items()
                                if r['argv'] == [contract['observer' if operation == 'watch' else 'controller'],
                                                  *contract['argv'][operation]])
                raw = fixture.pipeline([expected_argv])
                raw_score = score(before, fixture.state(), job, operation, raw,
                                  [expected_argv], expected_argv)
                fixture.reset(before)
                templated = invoke(graph, fixture, selected, bindings)
                template_score = score(before, fixture.state(), job, operation, templated['response'],
                                       [templated['argv']], expected_argv)
                if not raw_score['kata_effect_pass'] or not template_score['kata_effect_pass']:
                    raise RuntimeError('Oracle effect validation failed')
                task = dict(status=f'Read the current mode of the worker responsible for {job}. Do not change state.',
                    resume=f'Restore flow for the worker responsible for {job}, without restart, deletion or changes to other workers.',
                    watch=f'A bounded repair was just performed for {job}. Read a fresh job observation to verify its pending/archive counts. Do not mutate state.')[operation]
                case = dict(id=identity, family=family, split='train', seed=seed,
                    goal=task, before=before, target_job=job, operation=operation,
                    observations=dict(job=mapping, worker=worker_status, preparation=preparation),
                    graph_records=records, source_receipts=receipts,
                    examiner=dict(expected_argv=expected_argv, expected_subject=selected,
                        expected_bindings=bindings, raw=raw_score, template=template_score),
                    assistance='Oracle-prepared observations and exact command records; no candidate acquisition or recall.')
                write(dest/'case.json', case)
                cases.append(str(dest/'case.json'))
            finally:
                if graph:
                    graph.close()
                fixture.close()
    inputs = list(source.iterdir())
    inputs += [p for p in (root/'cases').rglob('*') if p.is_file()]
    inputs += [p for p in args.model.iterdir() if p.is_file()]
    for adapter in [args.base_adapter, args.repair_adapter]:
        inputs += [adapter/'adapters.safetensors', adapter/'adapter_config.json']
    plan = dict(created_at=now(), model=str(args.model.resolve()),
        adapters=dict(base=str(args.base_adapter.resolve()), repair=str(args.repair_adapter.resolve())),
        image=image, bridge=str(source/'akh_episode_bridge'), cases=cases, arms=ARMS,
        lock_file=str(args.lock_file.resolve()), files={str(p.resolve()):digest(p) for p in inputs},
        total_decisions=48, max_new_tokens=160, seed=83, timeout_seconds=900,
        training_updates=0, actual_candidate_actions_executed=True, certification=False,
        oracle_effects_passed=12, assistance='One-action diagnostic with oracle-prepared cold KG and observations. Not full episodes or autonomous KG learning.')
    write(root/'plan.json', plan)
    (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json', dict(stage='prepared', updated_at=now(), oracle_effects_passed=12))
    print(json.dumps(dict(stage='prepared', cases=len(cases), oracle_effects_passed=12)), flush=True)


def messages_and_tools(case, records, arm):
    if arm.startswith('structured'):
        knowledge = json.dumps(records, sort_keys=True)
    else:
        knowledge = '\n'.join(f'{s}: version {r["version"]}; command {r["source_line"]}; '
            f'description {r["description"]}; source {r["source"]} line {r["source_line_number"]}.'
            for s,r in records.items())
    if '_flat_' in arm:
        tools = [function('invoke_command', 'Choose one stored command subject and its actual argument. '
            'For a {worker} placeholder supply the observed worker identifier; for {job}, the job identifier. '
            'A command with no placeholder needs an empty argument. The executor substitutes that string '
            'as one literal argv token and checks the version. You choose the operation and target.',
            dict(subject=dict(type='string', enum=list(records)), argument=STRING), ['subject', 'argument'])]
    elif arm.endswith('template'):
        tools = [function('invoke_command', 'Choose one stored command subject and all its placeholder bindings. '
            'The executor substitutes each binding as one literal argv token and checks the installed version. '
            'You must choose the operation and target from the goal and observations.',
            dict(subject=dict(type='string', enum=list(records)),
                 bindings=dict(type='object', additionalProperties=STRING)), ['subject', 'bindings'])]
    else:
        tools = [function('terminal', 'Run a documented command or Unix pipeline. Supply actual target identifiers '
            'and executable syntax, without explanatory prose. No shell control operators or expansion.',
            dict(command=STRING), ['command'])]
    messages = [dict(role='system', content='One-action lab kata. The observations and persistent command '
        'records were prepared by an oracle. Choose one action that satisfies the requested goal. '
        'Use exactly one supplied tool call, without other prose. Use observed identifiers. '
        'Descriptions are explanations, not executable arguments. Do not invent commands or results.'),
        dict(role='user', content=case['goal']+'\nObserved receipts:\n'+json.dumps(case['observations'])+
             '\nCommand records read from the reopened KG:\n'+knowledge)]
    return messages, tools


def score_request(case, request, arm):
    """Separate interface, operation and target errors; no effect inference."""
    expected = case['examiner']
    tool = 'invoke_command' if arm.endswith('template') else 'terminal'
    available = isinstance(request, dict) and request.get('tool') == tool
    arguments = request.get('arguments', {}) if available else {}
    arguments = arguments if isinstance(arguments, dict) else {}
    if tool == 'invoke_command':
        correct_bindings = (arguments.get('argument') == next(iter(expected['expected_bindings'].values()))
                            if '_flat_' in arm else arguments.get('bindings') == expected['expected_bindings'])
        return dict(tool_available=available,
                    command_subject_correct=available and arguments.get('subject') == expected['expected_subject'],
                    target_binding_correct=available and correct_bindings)
    try:
        exact = available and parse_pipeline(arguments['command']) == [expected['expected_argv']]
    except (ValueError, KeyError, TypeError):
        exact = False
    return dict(tool_available=available, exact_argv=exact)


def prepare_additional_control(args):
    parent = args.parent.resolve()
    plan = load(parent/'plan.json')
    if load(parent/'status.json')['stage'] != 'completed' or plan['arms'] != ARMS:
        raise ValueError('Additional control requires the completed four-arm parent')
    if not all(digest(p) == h for p,h in plan['files'].items()):
        raise ValueError('Parent inputs changed')
    root = args.out.resolve()
    root.mkdir(exist_ok=False)
    source = root/'source'
    source.mkdir()
    for path in Path(__file__).parent.glob('*.py'):
        if not path.name.startswith('test_'):
            shutil.copy2(path, source/path.name)
    shutil.copy2(plan['bridge'], source/'akh_episode_bridge')
    # Same case files, observations, KG snapshots and weights; only interface/source changes.
    files = {p:h for p,h in plan['files'].items() if not Path(p).is_relative_to(parent/'source')}
    files.update({str(p):digest(p) for p in source.iterdir()})
    flat_arms = ['prose_flat_template', 'structured_flat_template']
    untuned = args.mode == 'prepare-untuned'
    if untuned:
        plan['adapters'] = {'untuned': None}
    plan.update(created_at=now(), parent=str(parent), parent_plan_sha256=digest(parent/'plan.json'),
                arms=ARMS+flat_arms if untuned else flat_arms, total_decisions=36 if untuned else 24,
                bridge=str(source/'akh_episode_bridge'), files=files,
                difference=('Untuned parent on all six conditions; exact same case observations, no adapter or weight updates.' if untuned
                            else 'Flat subject/argument interface instead of nested subject/bindings; no new case observations or weight updates.'))
    write(root/'plan.json', plan)
    (root/'plan.sha256').write_text(digest(root/'plan.json')+'\n')
    write(root/'status.json', dict(stage='prepared', updated_at=now(), parent=str(parent)))


def evaluate(root):
    import mlx.core as mx
    from mlx_lm import generate, load as load_model
    from mlx_lm.sample_utils import make_sampler
    plan = load(root/'plan.json')
    if digest(root/'plan.json') != (root/'plan.sha256').read_text().strip():
        raise ValueError('Plan changed')
    if load(root/'status.json')['stage'] != 'prepared':
        raise ValueError('Refusing replay')
    if not all(digest(p) == h for p,h in plan['files'].items()):
        raise ValueError('Frozen inputs changed')
    with Path(plan['lock_file']).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def stop(signum, frame):
            raise TimeoutError(f'Kata interrupted by signal {signum}')
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(plan['timeout_seconds'])
        results = []
        started = now()
        try:
            for label, adapter in plan['adapters'].items():
                model, tokenizer = load_model(plan['model'], adapter_path=adapter)
                for arm in plan['arms']:
                    for path in plan['cases']:
                        if shutil.disk_usage(root).free < 5*2**30:
                            raise RuntimeError('Desktop disk below five GiB')
                        case = load(path)
                        dest = root/label/arm/case['id']
                        dest.mkdir(parents=True)
                        shutil.copytree(Path(path).parent/'graph', dest/'graph')
                        graph = Graph(plan['bridge'], dest/'graph', dest/'engine.log')
                        fixture = None
                        try:
                            records = {s:read(graph, s) for s in case['graph_records']}
                            if records != case['graph_records']:
                                raise RuntimeError('Cold graph content changed')
                            messages, tools = messages_and_tools(case, records, arm)
                            write(dest/'input.json', dict(messages=messages, tools=tools))
                            write(root/'status.json', dict(stage='evaluating', pid=os.getpid(), started_at=started,
                                updated_at=now(), model=label, arm=arm, case=case['id'], completed=len(results)))
                            prompt = tokenizer.apply_chat_template(messages, tools=tools, tokenize=True,
                                        add_generation_prompt=True, enable_thinking=False)
                            mx.random.seed(plan['seed'])
                            raw = generate(model, tokenizer, prompt=prompt, max_tokens=plan['max_new_tokens'],
                                           sampler=make_sampler(temp=.7, top_p=.8, top_k=20))
                            fixture = fixture_for(plan['image'], case['before']['contract'], 'candidate')
                            fixture.reset(case['before'])
                            stages, request, execution = None, None, None
                            try:
                                request = parse_action(raw, 'native')
                                if arm.endswith('template') and request['tool'] == 'invoke_command':
                                    supplied = request['arguments']
                                    bindings = (single_binding(read(graph, supplied['subject']), supplied['argument'])
                                                if '_flat_' in arm else supplied['bindings'])
                                    execution = invoke(graph, fixture, request['arguments']['subject'],
                                                       bindings)
                                    stages, response = [execution['argv']], execution['response']
                                elif arm.endswith('raw') and request['tool'] == 'terminal':
                                    stages = parse_pipeline(request['arguments']['command'])
                                    response = fixture.pipeline(stages)
                                else:
                                    raise ValueError('Tool is outside the supplied interface')
                            except (ValueError, KeyError, TypeError) as error:
                                response = dict(success=False, error=str(error))
                            after = fixture.state()
                            result = dict(model=label, arm=arm, case=case['id'], raw=raw, request=request,
                                stages=stages, execution=execution, response=response, after=after,
                                score=score(case['before'], after, case['target_job'], case['operation'], response,
                                            stages, case['examiner']['expected_argv']),
                                prompt_tokens=len(prompt), peak_mlx_bytes=mx.get_peak_memory(),
                                expected_argv=case['examiner']['expected_argv'], created_at=now(), assisted=True)
                            result['components'] = score_request(case, request, arm)
                            write(dest/'result.json', result)
                            results.append(result)
                            print(json.dumps({k:result[k] for k in ['model', 'arm', 'case', 'score']}), flush=True)
                        finally:
                            if fixture:
                                fixture.close()
                            graph.close()
                del model, tokenizer
                mx.clear_cache()
            unchanged = all(digest(p) == h for p,h in plan['files'].items())
            if not unchanged:
                raise RuntimeError('Frozen inputs changed during the run')
            write(root/'results.json', dict(completed_at=now(), decisions=results,
                input_hashes_unchanged=True, certification=False, assistance=plan['assistance']))
            write(root/'status.json', dict(stage='completed', updated_at=now(), decisions=len(results), certification=False))
        except BaseException as error:
            write(root/'status.json', dict(stage='failed', updated_at=now(), completed=len(results), error=str(error)))
            raise
        finally:
            signal.alarm(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'prepare-flat', 'prepare-untuned', 'evaluate'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    for arg in ['model', 'base-adapter', 'repair-adapter', 'bridge', 'lock-file']:
        parser.add_argument('--'+arg, type=Path)
    parser.add_argument('--image', default='akh-medu/native-ocr-offline:recovery-20260910')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if any(getattr(args, name) is None for name in ['model', 'base_adapter', 'repair_adapter', 'bridge', 'lock_file']):
            parser.error('Preparation requires model, both adapters, bridge and lock-file')
        prepare(args)
    elif args.mode in ['prepare-flat', 'prepare-untuned']:
        if args.parent is None:
            parser.error('Additional control requires --parent')
        prepare_additional_control(args)
    else:
        evaluate(args.out.resolve())


if __name__ == '__main__':
    main()
