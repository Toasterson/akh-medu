#!/usr/bin/env python3
"""Executed, sourced one-action lessons; assistance is explicit, splits immutable."""
import argparse
import copy
import hashlib
import json
import random
import shutil
import subprocess
from pathlib import Path

from build_curriculum import FAMILIES, argv, digest, now
from command_templates import from_help, invoke, read, single_binding, store
from environment import Graph, world
from run_binding_kata import fixture_for, load, messages_and_tools, score, write

SEEDS = {'train': [211, 223, 227, 229, 233, 239, 241, 251],
         'validation': [307, 311], 'test': [401, 409]}
ARMS = ['prose_flat_template', 'structured_flat_template']


def distinct_world(family, seed):
    """Rotate the affected entity and give each split disjoint literal identifiers."""
    before, job = world(seed, ['acquire', 'cold_graph', 'revision'][(seed // 3) % 3])
    prefix = hashlib.sha256(f'{family}:{seed}'.encode()).hexdigest()[:8]
    jobs = {j: f'job-{prefix}-{i}' for i, j in enumerate(before['jobs'])}
    workers = {w: f'unit-{prefix}-{i}' for i, w in enumerate(before['workers'])}
    before['jobs'] = {jobs[j]: dict(r, worker=workers[r['worker']]) for j,r in before['jobs'].items()}
    before['workers'] = {workers[w]: r for w,r in before['workers'].items()}
    before.update(version=1, docs_available=True)
    return before, jobs[job]


def selected_subjects(family, seed, operation):
    rng = random.Random(f'{family}:{seed}:{operation}')
    keys = rng.sample(range(100, 999), 4)
    ops = ['catalog', 'status', 'resume', 'watch']
    rng.shuffle(ops)
    return dict(zip(ops, [f'command-{key}' for key in keys]))


def source_identity(identity, name, flag):
    # Case names contain examiner operation labels; do not expose them in receipts.
    opaque = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return f'lesson:{opaque}/{name}{flag}'


def goal_for(operation, job, variant):
    goals = {
        'status': [f'Read the current mode of the worker responsible for {job}. Do not change state.',
                   f'Find whether the worker handling {job} is paused or flowing. This is an observation only.'],
        'resume': [f'Restore flow for the worker responsible for {job}, without restart, deletion or changes to other workers.',
                   f'The observed worker for {job} is paused. Make it flowing with one bounded action; preserve archives and every other worker.'],
        'watch': [f'A bounded repair was just performed for {job}. Read a fresh job observation to verify its pending/archive counts. Do not mutate state.',
                  f'Verify the current pending and archive counts for {job} after the repair. Obtain a fresh timestamped observation; make no changes.']}
    return goals[operation][variant % 2]


def prepare_case(root, bridge, image, family, seed, operation):
    spec = FAMILIES[family]
    identity = f'{family}-{seed}-{operation}'
    dest = root/'cases'/identity
    dest.mkdir()
    contract = dict({k: copy.deepcopy(v) for k,v in spec.items() if k != 'split'}, version=1)
    before, job = distinct_world(family, seed)
    before['contract'] = contract
    fixture = fixture_for(image, contract, 'lesson')
    graph = None
    try:
        fixture.reset(before)
        graph = Graph(bridge, dest/'graph', dest/'engine.log')
        receipts, records = {}, {}
        subjects = selected_subjects(family, seed, operation)
        for name in [contract['observer'], contract['controller']]:
            for flag in ['--help', '--version']:
                receipt = fixture.pipeline([[name, flag]])
                assert receipt['success']
                receipts[name+flag] = dict(receipt, evidence_id=source_identity(identity,name,flag))
        for op, subject in subjects.items():
            executable = contract['observer' if op in ['catalog', 'watch'] else 'controller']
            help_receipt = receipts[executable+'--help']
            line = help_receipt['stdout'].splitlines().index(' '.join([executable, *contract['argv'][op]]))+1
            record = from_help(help_receipt, receipts[executable+'--version'], line,
                               {contract['observer'], contract['controller']})
            assert store(graph, subject, record)['success']
            records[subject] = record
        graph.close()
        graph = Graph(bridge, dest/'graph', dest/'engine-cold.log')
        assert records == {s:read(graph, s) for s in records}
        mapping = fixture.pipeline([argv(contract, 'watch', job=job)])
        assert mapping['success']
        worker = json.loads(mapping['stdout'])['worker']
        status = fixture.pipeline([argv(contract, 'status', worker=worker)])
        assert status['success']
        preparation = []
        if operation == 'watch':
            preparation.append(fixture.pipeline([argv(contract, 'resume', worker=worker)]))
            assert preparation[-1]['success']
        before = fixture.state()
        before['docs_available'] = False
        fixture.reset(before)
        argument = job if operation == 'watch' else worker
        subject = subjects[operation]
        expected_bindings = {'job' if operation == 'watch' else 'worker': argument}
        expected_argv = argv(contract, operation, **expected_bindings)
        execution = invoke(graph, fixture, subject, single_binding(read(graph, subject), argument))
        after = fixture.state()
        judged = score(before, after, job, operation, execution['response'], [execution['argv']], expected_argv)
        assert judged['kata_effect_pass']
        case = dict(id=identity, family=family, split=spec['split'], seed=seed,
            goal=goal_for(operation, job, seed // 3), before=before, target_job=job, operation=operation,
            observations=dict(job=mapping, worker=status, preparation=preparation),
            graph_records=records, source_receipts=receipts,
            examiner=dict(expected_argv=expected_argv, expected_subject=subject,
                          expected_bindings=expected_bindings, template=judged),
            assistance='Oracle-prepared receipts and cold KG; one candidate action, not autonomous acquisition.')
        write(dest/'case.json', case)
        write(dest/'oracle.json', dict(request=dict(tool='invoke_command', arguments=dict(subject=subject, argument=argument)),
            execution=execution, after=after, score=judged, created_at=now()))
        return dest/'case.json'
    finally:
        if graph:
            graph.close()
        fixture.close()


def row_for(case, arm):
    messages, tools = messages_and_tools(case, case['graph_records'], arm)
    expected = case['examiner']
    target = dict(name='invoke_command', arguments=dict(subject=expected['expected_subject'],
                    argument=next(iter(expected['expected_bindings'].values()))))
    messages.append(dict(role='assistant', content='<tool_call>\n'+json.dumps(target)+'\n</tool_call>'))
    return dict(messages=messages, tools=tools, episode_id=case['id'], decision_id=arm,
                family=case['family'], assisted_component=True)


def prepare(args):
    root = args.out.resolve()
    root.mkdir(exist_ok=False)
    (root/'cases').mkdir(); (root/'source').mkdir(); (root/'corpus').mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        if not p.name.startswith('test_'):
            shutil.copy2(p, root/'source'/p.name)
    bridge = root/'source'/'akh_episode_bridge'
    shutil.copy2(args.bridge, bridge)
    image = subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'], text=True).strip()
    cases, rows, evaluation = [], {'train': [], 'valid': [], 'test': []}, []
    identities = {'train': set(), 'valid': set(), 'test': set()}
    for family, spec in FAMILIES.items():
        split = 'valid' if spec['split'] == 'validation' else spec['split']
        for seed in SEEDS[spec['split']]:
            for operation in ['status', 'resume', 'watch']:
                path = prepare_case(root, bridge, image, family, seed, operation)
                case = load(path); cases.append(str(path))
                rows[split].extend(row_for(case, arm) for arm in ARMS)
                identities[split].update(case['before']['jobs']); identities[split].update(case['before']['workers'])
                if split != 'train' or seed == SEEDS['train'][0]:
                    evaluation.append(str(path))
                write(root/'status.json', dict(stage='building', updated_at=now(), cases=len(cases)))
    assert all(not identities[a] & identities[b] for a,b in [('train','valid'),('train','test'),('valid','test')])
    # Replay the complete original episode corpus, not randomly sampled fragments.
    rehearsal = args.rehearsal.resolve()
    old_hashes = {}
    for split, values in rows.items():
        old_file = rehearsal/(split+'.jsonl')
        old_hashes[str(old_file)] = digest(old_file)
        original = [json.loads(line) for line in old_file.read_text().splitlines()]
        combined = original+values
        (root/'corpus'/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in combined))
    manifest = dict(schema='akh-r1-binding-lessons-v1', created_at=now(), image=image,
        bridge=str(bridge), cases=cases, evaluation_cases=evaluation, arms=ARMS,
        authored_oracle_effects=len(cases), binding_rows={k:len(v) for k,v in rows.items()},
        rehearsal_files=old_hashes, split_identifiers_disjoint=True,
        assistance='Oracle-prepared evidence and KG; actions execute in isolated fixtures. No autonomous acquisition claim.',
        files={str(p):digest(p) for p in root.rglob('*') if p.is_file() and p.name!='status.json'},
        training_started=False, certification=False)
    write(root/'manifest.json', manifest)
    (root/'manifest.sha256').write_text(digest(root/'manifest.json')+'\n')
    write(root/'status.json', dict(stage='prepared', updated_at=now(), oracle_effects=len(cases), binding_rows=manifest['binding_rows']))
    print(json.dumps(dict(stage='prepared',cases=len(cases),binding_rows=manifest['binding_rows'],evaluation_cases=len(evaluation))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['out','bridge','rehearsal']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image',default='akh-medu/native-ocr-offline:recovery-20260910')
    prepare(p.parse_args())


if __name__=='__main__':
    main()
