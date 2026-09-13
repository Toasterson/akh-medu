#!/usr/bin/env python3
"""Execute authored teaching trajectories; never a learned-model/release score."""
import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from curriculum import SCHEMA, canonical_hash, compile_corpus
from environment import Fixture, Graph, score_effects, world
from flat_protocol import FLAT_TOOLS, parse_pipeline, project_fact


FAMILIES = {
    'named_flags_tsv': dict(split='train', observer='ravu', controller='tepo',
        delimiter='\t', columns=['job', 'worker', 'pending', 'archives'], argv=dict(
        catalog=['inventory', '--format', 'tsv'], watch=['observe', '--job', '{job}'],
        status=['status', '--worker', '{worker}'], resume=['resume', '--worker', '{worker}'])),
    'positional_csv': dict(split='train', observer='nali', controller='pemu',
        delimiter=',', columns=['pending', 'worker', 'archives', 'job'], argv=dict(
        catalog=['jobs', 'list'], watch=['job', 'show', '{job}'],
        status=['worker', 'inspect', '{worker}'], resume=['worker', 'continue', '{worker}'])),
    'scoped_flags_reordered': dict(split='validation', observer='sovu', controller='kira',
        delimiter='\t', columns=['worker', 'pending', 'job', 'archives'], argv=dict(
        catalog=['list', '--scope', 'jobs'], watch=['show', '--kind', 'job', '--key', '{job}'],
        status=['get', '--entity', 'worker', '--key', '{worker}'],
        resume=['update', '--entity', 'worker', '--key', '{worker}', '--mode', 'flowing'])),
    'subject_first_colon': dict(split='test', observer='fano', controller='zuri',
        delimiter=':', columns=['archives', 'job', 'worker', 'pending'], argv=dict(
        catalog=['jobs', 'export'], watch=['job', '{job}', 'read'],
        status=['worker', '{worker}', 'inspect'], resume=['worker', '{worker}', 'state', 'set', 'flowing'])),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def catalog(contract):
    tools = copy.deepcopy(FLAT_TOOLS)
    tools[0]['function']['description'] = tools[0]['function']['description'].replace(
        'miru, veln,', f'{contract["observer"]}, {contract["controller"]},')
    return tools


def argv(contract, operation, **bindings):
    name = contract['observer' if operation in {'catalog', 'watch'} else 'controller']
    return [name, *[word.format(**bindings) for word in contract['argv'][operation]]]


def contract_fact(contract, observer):
    name = contract['observer' if observer else 'controller']
    operations = ['catalog', 'watch'] if observer else ['status', 'resume']
    syntax = '; '.join(shlex.join([name, *contract['argv'][op]]) for op in operations)
    text = f'{name} {contract["version"]}: {syntax}.'
    if observer:
        text += f' Catalog delimiter {contract["delimiter"]!r}; columns {",".join(contract["columns"])}.'
    else:
        text += ' Resume sets flowing without restarting or deleting archives; idempotent.'
    return text


def check_expected_failure(response, expected, train_target, state_changed, tool, arguments):
    if train_target or response.get('success') or state_changed:
        raise RuntimeError('Injected failure must be excluded, unsuccessful and non-mutating')
    if expected == 'graph_lookup':
        wanted = f'symbol not found by label: "{arguments.get("symbol")}"'
        if tool != 'kg_query' or response.get('error') != wanted:
            raise RuntimeError('Expected an absent-symbol lookup, not an unrelated graph failure')
    elif response.get('exit_code') != expected:
        raise RuntimeError('Injected failure did not produce its required exit code')


def teach(family, spec, seed, phase, root, graph_path, bridge, image, acquisition_hash=None,
          repair_lessons=False):
    """The oracle knows the family; all target IDs and effect claims use receipts."""
    episode_id = f'{family}-{seed}-{phase}' + ('-repair' if repair_lessons else '')
    dest = root / episode_id
    dest.mkdir()
    before, job = world(seed, phase)
    contract = {k: copy.deepcopy(v) for k, v in spec.items() if k != 'split'}
    contract['version'] = 1
    before['contract'] = contract
    observer, controller = contract['observer'], contract['controller']
    portuguese = seed % 2 == 0
    goal = (f'Alert: {job} has pending backups. Diagnose and repair it in this lab; preserve '
            'archives and unrelated jobs. You are authorized to act. Learn the command '
            'contracts and retain reusable, sourced knowledge for another session.')
    if portuguese:
        goal = (f'Alerta: {job} tem cópias de segurança pendentes. Faça o diagnóstico e a reparação '
                'neste laboratório. Preserve os arquivos e os outros trabalhos. Tem autorização '
                'para agir. Aprenda os comandos e guarde conhecimento reutilizável com as fontes.')
    goal += f' Installed commands: {observer}, {controller}. This is a fresh conversation.'
    if phase == 'cold_graph':
        goal += ' Command documentation is offline. Version queries and other operations still work.'
    episode = dict(schema=SCHEMA, id=episode_id, family=family, split=spec['split'],
        provenance=dict(kind='authored_oracle_executed', learned_model=False,
            workflow='simulated_backup_progress', contract_sha256=canonical_hash(contract),
            seed=seed, phase=phase, language='pt' if portuguese else 'en',
            acquisition_episode_sha256=acquisition_hash, started_at=now()),
        initial_messages=[dict(role='system', content=(
            'You are an Akh handling an unfamiliar lab incident. Use the supplied tools, one call '
            'per response. Base operational claims on observations. Use report for progress and '
            'finish for the verified outcome and limits. Discover and store versioned command '
            'knowledge, retrieve it after restart, and check its applicability. Distinguish goals, '
            'hypotheses, observations and verified outcomes. Apply operator corrections to actions.')),
            dict(role='user', content=goal)], events=[], annotations=[], examiner={})
    graph = Graph(bridge, graph_path, dest / 'engine.log')
    fixture = None
    rows = []

    def event(role, content, **extra):
        identity = f'e{len(episode["events"]):03d}'
        episode['events'].append(dict(id=identity, role=role, content=content, **extra))
        return identity

    def act(tool, arguments, refs=(), kata=None, form=None, train_target=True, expected_failure=None):
        old = fixture.state()
        wire = dict(name=tool, arguments=arguments)
        action_id = event('assistant', '<tool_call>\n'+json.dumps(wire, ensure_ascii=False)+'\n</tool_call>',
                          evidence_refs=list(refs))
        if not train_target:
            episode['events'][-1]['train_target'] = False
        if kata or form:
            episode['annotations'].append(dict(at_event=action_id, kata=kata, form=form))
        request = dict(tool=tool, arguments=arguments)
        if tool == 'terminal':
            request = dict(tool=tool, arguments=dict(stages=parse_pipeline(arguments['command'])))
            response = fixture.pipeline(request['arguments']['stages'])
        elif tool == 'remember':
            response = graph.call('memory_write', project_fact(arguments))
        elif tool in {'kg_query', 'memory_recall'}:
            response = graph.call(tool, arguments)
        elif tool in {'report', 'finish'}:
            response = dict(success=True, result='Message recorded.')
        else:
            raise ValueError(tool)
        # Receipt identity is transport metadata, visible before any later citation.
        # It supplies no command, fact or examiner answer to the model.
        response = dict(response, evidence_id=f'episode:{episode_id}#e{len(episode["events"]):03d}')
        observation = event('tool', json.dumps(response, ensure_ascii=False), observes=action_id)
        rows.append(dict(request=request, wire_request=wire, response=response,
                         state_changed=fixture.state() != old, parsed=True, authored_oracle=True))
        if expected_failure is not None:
            check_expected_failure(response, expected_failure, train_target, fixture.state() != old,
                                   tool, arguments)
        elif not response.get('success'):
            raise RuntimeError(f'{episode_id} failed at {action_id}: {response}')
        return response, observation

    def terminal(stages, **kw):
        return act('terminal', dict(command=' | '.join(shlex.join(stage) for stage in stages)), **kw)

    def remember(subject, relation, value, reference):
        return act('remember', dict(subject=subject, relation=relation, value=value,
                                   source=f'episode:{episode_id}#{reference}'), refs=[reference])

    try:
        fixture = Fixture(image, f'akh-r1-teach-{os.getpid()}', commands=[observer, controller],
                          cli=Path(__file__).with_name('contract_cli.py'))
        fixture.reset(before)
        (dest / 'graph-before.json').write_text(json.dumps(graph.call('exam_export'), indent=2))
        act('report', dict(message=(
            f'I will inspect {job}, learn the installed command contracts, make a bounded repair '
            'if supported by observations, and verify progress. I will preserve archives and unrelated jobs.')),
            kata='Yes-And', form='Working Agreement')
        versions = {}
        for name in [observer, controller]:
            result, ref = terminal([[name, '--version']])
            versions[name] = (result['stdout'].strip(), ref)
        if repair_lessons:
            # Keep the repeated action/result as context, not a target to imitate.
            terminal([[controller, '--version']], train_target=False)
        if phase == 'acquire':
            for name, is_observer in [(observer, True), (controller, False)]:
                _, help_ref = terminal([[name, '--help']])
                if repair_lessons:
                    act('kg_query', dict(symbol=name, direction='from'), train_target=False,
                        expected_failure='graph_lookup')
                remember(name, 'versioned_contract', contract_fact(contract, is_observer), help_ref)
        # Cold trajectories really reopen the Engine. No old messages are copied.
        contract_refs = []
        for name, is_observer in [(observer, True), (controller, False)]:
            result, ref = act('kg_query', dict(symbol=name, direction='from'))
            expected = contract_fact(contract, is_observer)
            if expected not in result.get('result', '') or versions[name][0] not in expected:
                raise RuntimeError('Versioned contract absent from actual persistent KG receipt')
            contract_refs.append(ref)
        correction = ('Operator correction: I need progress visibility, not another permission question. '
                      'Do not restart any worker. Continue the authorized repair and tell me what you observe.')
        if portuguese:
            correction = ('Correção: preciso de saber o progresso, não de outra pergunta de autorização. '
                          'Não reinicie nenhum worker. Continue a reparação autorizada e diga o que observou.')
        correction_ref = event('user', correction)
        act('report', dict(message=(
            'Understood. I will continue within the existing authorization, use no restarts, '
            'and give updates tied to observed results. The cause is still unconfirmed.')),
            refs=[correction_ref], kata='Mirror', form='Response and Repair')
        if repair_lessons:
            act('report', dict(message='I will continue the authorized diagnosis and report progress.'),
                train_target=False)
        job_column = contract['columns'].index('job') + 1
        worker_column = contract['columns'].index('worker') + 1
        selection = [['awk', '-F', contract['delimiter'], f'${job_column}=="{job}" {{print ${worker_column}}}']]
        result, mapping_ref = terminal([argv(contract, 'catalog'), *selection], refs=contract_refs)
        worker = result['stdout'].strip()
        if not worker or len(worker.splitlines()) != 1:
            raise RuntimeError('Expected one worker from actual inventory selection')
        remember(job, 'observed_runs_on', worker, mapping_ref)
        status_refs = [mapping_ref, *contract_refs]
        if repair_lessons:
            result, remembered_ref = act('kg_query', dict(symbol=job, direction='from'))
            if worker not in result.get('result', ''):
                raise RuntimeError('Sourced job-to-worker mapping absent from graph')
            status_refs.append(remembered_ref)
            incomplete = [word for word in argv(contract, 'status', worker=worker) if word != worker]
            _, bad_ref = terminal([incomplete], refs=status_refs, train_target=False, expected_failure=64)
            status_refs.append(bad_ref)
        result, status_ref = terminal([argv(contract, 'status', worker=worker)], refs=status_refs,
                                      form='Response and Repair' if repair_lessons else None)
        status = json.loads(result['stdout'])
        if status['worker'] != worker or status['mode'] != 'paused':
            raise RuntimeError('Teaching world does not support the authored intervention')
        act('report', dict(message=(
            f'The inventory maps {job} to {worker}; its status is paused. A paused worker is '
            'one explanation for the pending backups; a downstream archive problem is another. '
            'Status alone cannot distinguish them. I will use the documented resume operation, '
            'then read a fresh job observation to test whether progress recovers.')),
            refs=[mapping_ref, status_ref], kata='Naming-the-Move', form='Competing Explanations')
        terminal([argv(contract, 'resume', worker=worker)], refs=[status_ref, *contract_refs],
                 form='Small Intervention')
        watch_argv = argv(contract, 'watch', job=job)
        result, verification_ref = terminal([watch_argv], refs=contract_refs)
        observed = json.loads(result['stdout'])
        if observed['job'] != job or observed['pending'] != 0:
            raise RuntimeError('Fresh verification did not confirm recovery')
        remember(job, 'observed_outcome', json.dumps(observed, sort_keys=True), verification_ref)
        act('finish', dict(message=(
            f'Fresh observation for {job}: pending={observed["pending"]}, archives={observed["archives"]}, '
            f'worker={observed["worker"]}. I resumed the observed paused worker without restarting it. '
            'I retained sourced versioned command contracts for later use. These are simulator archive '
            'counts; I have not checked real backup-file integrity. Use the stored observation command '
            'to check progress again, and rediscover the contract if the installed version changes.')),
            refs=[verification_ref], form='Return Ownership and Learning')
        episode['examiner'] = score_effects(before, fixture.state(), job, rows, verification_argv=watch_argv)
        episode['examiner'].update(learned_model=False, certification=False)
        return episode
    finally:
        episode['provenance']['finished_at'] = now()
        (dest / 'episode.json').write_text(json.dumps(episode, ensure_ascii=False, indent=2)+'\n')
        (dest / 'trace.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        (dest / 'examiner-before.json').write_text(json.dumps(before, indent=2))
        if fixture:
            (dest / 'examiner-after.json').write_text(json.dumps(fixture.state(), indent=2))
            fixture.close()
        try:
            (dest / 'graph-after.json').write_text(json.dumps(graph.call('exam_export'), indent=2))
        finally:
            graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--image', default='akh-medu/native-ocr-offline:recovery-20260910')
    parser.add_argument('--seeds', default='71,72,73')
    parser.add_argument('--frozen-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not args.frozen_worker:
        args.out.mkdir(parents=True, exist_ok=False)
        frozen = args.out / 'source'
        frozen.mkdir()
        sources = ['build_curriculum.py', 'curriculum.py', 'contract_cli.py', 'environment.py',
                   'flat_protocol.py', 'protocol.py']
        for name in sources:
            shutil.copy2(Path(__file__).with_name(name), frozen / name)
        shutil.copy2(args.bridge, frozen / 'akh_episode_bridge')
        image_id = subprocess.check_output(['docker', 'image', 'inspect', args.image,
                                            '--format', '{{.Id}}'], text=True).strip()
        manifest = dict(started_at=now(), image_id=image_id, authored_oracle=True, learned_model=False,
                        files={p.name: digest(p) for p in frozen.iterdir()}, seeds=args.seeds,
                        argv=sys.argv, release_certification=False)
        (args.out / 'execution-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result = subprocess.run([sys.executable, str(frozen / 'build_curriculum.py'), '--out', str(args.out),
                                 '--bridge', str(frozen / 'akh_episode_bridge'), '--image', image_id,
                                 '--seeds', args.seeds, '--frozen-worker'])
        manifest.update(finished_at=now(), exit_code=result.returncode,
                        source_hashes_unchanged=all(digest(frozen / p)==h for p,h in manifest['files'].items()))
        (args.out / 'execution-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        sys.exit(result.returncode)
    episodes = []
    for family, spec in FAMILIES.items():
        for seed in map(int, args.seeds.split(',')):
            graph_path = args.out / f'{family}-{seed}-acquire' / 'graph'
            acquired = teach(family, spec, seed, 'acquire', args.out, graph_path, args.bridge, args.image)
            episodes.append(acquired)
            cold_path = args.out / f'{family}-{seed}-cold-graph-state'
            shutil.copytree(graph_path, cold_path)
            episodes.append(teach(family, spec, seed, 'cold_graph', args.out, cold_path,
                                  args.bridge, args.image, canonical_hash(acquired)))
            print(json.dumps(dict(family=family, seed=seed, effects=[e['examiner']['effect_success']
                  for e in episodes[-2:]])), flush=True)
    manifest = compile_corpus(episodes, {k:v['split'] for k,v in FAMILIES.items()},
                              {k:catalog(v) for k,v in FAMILIES.items()}, args.out / 'corpus')
    manifest.update(workflow_families=1, command_contract_families=len(FAMILIES),
                    teacher='authored oracle with known contracts and actual execution receipts',
                    limitations=['Not learned model evidence', 'Not a sealed release holdout',
                                 'Archive counts, not actual backup-file integrity',
                                 'Kata annotations are curriculum labels, not quality judgments'])
    (args.out / 'corpus' / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['counts']), flush=True)


if __name__ == '__main__':
    main()
