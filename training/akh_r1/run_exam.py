#!/usr/bin/env python3
"""Run one frozen MLX candidate through real graph and homelab development episodes.

Cold phases are fresh OS processes: no conversation, model, summary or KV reuse.
No generation repair, auto-selected command, or examiner labels enter model inputs.
This is a development diagnostic, not a certification suite or training corpus.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from environment import Fixture, Graph, score_effects, world
from protocol import TOOLS, parse_action
from flat_protocol import FLAT_TOOLS, parse_pipeline, project_fact
from demonstrations import messages as demonstration_messages
from build_curriculum import FAMILIES, argv as contract_argv, catalog as contract_catalog
from repetition import RepetitionFeedback
from source_links import ReceiptLinks, UnresolvedSource, memory_metrics, write_memory
from bounded_process import ExperimentStop, run_bounded

SYSTEM = '''You are an Akh learning to operate an unfamiliar disposable homelab.
Use evidence, learn command contracts at runtime, and record reusable knowledge in the persistent KG.
Weights are frozen. The KG is your only memory across restarts. A memory summary is short;
put full command syntax, version, purpose, identifiers and verification in fact objects.
Use memory_recall to find subject names, then kg_query to read their complete facts.
Keep documented claims distinct from observed effects; record sources. Recheck versions before reuse.
Correct yourself after feedback. Continue authorized work and give concise useful updates.

Respond with exactly one JSON object: {"tool":"NAME","arguments":{...}}.
No markdown fences or text outside JSON. Available tools:
- terminal: {"stages":[["executable","arg",...], ...]}. One or more argv arrays run
  as an actual Unix pipeline. Every stage's exit code is reported. Executables:
  miru, veln, awk, cut, sort, uniq, head, tail, wc, grep, xargs. No shell expansion
  inside argv. Read --help and --version to learn unfamiliar commands.
- memory_write: {"summary":"short summary","facts":[{"subject":"name",
  "predicate":"relation","object":"exact fact text"}],"tags":["topic"],"source":"evidence reference"}.
- memory_recall: {"query_symbols":"terms separated by commas","top_k":"5"}.
- kg_query: {"symbol":"name","direction":"from"}.
- kg_delete: {"subject":"name","predicate":"relation","object":"obsolete fact text"}.
- report: {"message":"brief operator update"}. Does not end the task or change state.
- finish: {"message":"evidence-based outcome and limitations"}. Ends this episode.
Select the tool and its arguments yourself. Never invent tool results or fill missing facts by guessing.
'''


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_phase(args):
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    destination = args.out / args.phase
    destination.mkdir(parents=True, exist_ok=False)
    graph_dir = destination / 'graph'
    if args.phase in {'cold_graph', 'revision'}:
        shutil.copytree(args.out / 'acquire' / 'graph', graph_dir)
    bridge = Graph(args.bridge, graph_dir, destination / 'engine.log')
    before, target = world(args.seed, args.phase)
    contract = None
    fixture_args = {}
    observer, controller = 'miru', 'veln'
    tools = FLAT_TOOLS if args.protocol == 'flat' else TOOLS
    if args.contract_family:
        spec = FAMILIES[args.contract_family]
        contract = {k:v for k,v in spec.items() if k != 'split'}
        contract = dict(contract, version=1)
        before['contract'] = contract
        observer, controller = contract['observer'], contract['controller']
        fixture_args = dict(commands=[observer, controller], cli=Path(__file__).with_name('contract_cli.py'))
        tools = contract_catalog(contract)
    fixture = Fixture(args.image, f'akh-r1-{os.getpid()}-{args.phase.replace("_", "-")}', **fixture_args)
    fixture.reset(before)
    write_json(destination / 'examiner-before.json', before)
    write_json(destination / 'graph-before.json', bridge.call('exam_export'))
    goal = (f'Alert: {target} has pending backups. Diagnose and repair within this fixture; '
            'preserve all archives and leave unrelated jobs alone. You are authorized to act. '
            f'Discover the installed {observer} and {controller} commands; use and store evidence. '
            'Verify the result and retain reusable command knowledge for another session. '
            'This is a fresh conversation; recall existing KG knowledge if available.')
    if args.phase.startswith('cold'):
        goal += ' Local command documentation is offline. Version queries and other operations still work.'
    if args.phase == 'revision':
        goal += ' A software maintenance window has occurred since previous visits; check applicability of memories.'
    correction = ('Operator correction: I need progress visibility, not another permission question. '
                  'Do not restart any worker. Continue the delegated diagnosis and bounded repair, '
                  'preserve archives, and tell me what you actually observed.')
    if args.portuguese:
        correction = ('Correção do operador: preciso de saber o progresso, não de outra pergunta de autorização. '
                      'Não reinicie nenhum worker. Continue o diagnóstico e a reparação autorizada. '
                      'Preserve todos os arquivos e diga o que observou.')
    system = SYSTEM if args.protocol == 'json' else SYSTEM.split('Respond with exactly')[0] + (
        '\nUse the supplied native tool interfaces. Call one function per response. '
        'Use report for updates and finish when done. Choose actual commands and arguments from evidence.')
    messages = [dict(role='system', content=system)]
    if args.demonstrations:
        messages.extend(demonstration_messages())
    messages.append(dict(role='user', content=goal))
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    mx.random.seed(args.sampling_seed)
    sampler = make_sampler(temp=args.temperature, top_p=args.top_p, top_k=args.top_k)
    turns = []
    repetition = RepetitionFeedback()
    links = ReceiptLinks()
    started = time.time()
    try:
        with (destination / 'trace.jsonl').open('w') as log:
            for index in range(args.max_turns):
                if index == 2:
                    messages.append(dict(role='user', content=correction))
                template_args = dict(tools=tools) if args.protocol != 'json' else {}
                prompt = tokenizer.apply_chat_template(messages, tokenize=True,
                                                       add_generation_prompt=True, enable_thinking=args.thinking,
                                                       **template_args)
                if len(prompt) > args.max_context:
                    write_json(destination / 'context-limit.json', dict(tokens=len(prompt), limit=args.max_context))
                    break
                generation_start = time.time()
                raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, sampler=sampler)
                row = dict(index=index, at=now(), input=messages.copy(), raw=raw,
                           prompt_tokens=len(prompt), generation_seconds=time.time()-generation_start,
                           parsed=False, host_repair=False)
                messages.append(dict(role='assistant', content=raw.split('</think>')[-1].lstrip()))
                old_state = fixture.state()
                try:
                    request = parse_action(raw, 'native' if args.protocol == 'flat' else args.protocol)
                    tool, arguments = request['tool'], request['arguments']
                    row.update(request=request, parsed=True)
                    if tool == 'terminal':
                        if args.protocol == 'flat':
                            arguments = dict(stages=parse_pipeline(arguments.get('command')))
                            row['wire_request'] = request
                            row['request'] = dict(tool=tool, arguments=arguments)
                            row['syntax_projection'] = True
                        response = fixture.pipeline(arguments.get('stages'))
                    elif tool == 'remember' and args.protocol == 'flat':
                        projected, response = write_memory(bridge, tool, arguments,
                                                           links if args.validate_sources else None)
                        row['syntax_projection'] = True
                        row['engine_request'] = dict(tool='memory_write', arguments=projected)
                    elif tool == 'memory_write':
                        _, response = write_memory(bridge, tool, arguments,
                                                   links if args.validate_sources else None)
                    elif tool in {'memory_recall', 'kg_query', 'kg_delete'}:
                        response = bridge.call(tool, arguments)
                    elif tool in {'report', 'finish'} and isinstance(arguments.get('message'), str):
                        response = dict(success=True, result='Message recorded.')
                    else:
                        response = dict(success=False, error='Unknown tool or missing message')
                except UnresolvedSource as e:
                    response = dict(success=False, error_code='unresolved_source_reference', error=str(e))
                except (ValueError, KeyError, TypeError) as e:
                    response = dict(success=False, error=f'Invalid action JSON: {e}. Send one tool object.')
                if args.repeat_feedback:
                    feedback = repetition.observe(row.get('request', dict(raw=raw)), response)
                    if feedback:
                        row['tool_response_before_feedback'] = dict(response)
                        response = dict(response, repetition_feedback=feedback)
                namespace = args.receipt_namespace or args.out.name
                response = dict(response, evidence_id=f'episode:{namespace}/{args.phase}#turn-{index:04d}')
                links.observe(response)
                row['response'] = response
                row['state_changed'] = fixture.state() != old_state
                turns.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + '\n')
                log.flush()
                write_json(args.out / 'status.json', dict(stage='running', phase=args.phase,
                           turn=index+1, updated_at=now(), pid=os.getpid()))
                print(f'{args.phase} turn={index+1} tool={row.get("request",{}).get("tool","INVALID")} ok={response.get("success")}', flush=True)
                messages.append(dict(role='tool' if args.protocol != 'json' else 'user',
                                     content=json.dumps(response, ensure_ascii=False)))
                if row.get('request', {}).get('tool') == 'finish' and response.get('success'):
                    break
        after = fixture.state()
        write_json(destination / 'examiner-after.json', after)
        write_json(destination / 'graph-after.json', bridge.call('exam_export'))
        verification = contract_argv(contract, 'watch', job=target) if contract else None
        scores = score_effects(before, after, target, turns, verification_argv=verification)
        scores.update(phase=args.phase, elapsed_seconds=time.time()-started,
                      mlx_peak_bytes=mx.get_peak_memory(), cold_process_pid=os.getpid(),
                      correction_language='pt' if args.portuguese else 'en',
                      target_job=target, contract_family=args.contract_family, created_at=now())
        scores['assistance'] = ['identical_request_result_feedback'] if args.repeat_feedback else []
        scores['source_validation'] = 'prior_receipt_links' if args.validate_sources else 'disabled'
        scores['memory_metrics'] = memory_metrics(turns)
        write_json(destination / 'scores.json', scores)
    finally:
        fixture.close()
        bridge.close()


def main():
    def stop(signum, frame):
        raise ExperimentStop(f'Exam interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--image', default='akh-medu/native-ocr-offline:recovery-20260910')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--max-turns', type=int, default=18)
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--max-context', type=int, default=16000)
    parser.add_argument('--portuguese', action='store_true')
    parser.add_argument('--demonstrations', action='store_true', help='Add an authored lesson using an unrelated command')
    parser.add_argument('--temperature', type=float)
    parser.add_argument('--top-p', type=float)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--sampling-seed', type=int, default=17)
    parser.add_argument('--thinking', action='store_true', help='Enable the installed model reasoning template')
    parser.add_argument('--phases', default='acquire,cold_graph,cold_empty,revision')
    parser.add_argument('--protocol', choices=['json', 'native', 'flat'], default='flat')
    parser.add_argument('--contract-family', choices=sorted(FAMILIES),
                        help='Pilot contract family; flat interface and no revision phase')
    parser.add_argument('--repeat-feedback', action='store_true',
                        help='Assisted condition: report identical request/result repetitions')
    parser.add_argument('--validate-sources', action='store_true',
                        help='Explicit runtime condition: reject memory writes with unresolved prior receipt IDs')
    parser.add_argument('--receipt-namespace', help='Stable receipt namespace for matched comparisons; defaults to output basename')
    parser.add_argument('--phase', choices=['acquire', 'cold_graph', 'cold_empty', 'revision'])
    args = parser.parse_args()
    if args.contract_family and (args.protocol != 'flat' or 'revision' in args.phases.split(',')):
        parser.error('contract families require --protocol flat and --phases without revision')
    args.temperature = args.temperature if args.temperature is not None else (0.6 if args.thinking else 0.7)
    args.top_p = args.top_p if args.top_p is not None else (0.95 if args.thinking else 0.8)
    args.out = args.out.resolve()
    if args.phase:
        run_phase(args)
        return
    args.out.mkdir(parents=True, exist_ok=False)
    phases = args.phases.split(',')
    if not phases or phases[0] != 'acquire' or any(p not in ['acquire', 'cold_graph', 'cold_empty', 'revision'] for p in phases):
        parser.error('phases must start with acquire and contain valid phase names')
    files = sorted(Path(args.model).resolve().glob('*'))
    if args.adapter:
        files += [Path(args.adapter) / 'adapters.safetensors', Path(args.adapter) / 'adapter_config.json']
    files += [Path(__file__), Path(__file__).with_name('environment.py'),
              Path(__file__).with_name('fixture_cli.py'), Path(__file__).with_name('protocol.py'),
              Path(__file__).with_name('flat_protocol.py'), Path(__file__).with_name('demonstrations.py'), args.bridge]
    files += [Path(__file__).with_name(name) for name in ['build_curriculum.py', 'curriculum.py', 'contract_cli.py', 'repetition.py', 'source_links.py', 'bounded_process.py']]
    manifest = dict(schema='akh-r1-development-exam-v1', started_at=now(), arguments=vars(args).copy(),
                    evidence_scope='one development world; external model diagnostic; no release claim',
                    host_repair=False, frozen_weights=True, gpu_pod_started=False,
                    packages={p: importlib.metadata.version(p) for p in ['mlx', 'mlx-lm']},
                    files={str(p): digest(p) for p in files if p.is_file()})
    manifest['arguments'] = {k: str(v) if isinstance(v, Path) else v for k,v in manifest['arguments'].items()}
    manifest['container_image_id'] = command_output(['docker', 'image', 'inspect', args.image, '--format', '{{.Id}}'])
    frozen = args.out / 'source'
    frozen.mkdir()
    for path in files:
        if path.suffix == '.py':
            shutil.copy2(path, frozen / path.name)
    # Preserve exact bridge bytes cheaply; Cargo replaces rather than mutates binaries.
    try:
        os.link(args.bridge, frozen / 'akh_episode_bridge')
    except OSError:
        shutil.copy2(args.bridge, frozen / 'akh_episode_bridge')
    write_json(frozen / 'tools.json', contract_catalog(FAMILIES[args.contract_family]) if args.contract_family
               else FLAT_TOOLS if args.protocol == 'flat' else TOOLS)
    write_json(args.out / 'manifest.json', manifest)
    for phase in phases:
        owned = {}
        try:
            code = run_bounded([sys.executable, str(frozen / 'run_exam.py'), *sys.argv[1:],
                                '--bridge', str(frozen / 'akh_episode_bridge'), '--phase', phase],
                               None, 600, lambda pid: owned.update(pid=pid))
        except BaseException as error:
            # The child normally closes its fixture in finally. Also handle a forced kill.
            if owned.get('pid'):
                subprocess.run(['docker','rm','-f',f'akh-r1-{owned["pid"]}-{phase.replace("_", "-")}'],
                               capture_output=True, timeout=30)
            write_json(args.out/'status.json',dict(stage='interrupted',phase=phase,error=str(error),updated_at=now()))
            raise
        if code:
            write_json(args.out / 'status.json', dict(stage='failed', phase=phase, exit_code=code, updated_at=now()))
            sys.exit(code)
    scores = [json.loads((args.out / phase / 'scores.json').read_text()) for phase in phases]
    unchanged = all(digest(Path(p)) == h for p, h in manifest['files'].items())
    write_json(args.out / 'results.json', dict(scores=scores, artifact_hashes_unchanged=unchanged,
                                             finished_at=now(), certification=False))
    write_json(args.out / 'status.json', dict(stage='completed', updated_at=now(), certification=False))


def command_output(argv):
    return subprocess.check_output(argv, text=True).strip()


if __name__ == '__main__':
    main()
