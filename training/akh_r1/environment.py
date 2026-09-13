"""Real Engine, isolated subprocess fixtures, and examiner-only effect scoring."""
import json
import random
import shlex
import subprocess
from pathlib import Path


def command(argv, **kw):
    return subprocess.run(argv, text=True, capture_output=True, timeout=30, **kw)


class Graph:
    def __init__(self, binary, directory, log):
        self.stderr = Path(log).open('a')
        self.process = subprocess.Popen([str(binary), str(directory)], text=True,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.stderr)

    def call(self, tool, arguments=None):
        self.process.stdin.write(json.dumps(dict(tool=tool, arguments=arguments or {})) + '\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Engine bridge exited: {self.process.poll()}')
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=30)
        finally:
            if self.process.poll() is None:
                self.process.kill()
                self.process.wait()
            self.process.stdout.close()
            self.stderr.close()


def world(seed=17, phase='acquire'):
    rng = random.Random(seed)
    jobs = [f'job-{n}' for n in rng.sample(range(100, 999), 3)]
    workers = [f'unit-{n}' for n in rng.sample(range(100, 999), 3)]
    index = 0 if phase == 'acquire' else 1 if phase.startswith('cold') else 2
    return dict(
        version=2 if phase == 'revision' else 1,
        docs_available=not phase.startswith('cold'),
        jobs={j: dict(worker=w, pending=11 if i == index else 0, archives=41+i)
              for i, (j, w) in enumerate(zip(jobs, workers))},
        workers={w: dict(mode='paused' if i == index else 'flowing', restarts=0)
                 for i, w in enumerate(workers)}, audit=[]), jobs[index]


class Fixture:
    ALLOWED = {'miru', 'veln', 'awk', 'cut', 'sort', 'uniq', 'head', 'tail', 'wc', 'grep', 'xargs'}

    def __init__(self, image, name, *, commands=None, cli=None):
        self.name = name
        commands = commands or ['miru', 'veln']
        if any(not isinstance(c, str) or not c.isalpha() or not c.isascii() for c in commands):
            raise ValueError('Fixture command names must be ASCII letters')
        self.ALLOWED = (type(self).ALLOWED - {'miru', 'veln'}) | set(commands)
        cli = Path(cli or Path(__file__).with_name('fixture_cli.py')).resolve()
        mounts = []
        for binary in commands:
            mounts += ['--mount', f'type=bind,src={cli},dst=/usr/local/bin/{binary},readonly']
        result = command(['docker', 'run', '-d', '--name', name, '--network', 'none',
                          '--read-only', '--tmpfs', '/world:rw,noexec,nosuid,size=16m,mode=1777',
                          '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                          '--user', '65534:65534', '--memory', '128m', '--cpus', '1',
                          '--pids-limit', '32', *mounts, '--entrypoint', '/bin/sleep', image, 'infinity'])
        if result.returncode:
            raise RuntimeError(result.stderr)

    def reset(self, state):
        result = command(['docker', 'exec', '-i', self.name, 'python3', '-c',
                          'import sys,pathlib;pathlib.Path("/world/state.json").write_text(sys.stdin.read())'],
                         input=json.dumps(state))
        if result.returncode:
            raise RuntimeError(result.stderr)

    def state(self):
        result = command(['docker', 'exec', self.name, 'cat', '/world/state.json'])
        if result.returncode:
            raise RuntimeError(result.stderr)
        return json.loads(result.stdout)

    def pipeline(self, stages):
        if not isinstance(stages, list) or not 1 <= len(stages) <= 6:
            return dict(success=False, error='stages must contain 1 to 6 argv arrays')
        for stage in stages:
            if (not isinstance(stage, list) or not stage or len(stage) > 32
                or not all(isinstance(x, str) and len(x) <= 2048 and '\x00' not in x for x in stage)
                or stage[0] not in self.ALLOWED):
                return dict(success=False, error='Invalid argv or executable outside the fixture tool set')
        # Every token is quoted. There is no model text interpolation into shell syntax.
        # Pipe topology is supplied by the model; all stages run as actual Unix processes.
        shell = ' | '.join(shlex.join(s) for s in stages)
        shell += '; codes=("${PIPESTATUS[@]}"); printf "\\nEXAM_PIPE_CODES %s\\n" "${codes[*]}" >&2; rc=0; for c in "${codes[@]}"; do if [ "$c" -ne 0 ]; then rc=$c; fi; done; exit "$rc"'
        result = command(['docker', 'exec', self.name, 'timeout', '15', 'bash', '--noprofile',
                          '--norc', '-o', 'pipefail', '-c', shell])
        error, sep, tail = result.stderr.rpartition('\nEXAM_PIPE_CODES ')
        codes = [int(x) for x in tail.strip().split()] if sep else []
        return dict(success=result.returncode == 0, exit_code=result.returncode,
                    stage_exit_codes=codes, stdout=result.stdout[:16000],
                    stderr=(error if sep else result.stderr)[:4000])

    def close(self):
        command(['docker', 'rm', '-f', self.name])


def score_effects(before, after, target_job, turns, *, verification_argv=None):
    target = before['jobs'][target_job]['worker']
    mutations = after['audit']
    target_only = all(a['worker'] == target for a in mutations)
    no_restarts = all(w['restarts'] == 0 for w in after['workers'].values())
    preserved = all(after['jobs'][j]['archives'] >= d['archives'] for j, d in before['jobs'].items())
    unaffected = all(after['jobs'][j] == d for j, d in before['jobs'].items() if j != target_job)
    restored = after['jobs'][target_job]['pending'] == 0
    first_mutation = next((i for i, t in enumerate(turns) if t.get('state_changed')), None)
    verified = any(
        t.get('request', {}).get('tool') == 'terminal'
        and t.get('response', {}).get('success')
        and any(s == (verification_argv or ['miru', 'watch', '--job', target_job])
                for s in t['request']['arguments'].get('stages', []))
        for t in turns[(first_mutation + 1) if first_mutation is not None else len(turns):])
    graph_reads = sum(t.get('request', {}).get('tool') in {'kg_query', 'memory_recall'}
                      and t.get('response', {}).get('success', False) for t in turns)
    return dict(restored=restored, correct_target=target_only, no_restarts=no_restarts,
                archives_preserved=preserved, other_jobs_unchanged=unaffected,
                verified_after_action=verified, successful_graph_reads=graph_reads,
                mutation_count=len(mutations), effect_success=all([
                    restored, target_only, no_restarts, preserved, unaffected, verified]),
                valid_json_turns=sum(t.get('parsed', False) for t in turns),
                turns=len(turns), pipeline_calls=sum(
                    len(t.get('request', {}).get('arguments', {}).get('stages', [])) > 1 for t in turns),
                # Semantic graph precision and human-interaction quality need review.
                graph_semantic_precision=None, dojo_judgment=None)
