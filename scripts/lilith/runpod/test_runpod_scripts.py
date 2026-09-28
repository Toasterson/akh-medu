"""Fail-closed behaviour of the pod shell scripts and payload provenance, without a GPU."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
sys.path.insert(0, str(HERE))

import verify_payload  # noqa: E402

GIT_ENV = {'GIT_AUTHOR_NAME': 'test', 'GIT_AUTHOR_EMAIL': 'test@example.invalid',
           'GIT_COMMITTER_NAME': 'test', 'GIT_COMMITTER_EMAIL': 'test@example.invalid',
           'GIT_CONFIG_GLOBAL': os.devnull, 'GIT_CONFIG_NOSYSTEM': '1'}


def clean_environ(**extra):
    environ = {k: v for k, v in os.environ.items()
               if not k.startswith('LILITH_') and k not in ('HF_TOKEN', 'GIT_DIR', 'GIT_WORK_TREE')}
    environ.update(extra)
    return environ


def run(script, *args, root=None, allow_non_workspace=True, script_dir=HERE, **extra):
    environ = clean_environ(**extra)
    if root is not None:
        environ['LILITH_ROOT'] = str(root)
    if allow_non_workspace:
        environ['LILITH_ALLOW_NON_WORKSPACE'] = '1'
    return subprocess.run(['bash', str(script_dir / script), *args], capture_output=True, text=True,
                          env=environ, timeout=120)


def git(cwd, *args):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=clean_environ(**GIT_ENV)).stdout.strip()


def make_origin_checkout(base: Path) -> Path:
    """A throwaway origin with every payload file (plus make_payload.sh) committed on main."""
    origin, work = base / 'origin.git', base / 'work'
    git(base, 'init', '-q', '--bare', '-b', 'main', str(origin))
    git(base, 'init', '-q', '-b', 'main', str(work))
    for name in verify_payload.PAYLOAD_FILES + ('scripts/lilith/runpod/make_payload.sh',):
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPOSITORY / name, target)
    git(work, 'add', '-A')
    git(work, 'commit', '-q', '-m', 'payload sources')
    git(work, 'remote', 'add', 'origin', str(origin))
    git(work, 'push', '-q', 'origin', 'main')
    return work


def make_payload(work: Path, out: Path):
    return subprocess.run(['bash', str(work / 'scripts/lilith/runpod/make_payload.sh'), str(out)],
                          capture_output=True, text=True, env=clean_environ(**GIT_ENV), timeout=120)


class ScriptTests(unittest.TestCase):
    def test_scripts_parse(self):
        for script in ('bootstrap.sh', 'run_smoke.sh', 'make_payload.sh'):
            result = subprocess.run(['bash', '-n', str(HERE / script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_smoke_refuses_an_existing_run_directory_without_touching_it(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'runs/c6-smoke-a'
            existing.mkdir(parents=True)
            (existing / 'evidence.json').write_text('{}')
            result = run('run_smoke.sh', 'c6-smoke-a', root=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lilith::smoke::exists', result.stderr)
            self.assertEqual(sorted(p.name for p in existing.iterdir()), ['evidence.json'])

    def test_smoke_refuses_invalid_run_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            for run_id in ('', '../escape', 'a/b', '-flag'):
                result = run('run_smoke.sh', run_id, root=directory)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('lilith::smoke::usage', result.stderr)
            self.assertFalse((Path(directory) / 'runs').exists())

    def test_scripts_refuse_roots_outside_the_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('bootstrap.sh', '--check', root=directory, allow_non_workspace=False)
            self.assertIn('lilith::bootstrap::root', result.stderr)
            result = run('run_smoke.sh', 'c6-x', root=directory, allow_non_workspace=False)
            self.assertIn('lilith::smoke::root', result.stderr)

    def test_bootstrap_check_fails_closed_without_an_env(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('bootstrap.sh', '--check', root=directory, LILITH_ALLOW_NO_CUDA='1')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lilith::bootstrap::env_missing', result.stderr)
            self.assertIn('readiness:', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])   # --check creates nothing

    @unittest.skipIf(shutil.which('nvidia-smi'), 'host has a GPU')
    def test_bootstrap_refuses_a_host_without_a_gpu_before_anything_else(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run('bootstrap.sh', root=directory)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('lilith::bootstrap::no_gpu', result.stderr)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_bootstrap_rejects_unknown_arguments(self):
        result = run('bootstrap.sh', '--resume', root='/workspace/lilith')
        self.assertIn('lilith::bootstrap::usage', result.stderr)

    def test_uv_is_pinned_by_sha256_and_never_piped_to_a_shell(self):
        text = (HERE / 'bootstrap.sh').read_text()
        self.assertIn('UV_SHA256="2e2b37e9811e17675a9e70bed5e1a58fc8c0388be63d751d72cc735188c149ff"', text)
        self.assertNotRegex(text, r'curl[^\n]*\|\s*(env[^\n]*)?sh\b')
        self.assertNotIn('astral.sh/uv', text)

    def test_token_never_appears_in_scripts_as_a_literal_write(self):
        for script in ('bootstrap.sh', 'run_smoke.sh', 'make_payload.sh'):
            text = (HERE / script).read_text()
            self.assertNotIn('set -x', text)
            self.assertNotIn('huggingface-cli login', text)
            self.assertNotIn('hf auth login', text)
            self.assertNotRegex(text, r'HF_TOKEN[^\n]*>')


class VerifyPayloadTests(unittest.TestCase):
    sha = 'a' * 40

    def payload(self, directory: Path) -> Path:
        for name in verify_payload.PAYLOAD_FILES:
            (directory / name).parent.mkdir(parents=True, exist_ok=True)
            (directory / name).write_text(name)
        manifest = directory / 'source-manifest.json'
        manifest.write_text(json.dumps(verify_payload.build(directory, self.sha)))
        return manifest

    def test_matching_payload_verifies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = verify_payload.verify(root, self.payload(root), self.sha, False)
            self.assertEqual(record['status'], 'verified', record['problems'])

    def test_every_failure_mode_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.payload(root)
            for expected in ('', 'not-a-sha', 'b' * 40):
                self.assertEqual(verify_payload.verify(root, manifest, expected, False)['status'], 'failed')
            self.assertEqual(verify_payload.verify(root, root / 'missing.json', self.sha, False)['status'], 'failed')
            (root / verify_payload.PAYLOAD_FILES[0]).write_text('tampered')
            record = verify_payload.verify(root, manifest, self.sha, False)
            self.assertEqual(record['status'], 'failed')
            self.assertTrue(any('differs' in problem for problem in record['problems']))
            (root / verify_payload.PAYLOAD_FILES[1]).unlink()
            self.assertEqual(verify_payload.verify(root, manifest, self.sha, False)['status'], 'failed')

    def test_allow_unmerged_is_recorded_not_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = verify_payload.verify(root, root / 'missing.json', '', True)
            self.assertEqual(record['status'], 'unverified_allowed')
            self.assertTrue(record['allow_unmerged'])
            self.assertTrue(record['problems'])

    def test_manifest_file_set_must_equal_the_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.payload(root)
            data = json.loads(manifest.read_text())
            data['files'].pop(verify_payload.PAYLOAD_FILES[-1])
            manifest.write_text(json.dumps(data))
            self.assertEqual(verify_payload.verify(root, manifest, self.sha, False)['status'], 'failed')

    def test_payload_names_every_module_the_smoke_imports(self):
        required = {'lilith_training_engine.py', 'lilith_training_cache_control.py', 'lilith_rl.py',
                    'lilith_cache_path_diagnostic.py', 'lilith_numerics.py',
                    'lilith_adapter_mechanics.py', 'lilith_lora_trainer.py', 'reference_microrun.py',
                    'verify_payload.py', 'bootstrap.sh', 'run_smoke.sh', 'requirements-cuda.txt'}
        self.assertTrue(required <= {Path(name).name for name in verify_payload.PAYLOAD_FILES})


@unittest.skipUnless(shutil.which('git'), 'git not installed')
class MakePayloadTests(unittest.TestCase):
    def test_payload_from_main_verifies_on_the_pod_and_the_smoke_uses_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            work = make_origin_checkout(base)
            main_sha = git(work, 'rev-parse', 'HEAD')
            result = make_payload(work, base / 'out')
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary['main_sha'], main_sha)
            with tarfile.open(summary['tarball']) as archive:
                names = {member.name.removeprefix('./') for member in archive.getmembers() if member.isfile()}
                self.assertEqual(names, set(verify_payload.PAYLOAD_FILES) | {'source-manifest.json'})
                pod = base / 'pod-src'
                archive.extractall(pod, filter='data')
            self.assertEqual(hashlib.sha256(Path(summary['manifest']).read_bytes()).hexdigest(),
                             summary['manifest_sha256'])
            self.assertEqual(make_payload(work, base / 'out').returncode, 1)   # write-once

            # The smoke verifies the shipped payload; later steps fail here (no GPU/env).
            pod_scripts = pod / 'scripts/lilith/runpod'
            root = base / 'lilith'
            ok = run('run_smoke.sh', 'prov-ok', root=root, script_dir=pod_scripts,
                     LILITH_EXPECTED_MAIN_SHA=main_sha)
            self.assertNotEqual(ok.returncode, 0)
            result = json.loads((root / 'runs/prov-ok/smoke-result.json').read_text())
            self.assertEqual(result['steps'][0]['step'], 'provenance')
            self.assertEqual(result['steps'][0]['status'], 'ok')
            self.assertEqual(result['provenance']['status'], 'verified')
            self.assert_sums_are_last(root / 'runs/prov-ok')

            empty = run('run_smoke.sh', 'prov-empty', root=root, script_dir=pod_scripts)
            self.assertNotEqual(empty.returncode, 0)
            result = json.loads((root / 'runs/prov-empty/smoke-result.json').read_text())
            self.assertEqual([s['status'] for s in result['steps']], ['failed'])
            self.assertEqual(result['provenance']['status'], 'failed')

            (pod_scripts / 'lilith_lora_trainer.py').write_text('# tampered\n')
            tampered = run('run_smoke.sh', 'prov-tampered', root=root, script_dir=pod_scripts,
                           LILITH_EXPECTED_MAIN_SHA=main_sha)
            result = json.loads((root / 'runs/prov-tampered/smoke-result.json').read_text())
            self.assertNotEqual(tampered.returncode, 0)
            self.assertEqual([s['status'] for s in result['steps']], ['failed'])

            allowed = run('run_smoke.sh', 'prov-allowed', root=root, script_dir=pod_scripts,
                          LILITH_EXPECTED_MAIN_SHA=main_sha, LILITH_ALLOW_UNMERGED='1')
            self.assertNotEqual(allowed.returncode, 0)   # fails later, at bootstrap-check
            result = json.loads((root / 'runs/prov-allowed/smoke-result.json').read_text())
            self.assertEqual(result['steps'][0]['status'], 'ok')
            self.assertEqual(result['provenance']['status'], 'unverified_allowed')
            self.assertEqual(result['knobs']['LILITH_ALLOW_UNMERGED'], '1')

    def assert_sums_are_last(self, run_dir: Path):
        assert_packaging(self, run_dir)

    def test_refuses_a_dirty_tree_and_a_head_not_on_main(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            work = make_origin_checkout(base)
            target = work / verify_payload.PAYLOAD_FILES[0]
            target.write_text(target.read_text() + '\n# local edit\n')
            dirty = make_payload(work, base / 'out')
            self.assertNotEqual(dirty.returncode, 0)
            self.assertIn('lilith::payload::dirty', dirty.stderr)
            git(work, 'checkout', '-q', '--', verify_payload.PAYLOAD_FILES[0])
            git(work, 'checkout', '-q', '-b', 'feature')
            git(work, 'commit', '-q', '--allow-empty', '-m', 'unmerged')
            unmerged = make_payload(work, base / 'out')
            self.assertNotEqual(unmerged.returncode, 0)
            self.assertIn('lilith::payload::not_main', unmerged.stderr)
            self.assertFalse((base / 'out').exists())



def assert_packaging(test: unittest.TestCase, run_dir: Path):
    """smoke-result.json is the completion marker: last written, covered by SHA256SUMS and the tarball."""
    test.assertFalse((run_dir / 'smoke.pid').exists())
    test.assertFalse((run_dir.parent / f'.stage-{run_dir.name}').exists())
    sums = dict(reversed(line.split(maxsplit=1)) for line in (run_dir / 'SHA256SUMS').read_text().splitlines())
    test.assertIn('./smoke.log', sums)
    test.assertIn('./smoke-result.json', sums)
    test.assertNotIn('./smoke.pid', sums)
    test.assertFalse(any('results-' in name for name in sums))
    for name, digest in sums.items():
        test.assertEqual(hashlib.sha256((run_dir / name).read_bytes()).hexdigest(), digest, name)
    test.assertIn('finished rc=', (run_dir / 'smoke.log').read_text().splitlines()[-1])
    result = run_dir / 'smoke-result.json'
    tarball = run_dir / f'results-{run_dir.name}.tar.gz'
    with tarfile.open(tarball) as archive:
        packed = archive.extractfile(f'{run_dir.name}/smoke-result.json').read()
        names = archive.getnames()
    test.assertEqual(packed, result.read_bytes())
    test.assertFalse(any('.stage-' in name or name.endswith('.tar.gz') for name in names), names)


FAKE_PYTHON = r"""#!/usr/bin/env bash
# Test double for $ROOT/env/bin/python: fakes the GPU steps, runs everything else for real.
printf '%s\n' "$*" >> "$FAKE_LOG"
script="$(basename "${1:-}")"
output=""; mode=""; previous=""
for argument in "$@"; do
  [[ "$previous" == --output ]] && output="$argument"
  [[ "$previous" == --mode ]] && mode="$argument"
  previous="$argument"
done
case "$script" in
  lilith_training_cache_control.py)
    printf '{"status": "%s", "numerics": {"label": "fake"}}\n' "${FAKE_CACHE_STATUS:-failed_equivalence}" > "$output"
    [[ "${FAKE_CACHE_STATUS:-failed_equivalence}" == passed ]]; exit $? ;;
  lilith_cache_path_diagnostic.py)
    printf '{"status": "finished", "verdict": {"cause": "run_to_run_nondeterminism", "recommended_numerics_flags": ["--numerics-deterministic"]}}\n' > "$output"
    exit 0 ;;
  reference_microrun.py)
    [[ "$mode" == "${FAKE_FAIL_MODE:-none}" ]] && exit 2
    printf '{"status": "ok", "contexts": {}}\n' > "$output"
    exit 0 ;;
esac
exec "$REAL_PYTHON" "$@"
"""


class SmokeSequencingTests(unittest.TestCase):
    """run_smoke.sh on a CPU host with faked GPU steps: independent steps survive a failed control."""

    def smoke(self, run_id, **extra):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base, True)
        scripts = base / 'pod/scripts/lilith/runpod'
        scripts.mkdir(parents=True)
        for name in ('run_smoke.sh', 'verify_payload.py'):
            shutil.copy2(HERE / name, scripts / name)
        (scripts / 'bootstrap.sh').write_text('echo \'{"status": "ok"}\'\n')
        bin_dir = base / 'bin'
        bin_dir.mkdir()
        (bin_dir / 'nvidia-smi').write_text('#!/usr/bin/env bash\necho "NVIDIA H100 80GB HBM3, 580.0, 81559 MiB, GPU-x"\n')
        (bin_dir / 'timeout').write_text('#!/usr/bin/env bash\nshift 3\nexec "$@"\n')
        root = base / 'lilith'
        (root / 'env/bin').mkdir(parents=True)
        (root / 'env/bin/python').write_text(FAKE_PYTHON)
        for path in (bin_dir / 'nvidia-smi', bin_dir / 'timeout', root / 'env/bin/python'):
            path.chmod(0o755)
        log = base / 'fake.log'
        result = run('run_smoke.sh', run_id, root=root, script_dir=scripts,
                     PATH=f'{bin_dir}{os.pathsep}{os.environ["PATH"]}', FAKE_LOG=str(log),
                     REAL_PYTHON=sys.executable, LILITH_ALLOW_UNMERGED='1', LILITH_SMOKE_SYNTHETIC='1', **extra)
        run_dir = root / 'runs' / run_id
        smoke_result = json.loads((run_dir / 'smoke-result.json').read_text())
        return result, smoke_result, run_dir, log.read_text() if log.exists() else ''

    def statuses(self, smoke_result):
        return {step['step']: step['status'] for step in smoke_result['steps']}

    def test_failed_control_runs_the_diagnostic_and_every_independent_step(self):
        result, smoke, run_dir, calls = self.smoke('seq-fail', LILITH_NUMERICS_TF32='off',
                                                   LILITH_NUMERICS_DETERMINISTIC='1')
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertEqual(smoke['status'], 'failed')
        self.assertEqual(smoke['failed_steps'], ['cache-control'])
        self.assertEqual(list(self.statuses(smoke)), [
            'provenance', 'bootstrap-check', 'gpu', 'data-pin', 'cache-control', 'cache-diagnostic',
            'memory-probe', 'microrun', 'cold-reload'])
        self.assertEqual(self.statuses(smoke)['cache-diagnostic'], 'ok')
        self.assertEqual(smoke['cache_diagnostic_verdict']['cause'], 'run_to_run_nondeterminism')
        self.assertEqual(smoke['knobs']['LILITH_NUMERICS_TF32'], 'off')
        control_call = next(line for line in calls.splitlines() if 'lilith_training_cache_control.py' in line)
        self.assertIn('--numerics-tf32 off', control_call)
        self.assertIn('--numerics-deterministic', control_call)
        self.assertIn('FAILED steps: cache-control', (run_dir / 'smoke.log').read_text())
        assert_packaging(self, run_dir)

    def test_passing_control_skips_the_diagnostic(self):
        result, smoke, run_dir, calls = self.smoke('seq-pass', FAKE_CACHE_STATUS='passed')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(smoke['status'], 'passed')
        self.assertEqual(smoke['failed_steps'], [])
        self.assertNotIn('cache-diagnostic', self.statuses(smoke))
        self.assertNotIn('lilith_cache_path_diagnostic.py', calls)
        control_call = next(line for line in calls.splitlines() if 'lilith_training_cache_control.py' in line)
        self.assertNotIn('--numerics', control_call)   # default = qualified behaviour
        assert_packaging(self, run_dir)

    def test_dependent_step_is_skipped_and_listed(self):
        _, smoke, run_dir, _ = self.smoke('seq-dep', FAKE_CACHE_STATUS='passed', FAKE_FAIL_MODE='microrun')
        statuses = self.statuses(smoke)
        self.assertEqual(statuses['microrun'], 'failed')
        self.assertEqual(statuses['cold-reload'], 'skipped_dependency')
        self.assertEqual(statuses['memory-probe'], 'ok')
        self.assertEqual(smoke['failed_steps'], ['microrun', 'cold-reload'])
        assert_packaging(self, run_dir)

    def test_exhausted_budget_is_recorded_per_step(self):
        _, smoke, run_dir, _ = self.smoke('seq-budget', LILITH_SMOKE_BUDGET_SECONDS='100')
        statuses = self.statuses(smoke)
        self.assertEqual(statuses, {'provenance': 'budget_exhausted'})
        self.assertEqual(smoke['status'], 'failed')
        assert_packaging(self, run_dir)


FAKE_UV = r"""#!/usr/bin/env bash
# Test double for the pinned uv: `venv` creates a python shim, `pip install` succeeds.
case "$1" in
  --version) echo "uv 0.12.8" ;;
  venv)
    target="${@: -1}"
    mkdir -p "$target/bin" "$target/fake-site"
    printf '#!/usr/bin/env bash\nPYTHONPATH="%s/fake-site" exec "%s" "$@"\n' "$target" "$REAL_PYTHON" > "$target/bin/python"
    chmod 755 "$target/bin/python"
    printf '%s\n' '__version__ = "0.0+fake"' 'class version:' '    cuda = None' '    hip = None' \
      'class cuda:' '    @staticmethod' '    def is_available():' '        return False' \
      '    @staticmethod' '    def get_device_name(index):' '        return "fake"' > "$target/fake-site/torch.py" ;;
  pip) exit 0 ;;
  *) exit 64 ;;
esac
"""


class BootstrapEnvStateTests(unittest.TestCase):
    """An interrupted env build is rebuilt; a complete env from another lock is refused."""

    def root_with_uv(self):
        base = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, base, True)
        root = base / 'lilith'
        (root / 'bin').mkdir(parents=True)
        (root / 'bin/uv').write_text(FAKE_UV)
        (root / 'bin/uv').chmod(0o755)
        return root

    def bootstrap(self, root, *args):
        return run('bootstrap.sh', *args, root=root, LILITH_ALLOW_NO_CUDA='1', REAL_PYTHON=sys.executable)

    def test_empty_env_is_moved_aside_and_rebuilt(self):
        for partial in ('empty', 'python-only'):
            with self.subTest(partial=partial):
                root = self.root_with_uv()
                (root / 'env').mkdir()
                if partial == 'python-only':
                    (root / 'env/bin').mkdir()
                    (root / 'env/bin/python').write_text('#!/bin/sh\n')
                    (root / 'env/bin/python').chmod(0o755)
                result = self.bootstrap(root)
                self.assertNotIn('env_missing', result.stderr)
                self.assertIn('never finished building', result.stderr)
                aside = sorted(root.glob('env.incomplete-*'))
                self.assertEqual(len(aside), 1, result.stderr)
                lock = hashlib.sha256((HERE / 'requirements-cuda.txt').read_bytes()).hexdigest()
                self.assertEqual((root / 'env/.lilith-lock-sha256').read_text().strip(), lock)
                self.assertTrue((root / 'env/versions.json').is_file())
                # The run then stops at the parent download (no HF_TOKEN in tests).
                self.assertIn('lilith::bootstrap::hf_token', result.stderr)

    def test_check_reports_an_incomplete_env_without_moving_it(self):
        root = self.root_with_uv()
        (root / 'env').mkdir()
        result = self.bootstrap(root, '--check')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lilith::bootstrap::env_incomplete', result.stderr)
        self.assertTrue((root / 'env').is_dir())
        self.assertFalse(list(root.glob('env.incomplete-*')))

    def test_complete_env_from_another_lock_is_refused(self):
        root = self.root_with_uv()
        (root / 'env/bin').mkdir(parents=True)
        (root / 'env/bin/python').write_text('#!/bin/sh\n')
        (root / 'env/bin/python').chmod(0o755)
        (root / 'env/.lilith-lock-sha256').write_text('0' * 64 + '\n')
        result = self.bootstrap(root)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('lilith::bootstrap::env_lock', result.stderr)
        self.assertEqual((root / 'env/.lilith-lock-sha256').read_text(), '0' * 64 + '\n')
        self.assertFalse(list(root.glob('env.incomplete-*')))


class ChatTemplatePinConventionTest(unittest.TestCase):
    """The admission manifest pins the template as sha256(json.dumps(text)); the file pin is bytes."""

    def _setup(self, text, pin):
        import tempfile, json as _json
        d = Path(tempfile.mkdtemp())
        (d / 'chat_template.jinja').write_text(text, encoding='utf-8')
        m = d / 'manifest.json'
        m.write_text(_json.dumps({'chat_template_sha256': pin}))
        return d, m

    def test_json_string_convention_matches(self):
        import hashlib, json as _json
        import reference_microrun as rm
        text = '{{ bos_token }}{% for m in messages %}{{ m.content }}{% endfor %}\n'
        d, m = self._setup(text, hashlib.sha256(_json.dumps(text).encode()).hexdigest())
        self.assertEqual(rm.check_chat_template(d, m)['pin_convention'], 'json_string')

    def test_file_bytes_convention_matches(self):
        import hashlib
        import reference_microrun as rm
        text = 'template'
        d, m = self._setup(text, hashlib.sha256(text.encode()).hexdigest())
        self.assertEqual(rm.check_chat_template(d, m)['pin_convention'], 'file_bytes')

    def test_other_template_refused(self):
        import reference_microrun as rm
        d, m = self._setup('template', '0' * 64)
        with self.assertRaises(Exception):
            rm.check_chat_template(d, m)


if __name__ == '__main__':
    unittest.main()
