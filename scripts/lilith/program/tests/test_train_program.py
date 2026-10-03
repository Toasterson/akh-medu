"""CPU-only tests for the Lilith training program: manifest guards, stage loop, checkpoint/resume,
on-policy handshake, window exits, pod and host helpers. A fake engine stands in for the GPU; the
torch engine is exercised on a tiny random Gemma-4 only when torch/transformers/peft are installed.

Run: cd scripts/lilith && python3 -m pytest program/tests/test_train_program.py runpod/test_runpod_scripts.py -q
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

PROGRAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROGRAM))
sys.path.insert(0, str(PROGRAM.parents[0] / 'runpod'))
sys.path.insert(0, str(PROGRAM.parents[0] / 'serve'))
sys.path.insert(0, str(PROGRAM.parents[1] / 'session-platform'))

import checkpoint as ckpt  # noqa: E402
import run_program_pod  # noqa: E402
import serve_adapter  # noqa: E402
import supervise_program  # noqa: E402
import train_program as tp  # noqa: E402
from lilith_adapter_mechanics import PARENT_REVISION  # noqa: E402

STAGES = ('white', 'yellow', 'green', 'brown', 'black')
MAIN_SHA = 'a' * 40
OK_PAYLOAD = lambda root, sha: {'status': 'verified', 'manifest_sha256': 'f' * 64, 'problems': []}  # noqa: E731


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sft_record(rid, stage, split='train', weight=1.0, rng=None, multi_turn=False):
    rng = rng or random.Random(rid)
    n = rng.randint(12, 40)
    ids = [rng.randint(3, 500) for _ in range(n)]
    prompt = rng.randint(3, 8)
    mask = [False] * prompt + [True] * (n - prompt)
    if multi_turn:
        mask = [False] * 4 + [True] * 4 + [False] * 4 + [True] * (n - 12)
    return {'record_id': rid, 'stage': stage, 'kind': 'sft', 'weight': weight, 'source': 'test', 'families': ['f'], 'voice': 'maker',
            'language': 'en', 'component_id': 'c', 'split': split, 'input_ids': ids, 'target_mask': mask}


def pref_record(rid, stage, split='train', weight=1.0, rng=None):
    rng = rng or random.Random(rid)
    prompt = [rng.randint(3, 500) for _ in range(rng.randint(4, 9))]
    chosen = prompt + [rng.randint(3, 500) for _ in range(rng.randint(3, 10))]
    rejected = prompt + [rng.randint(3, 500) for _ in range(rng.randint(3, 10))]
    side = lambda ids: {'input_ids': ids, 'target_mask': [False] * len(prompt) + [True] * (len(ids) - len(prompt))}  # noqa: E731
    return {'record_id': rid, 'stage': stage, 'kind': 'preference', 'weight': weight, 'source': 'test', 'families': ['f'], 'voice': 'maker',
            'language': 'en', 'component_id': 'c', 'split': split, 'chosen': side(chosen), 'rejected': side(rejected)}


def archive_lines(stage, n_sft=14, n_pref=6):
    rows = [sft_record(f'{stage}-s{i}', stage) for i in range(n_sft)]
    rows += [pref_record(f'{stage}-p{i}', stage) for i in range(n_pref)]
    rows += [sft_record(f'{stage}-sd{i}', stage, 'dev') for i in range(2)] + [pref_record(f'{stage}-pd{i}', stage, 'dev') for i in range(2)]
    rows += [sft_record(f'{stage}-z', stage, weight=0.0), sft_record(f'{stage}-t', stage, 'test')]
    return rows


class World:
    """A frozen manifest, archives, parent pin files, approval and run root in a temp directory."""

    def __init__(self, base: Path, onpolicy=('green', 'black'), sft_passes=2, preference_passes=1, main_sha=MAIN_SHA):
        self.base, self.main_sha = base, main_sha
        self.data = base / 'data'
        self.data.mkdir()
        self.root = base / 'workspace'
        self.root.mkdir()
        self.parent = base / 'parent'
        self.parent.mkdir()
        pins = {}
        for name in tp.TOKENIZER_FILES:
            (self.parent / name).write_text(f'pinned {name}')
            pins[name] = sha(f'pinned {name}'.encode())
        stages = []
        for order, stage in enumerate(STAGES):
            body = ''.join(json.dumps(r) + '\n' for r in archive_lines(stage)).encode()
            (self.data / f'{stage}.tokens.jsonl').write_bytes(body)
            entry = {'name': stage, 'order': order, 'archive': f'{stage}.tokens.jsonl', 'archive_sha256': sha(body),
                     'sft_passes': sft_passes, 'preference_passes': preference_passes, 'lr_hint': 1e-4,
                     'onpolicy': {'families': ['evidence'], 'quota': 3, 'snapshot_until': '2026-10-02T00:00:00Z'} if stage in onpolicy else None}
            stages.append(entry)
        self.manifest = {'schema': tp.MANIFEST_SCHEMA, 'run_id': 'lilith-program-test', 'parent_revision': PARENT_REVISION,
                         'tokenizer_pins': pins, 'stages': stages}
        self.manifest_path = self.data / 'program-manifest.json'
        self.write_manifest()
        self.approval_path = base / 'approval.json'
        self.write_approval()

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest, indent=2))

    def manifest_sha(self):
        return sha(self.manifest_path.read_bytes())

    def write_approval(self, **overrides):
        record = {'decision': 'go', 'run_id': self.manifest['run_id'], 'manifest_sha256': self.manifest_sha(), 'main_sha': self.main_sha,
                  'operator': 'toasty', 'approved_at': '2026-10-03T12:00:00+00:00'}
        record.update(overrides)
        self.approval_path.write_text(json.dumps(record))

    @property
    def run_dir(self):
        return self.root / 'program' / self.manifest['run_id']

    def args(self, **overrides) -> tp.ProgramArgs:
        values = dict(manifest=self.manifest_path, parent=self.parent, approval=self.approval_path, main_sha=self.main_sha, root=self.root,
                      max_wall_seconds=100000, checkpoint_steps=3, token_budget=64, max_batch_records=4, dev_max=2,
                      onpolicy_timeout=3600, onpolicy_poll_seconds=1.0, early_stop_seconds=300)
        values.update(overrides)
        return tp.ProgramArgs(**values)

    def respond(self, stage, *, adapter_sha=None, archive_sha=None, name=None):
        """The host: deliver an on-policy archive and ready file for ``stage``."""
        stage_dir = self.run_dir / 'stages' / stage
        needed = json.loads((stage_dir / 'onpolicy-needed.json').read_text())
        rows = [sft_record(f'{stage}-op-s{i}', stage) for i in range(5)] + [pref_record(f'{stage}-op-p{i}', stage) for i in range(3)]
        body = ''.join(json.dumps(r) + '\n' for r in rows).encode()
        archive = stage_dir / (name or needed['archive_name'])
        archive.write_bytes(body)
        (stage_dir / 'onpolicy-ready.json').write_text(json.dumps({
            'schema': tp.READY_SCHEMA, 'run_id': self.manifest['run_id'], 'stage': stage, 'archive': name or needed['archive_name'],
            'sha256': archive_sha or sha(body), 'adapter_sha256': adapter_sha or needed['adapter_sha256']}))


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.slept = []
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()


class FakeEngine:
    """Deterministic stand-in: a hash chain over every step; checkpoints and adapters are small files."""

    def __init__(self, ctx=None, clock: FakeClock | None = None, step_seconds=0.0, fail_at_step=None):
        self.context, self.clock, self.step_seconds, self.fail_at_step = ctx, clock, step_seconds, fail_at_step
        self.digest = sha(b'init')
        self.calls: list[tuple] = []
        self.steps = 0
        self.released = 0

    def _advance(self, kind, sources, lr, step):
        if self.fail_at_step is not None and step == self.fail_at_step:
            raise RuntimeError('simulated crash')
        self.digest = sha(f'{self.digest}|{kind}|{",".join(sources)}|{lr:.12g}'.encode())
        self.calls.append((kind, tuple(sources), lr, step))
        self.steps += 1
        if self.clock:
            self.clock.now += self.step_seconds
        return tp.StepOutcome(1.0 + int(self.digest[:4], 16) / 65535, 1.0, len(sources) * 10, 0.0)

    def sft_step(self, examples, *, lr, step):
        return self._advance('sft', [e.source for e in examples], lr, step)

    def preference_step(self, pairs, *, lr, step):
        return self._advance('pref', [p.source for p in pairs], lr, step)

    def dev_metrics(self, sft, pairs):
        return {'sft_examples': len(sft), 'preference_pairs': len(pairs), 'dev_digest': self.digest[:8]}

    def save_training_state(self, directory):
        (directory / 'fake.json').write_text(json.dumps({'digest': self.digest}))

    def load_training_state(self, directory):
        self.digest = json.loads((directory / 'fake.json').read_text())['digest']

    def save_adapter(self, directory):
        directory.mkdir()
        weights = f'weights:{self.digest}'.encode()
        (directory / 'adapter_model.safetensors').write_bytes(weights)
        (directory / 'adapter_config.json').write_text(json.dumps({'r': 16}))
        return {'directory': str(directory), 'files_sha256': {'adapter_model.safetensors': sha(weights)}, 'tensor_sha256': self.digest, 'rank': 16}

    def write_cold_reload_probe(self, path, adapter, input_ids):
        path.write_text(json.dumps({'ids': list(input_ids)}))

    def describe(self):
        return {'parent_pins': {'model.safetensors': 'x'}, 'trainer_config': {'rank': 16}}

    def release_cache(self):
        self.released += 1


def run(world: World, *, engine=None, clock=None, payload=OK_PAYLOAD, **overrides):
    """Run one window; returns (exit code, engine). A fresh FakeEngine unless one is supplied."""
    clock = clock or FakeClock()
    engines = []

    def factory(ctx):
        engines.append(engine or FakeEngine(ctx, clock))
        return engines[-1]

    stop = overrides.pop('stop_requested', lambda: False)
    code = tp.run_program(world.args(**overrides), engine_factory=factory, payload_verifier=payload, clock=clock, sleep=clock.sleep,
                          stop_requested=stop)
    return code, (engines[-1] if engines else None)       # no engine is built for an already-complete run


def receipts(world):
    path = world.run_dir / 'steps.jsonl'
    return [json.loads(line) for line in path.read_text().splitlines()]


def signature(world):
    return [(r['step'], r['stage'], r['phase'], r['pass'], r['batch'], r['batch_digest'], r['lr']) for r in receipts(world)]


class Temp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        os.environ['LILITH_ALLOW_NON_WORKSPACE'] = '1'
        self.addCleanup(os.environ.pop, 'LILITH_ALLOW_NON_WORKSPACE', None)

    def world(self, **kwargs) -> World:
        return World(self.base, **kwargs)

    def assertRefused(self, code, call):
        with self.assertRaises(ckpt.ProgramError) as caught:
            call()
        self.assertEqual(caught.exception.code, f'lilith::program::{code}')


class GuardTests(Temp):
    def test_archive_hash_mismatch_is_refused(self):
        world = self.world()
        (world.data / 'white.tokens.jsonl').write_text('tampered\n')
        self.assertRefused('archive_hash', lambda: run(world))
        self.assertFalse(world.run_dir.exists())

    def test_manifest_changed_after_approval_is_refused(self):
        world = self.world()
        world.manifest['stages'][0]['sft_passes'] = 3     # same archives, different program
        world.write_manifest()
        self.assertRefused('approval_manifest', lambda: run(world))

    def test_approval_main_sha_and_decision_and_init_binding(self):
        world = self.world()
        world.write_approval(main_sha='b' * 40)
        self.assertRefused('approval_main', lambda: run(world))
        world.write_approval(decision='hold')
        self.assertRefused('approval_decision', lambda: run(world))
        world.write_approval(init_adapter_sha256='c' * 64)
        self.assertRefused('approval_init_adapter', lambda: run(world))

    def test_parent_pin_and_revision_mismatch(self):
        world = self.world()
        (world.parent / 'chat_template.jinja').write_text('another template')
        self.assertRefused('parent_pin', lambda: run(world))
        world = World(self.base / 'second' if (self.base / 'second').mkdir() is None else self.base)
        world.manifest['parent_revision'] = '0' * 40
        world.write_manifest()
        world.write_approval()
        self.assertRefused('parent_revision', lambda: run(world))

    def test_unverified_payload_is_refused(self):
        world = self.world()
        bad = lambda root, sha: {'status': 'failed', 'problems': ['file differs']}  # noqa: E731
        self.assertRefused('payload', lambda: run(world, payload=bad))

    def test_stage_order_is_canonical_and_complete(self):
        world = self.world()
        world.manifest['stages'][0]['order'], world.manifest['stages'][1]['order'] = 1, 0     # yellow before white
        world.write_manifest()
        self.assertRefused('manifest_stage_order', lambda: tp.parse_manifest(world.manifest_path))
        world = World(self.base / 'b' if (self.base / 'b').mkdir() is None else self.base)
        world.manifest['stages'].pop()
        world.write_manifest()
        self.assertRefused('manifest_stage_order', lambda: tp.parse_manifest(world.manifest_path))

    def test_record_validation(self):
        world = self.world()
        path = world.data / 'bad.jsonl'
        bad_weight = sft_record('x', 'white', weight=1.5)
        path.write_text(json.dumps(bad_weight) + '\n')
        self.assertRefused('record_weight', lambda: tp.load_archive(path, 'white'))
        path.write_text(json.dumps(sft_record('x', 'yellow')) + '\n')
        self.assertRefused('record_stage', lambda: tp.load_archive(path, 'white'))
        broken = pref_record('p', 'white')
        broken['rejected']['input_ids'][0] += 1               # no shared prefix
        path.write_text(json.dumps(broken) + '\n')
        self.assertRefused('pair_prefix', lambda: tp.load_archive(path, 'white'))
        path.write_text(json.dumps(sft_record('x', 'white')) + '\n' + json.dumps(sft_record('x', 'white')) + '\n')
        self.assertRefused('record_duplicate', lambda: tp.load_archive(path, 'white'))

    def test_archive_parsing_splits_weights_and_mask_shapes(self):
        world = self.world()
        path = world.data / 'mix.jsonl'
        rows = archive_lines('white') + [sft_record('mt', 'white', multi_turn=True)]
        path.write_text(json.dumps({'schema': tp.TOKENS_SCHEMA}) + '\n' + ''.join(json.dumps(r) + '\n' for r in rows))
        data = tp.load_archive(path, 'white')
        self.assertEqual((len(data.sft_train), len(data.pref_train), len(data.sft_dev), len(data.pref_dev)), (15, 6, 2, 2))
        self.assertEqual(data.skipped, {'zero_weight': 1, 'other_split': 1})        # train only; weight 0 teaches nothing
        multi = next(i for i in data.sft_train if i.record_id == 'mt')
        self.assertEqual(multi.example.target_mask.count(True), 4 + len(multi.example.target_mask) - 12)

    def test_declared_counts_must_match(self):
        world = self.world()
        spec = tp.parse_manifest(world.manifest_path).stages[0]
        data = tp.load_archive(world.data / spec.archive, 'white')
        tp.verify_counts(tp.dataclasses.replace(spec, counts={'train': 21, 'dev': 4, 'sft': 18, 'preference': 8}), data)
        self.assertRefused('counts', lambda: tp.verify_counts(tp.dataclasses.replace(spec, counts={'train': 99}), data))

    def test_fresh_start_refuses_existing_run_and_resume_needs_one(self):
        world = self.world()
        self.assertEqual(run(world, stop_requested=lambda: True)[0], tp.EXIT_WINDOW)
        self.assertRefused('run_exists', lambda: run(world))
        other = World(self.base / 'c' if (self.base / 'c').mkdir() is None else self.base)
        self.assertRefused('resume_nothing', lambda: run(other, resume=True))


class InitAdapterTests(Temp):
    def adapter(self, **config):
        directory = self.base / 'init'
        directory.mkdir(exist_ok=True)
        (directory / 'adapter_model.safetensors').write_bytes(b'weights')
        cfg = {'peft_type': 'LORA', 'r': 16, 'lora_alpha': 16, 'target_modules': tp.trainer.LANGUAGE_MODEL_TARGETS, 'bias': 'none'}
        cfg.update(config)
        (directory / 'adapter_config.json').write_text(json.dumps(cfg))
        return directory

    def test_compatible_adapter_is_adopted_and_recorded(self):
        record = tp.check_init_adapter(self.adapter(), sha(b'weights'), tp.trainer.TrainerConfig())
        self.assertEqual(record['decision'], 'adopted_as_starting_point')
        self.assertEqual(record['adapter_model_sha256'], sha(b'weights'))

    def test_incompatible_rank_alpha_targets_and_hash_are_refused(self):
        config = tp.trainer.TrainerConfig()
        self.assertRefused('init_adapter_incompatible', lambda: tp.check_init_adapter(self.adapter(r=32), sha(b'weights'), config))
        self.assertRefused('init_adapter_incompatible', lambda: tp.check_init_adapter(self.adapter(lora_alpha=32), sha(b'weights'), config))
        self.assertRefused('init_adapter_incompatible', lambda: tp.check_init_adapter(self.adapter(target_modules=['q_proj']), sha(b'weights'), config))
        self.assertRefused('init_adapter_hash', lambda: tp.check_init_adapter(self.adapter(), 'e' * 64, config))

    def test_run_with_init_adapter_binds_approval_and_records_decision(self):
        world = self.world()
        init = self.adapter()
        world.write_approval(init_adapter_sha256=sha(b'weights'))
        engines = []
        code = tp.run_program(world.args(init_adapter=init, init_adapter_sha256=sha(b'weights'), onpolicy_wait_mode='exit'),
                              engine_factory=lambda ctx: engines.append(ctx) or FakeEngine(ctx), payload_verifier=OK_PAYLOAD,
                              stop_requested=lambda: True)
        self.assertEqual(code, tp.EXIT_WINDOW)
        self.assertEqual(engines[0].init_adapter['decision'], 'adopted_as_starting_point')
        identity = json.loads((world.run_dir / 'identity.json').read_text())
        self.assertEqual(identity['init_adapter']['adapter_model_sha256'], sha(b'weights'))


class PlanningTests(unittest.TestCase):
    def test_batches_cover_every_record_once_within_budget_and_are_seeded(self):
        lengths = [random.Random(i).randint(5, 60) for i in range(100)]
        plan = tp.plan_batches(lengths, seed=7, token_budget=120, max_records=8)
        flat = sorted(i for batch in plan for i in batch)
        self.assertEqual(flat, list(range(100)))
        for batch in plan:
            self.assertTrue(len(batch) <= 8 and (len(batch) == 1 or sum(lengths[i] for i in batch) <= 120))
        self.assertEqual(plan, tp.plan_batches(lengths, seed=7, token_budget=120, max_records=8))
        self.assertNotEqual(plan, tp.plan_batches(lengths, seed=8, token_budget=120, max_records=8))

    def test_oversized_record_gets_its_own_batch(self):
        plan = tp.plan_batches([500, 5, 5], seed=1, token_budget=100, max_records=8)
        self.assertIn((0,), plan)

    def test_lr_schedule_warms_up_then_cosines_to_the_floor(self):
        total = 100
        lrs = [tp.cosine_lr(1e-4, s, total) for s in range(total)]
        self.assertAlmostEqual(lrs[0], 1e-4 / 5)                    # 5% warmup = 5 steps
        self.assertAlmostEqual(lrs[4], 1e-4)
        self.assertTrue(all(a >= b for a, b in zip(lrs[4:], lrs[5:])))
        self.assertAlmostEqual(lrs[-1], 1e-5, delta=2e-7)           # 10% floor
        self.assertEqual(tp.cosine_lr(1e-4, 0, 1), 1e-4)


class ProgramRunTests(Temp):
    def handshake_host(self, world, clock, delivered):
        def on_sleep():
            for stage in ('green', 'black'):
                needed = world.run_dir / 'stages' / stage / 'onpolicy-needed.json'
                ready = world.run_dir / 'stages' / stage / 'onpolicy-ready.json'
                if needed.is_file() and not ready.is_file():
                    world.respond(stage)
                    delivered.append(stage)
        clock.on_sleep = on_sleep

    def test_full_run_trains_five_stages_in_order_with_handshake_and_writes_the_candidate(self):
        world = self.world()
        clock, delivered = FakeClock(), []
        self.handshake_host(world, clock, delivered)
        code, engine = run(world, clock=clock)
        self.assertEqual(code, tp.EXIT_COMPLETE)
        self.assertEqual(delivered, ['green', 'black'])
        stage_sequence = [r['stage'] for r in receipts(world)]
        self.assertEqual(sorted(set(stage_sequence), key=stage_sequence.index), list(STAGES))
        self.assertEqual(stage_sequence, sorted(stage_sequence, key=STAGES.index))          # never back to an earlier stage
        phases = {s: [r['phase'] for r in receipts(world) if r['stage'] == s] for s in STAGES}
        self.assertEqual(phases['white'], sorted(phases['white'], key=['sft', 'preference'].index))
        self.assertEqual(set(phases['green']), {'sft', 'preference', 'onpolicy_sft', 'onpolicy_preference'})
        self.assertNotIn('onpolicy_sft', phases['yellow'])
        self.assertGreaterEqual(engine.released, 1)
        done = json.loads((world.run_dir / 'program-complete.json').read_text())
        self.assertEqual(set(done['stages']), set(STAGES))
        self.assertEqual(done['total_steps'], len(receipts(world)))
        self.assertTrue(all('dev' in v for v in done['stages'].values()))
        candidate = json.loads((world.run_dir / 'candidate-manifest.json').read_text())
        self.assertEqual(candidate['status'], 'candidate_inactive')
        self.assertFalse(candidate['activated'])
        self.assertEqual(candidate['reward_policy'], 'cos-v1')
        self.assertEqual(candidate['program_manifest_sha256'], world.manifest_sha())
        self.assertEqual(candidate['code']['main_sha'], MAIN_SHA)
        self.assertEqual(candidate['stage_steps']['green']['onpolicy_sft'] > 0, True)
        self.assertEqual(candidate['adapter']['adapter_model_sha256'], done['final_adapter_sha256'])
        self.assertEqual(done['candidate_manifest_sha256'], ckpt.sha256_file(world.run_dir / 'candidate-manifest.json'))
        self.assertTrue((world.run_dir / 'stages' / 'green' / 'adapter' / 'adapter_model.safetensors').is_file())
        self.assertTrue((world.run_dir / 'stages' / 'green' / 'adapter-final').is_dir())
        self.assertTrue((world.run_dir / 'stages' / 'white' / 'stage-result.json').is_file())
        budget = json.loads((world.run_dir / 'budget.json').read_text())
        self.assertEqual(budget['done_steps'], done['total_steps'])
        self.assertEqual(run(world, resume=True)[0], tp.EXIT_COMPLETE)                       # idempotent after completion

    def test_every_receipt_carries_loss_weight_tokens_lr_and_stage(self):
        world = self.world(onpolicy=())
        self.assertEqual(run(world)[0], tp.EXIT_COMPLETE)
        for record in receipts(world):
            for key in ('loss', 'weight_sum', 'tokens', 'lr', 'stage', 'step', 'batch_digest'):
                self.assertIn(key, record)
        lrs = [r['lr'] for r in receipts(world) if r['stage'] == 'white']
        self.assertAlmostEqual(max(lrs), 1e-4)

    def test_zero_weight_records_never_enter_a_batch(self):
        world = self.world(onpolicy=())
        code, engine = run(world)
        self.assertEqual(code, tp.EXIT_COMPLETE)
        used = {s for _, sources, _, _ in engine.calls for s in sources}
        self.assertFalse(any(name.endswith('-z') or name.endswith('-t') or '-sd' in name or '-pd' in name for name in used))

    def test_resume_mid_stage_reproduces_batch_order_and_final_weights(self):
        reference = World(self.base / 'ref' if (self.base / 'ref').mkdir() is None else self.base, onpolicy=())
        # reference world lives in its own subtree
        ref_code, ref_engine = run(reference)
        self.assertEqual(ref_code, tp.EXIT_COMPLETE)
        base2 = self.base / 'interrupted'
        base2.mkdir()
        world = World(base2, onpolicy=())
        engine_a = FakeEngine()
        code, _ = run(world, engine=engine_a, stop_requested=lambda: engine_a.steps >= 7)
        self.assertEqual(code, tp.EXIT_WINDOW)
        state = ckpt.CheckpointStore(world.run_dir).newest_valid()[1]
        self.assertEqual(state.global_step, 7)
        self.assertEqual((state.cursor.stage_index, state.cursor.phase), (0, 'sft'))      # stopped mid-stage
        code, engine_b = run(world, resume=True)
        self.assertEqual(code, tp.EXIT_COMPLETE)
        self.assertEqual(signature(world), signature(reference))
        self.assertEqual(engine_a.steps + engine_b.steps, ref_engine.steps)
        self.assertEqual(engine_b.digest, ref_engine.digest)
        self.assertEqual(json.loads((world.run_dir / 'program-complete.json').read_text())['final_adapter_sha256'],
                         json.loads((reference.run_dir / 'program-complete.json').read_text())['final_adapter_sha256'])
        self.assertEqual([c[:2] + (c[3],) for c in engine_a.calls + engine_b.calls], [c[:2] + (c[3],) for c in ref_engine.calls])

    def test_crash_between_checkpoints_replays_without_duplicate_receipts(self):
        reference = World(self.base / 'ref' if (self.base / 'ref').mkdir() is None else self.base, onpolicy=())
        ref_engine = run(reference)[1]
        base2 = self.base / 'crashed'
        base2.mkdir()
        world = World(base2, onpolicy=())
        with self.assertRaises(RuntimeError):
            run(world, engine=FakeEngine(fail_at_step=8))                       # periodic checkpoints at steps 3 and 6
        self.assertEqual(len(receipts(world)), 8)
        code, engine = run(world, resume=True)
        self.assertEqual(code, tp.EXIT_COMPLETE)
        self.assertEqual([r['step'] for r in receipts(world)], list(range(ref_engine.steps)))
        self.assertEqual(signature(world), signature(reference))
        self.assertEqual(engine.digest, ref_engine.digest)

    def test_resume_refuses_different_manifest_or_main_sha_or_recipe(self):
        world = self.world(onpolicy=())
        self.assertEqual(run(world, stop_requested=lambda: True)[0], tp.EXIT_WINDOW)
        other = World(self.base / 'o' if (self.base / 'o').mkdir() is None else self.base, onpolicy=(), main_sha='b' * 40)
        other.manifest['run_id'] = world.manifest['run_id']
        other.root = world.root
        other.write_manifest()
        other.write_approval()
        self.assertRefused('resume_identity', lambda: run(other, resume=True))               # different main SHA and manifest
        again = World(self.base / 'p' if (self.base / 'p').mkdir() is None else self.base, onpolicy=())
        again.root = world.root
        self.assertRefused('resume_identity', lambda: run(again, resume=True, token_budget=65))   # same program, other batch recipe

    def test_window_deadline_exits_75_and_the_next_window_finishes(self):
        world = self.world(onpolicy=())
        clock = FakeClock()
        engine = FakeEngine(clock=clock, step_seconds=100)
        code, _ = run(world, engine=engine, clock=clock, max_wall_seconds=1300)          # 1300 - 300 early = 1000 s = 10 steps
        self.assertEqual(code, tp.EXIT_WINDOW)
        self.assertEqual(engine.steps, 10)
        exit_record = json.loads((world.run_dir / 'window-exit.json').read_text())
        self.assertEqual((exit_record['exit_code'], exit_record['reason']), (75, 'deadline'))
        self.assertEqual(ckpt.CheckpointStore(world.run_dir).newest_valid()[1].global_step, 10)
        self.assertEqual(run(world, resume=True)[0], tp.EXIT_COMPLETE)
        windows = [json.loads(line) for line in (world.run_dir / 'windows.jsonl').read_text().splitlines()]
        self.assertEqual([w['event'] for w in windows], ['start', 'end', 'start', 'end'])

    def test_sigterm_style_stop_checkpoints_at_the_step_boundary(self):
        world = self.world(onpolicy=())
        engine = FakeEngine()
        code, _ = run(world, engine=engine, stop_requested=lambda: engine.steps >= 2)
        self.assertEqual(code, tp.EXIT_WINDOW)
        self.assertEqual(ckpt.CheckpointStore(world.run_dir).newest_valid()[1].global_step, 2)
        self.assertEqual(json.loads((world.run_dir / 'window-exit.json').read_text())['reason'], 'signal')

    def test_corrupt_newest_checkpoint_falls_back_to_the_previous_one(self):
        world = self.world(onpolicy=())
        engine = FakeEngine()
        run(world, engine=engine, stop_requested=lambda: engine.steps >= 7)
        store = ckpt.CheckpointStore(world.run_dir)
        newest = list(store.directories())[-1]
        (newest / 'fake.json').write_text('corrupt')
        found = store.newest_valid()
        self.assertLess(found[1].global_step, 7)

    def test_nonfinite_loss_is_a_failure_not_a_checkpoint(self):
        world = self.world(onpolicy=())

        class Nan(FakeEngine):
            def _advance(self, *a):
                outcome = super()._advance(*a)
                return tp.StepOutcome(float('nan'), 1.0, outcome.tokens, 0.0)
        self.assertRefused('nonfinite', lambda: run(world, engine=Nan()))


class OnPolicyTests(Temp):
    def window_after_request(self, world, **overrides):
        """Run until the first on-policy request, exit mode so the window ends there."""
        return run(world, onpolicy_wait_mode='exit', **overrides)

    def test_needed_file_names_adapter_families_quota_and_snapshot(self):
        world = self.world()
        code, _ = self.window_after_request(world)
        self.assertEqual(code, tp.EXIT_WINDOW)
        needed = json.loads((world.run_dir / 'stages' / 'green' / 'onpolicy-needed.json').read_text())
        self.assertEqual((needed['stage'], needed['families'], needed['quota'], needed['snapshot_until']),
                         ('green', ['evidence'], 3, '2026-10-02T00:00:00Z'))
        adapter = world.run_dir / 'stages' / 'green' / 'adapter' / 'adapter_model.safetensors'
        self.assertEqual(needed['adapter_sha256'], ckpt.sha256_file(adapter))
        self.assertEqual(json.loads((world.run_dir / 'window-exit.json').read_text())['reason'], 'onpolicy_pending')
        state = ckpt.CheckpointStore(world.run_dir).newest_valid()[1]
        self.assertEqual((state.cursor.stage_index, state.cursor.phase), (2, 'onpolicy_wait'))

    def test_exit_mode_pending_then_resume_after_ready(self):
        world = self.world()
        self.window_after_request(world)
        self.assertEqual(run(world, resume=True, onpolicy_wait_mode='exit')[0], tp.EXIT_WINDOW)     # still nothing delivered
        world.respond('green')
        code, _ = run(world, resume=True, onpolicy_wait_mode='exit')
        self.assertEqual(code, tp.EXIT_WINDOW)                                                      # now waits on black
        self.assertTrue(any(r['phase'] == 'onpolicy_sft' and r['stage'] == 'green' for r in receipts(world)))
        self.assertTrue((world.run_dir / 'stages' / 'black' / 'onpolicy-needed.json').is_file())

    def test_poll_mode_waits_then_trains_the_archive(self):
        world = self.world(onpolicy=('white',))
        clock = FakeClock()
        clock.on_sleep = lambda: (world.respond('white') if not (world.run_dir / 'stages/white/onpolicy-ready.json').is_file() else None)
        code, _ = run(world, clock=clock)
        self.assertEqual(code, tp.EXIT_COMPLETE)
        self.assertTrue(clock.slept)
        steps = json.loads((world.run_dir / 'stages/white/stage-result.json').read_text())['steps']
        self.assertGreater(steps['onpolicy_sft'], 0)
        self.assertGreater(steps['onpolicy_preference'], 0)

    def test_timeout_ends_the_window_cleanly(self):
        world = self.world(onpolicy=('white',))
        code, _ = run(world, onpolicy_timeout=5, onpolicy_poll_seconds=2)
        self.assertEqual(code, tp.EXIT_WINDOW)
        self.assertEqual(json.loads((world.run_dir / 'window-exit.json').read_text())['reason'], 'onpolicy_timeout')
        self.assertEqual(ckpt.CheckpointStore(world.run_dir).newest_valid()[1].cursor.phase, 'onpolicy_wait')

    def test_ready_for_another_adapter_or_with_a_bad_hash_or_name_is_refused(self):
        for override, code in (({'adapter_sha': 'd' * 64}, 'onpolicy_adapter'), ({'archive_sha': 'd' * 64}, 'onpolicy_archive_hash'),
                               ({'name': 'other.tokens.jsonl'}, 'onpolicy_archive_name')):
            base = self.base / code
            base.mkdir()
            world = World(base, onpolicy=('white',))
            self.window_after_request(world)
            world.respond('white', **override)
            self.assertRefused(code, lambda: run(world, resume=True, onpolicy_wait_mode='exit'))


class CheckpointTests(Temp):
    def test_commit_is_atomic_and_pruned(self):
        store = ckpt.CheckpointStore(self.base, keep=2)
        state = ckpt.ProgramState()
        for seq in range(1, 5):
            state.checkpoint_seq, state.global_step = seq, seq * 10
            tmp = store.begin(seq)
            (tmp / 'blob').write_bytes(b'x' * seq)
            store.commit(tmp, state)
        self.assertEqual([d.name for d in store.directories()], ['ckpt-000003', 'ckpt-000004'])
        self.assertEqual(store.newest_valid()[1].global_step, 40)
        (self.base / 'checkpoints' / '.tmp-ckpt-000005').mkdir()
        self.assertEqual(store.clean_stale(), ['.tmp-ckpt-000005'])

    def test_truncate_jsonl_drops_replayed_and_torn_lines(self):
        path = self.base / 'steps.jsonl'
        path.write_text('{"step": 0}\n{"step": 1}\n{"step": 2}\n{"step": 3')
        self.assertEqual(ckpt.truncate_jsonl(path, lambda r: r['step'] < 2), 2)
        self.assertEqual(path.read_text().splitlines(), ['{"step": 0}', '{"step": 1}'])


class PodScriptTests(Temp):
    def test_status_reports_liveness_by_marker_and_pending_handshake(self):
        root = self.base / 'ws'
        run_dir = root / 'program' / 'run-abc'
        (run_dir / 'stages' / 'green').mkdir(parents=True)
        (run_dir / 'stages' / 'green' / 'onpolicy-needed.json').write_text('{}')
        (run_dir / 'progress.json').write_text(json.dumps({'status': 'training', 'global_step': 4}))
        ckpt.write_json_atomic(run_dir / 'program.pid', {'pid': os.getpid(), 'marker': 'not-in-this-cmdline-zzz'})
        report = run_program_pod.status(root, 'run-abc')
        self.assertFalse(report['alive'])                      # pid exists but the marker is absent: not our trainer
        self.assertEqual(report['onpolicy_pending'], ['green'])
        self.assertEqual(report['progress']['global_step'], 4)
        marker = f'unique-marker-{os.getpid()}'
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', marker])
        self.addCleanup(child.kill)
        self.assertTrue(run_program_pod.process_matches(child.pid, marker))
        self.assertFalse(run_program_pod.process_matches(child.pid, 'another-marker'))
        child.kill()
        child.wait()
        self.assertFalse(run_program_pod.process_matches(child.pid, marker))

    def test_onpolicy_ready_validates_hash_name_and_adapter(self):
        root = self.base / 'ws'
        stage = root / 'program' / 'run-abc' / 'stages' / 'green'
        stage.mkdir(parents=True)
        (stage / 'onpolicy-needed.json').write_text(json.dumps({'archive_name': 'onpolicy-green.tokens.jsonl', 'adapter_sha256': 'a' * 64}))
        (stage / 'onpolicy-green.tokens.jsonl').write_text('rows')
        parser = run_program_pod.build_parser()
        good = parser.parse_args(['--root', str(root), 'onpolicy-ready', '--run-id', 'run-abc', '--stage', 'green',
                                  '--archive', 'onpolicy-green.tokens.jsonl', '--sha256', sha(b'rows'), '--adapter-sha256', 'a' * 64])
        record = run_program_pod.onpolicy_ready(good)
        self.assertTrue((stage / 'onpolicy-ready.json').is_file())
        self.assertEqual(record['archive'], 'onpolicy-green.tokens.jsonl')
        for flag, value in (('--sha256', sha(b'other')), ('--adapter-sha256', 'b' * 64), ('--archive', 'x.jsonl')):
            argv = ['--root', str(root), 'onpolicy-ready', '--run-id', 'run-abc', '--stage', 'green', '--archive', 'onpolicy-green.tokens.jsonl',
                    '--sha256', sha(b'rows'), '--adapter-sha256', 'a' * 64]
            argv[argv.index(flag) + 1] = value
            with self.assertRaises(ckpt.ProgramError):
                run_program_pod.onpolicy_ready(parser.parse_args(argv))


class HostSupervisorTests(Temp):
    def approval(self, **overrides):
        record = {'decision': 'go', 'run_id': 'lilith-program-test', 'pod_id': 'pod123', 'main_sha': MAIN_SHA, 'manifest_sha256': '1' * 64,
                  'stop_pod_at_completion_or_deadline': True, 'max_window_seconds': 20000}
        record.update(overrides)
        path = self.base / 'host-approval.json'
        path.write_text(json.dumps(record))
        return path

    def test_approval_must_bind_pod_stop_and_window(self):
        supervise_program.load_host_approval(self.approval(), 'pod123', 18000)
        for overrides, window in (({'decision': 'hold'}, 18000), ({'pod_id': 'other'}, 18000), ({'stop_pod_at_completion_or_deadline': False}, 18000),
                                  ({'max_window_seconds': 100}, 18000)):
            with self.assertRaises(supervise_program.HostError):
                supervise_program.load_host_approval(self.approval(**overrides), 'pod123', window)

    def test_decide_reads_one_status(self):
        decide = lambda probe, strikes=0: supervise_program.decide(probe, stall_seconds=1800, stalled_strikes=strikes).action  # noqa: E731
        self.assertEqual(decide({'program_complete': True}), 'complete')
        self.assertEqual(decide({'alive': True, 'onpolicy_pending': ['green']}), 'handshake')
        self.assertEqual(decide({'alive': True, 'newest_write_age_seconds': 30, 'gpu_util': 90}), 'continue')
        stalled = {'alive': True, 'newest_write_age_seconds': 4000, 'gpu_util': 0}
        self.assertEqual((decide(stalled, 0), decide(stalled, 1)), ('continue', 'stalled'))
        self.assertEqual(decide({'alive': False, 'window_exit': {'exit_code': 75, 'reason': 'deadline'}}), 'window_ended')
        self.assertEqual(decide({'alive': False, 'window_exit': {'exit_code': 1, 'reason': 'x'}}), 'failed')
        self.assertEqual(decide({'alive': False}), 'crashed')

    def test_window_runs_handshake_stops_the_pod_and_records_spend(self):
        clock = FakeClock()
        local_pod = {'status': 0}
        stops, commands = [], []
        host_dest = self.base / 'evidence'
        args = supervise_program.HostArgs(pod_id='pod123', ssh='root@1.2.3.4', port=2222, identity=Path('/nonexistent'), window_seconds=18000,
                                          approval=self.approval(), dest=host_dest, hourly_usd=2.5, poll_seconds=300, driver_poll_seconds=10)
        approval = supervise_program.load_host_approval(args.approval, 'pod123', 18000)

        class FakeSsh:
            target, port, opts, runner = 'root@1.2.3.4', 2222, [], None

            def __init__(inner):
                inner.polls = 0

            def run(inner, command, *, timeout, input=None):
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, '{}', '')

            def json(inner, command, *, timeout=60):
                commands.append(command)
                if ' status ' in command:
                    inner.polls += 1
                    if inner.polls == 1:
                        return {'alive': True, 'onpolicy_pending': ['green']}
                    if inner.polls == 2:
                        return {'alive': True, 'newest_write_age_seconds': 5}
                    return {'alive': False, 'program_complete': True}
                if command.startswith('cat'):
                    return {'adapter_sha256': 'a' * 64, 'stage': 'green'}
                if ' start ' in command:
                    return {'marker': 'm1', 'window': 1, 'log': '/x.log', 'exited_immediately': False}
                if 'onpolicy-ready' in command:
                    return {'stage': 'green'}
                return {}

            def put(inner, local, remote, *, timeout=900):
                commands.append(f'put {Path(local).name} {remote}')

            def get(inner, remote, local, *, timeout=300):
                return False

        def driver():                                        # the supervising session answers once vLLM is "served"
            ready = host_dest / 'onpolicy' / 'green'
            if (ready / 'serve-ready.json').is_file() and not (ready / 'archive-ready.json').is_file():
                body = b'archive'
                (ready / 'onpolicy-green.tokens.jsonl').write_bytes(body)
                (ready / 'archive-ready.json').write_text(json.dumps({'archive': 'onpolicy-green.tokens.jsonl', 'sha256': sha(body)}))
        clock.on_sleep = driver
        supervisor = supervise_program.Supervisor(args, approval, ssh=FakeSsh(), clock=clock, sleep=clock.sleep,
                                                  stop_pod=lambda pod: stops.append(pod) or {'runtimeStatus': 'stopped', 'desiredStatus': 'EXITED'})
        code = supervisor.run()
        self.assertEqual(code, supervise_program.EXIT_COMPLETE)
        self.assertEqual(stops, ['pod123'])
        serve = json.loads((host_dest / 'onpolicy' / 'green' / 'serve-ready.json').read_text())
        self.assertEqual(serve['user_agent_required'], 'curl/8.7.1')
        self.assertTrue(serve['proxy_url'].startswith('https://pod123-8000.proxy.runpod.net'))
        self.assertFalse((host_dest / 'onpolicy' / 'green' / 'serve-key').exists())          # key removed after the round
        order = [c for c in commands if 'serve_candidate.sh' in c or 'onpolicy-ready' in c]
        self.assertIn('serve_candidate.sh start', order[0])
        self.assertIn('serve_candidate.sh stop', order[1])                                   # vLLM stops before ready is written
        self.assertIn('onpolicy-ready', order[2])
        spend = json.loads((host_dest / 'spend.jsonl').read_text().splitlines()[0])
        self.assertEqual(spend['pod_id'], 'pod123')
        self.assertIsNotNone(spend['estimated_usd'])

    def test_pod_is_stopped_even_when_the_window_fails(self):
        class DeadSsh:
            target, port, opts, runner = 'x', 1, [], None

            def run(self, command, *, timeout, input=None):
                raise OSError('unreachable')

            def json(self, command, *, timeout=60):
                raise OSError('unreachable')

            def get(self, *a, **k):
                return False
        stops = []
        args = supervise_program.HostArgs(pod_id='pod123', ssh='root@1.2.3.4', port=2222, identity=Path('/nonexistent'), window_seconds=18000,
                                          approval=self.approval(), dest=self.base / 'ev2', ssh_ready_seconds=0)
        supervisor = supervise_program.Supervisor(args, supervise_program.load_host_approval(args.approval, 'pod123', 18000), ssh=DeadSsh(),
                                                  clock=FakeClock(), sleep=FakeClock().sleep, stop_pod=lambda pod: stops.append(pod) or {})
        self.assertEqual(supervisor.run(), supervise_program.EXIT_FAILURE)
        self.assertEqual(stops, ['pod123'])
        self.assertIn('error', json.loads((self.base / 'ev2' / 'supervisor.json').read_text()))


class CandidateServingTests(Temp):
    def stage_adapter(self, rank=16):
        root = self.base / 'ws'
        directory = root / 'program' / 'run-abc' / 'stages' / 'green' / 'adapter'
        directory.mkdir(parents=True)
        (directory / 'adapter_model.safetensors').write_bytes(b'lora')
        (directory / 'adapter_config.json').write_text(json.dumps({'r': rank}))
        selection = self.base / 'selection.json'
        selection.write_text(json.dumps({'schema': serve_adapter.CANDIDATE_SCHEMA, 'adapter_dir': str(directory), 'served_model_name': 'lilith',
                                         'adapter_model_sha256': sha(b'lora'), 'max_lora_rank': 16, 'run_id': 'run-abc'}))
        return root, selection

    def test_candidate_selection_serves_one_pinned_uncertified_adapter_under_program(self):
        root, selection = self.stage_adapter()
        chosen = serve_adapter.select_candidate(selection, root)
        self.assertEqual((chosen.mode, chosen.lora_rank, chosen.max_lora_rank), ('adapter', 16, 16))
        self.assertIn('UNCERTIFIED', chosen.reason)

    def test_candidate_refuses_hash_path_and_rank_drift(self):
        root, selection = self.stage_adapter()
        data = json.loads(selection.read_text())
        for key, value in (('adapter_model_sha256', 'f' * 64), ('adapter_dir', str(self.base)), ('max_lora_rank', 32), ('served_model_name', 'x')):
            selection.write_text(json.dumps({**data, key: value}))
            with self.assertRaises(serve_adapter.SelectionError):
                serve_adapter.select_candidate(selection, root)

    def test_serve_candidate_shell_refuses_missing_inputs(self):
        script = PROGRAM / 'serve_candidate.sh'
        env = {**os.environ, 'LILITH_ALLOW_NON_WORKSPACE': '1'}
        bad_stage = subprocess.run(['bash', str(script), 'start', '--root', str(self.base), '--run-id', 'run-abc', '--stage', 'purple',
                                    '--main-sha', MAIN_SHA], capture_output=True, text=True, env=env, input='')
        self.assertEqual(bad_stage.returncode, 1)
        self.assertIn('lilith::candidate::usage', bad_stage.stderr)
        no_adapter = subprocess.run(['bash', str(script), 'start', '--root', str(self.base), '--run-id', 'run-abc', '--stage', 'green',
                                     '--main-sha', MAIN_SHA], capture_output=True, text=True, env=env, input='')
        self.assertIn('lilith::candidate::adapter', no_adapter.stderr)
        root, _ = self.stage_adapter()
        no_key = subprocess.run(['bash', str(script), 'start', '--root', str(root), '--run-id', 'run-abc', '--stage', 'green',
                                 '--main-sha', MAIN_SHA], capture_output=True, text=True, env=env, input='short\n')
        self.assertIn('lilith::candidate::api_key', no_key.stderr)
        self.assertEqual(subprocess.run(['bash', str(script), 'stop', '--root', str(root)], capture_output=True, text=True, env=env).stdout.strip(),
                         '{"stopped": true, "was_running": false}')


try:
    import torch
    import transformers  # noqa: F401
    import peft  # noqa: F401
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedForConditionalGeneration
    HAVE_STACK = True
except ImportError:   # pragma: no cover - CPU laptop without the training stack
    HAVE_STACK = False


@unittest.skipUnless(HAVE_STACK, 'torch/transformers/peft not installed')
class TorchEngineTests(Temp):
    """The real engine on a tiny random Gemma-4: steps, checkpoint round trip, adapter write-once."""

    def engine(self):
        text = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                    head_dim=16, global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True, sliding_window=8,
                    layer_types=['sliding_attention'] * 3 + ['full_attention'], final_logit_softcapping=30.0, max_position_embeddings=4096)
        config = Gemma4UnifiedConfig(text_config=text, vision_config=dict(mm_embed_dim=32, output_proj_dims=64), audio_config=dict(audio_embed_dim=32))
        config._attn_implementation = 'eager'
        torch.manual_seed(0)
        trainer_config = tp.trainer.TrainerConfig(rank=16, base_precision='fp32', ce_chunk_tokens=5, attention_query_chunk=3, seed=11)
        tp.trainer.register_chunked_attention(3)
        base = Gemma4UnifiedForConditionalGeneration(config).float()
        base.set_attn_implementation(tp.trainer.ATTENTION_IMPLEMENTATION)
        model, surface = tp.trainer.prepare_lora_model(base, trainer_config)
        import lilith_numerics
        return tp.TorchEngine(model, trainer_config, {'surface': surface}, lilith_numerics.NumericsConfig()), trainer_config

    def examples(self):
        rng = random.Random(3)
        sft = [tp.trainer.build_sft_example([rng.randint(3, 500) for _ in range(6)], [rng.randint(3, 500) for _ in range(8)], source=f's{i}')
               for i in range(2)]
        pair = tp.trainer.PreferencePair(tuple(rng.randint(3, 500) for _ in range(6)), (11, 12, 13), (21, 22), 1.0, 'p0')
        return sft, [pair]

    def test_steps_checkpoint_round_trip_and_adapter(self):
        engine, config = self.engine()
        sft, pairs = self.examples()
        first = engine.sft_step(sft, lr=1e-3, step=0)
        second = engine.preference_step(pairs, lr=1e-3, step=1)
        self.assertTrue(all(map(lambda o: o.loss == o.loss and o.grad_norm > 0, (first, second))))
        directory = self.base / 'ckpt'
        directory.mkdir()
        engine.save_training_state(directory)
        before = tp.trainer.adapter_tensor_sha256(engine.model)
        engine.sft_step(sft, lr=1e-3, step=2)
        self.assertNotEqual(tp.trainer.adapter_tensor_sha256(engine.model), before)
        engine.load_training_state(directory)
        self.assertEqual(tp.trainer.adapter_tensor_sha256(engine.model), before)
        replay_a = engine.sft_step(sft, lr=1e-3, step=2).loss
        engine.load_training_state(directory)
        replay_b = engine.sft_step(sft, lr=1e-3, step=2).loss
        self.assertAlmostEqual(replay_a, replay_b, places=5)                  # optimizer state restored too
        dev = engine.dev_metrics(sft, pairs)
        self.assertGreater(dev['sft_supervised_tokens'], 0)
        record = engine.save_adapter(self.base / 'adapter')
        self.assertIn('adapter_model.safetensors', record['files_sha256'])


if __name__ == '__main__':
    unittest.main()
