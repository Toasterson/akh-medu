"""Read-only replay of RL experiment receipts; no model execution or training."""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import lilith_rl as trajectory
import lilith_rl_tasks as exercises


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wrapper_diagnostic(source, task):
    """Unscored extraction of one complete fenced artifact; no program rewriting."""
    match = re.fullmatch(r"\s*```(?:python)?[ \t]*\n(.*?)\n```\s*", source, re.DOTALL)
    if not match or '```' in match.group(1):
        return None
    artifact = match.group(1)
    return dict(scored=False, transformation='remove one enclosing Markdown code fence',
                raw_sha256=trajectory.digest(source), artifact_sha256=trajectory.digest(artifact),
                verification=exercises.verify(artifact, task))


def audit(path):
    path = Path(path)
    config = json.loads((path / 'config.json').read_text())
    result = json.loads((path / 'result.json').read_text())
    for module, key in ((exercises, 'task_module_sha256'), (trajectory, 'trajectory_module_sha256')):
        if file_hash(module.__file__) != config[key]:
            raise ValueError('Replay module differs from frozen collection: ' + key)
    manifest = Path(__file__).resolve().parents[2] / 'training/lilith/rl-20260921/tasks.json'
    if file_hash(manifest) != config['task_manifest_sha256']:
        raise ValueError('Task manifest differs from collection')
    tasks = {task.id: task for task in exercises.tasks()}
    report = {'schema': 'lilith/rl-independent-replay@1', 'runner_status': result['status'],
              'quality_improvement_established': False, 'native_illumos': False,
              'arms': {}, 'outcomes': [], 'complete': True}
    for arm in ('base', 'sft', 'klpo'):
        record_path = path / (arm + '-trajectories.json')
        rows = json.loads(record_path.read_text()) if record_path.exists() else []
        seen = set()
        counts = {split: dict(observed=0, expected=sum(t.split == split for t in tasks.values()),
                              passed=0, completed=0, generated_tokens=0, failures=Counter())
                  for split in ('train', 'development_holdout')}
        for row in rows:
            task = tasks[row['task']['id']]
            if task.id in seen:
                raise ValueError('Repeated task within arm')
            seen.add(task.id)
            expected_task = dict(id=task.id, family=task.family, split=task.split,
                                 prompt_sha256=trajectory.digest(task.prompt))
            policy = dict(result['initial_policy'])
            if arm != 'base':
                if arm not in result:
                    raise ValueError('Incomplete arm lacks confirmed adapter fingerprint')
                policy['adapter_sha256'] = result[arm]['adapter_sha256']
                policy['version'] = arm + '-' + policy['adapter_sha256'][:16]
            trajectory.validate_record(row, expected_policy=policy,
                expected_verifier_sha256=config['task_module_sha256'], for_update=False,
                expected_task=expected_task)
            expected_evidence = exercises.verify(row['completion']['generated_text'], task)
            expected_evidence['generated_text_sha256'] = trajectory.digest(row['completion']['generated_text'])
            status = row['completion']['status']
            if status != 'completed':
                expected_evidence['artifact_verifier_passed_before_completion_gate'] = expected_evidence['passed']
                expected_evidence['passed'] = False
                expected_evidence['completion_gate'] = status
            if expected_evidence != row['reward']['evidence']:
                raise ValueError('Independent verifier replay differs: ' + row['id'])
            generated = sum(row['policy_mask'])
            if generated > config['max_new_tokens'] or len(row['token_ids']) > config['max_context_tokens']:
                raise ValueError('Recorded token budget exceeded')
            if any(len(samples) != config['mc_samples'] for active, samples in zip(row['policy_mask'], row['mc_token_ids']) if active):
                raise ValueError('Auxiliary sample count differs from protocol')
            if row['sampler']['termination_token_ids'] != config['eos_ids']:
                raise ValueError('Termination policy differs from protocol')
            passed = expected_evidence['passed']
            category = expected_evidence['category'] if status == 'completed' else status
            count = counts[task.split]
            count['observed'] += 1
            count['completed'] += status == 'completed'
            count['passed'] += passed
            count['generated_tokens'] += generated
            if not passed:
                count['failures'][category] += 1
            diagnostic = wrapper_diagnostic(row['completion']['generated_text'], task) if not passed else None
            report['outcomes'].append(dict(wrapper_diagnostic=diagnostic, arm=arm, task=task.id, split=task.split,
                passed=passed, category=category, counterexample=expected_evidence['counterexample'],
                generated_tokens=generated, record_sha256=row['record_sha256']))
        report['arms'][arm] = counts
        report['complete'] &= len(seen) == len(tasks)
    report['complete'] &= result['status'] == 'passed'
    report['updates'] = {arm: result.get(arm + '_update') for arm in ('sft', 'klpo')}
    report['interpretation'] = ('One public development task pack and paired RNG seeds; descriptive only. '
        'Supervised references and RL outcome rewards supply different information. '
        'Incomplete trajectories remain evaluation failures and cannot update.')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.run), indent=2))
