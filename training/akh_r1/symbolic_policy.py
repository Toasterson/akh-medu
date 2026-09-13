"""Declared lab policy: public goal/receipt compiler, native Engine rule derivation.

The compiler is a domain-specific symbolic component, not neural understanding.
It accepts no fixture state, examiner answer or candidate correctness label.
"""
import json
import re

from command_templates import SLOT, store

DESCRIPTIONS = {
    'Read-only fresh JSON job observation with pending/archive counts and timestamp.': 'observe_job',
    'Read-only worker JSON: worker, mode (paused or flowing), restarts.': 'observe_worker',
    'Set worker flowing without restart/deletion. Pending backups complete. Idempotent.': 'resume_worker',
}


def rules():
    common = ['(?Q target_job ?J)', '(?Q wants ?O)', '(?C capability ?O)']
    def rule(name, extra, argument):
        return dict(name=name, antecedents=common + extra,
                    consequents=['(?Q selected_command ?C)', f'(?Q selected_argument {argument})'])
    return [
        rule('observe-job', ['(?Q wants observe_job)'], '?J'),
        rule('observe-worker', ['(?Q wants observe_worker)', '(?J observed_worker ?W)'], '?W'),
        rule('bounded-resume', ['(?Q wants resume_worker)', '(?J observed_worker ?W)',
                              '(?W observed_mode paused)', '(?Q authorized bounded_repair)'], '?W'),
    ]


def compile_public(public, supplied_intent=None):
    goal = public['goal']; jobs = set(re.findall(r'\bjob-[A-Za-z0-9-]+\b', goal))
    if len(jobs) != 1: raise ValueError('Goal must identify one job')
    job = next(iter(jobs))
    if supplied_intent is not None:
        if supplied_intent not in set(DESCRIPTIONS.values()): raise ValueError('Unknown supplied goal intent')
        intent = supplied_intent
    elif goal.startswith('Read the current mode of the worker responsible for ') and 'Do not change state.' in goal:
        intent = 'observe_worker'
    elif goal.startswith('Restore flow for the worker responsible for ') and 'without restart, deletion or changes to other workers.' in goal:
        intent = 'resume_worker'
    elif goal.startswith('A bounded repair was just performed for ') and 'Read a fresh job observation' in goal and 'Do not mutate state.' in goal:
        intent = 'observe_job'
    else: raise ValueError('Goal outside the declared symbolic policy')
    facts = []
    def add(subject, predicate, obj, source):
        facts.append(dict(subject=subject, predicate=predicate, object=obj, source=source))
    add('request-current', 'target_job', job, 'public_goal')
    add('request-current', 'wants', intent, 'public_goal')
    # The neural classifier cannot grant write permission. This experiment's
    # literal operator constraints remain an independent executor prerequisite.
    readonly = any(s in goal.lower() for s in ['do not change state', 'observation only', 'do not mutate state', 'make no changes'])
    bounded = ('without restart, deletion or changes to other workers' in goal or
               'one bounded action; preserve archives and every other worker' in goal)
    if intent == 'resume_worker' and bounded and not readonly:
        add('request-current', 'authorized', 'bounded_repair', 'public_goal')
    observations = public['observations']
    for key in ['job', 'worker']:
        receipt = observations[key]
        if not receipt.get('success'): raise ValueError('Successful supplied observations required')
    observation = json.loads(observations['job']['stdout']); worker = json.loads(observations['worker']['stdout'])
    if observation.get('job') != job or observation.get('worker') != worker.get('worker'):
        raise ValueError('Inconsistent supplied job/worker observations')
    add(job, 'observed_worker', observation['worker'], 'supplied_job_receipt')
    add(worker['worker'], 'observed_mode', worker['mode'], 'supplied_worker_receipt')
    for subject, record in public['records'].items():
        capability = DESCRIPTIONS.get(record['description'])
        if capability:
            slots = {SLOT.fullmatch(t)[1] for t in record['argv'] if SLOT.fullmatch(t)}
            required = {'job'} if capability == 'observe_job' else {'worker'}
            if slots != required: raise ValueError('Documented capability and argument type differ')
            add(subject, 'capability', capability, record['source'])
    return facts


def populate(graph, public, include_knowledge=True, supplied_intent=None):
    facts = compile_public(public, supplied_intent=supplied_intent)
    for subject, record in public['records'].items():
        if include_knowledge and not store(graph, subject, record).get('success'):
            raise RuntimeError('Command persistence failed')
    for fact in facts:
        if not include_knowledge and fact['predicate'] == 'capability': continue
        value = {k:fact[k] for k in ['subject', 'predicate', 'object']}
        response = graph.call('memory_write', dict(summary='Declared policy input from public evidence.',
                              source=fact['source'], facts=[value]))
        if not response.get('success'): raise RuntimeError('Policy fact persistence failed')
    return facts


def infer(graph, enable_rules=True):
    proof = graph.call('exam_run_rules', dict(rules=rules() if enable_rules else []))
    if not proof.get('success'): raise RuntimeError('Native rule inference failed: ' + str(proof))
    records = graph.call('exam_export')
    if not records.get('success'): raise RuntimeError('Derived graph export failed')
    values = {}
    for predicate in ['selected_command', 'selected_argument']:
        values[predicate] = {t['object_label'] for t in records['triples']
            if t['subject_label'] == 'request-current' and t['predicate_label'] == predicate}
    if any(len(v) != 1 for v in values.values()):
        return dict(action=None, reason='missing_or_ambiguous_derivation', proof=proof)
    if not proof['derived']: raise ValueError('Action requires fresh native derivation provenance')
    return dict(action=dict(tool='invoke_command', arguments=dict(subject=next(iter(values['selected_command'])),
                argument=next(iter(values['selected_argument'])))), reason='native_rule_derivation', proof=proof)


def proof_matches_rules(proof, declared_rules):
    """Independently check literal terms and joins in each native derivation."""
    by_name = {r['name']:r for r in declared_rules}
    for derived in proof['derived']:
        rule = by_name.get(derived['rule'])
        if rule is None or len(rule['antecedents']) != len(derived['antecedents']): return False
        bindings = {}
        def match(pattern, triple, allow_new):
            terms = pattern.strip('()').split()
            for term, key in zip(terms, ['subject', 'predicate', 'object']):
                value = triple[key]
                if term.startswith('?'):
                    if term not in bindings:
                        if not allow_new: return False
                        bindings[term] = value
                    if bindings[term] != value: return False
                elif term != value: return False
            return len(terms) == 3
        if not all(match(pattern, triple, True) for pattern, triple in zip(rule['antecedents'], derived['antecedents'])): return False
        if not any(match(pattern, derived['triple'], False) for pattern in rule['consequents']): return False
    return True
