"""Persistent operator goals, runtime command learning and native completion proofs.

This is an explicit symbolic lab policy. Its inputs are an operator goal contract,
an advertised executable list, a real Engine and a subprocess-call capability.
It has no fixture state or examiner interface.
"""
import copy
import hashlib
import json
import secrets
import shlex

from command_registry import CommandRegistry, registry_records
from symbolic_policy import DESCRIPTIONS, proof_matches_rules


def validate_goal(goal):
    if (goal.get('schema') != 'akh-persistent-goal-v1' or
        not isinstance(goal.get('goal_id'), str) or not goal['goal_id'].startswith('goal-') or
        not isinstance(goal.get('target_job'), str) or not goal['target_job'].startswith('job-') or
        goal.get('desired') != dict(pending=0, worker_mode='flowing') or
        type((goal.get('desired') or {}).get('pending')) is not int or
        goal.get('constraints') != dict(no_restart=True, preserve_archives=True) or
        any(type(v) is not bool for v in (goal.get('constraints') or {}).values()) or
        type(goal.get('allow_repair')) is not bool or
        not isinstance(goal.get('source'), str) or not goal['source'].startswith('operator:')):
        raise ValueError('Unsupported or incomplete public goal contract')
    return copy.deepcopy(goal)


def goal_rules(completion=True, actions=True):
    result = []
    if actions:
        for need, capability, binding in [('job', 'observe_job', '?J'), ('worker', 'observe_worker', '?W'), ('repair', 'resume_worker', '?W')]:
            antecedents = ['(?X evaluates ?G)', f'(?X needs {need})', '(?X eligible_command ?C)', f'(?C capability {capability})']
            antecedents += ['(?G target_job ?J)'] if binding == '?J' else ['(?X bound_worker ?W)']
            if need == 'repair': antecedents += ['(?G repair_authorized true)']
            result.append(dict(name='next-'+need, antecedents=antecedents,
                consequents=['(?X selected_command ?C)', f'(?X selected_argument {binding})']))
    result += [
        dict(name='job-clear', antecedents=['(?X evaluates ?G)', '(?G target_job ?J)', '(?X job_receipt ?R)',
            '(?R observed_job ?J)', '(?R observed_worker ?W)', '(?X bound_worker ?W)', '(?R pending clear)'],
            consequents=['(?X job_clear true)']),
        dict(name='worker-flowing', antecedents=['(?X worker_receipt ?R)', '(?R observed_worker ?W)',
            '(?X bound_worker ?W)', '(?R observed_mode flowing)'], consequents=['(?X worker_flowing true)']),
    ]
    for kind in ['job', 'worker']:
        result += [
            dict(name='fresh-'+kind, antecedents=[f'(?X {kind}_receipt ?R)', '(?X session ?S)', '(?R receipt_session ?S)'],
                consequents=[f'(?X fresh_{kind} true)']),
            dict(name='after-action-'+kind, antecedents=[f'(?X {kind}_receipt ?R)', '(?R after_latest_action true)'],
                consequents=[f'(?X after_action_{kind} true)']),
        ]
    if completion:
        result.append(dict(name='goal-complete', antecedents=[f'(?X {p} true)' for p in
            ['job_clear', 'worker_flowing', 'fresh_job', 'fresh_worker', 'after_action_job',
             'after_action_worker', 'archives_preserved', 'restarts_preserved']],
            consequents=['(?X goal_complete proved)']))
    return result


class PipelinePort:
    def __init__(self, call): self.pipeline = call


class PersistentGoalRuntime:
    def __init__(self, graph, call, commands, *, session=None, record_event=None):
        self.graph = graph; self.port = PipelinePort(call); self.commands = sorted(commands)
        self.session = session or 'session-' + secrets.token_hex(8)
        self.record_event = record_event or (lambda event: None)
        self.registry = CommandRegistry(graph, self.commands)
        self.state = None; self.versions = {}; self.evaluations = 0; self.learned = []

    def export(self):
        value = self.graph.call('exam_export')
        if not value.get('success'): raise RuntimeError('Engine export failed')
        return value['triples']

    def put(self, subject, predicate, obj, source):
        result = self.graph.call('memory_write', dict(summary='Persistent goal runtime evidence.', source=source,
            facts=[dict(subject=subject, predicate=predicate, object=str(obj))]))
        if not result.get('success'): raise RuntimeError('Engine evidence write failed')

    def get(self, subject, predicate):
        return [t['object_label'] for t in self.export() if t['subject_label'] == subject and t['predicate_label'] == predicate]

    def replace(self, subject, predicate, value, source):
        for old in self.get(subject, predicate):
            if str(value) == old: continue
            result = self.graph.call('kg_delete', dict(subject=subject, predicate=predicate, object=old))
            if not result.get('success'): raise RuntimeError('Current goal pointer retraction failed')
        if value is not None: self.put(subject, predicate, value, source)

    def save(self):
        goal = self.state['request']; payload = json.dumps(self.state, sort_keys=True)
        revision = goal['goal_id'] + '-revision-' + hashlib.sha256(payload.encode()).hexdigest()[:16]
        self.put(revision, 'goal_revision_payload', payload, goal['source'])
        self.put(goal['goal_id'], 'goal_revision', revision, goal['source'])
        self.replace(goal['goal_id'], 'goal_record', payload, goal['source'])
        self.replace(goal['goal_id'], 'target_job', goal['target_job'], goal['source'])
        self.replace(goal['goal_id'], 'repair_authorized', 'true' if goal['allow_repair'] else None, goal['source'])

    def open_goal(self, goal_id, public_goal=None):
        if public_goal is not None:
            goal = validate_goal(public_goal)
            if goal['goal_id'] != goal_id: raise ValueError('Goal identity mismatch')
            if self.get(goal_id, 'goal_record'): raise ValueError('Refusing to overwrite an existing goal')
            self.state = dict(request=goal, sequence=0, last_mutation_sequence=-1, baseline=None,
                              latest={}, status='active', completion_history=[])
            self.save()
        else:
            values = self.get(goal_id, 'goal_record')
            if len(values) != 1: raise ValueError('One persisted goal record required')
            self.state = json.loads(values[0]); validate_goal(self.state['request'])
            if self.state['request']['goal_id'] != goal_id: raise ValueError('Stored goal identity differs')
        self.record_event(dict(kind='goal_opened', session=self.session, goal=copy.deepcopy(self.state),
                               supplied_goal=public_goal is not None))

    def revoke_repair(self, source):
        if not source.startswith('operator:'): raise ValueError('Operator correction source required')
        self.state['request']['allow_repair'] = False
        self.state['request']['source'] = source
        self.state['sequence'] += 1; self.save()
        self.record_event(dict(kind='operator_correction', session=self.session, source=source,
                               allow_repair=False))

    def receipt(self, response, kind, *, stages=None, execution=None):
        self.state['sequence'] += 1
        # A new goal resets its sequence while the runtime session can continue.
        # Include goal identity so its observations cannot alias older evidence.
        identity = 'receipt-' + self.session + '-' + self.state['request']['goal_id'] + '-' + str(self.state['sequence'])
        response = dict(response, evidence_id=identity)
        record = dict(id=identity, sequence=self.state['sequence'], session=self.session,
                      kind=kind, response=response, stages=stages, execution=execution)
        self.put(identity, 'runtime_receipt', json.dumps(record, sort_keys=True), identity)
        self.record_event(dict(kind='receipt', receipt=record))
        self.save()
        return record

    def documentation(self, command, flag):
        stages = [[command, flag]]
        record = self.receipt(self.port.pipeline(stages), 'documentation_'+flag[2:], stages=stages)
        self.registry.observe(stages, record['response'])
        return record

    def current_records(self):
        return {subject:record for subject,record in registry_records(self.graph).items()
                if record['argv'][0] in self.versions and record['version'] == self.versions[record['argv'][0]]['response']['stdout'].strip()}

    def discover(self):
        before = registry_records(self.graph)
        self.record_event(dict(kind='registry_read', session=self.session, records=before))
        for command in self.commands:
            record = self.documentation(command, '--version')
            if not record['response'].get('success'): return 'version_query_failed'
            self.versions[command] = record
        def capabilities(): return {DESCRIPTIONS.get(r['description']) for r in self.current_records().values()}
        required = set(DESCRIPTIONS.values())
        if not required <= capabilities():
            for command in self.commands:
                help_record = self.documentation(command, '--help')
                if not help_record['response'].get('success'): continue
                lines = help_record['response']['stdout'].splitlines()
                for i, line in enumerate(lines[:-1]):
                    try: words = shlex.split(line)
                    except ValueError: continue
                    if not words or words[0] != command or lines[i+1] not in DESCRIPTIONS: continue
                    learned = self.registry.learn(command, i+1)
                    self.put(learned['subject'], 'learned_version_receipt', self.versions[command]['id'], help_record['id'])
                    self.learned.append(learned['subject'])
                    self.record_event(dict(kind='command_learned', session=self.session, result=learned))
        records = self.current_records()
        for subject, record in records.items():
            capability = DESCRIPTIONS.get(record['description'])
            if capability: self.put(subject, 'capability', capability, record['source'])
        self.record_event(dict(kind='registry_read', session=self.session, records=registry_records(self.graph), current_records=records))
        return None if required <= capabilities() else 'missing_current_command_knowledge'

    def observation(self, record, capability):
        # Mutation responses are not independent observation/verification receipts.
        if capability not in ['observe_job', 'observe_worker'] or not record['response'].get('success'): return
        try: value = json.loads(record['response']['stdout'])
        except (ValueError, TypeError): raise ValueError('Observation is not JSON') from None
        if not isinstance(value, dict): raise ValueError('Observation must be an object')
        if capability == 'observe_job':
            if (value.get('job') != self.state['request']['target_job'] or
                not isinstance(value.get('worker'), str) or
                any(type(value.get(k)) not in [int, float] for k in ['pending', 'archives', 'observed_at'])):
                raise ValueError('Invalid job observation identity or fields')
            self.state['latest']['job'] = dict(record, value=value)
        else:
            job = self.state['latest'].get('job')
            if (not job or value.get('worker') != job['value']['worker'] or
                value.get('mode') not in ['paused', 'flowing'] or type(value.get('restarts')) is not int):
                raise ValueError('Invalid worker observation identity or fields')
            self.state['latest']['worker'] = dict(record, value=value)
        job = self.state['latest'].get('job'); worker = self.state['latest'].get('worker')
        if (self.state['baseline'] is None and job and worker and
            job['session'] == worker['session'] == self.session):
            self.state['baseline'] = dict(worker=job['value']['worker'], archives=job['value']['archives'], restarts=worker['value']['restarts'])
        self.save()

    def snapshot(self, *, completion=True, actions=True):
        self.evaluations += 1; context = 'evaluation-' + self.session + '-' + str(self.evaluations)
        goal = self.state['request']; latest = self.state['latest']; baseline = self.state['baseline']
        facts = []
        def add(s,p,o,source): facts.append(dict(subject=s,predicate=p,object=str(o),source=source))
        add(context, 'evaluates', goal['goal_id'], goal['source']); add(context, 'session', self.session, 'runtime_session')
        records = self.current_records()
        for subject, record in records.items():
            add(context, 'eligible_command', subject, self.versions[record['argv'][0]]['id'])
        worker_id = latest.get('job', {}).get('value', {}).get('worker')
        if worker_id: add(context, 'bound_worker', worker_id, latest['job']['id'])
        for kind in ['job','worker']:
            if kind not in latest: continue
            record = latest[kind]; value = record['value']
            # Per-evaluation projection makes freshness relative to the current
            # latest action; a later action cannot reuse an old "after" assertion.
            projected = context + '-' + kind
            add(context, kind+'_receipt', projected, record['id'])
            add(projected, 'receipt_session', record['session'], record['id'])
            add(projected, 'observed_worker', value['worker'], record['id'])
            if record['sequence'] > self.state['last_mutation_sequence']:
                add(projected, 'after_latest_action', 'true', record['id'])
            if kind == 'job':
                add(projected, 'observed_job', value['job'], record['id'])
                add(projected, 'pending', 'clear' if value['pending'] == 0 else 'nonzero', record['id'])
                if baseline and value['archives'] >= baseline['archives']:
                    add(context, 'archives_preserved', 'true', record['id'])
            else:
                add(projected, 'observed_mode', value['mode'], record['id'])
                if baseline and value['worker'] == baseline['worker'] and value['restarts'] == baseline['restarts']:
                    add(context, 'restarts_preserved', 'true', record['id'])
        need = None
        for kind in ['job','worker']:
            record = latest.get(kind)
            if not record or record['session'] != self.session or record['sequence'] <= self.state['last_mutation_sequence']:
                need = kind; break
        if need is None and baseline and worker_id != baseline['worker']:
            raise ValueError('Observed job-worker identity changed from goal baseline')
        if need is None and latest['worker']['value']['mode'] == 'paused': need = 'repair'
        if need: add(context, 'needs', need, 'declared_goal_dependency_policy')
        for fact in facts: self.put(fact['subject'], fact['predicate'], fact['object'], fact['source'])
        declared = goal_rules(completion=completion, actions=actions)
        proof = self.graph.call('exam_run_rules', dict(rules=declared))
        if not proof.get('success') or not proof_matches_rules(proof, declared): raise RuntimeError('Invalid native goal proof')
        def values(predicate): return {d['triple']['object'] for d in proof['derived']
            if d['triple']['subject'] == context and d['triple']['predicate'] == predicate}
        complete = values('goal_complete') == {'proved'}
        commands = values('selected_command'); arguments = values('selected_argument')
        action = dict(subject=next(iter(commands)), argument=next(iter(arguments))) if len(commands)==len(arguments)==1 else None
        if len(commands)>1 or len(arguments)>1: raise ValueError('Ambiguous native action derivation')
        result = dict(context=context, session=self.session, facts=facts, rules=declared, proof=proof,
                      complete=complete, action=action, need=need, state=copy.deepcopy(self.state))
        self.record_event(dict(kind='evaluation', evaluation=result))
        return result

    def run(self, *, stop_after_repair=False, withhold_observations=False, completion=True, actions=True, max_actions=8):
        error = self.discover()
        if error: return self.finish('blocked', error)
        for _ in range(max_actions+1):
            evaluation = self.snapshot(completion=completion, actions=actions)
            if evaluation['complete']:
                self.state['completion_history'].append(dict(context=evaluation['context'], session=self.session,
                    source_receipts={k:r['id'] for k,r in self.state['latest'].items()}))
                return self.finish('complete', 'fresh_native_goal_proof')
            action = evaluation['action']
            if withhold_observations: return self.finish('blocked', 'fresh_observations_withheld')
            if action is None:
                reason = 'repair_authority_revoked' if evaluation['need']=='repair' and not self.state['request']['allow_repair'] else 'no_supported_native_action_or_completion'
                return self.finish('blocked', reason)
            if _ == max_actions: return self.finish('blocked', 'action_budget_exhausted')
            record = self.current_records()[action['subject']]; capability = DESCRIPTIONS[record['description']]
            try: execution = self.registry.invoke(self.port, **action)
            except ValueError as error: return self.finish('blocked', str(error))
            observed = self.receipt(execution['response'], capability, stages=[execution['argv']], execution=execution)
            if not observed['response'].get('success'): return self.finish('blocked', 'command_failed')
            if capability == 'resume_worker':
                self.state['last_mutation_sequence'] = observed['sequence']; self.save()
                if stop_after_repair: return self.finish('checkpoint', 'repair_completed_verification_pending')
            else: self.observation(observed, capability)
        raise AssertionError('Unreachable action budget')

    def finish(self, status, reason):
        self.state['status'] = status; self.save()
        result = dict(status=status, reason=reason, session=self.session, goal=copy.deepcopy(self.state),
                      learned_subjects=self.learned, records=registry_records(self.graph))
        self.record_event(dict(kind='finished', result=result))
        return result
