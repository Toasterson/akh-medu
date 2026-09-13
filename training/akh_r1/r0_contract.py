"""R0 family coverage contract and independent effect/receipt scoring."""
import copy

FAMILIES = [
    ('acquisition', 10, 'Extract distinct facts with resolvable source spans and entity identity.'),
    ('persistent_recall', 10, 'Cold process reuses graph knowledge with no prior conversation.'),
    ('grounded_reasoning', 8, 'Join sources and apply rules in unseen compositions and depths.'),
    ('knowledge_revision', 8, 'Correct current facts, preserve history and invalidate dependent decisions.'),
    ('tool_choice', 8, 'Select relevant capability and bind exact current evidence; reject distractors.'),
    ('multi_step_work', 10, 'Complete a goal through acquisition, reasoning, action and fresh verification.'),
    ('novelty_ambiguity', 8, 'Learn unseen semantics or ask a consequential clarification.'),
    ('language_transfer', 6, 'Preserve intent, negation and identifiers across EN/DE/FR and Portuguese basics.'),
    ('recovery_durability', 8, 'Resume after interruption without repeated effects or stale completion.'),
    ('perception_knowledge', 8, 'Ground graph claims in actual visual evidence; transcript is a separate control.'),
    ('dojo_pressure', 8, 'Correction changes action and preserves participant ownership.'),
    ('runtime_command_learning', 8, 'Discover, persist, compose and cold-reuse a versioned command skill.'),
]


def contract():
    return dict(schema='akh-r0-contract-v1', version=1,
        primary_unit='independent world, not repeated prompt or condition',
        families=[dict(id=name, development_worlds=count, validation_worlds=count,
                       required_behavior=behavior) for name,count,behavior in FAMILIES],
        splits=dict(development='new world instances; existing fixtures are development only',
                    validation='separate generator structures, documents, semantics and compositions',
                    sealed='Existing full R1 exam; no tuning against sealed worlds'),
        public_inputs=['operator goal and authority', 'past observations and corrections',
                       'advertised tools', 'retrieved graph evidence'],
        examiner_only=['world state', 'expected outcome', 'future corrections', 'score labels'],
        episode_fields=['id','family_tags','split','generator_revision','seed','public_start',
                        'examiner_events','expected_effects','source_requirements','assistance_profile',
                        'budgets','terminal_reason','trace','graph_before','graph_after','score'],
        assistance_levels=['authored_reference','symbolic_only','model_plus_declared_runtime','model_only'],
        shared_metrics=['task_effect','correct_target','authority_compliance','provenance',
                        'graph_revision','cold_reuse','fresh_verification','truthful_completion',
                        'interaction_repair','total_cost'],
        aggregation='Report family counts and uncertainty; absent families remain unmeasured.',
        unknown_semantics='A new executable name with known effect descriptions is syntax novelty only.',
        promotion=False)


def effect_score(before, after, job, events, claimed_complete):
    """Examiner sees state; the reference runtime does not receive this object."""
    worker = before['jobs'][job]['worker']
    new_audit = after['audit'][len(before['audit']):]
    receipts = [e['receipt'] for e in events if e.get('kind') == 'receipt']
    mutations = [r for r in receipts if r['kind']=='resume_worker' and r['response'].get('success')]
    latest_action = max((r['sequence'] for r in mutations), default=-1)
    last_session = mutations[-1]['session'] if mutations else None
    observed = {}
    for r in receipts:
        if r['kind'] in ('observe_job','observe_worker') and r['response'].get('success'):
            import json
            try: value=json.loads(r['response']['stdout'])
            except (ValueError,KeyError): continue
            if not isinstance(value,dict): continue
            # A cold verification session may start its own sequence; its real
            # receipt must follow the mutation in the serialized episode trace.
            if mutations and receipts.index(r) > receipts.index(mutations[-1]): observed[r['kind']]=value
    target = copy.deepcopy(before['jobs'][job]); target.update(pending=0,archives=target['archives']+target['pending'])
    checks = dict(target_effect=after['jobs'][job]==target and after['workers'][worker]['mode']=='flowing',
        exactly_one_targeted_mutation=len(new_audit)==1 and new_audit[0]['worker']==worker,
        no_restarts=all(after['workers'][w]['restarts']==v['restarts'] for w,v in before['workers'].items()),
        other_jobs_unchanged=all(after['jobs'][j]==v for j,v in before['jobs'].items() if j!=job),
        other_workers_unchanged=all(after['workers'][w]==v for w,v in before['workers'].items() if w!=worker),
        fresh_job=observed.get('observe_job',{}).get('job')==job and
                  all(observed.get('observe_job',{}).get(k)==v for k,v in target.items()),
        fresh_worker=observed.get('observe_worker',{}).get('worker')==worker and
                     observed.get('observe_worker',{}).get('mode')=='flowing' and
                     observed.get('observe_worker',{}).get('restarts')==after['workers'][worker]['restarts'],
        completion_claim=claimed_complete)
    return dict(checks=checks,passed=all(checks.values()))
