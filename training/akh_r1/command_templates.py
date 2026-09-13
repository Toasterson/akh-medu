"""Exact documented templates; candidate chooses both the template and bindings."""
import json
import re
import shlex


SLOT = re.compile(r'^\{([a-z][a-z0-9_]*)\}$')


def from_help(receipt, version_receipt, line_number, allowed):
    if not receipt.get('success') or not version_receipt.get('success'):
        raise ValueError('Successful help and version receipts are required')
    if not isinstance(receipt.get('evidence_id'), str) or not receipt['evidence_id']:
        raise ValueError('Missing source identity')
    lines = receipt.get('stdout', '').splitlines()
    if type(line_number) is not int or not 1 <= line_number < len(lines):
        raise ValueError('Choose a one-based line with a following description')
    line = lines[line_number-1]
    argv = shlex.split(line)
    if not argv or argv[0] not in allowed:
        raise ValueError('Selected line is outside the documented command set')
    version = version_receipt.get('stdout', '').strip()
    if not version.startswith(argv[0]+' '):
        raise ValueError('Version receipt is for another command')
    for word in argv:
        if ('{' in word or '}' in word) and not SLOT.fullmatch(word):
            raise ValueError('Only whole-argument placeholders are supported')
    return dict(schema='akh-command-template-v1', source=receipt['evidence_id'],
                source_line_number=line_number, source_line=line, argv=argv,
                description=lines[line_number], version=version)


def bind(record, bindings):
    argv = record.get('argv')
    if (record.get('schema') != 'akh-command-template-v1' or not isinstance(argv, list)
        or not argv or not all(isinstance(w, str) and w for w in argv)
        or shlex.split(record.get('source_line', '')) != argv):
        raise ValueError('Template differs from its recorded source line')
    slots = {SLOT.fullmatch(w)[1] for w in argv if SLOT.fullmatch(w)}
    if (not isinstance(bindings, dict) or set(bindings) != slots
        or not all(isinstance(v, str) and v and '\x00' not in v and len(v) <= 2048
                   for v in bindings.values())):
        raise ValueError('Supply exactly the required nonempty bindings')
    return [bindings[SLOT.fullmatch(w)[1]] if SLOT.fullmatch(w) else w for w in argv]


def store(graph, subject, record):
    return graph.call('memory_write', dict(summary='Exact documented command template.',
        source=record['source'], facts=[dict(subject=subject, predicate='command_template',
                                           object=json.dumps(record, sort_keys=True))]))


def read(graph, subject):
    # Structured Engine export avoids trying to parse the human-readable query renderer.
    response = graph.call('exam_export')
    values = [t['object_label'] for t in response.get('triples', [])
              if t['subject_label'] == subject and t['predicate_label'] == 'command_template']
    if not response.get('success') or len(values) != 1:
        raise ValueError('Select one unambiguous stored command subject')
    return json.loads(values[0])


def invoke(graph, fixture, subject, bindings):
    record = read(graph, subject)
    argv = bind(record, bindings)
    version = fixture.pipeline([[argv[0], '--version']])
    if not version.get('success') or version.get('stdout', '').strip() != record['version']:
        raise ValueError('Installed command version differs from the stored template')
    return dict(argv=argv, version_check=version, response=fixture.pipeline([argv]))


def single_binding(record, argument):
    """Project a single candidate argument to the only slot; never infer a value."""
    slots = {SLOT.fullmatch(w)[1] for w in record['argv'] if SLOT.fullmatch(w)}
    if len(slots) == 1 and isinstance(argument, str) and argument:
        return {next(iter(slots)): argument}
    if not slots and argument == '':
        return {}
    raise ValueError('Flat invocation needs one placeholder, or an empty argument for a zero-slot command')


def conditional_from_help(help_receipt,version_receipt):
    """Admit observed typed syntax without supplying task-specific arguments."""
    if not help_receipt.get('success') or not help_receipt.get('evidence_id') or not version_receipt.get('success') or not version_receipt.get('evidence_id'):
        raise ValueError('Sourced help and version required')
    document=json.loads(help_receipt['stdout'])
    if set(document) not in ({'command','version','modes'},{'command','version','modes','acquisition'}) or document['version']!=version_receipt['stdout'].strip():raise ValueError('Help/version mismatch')
    if not document['version'].startswith(document['command']+' '):raise ValueError('Wrong command version')
    for mode,spec in document['modes'].items():
        if not mode or set(spec)!={'argv','requires_receipt'} or type(spec['requires_receipt']) is not bool:raise ValueError('Invalid mode contract')
        if not spec['argv'] or spec['argv'][0]!=document['command']:raise ValueError('Wrong executable')
        for word in spec['argv']:
            if ('{' in word or '}' in word) and not SLOT.fullmatch(word):raise ValueError('Only whole argument slots')
    if 'acquisition' in document:
        acquisition=document['acquisition']
        if set(acquisition)!={'argv','receipt_fields','validity'} or acquisition['receipt_fields']!=['ticket','target','version'] or acquisition['validity']!='current-target-and-version-single-use':raise ValueError('Unsupported capability contract')
        argv=acquisition['argv']
        if not isinstance(argv,list) or not argv or argv[0]!=document['command'] or argv.count('{target}')!=1 or any(('{' in w or '}' in w) and w!='{target}' for w in argv):raise ValueError('Invalid acquisition syntax')
    return dict(schema='akh-conditional-command-v1',document=document,source=help_receipt['evidence_id'],version_source=version_receipt['evidence_id'])


def bind_conditional(record,mode,arguments,current_version,receipt=None):
    if record.get('schema')!='akh-conditional-command-v1' or not record.get('source') or not record.get('version_source'):raise ValueError('Unobserved command contract')
    doc=record['document']
    if current_version!=doc['version'] or mode not in doc['modes']:raise ValueError('Unknown mode or stale documentation')
    spec=doc['modes'][mode];values=dict(arguments)
    if spec['requires_receipt']:
        if not receipt or not receipt.get('success') or not receipt.get('evidence_id') or receipt.get('version')!=current_version or receipt.get('target')!=values.get('target') or not receipt.get('ticket'):raise ValueError('Current target-bound capability receipt required')
        if 'ticket' in values:raise ValueError('Ticket must come from observed receipt')
        values['ticket']=receipt['ticket']
    elif receipt is not None:raise ValueError('Receipt supplied to non-capability mode')
    argv=spec['argv'];template=dict(schema='akh-command-template-v1',argv=argv,source_line=shlex.join(argv))
    return bind(template,values)


def generate_command_descriptor(split,seed):
    """Finite grammar descriptors; seed/alias changes earn no world credit."""
    import random
    if split not in ('development','validation'):raise ValueError('Unknown split')
    rng=random.Random(seed);width=rng.randint(1,3);version='sana '+str(rng.randint(2,9))
    if split=='development':
        modes={'direct':dict(argv=['sana','repair','{target}'],requires_receipt=False),'scoped':dict(argv=['sana','repair','{target}','--scope','local']+sum(([f'--require-{i}','yes'] for i in range(width)),[]),requires_receipt=False)}
        mechanism='conditional_required_arguments';selected='scoped'
    else:
        modes={'commit':dict(argv=['sana','commit','{target}']+sum(([f'--require-{i}','yes'] for i in range(width)),[])+['--ticket','{ticket}'],requires_receipt=True)}
        mechanism='observed_capability_dependency';selected='commit'
    return dict(schema='akh-command-distribution-v1',split=split,seed=seed,mechanism=mechanism,
      public_start=dict(goal='Discover the documented command and repair the observed target; verify the effect',command='sana'),
      document=dict(command='sana',version=version,modes=modes),
      delivery=dict(initial=['observe_target','read_help','read_version'],cold=['read_current_version','observe_target'],later_trigger='after_initial_verified_effect',later=['version_change','new_target_observation'],capability_observation=['sana','inspect','{target}'] if split=='validation' else None),
      examiner=dict(target='unit-a',later_target='unit-b',selected_mode=selected,required_pairs=width,expected_effect_count=1,requires_ticket=split=='validation'),
      controls=['no-help','stale-version']+(['consumed-ticket'] if split=='validation' else ['missing-required-argument','mixed-mode-arguments']),
      assistance=['Typed help interpretation','Explicit target/version authority','Declared inspect receipt protocol'],qualified=False,new_world_credit=0)


def bind_acquisition(record,target,current_version):
    if record.get('schema')!='akh-conditional-command-v1' or not record.get('source') or not record.get('version_source'):raise ValueError('Observed help required')
    doc=record['document']
    if current_version!=doc['version'] or 'acquisition' not in doc:raise ValueError('Missing current acquisition documentation')
    argv=doc['acquisition']['argv']
    return bind(dict(schema='akh-command-template-v1',argv=argv,source_line=shlex.join(argv)),dict(target=target))
