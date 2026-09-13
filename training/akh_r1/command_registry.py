"""Candidate-selected source lines, persistent exact templates, explicit retrieval."""
import hashlib
import json

from command_templates import from_help, invoke, read, single_binding, store
from protocol import STRING, function


def registry_records(graph):
    exported=graph.call('exam_export')
    if not exported.get('success'):
        raise ValueError('Cannot read command registry')
    records={}
    for triple in exported.get('triples',[]):
        if triple['predicate_label']!='command_template':
            continue
        subject=triple['subject_label']
        if subject in records:
            raise ValueError('Ambiguous command record')
        records[subject]=json.loads(triple['object_label'])
    return records


def registry_tools(commands):
    return [
        function('terminal',f'Run a documented Unix command in the lab. Installed commands: {", ".join(commands)}. '
                 'Use --help and --version for discovery. Help lines are one-based. All argv are literal.',
                 dict(command=STRING),['command']),
        function('learn_command','Register one command from a selected line of help already returned by terminal. '
                 'First read this executable with --help and --version. Select the one-based executable syntax line, '
                 'not a description. The host copies that exact line and returns a persistent subject ID. '
                 'Choose useful commands yourself; multiple registrations may be necessary.',
                 dict(command=STRING,line_number=dict(type='integer',minimum=1)),['command','line_number']),
        function('list_commands','Retrieve stored command subjects, exact syntax, descriptions and versions from the persistent KG. '
                 'Use this in a fresh conversation; no old command records are in the prompt.',{},[]),
        function('invoke_command','Invoke a stored command. Choose its subject from the registry and supply the actual worker or job '
                 'identifier as argument, not a whole command. Use an empty argument for a command with no placeholder. '
                 'The executor checks the installed version and substitutes one literal argv token.',
                 dict(subject=STRING,argument=STRING),['subject','argument']),
        function('report','Report observed progress to the operator and continue the task.',dict(message=STRING),['message']),
        function('finish','End with the verified result or an accurate explanation of what remains blocked.',dict(message=STRING),['message']),
    ]


class CommandRegistry:
    def __init__(self,graph,commands):
        self.graph=graph
        self.commands=set(commands)
        self.receipts={}

    def observe(self,stages,response):
        if (len(stages)==1 and len(stages[0])==2 and stages[0][0] in self.commands
            and stages[0][1] in {'--help','--version'} and response.get('success')
            and response.get('evidence_id')):
            self.receipts[tuple(stages[0])]=dict(response)

    def learn(self,command,line_number):
        if command not in self.commands:
            raise ValueError('Choose an installed command')
        try:
            help_receipt=self.receipts[(command,'--help')]
            version_receipt=self.receipts[(command,'--version')]
        except KeyError:
            raise ValueError('Read this command with --help and --version first') from None
        record=from_help(help_receipt,version_receipt,line_number,self.commands)
        # IDs are deterministic syntax/version identities; allocation chooses no operation.
        identity=json.dumps([record['argv'],record['version']],sort_keys=True)
        subject='command-'+hashlib.sha256(identity.encode()).hexdigest()[:12]
        existing=registry_records(self.graph)
        if subject in existing:
            if existing[subject]['argv']!=record['argv'] or existing[subject]['version']!=record['version']:
                raise ValueError('Command identity collision')
            return dict(success=True,subject=subject,record=existing[subject],already_present=True)
        result=store(self.graph,subject,record)
        if not result.get('success'):
            raise ValueError('Command registration failed')
        return dict(success=True,subject=subject,record=read(self.graph,subject),already_present=False)

    def invoke(self,fixture,subject,argument):
        return invoke(self.graph,fixture,subject,single_binding(read(self.graph,subject),argument))
