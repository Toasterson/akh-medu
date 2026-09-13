"""Finite calls derived only from public goals and successful observed receipts.

This is an explicitly assisted interface, not a planner. It never receives the
fixture state, examiner target, operation labels or desired outcome.
"""
import copy
import csv
import hashlib
import io
import json
import re

from command_templates import SLOT, from_help


def wire(request):
    return '<tool_call>\n'+json.dumps(dict(name=request['tool'],arguments=request['arguments']))+'\n</tool_call>'


class TypedChoices:
    def __init__(self,commands,public_goal):
        self.commands=tuple(commands);self.receipts={};self.records={}
        self.ids={'job':set(re.findall(r'\bjob-[A-Za-z0-9-]+\b',public_goal)),'worker':set()}
        self.sources={('job',j):'public_goal' for j in self.ids['job']}

    def observe(self,request,response,stages=None):
        if not response.get('success') or not response.get('evidence_id'):return
        tool=request['tool'];rid=response['evidence_id']
        if (tool=='terminal' and stages and len(stages)==1 and len(stages[0])==2
            and stages[0][0] in self.commands and stages[0][1] in {'--help','--version'}):
            self.receipts[tuple(stages[0])]=copy.deepcopy(response)
        if tool=='learn_command' and 'subject' in response:
            self.records[response['subject']]=copy.deepcopy(response['record'])
        if tool=='list_commands':self.records=copy.deepcopy(response['records'])
        stdout=response.get('stdout','')
        try:
            value=json.loads(stdout);rows=[value] if isinstance(value,dict) else []
        except ValueError:
            rows=[]
            # The fixture's three documented inventory delimiters; headers type
            # the identifiers. No positional/target inference or state access.
            for delimiter in ['\t',',',':']:
                reader=csv.DictReader(io.StringIO(stdout),delimiter=delimiter)
                if reader.fieldnames and {'job','worker'}<=set(reader.fieldnames):
                    rows=list(reader);break
        for row in rows:
            for kind in self.ids:
                value=row.get(kind)
                if isinstance(value,str) and value and len(value)<=2048 and '\x00' not in value:
                    self.ids[kind].add(value);self.sources[(kind,value)]=rid

    def choices(self):
        choices=[]
        def add(tool,description,**arguments):
            choices.append(dict(request=dict(tool=tool,arguments=arguments),description=description))
        for command in self.commands:
            for flag in ['--help','--version']:
                add('terminal','Read installed command documentation or version.',command=command+' '+flag)
            help_receipt=self.receipts.get((command,'--help'));version=self.receipts.get((command,'--version'))
            if help_receipt and version:
                header=version['stdout'].strip().split(maxsplit=1)
                header_line=f'{command} version {header[1]}.' if len(header)==2 else None
                for line_number,line in enumerate(help_receipt['stdout'].splitlines(),1):
                    if line==header_line:continue
                    try:record=from_help(help_receipt,version,line_number,set(self.commands))
                    except ValueError:continue
                    add('learn_command',record['source_line']+' — '+record['description'],command=command,line_number=line_number)
        add('list_commands','Retrieve the persistent command registry.')
        for subject,record in self.records.items():
            slots={SLOT.fullmatch(w)[1] for w in record['argv'] if SLOT.fullmatch(w)}
            arguments=[''] if not slots else (sorted(self.ids.get(next(iter(slots)),[])) if len(slots)==1 else [])
            for argument in arguments:
                add('invoke_command',record['source_line']+' — '+record['description'],subject=subject,argument=argument)
        add('report','Continue without making an outcome claim.',message='Continuing; see the recorded observations.')
        add('finish','Stop without making an outcome claim.',message='Stopping; the execution trace records what I observed and changed.')
        # Stable ordering has no information about the goal or correct operation.
        choices.sort(key=lambda c:hashlib.sha256(wire(c['request']).encode()).hexdigest())
        return choices

    def evidence(self):
        return copy.deepcopy(dict(records=self.records,identifiers={k:sorted(v) for k,v in self.ids.items()},
                    identifier_sources=[dict(kind=k,value=v,source=s) for (k,v),s in sorted(self.sources.items())],
                    discovery_receipts={' '.join(k):v for k,v in self.receipts.items()}))


class TokenChoiceMask:
    """Prefix trie over complete canonical calls; never chooses a semantic answer."""
    def __init__(self,tokenizer,requests):
        self.trie={};self.base_length=None;self.forced_steps=0;self.decision_steps=0
        self.eos=set(tokenizer.eos_token_ids)
        for request in requests:
            tokens=tokenizer.encode(wire(request),add_special_tokens=False)
            if tokenizer.decode(tokens)!=wire(request):raise ValueError('Choice tokenization must round-trip exactly')
            node=self.trie
            for token in tokens:node=node.setdefault(token,{})
            node[None]=True

    def allowed(self,generated):
        node=self.trie
        for index,token in enumerate(generated):
            # MLX computes one token ahead before yielding EOS to its caller.
            if token in self.eos and None in node and all(t in self.eos for t in generated[index:]):
                return sorted(self.eos)
            if token not in node:raise ValueError('Generated prefix left the prepared choice set')
            node=node[token]
        return [k for k in node if k is not None]+(sorted(self.eos) if None in node else [])

    def __call__(self,tokens,logits):
        import mlx.core as mx
        # Installed MLX calls processors after prompt prefill. The first call
        # can contain only the final prompt token, so learn its actual length.
        if self.base_length is None:self.base_length=len(tokens)
        allowed=self.allowed(tokens[self.base_length:].tolist())
        if not allowed:raise ValueError('No valid continuation')
        self.forced_steps+=len(allowed)==1;self.decision_steps+=len(allowed)>1
        result=mx.full_like(logits,-float('inf'));indices=mx.array(allowed)
        result[:,indices]=logits[:,indices]
        return result
