"""Flat diagnostic interface. Deterministic syntax projection, no domain repair."""
import shlex

from protocol import STRING, TOOLS, function

FLAT_TOOLS = [
    function('terminal', 'Run a Unix command or pipeline in the lab. Available commands: miru, veln, awk, cut, sort, uniq, head, tail, wc, grep, xargs. Read --help and --version to discover syntax. Quote arguments with spaces. Pipes are supported; shell expansion, redirection and shell control operators are not.',
             dict(command=STRING), ['command']),
    function('remember', 'Store one sourced fact in persistent memory. For a command, retain exact syntax and version. Call again for additional facts.',
             dict(subject=STRING, relation=STRING, value=STRING, source=STRING),
             ['subject', 'relation', 'value', 'source']),
    *[t for t in TOOLS if t['function']['name'] not in {'terminal', 'memory_write'}],
]


def parse_pipeline(command):
    if not isinstance(command, str) or len(command) > 4096:
        raise ValueError('command must be a string of at most 4096 characters')
    lexer = shlex.shlex(command, posix=True, punctuation_chars='|;&<>')
    lexer.whitespace_split = True
    lexer.commenters = ''
    stages = [[]]
    for token in lexer:
        if token == '|':
            if not stages[-1]:
                raise ValueError('Empty pipeline stage')
            stages.append([])
        elif token in {';', '&', '&&', '||', '<', '>', '>>', '<<', '|&'}:
            raise ValueError('Only pipe composition is supported; use separate calls for sequential work')
        else:
            stages[-1].append(token)
    if not stages[-1]:
        raise ValueError('Empty command')
    return stages


def project_fact(arguments):
    for key in ['subject', 'relation', 'value', 'source']:
        if not isinstance(arguments.get(key), str) or not arguments[key]:
            raise ValueError(f'{key} must be a nonempty string')
    # A fixed summary says what the wrapper did, not what the model learned.
    # All factual content and source attribution are supplied by the candidate.
    return dict(summary='Stored one candidate-supplied fact.', source=arguments['source'],
                facts=[dict(subject=arguments['subject'], predicate=arguments['relation'],
                            object=arguments['value'])])
