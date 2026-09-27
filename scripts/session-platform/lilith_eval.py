"""Offline, family-separated apprenticeship evaluation. Never launches models."""
import argparse
import ast
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

ARMS = ('none', 'text', 'vsa', 'independent_reference')
FAMILIES = ('alignment', 'interval_intersection', 'modular_sequence_order')
COMMON_INSTRUCTION = ('Return only one Python function, with no Markdown fences or explanatory prose. '
    'Use integer/bool literals, local assignment, if/else, return, comparisons, and/or/not, '
    'conditional expressions, +, -, *, //, %, and bitwise & only. No calls (including min/max), '
    'imports, loops, shifts, exponentiation, decorators, helper functions, containers or strings. '
    'Keep the source under 4096 bytes. These are local Python reference tasks, not native OS validation.')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def verifier_sha256():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def source_fingerprint(source):
    return digest(ast.dump(ast.parse(source), include_attributes=False))


def task_spec(family, bits):
    limit = (1 << bits) - 1
    domain = f'Inputs are integers. Reject any input outside 0..{limit} by returning -1. '
    if family == 'alignment':
        return domain + ('Write align_up(value: int, alignment: int) -> int. Also reject alignment zero '
            'or alignment not a power of two. Return the smallest multiple of alignment at least value, '
            f'or -1 if the result exceeds {limit}.')
    if family == 'interval_intersection':
        return domain + ('Write overlap_size(a_start: int, a_end: int, b_start: int, b_end: int) -> int. '
            'Reject an end smaller than its start. Return the number of integer positions in the '
            'intersection of half-open intervals [a_start,a_end) and [b_start,b_end). Empty intervals '
            'and touching endpoints have zero overlap.')
    if family == 'modular_sequence_order':
        return domain + ('Write ring_distance(start: int, end: int, modulus: int) -> int. Also reject '
            'modulus zero or either position at least modulus. Return the forward distance from start '
            'to end wrapping at modulus; equal positions have distance zero.')
    raise ValueError('unknown task family')


def validate_manifest(m):
    if m.get('schema') != 'lilith/evaluation@2':
        raise ValueError('unsupported manifest')
    if m.get('claims') != {'qwen_certified': False, 'illumos_native': False, 'sealed': False}:
        raise ValueError('this development harness cannot certify release claims')
    if m.get('common_instruction') != COMMON_INSTRUCTION:
        raise ValueError('common instruction drift')
    if m['verifier_sha256'] != verifier_sha256():
        raise ValueError('verifier changed; freeze a new manifest')
    if m['arms'] != list(ARMS):
        raise ValueError('all three retrieval arms and independent reference are required')
    train, held = set(m['training_families']), set(m['held_out_families'])
    if train & held or held != set(FAMILIES):
        raise ValueError('families must be separated and registered')
    if not m['seeds'] or len(set(m['seeds'])) != len(m['seeds']) or any(type(s) is not int or s < 0 for s in m['seeds']):
        raise ValueError('distinct nonnegative seeds required')
    budget = m['budget']
    if set(budget) != {'attempts', 'tools', 'max_new_tokens', 'context_tokens', 'lesson_bytes', 'wall_seconds'}:
        raise ValueError('exact common budget required')
    if budget['attempts'] != 1 or budget['tools'] != 0 or any(type(v) is not int or v < 0 for v in budget.values()):
        raise ValueError('one attempt and zero tools required')
    if any(budget[k] <= 0 for k in ('max_new_tokens','context_tokens','lesson_bytes','wall_seconds')):
        raise ValueError('positive resource caps required')
    models = m['models']
    for role in ('candidate','reference'):
        for key in ('sha256','tokenizer_sha256'):
            value = models[role][key]
            if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('model and tokenizer hashes must be pinned')
    if models['candidate']['sha256'] == models['reference']['sha256']:
        raise ValueError('reference must be independent of candidate')
    if models['reference'].get('upstream_qwen_equivalence') is not False:
        raise ValueError('local reference equivalence remains unresolved')
    ids = set()
    for task in m['tasks']:
        if task['id'] in ids or task['family'] not in held or task['bits'] not in (8,16):
            raise ValueError('duplicate or invalid task')
        ids.add(task['id'])
        if task['prompt'] != task_spec(task['family'],task['bits']) or task['prompt_sha256'] != digest(task['prompt']):
            raise ValueError('task specification drift')
    expected = set(itertools.product(FAMILIES,(8,16)))
    if {(t['family'],t['bits']) for t in m['tasks']} != expected or len(ids) != len(expected):
        raise ValueError('all six family/width tasks are required')
    return m


def schedule(manifest):
    m = validate_manifest(manifest)
    rows = []
    for index, (task, seed) in enumerate(itertools.product(m['tasks'], m['seeds'])):
        rotation = index % len(ARMS)
        for arm in ARMS[rotation:] + ARMS[:rotation]:
            rows.append(dict(task=task['id'], arm=arm, seed=seed,
                model_role='reference' if arm == 'independent_reference' else 'candidate',
                fresh_session=True, budget=m['budget']))
    return rows


class UnsupportedSource(ValueError):
    pass


def compile_bounded(source, family):
    """Interpret a tiny arithmetic language; no exec, imports, calls or loops."""
    if len(source.encode()) > 4096:
        raise UnsupportedSource('source exceeds 4096 bytes')
    tree = ast.parse(source)
    allowed = (ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,
        ast.Assign,ast.Name,ast.Load,ast.Store,ast.Constant,ast.Compare,ast.Lt,ast.Gt,
        ast.LtE,ast.GtE,ast.Eq,ast.NotEq,ast.BoolOp,ast.And,ast.Or,ast.UnaryOp,
        ast.USub,ast.UAdd,ast.Not,ast.BinOp,ast.Add,ast.Sub,ast.Mult,ast.FloorDiv,
        ast.Mod,ast.BitAnd,ast.IfExp)
    nodes = list(ast.walk(tree))
    if len(nodes) > 512 or any(not isinstance(n,allowed) for n in nodes):
        raise UnsupportedSource('unsupported syntax or AST size')
    if len(tree.body) != 1 or not isinstance(tree.body[0],ast.FunctionDef):
        raise UnsupportedSource('one function required')
    fn = tree.body[0]
    signature = {'alignment':('align_up',['value','alignment']),
        'interval_intersection':('overlap_size',['a_start','a_end','b_start','b_end']),
        'modular_sequence_order':('ring_distance',['start','end','modulus'])}[family]
    if (fn.name != signature[0] or [a.arg for a in fn.args.args] != signature[1]
        or fn.decorator_list or fn.args.defaults or fn.args.kw_defaults or fn.args.vararg
        or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs
        or sum(isinstance(n,ast.FunctionDef) for n in nodes) != 1):
        raise UnsupportedSource('wrong function signature')
    for node in nodes:
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,bool) or abs(node.value) > 2**64):
            raise UnsupportedSource('unsupported literal')
        if isinstance(node,ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0],ast.Name)):
            raise UnsupportedSource('only simple local assignment allowed')
    binary = {ast.Add:lambda a,b:a+b,ast.Sub:lambda a,b:a-b,ast.Mult:lambda a,b:a*b,
        ast.FloorDiv:lambda a,b:a//b,ast.Mod:lambda a,b:a%b,ast.BitAnd:lambda a,b:a&b}
    compare = {ast.Lt:lambda a,b:a<b,ast.Gt:lambda a,b:a>b,ast.LtE:lambda a,b:a<=b,
        ast.GtE:lambda a,b:a>=b,ast.Eq:lambda a,b:a==b,ast.NotEq:lambda a,b:a!=b}
    def run(*args):
        env = dict(zip(signature[1],args))
        def expr(n):
            if isinstance(n,ast.Constant): value=n.value
            elif isinstance(n,ast.Name): value=env[n.id]
            elif isinstance(n,ast.BinOp): value=binary[type(n.op)](expr(n.left),expr(n.right))
            elif isinstance(n,ast.UnaryOp):
                value=expr(n.operand)
                value=not value if isinstance(n.op,ast.Not) else -value if isinstance(n.op,ast.USub) else +value
            elif isinstance(n,ast.Compare):
                left=expr(n.left);value=True
                for op,right_node in zip(n.ops,n.comparators):
                    right=expr(right_node)
                    if not compare[type(op)](left,right):value=False;break
                    left=right
            elif isinstance(n,ast.BoolOp):
                value=expr(n.values[0])
                for child in n.values[1:]:
                    if isinstance(n.op,ast.And) and not value:break
                    if isinstance(n.op,ast.Or) and value:break
                    value=expr(child)
            elif isinstance(n,ast.IfExp):value=expr(n.body if expr(n.test) else n.orelse)
            else:raise UnsupportedSource('unsupported expression')
            if type(value) not in (int,bool) or abs(value).bit_length() > 128:
                raise ArithmeticError('bounded interpreter numeric limit')
            return value
        def block(body):
            for node in body:
                if isinstance(node,ast.Return):return True,expr(node.value)
                if isinstance(node,ast.Assign):env[node.targets[0].id]=expr(node.value)
                elif isinstance(node,ast.If):
                    returned,value=block(node.body if expr(node.test) else node.orelse)
                    if returned:return True,value
                elif not isinstance(node,ast.Return):raise UnsupportedSource('unsupported statement')
            return False,None
        return block(fn.body)[1]
    return run


def cases(task):
    maximum = (1 << task['bits']) - 1
    values = sorted(set([-1,0,1,2,3,7,8,15,16,maximum//2,maximum-1,maximum,maximum+1]))
    arity = 4 if task['family'] == 'interval_intersection' else 2 if task['family'] == 'alignment' else 3
    for args in itertools.product(values,repeat=arity):
        invalid = any(v < 0 or v > maximum for v in args)
        if task['family'] == 'alignment':
            value,alignment = args
            invalid |= alignment == 0 or alignment & (alignment-1) != 0
            expected = -1 if invalid else ((value+alignment-1)//alignment)*alignment
            if expected > maximum:expected=-1
        elif task['family'] == 'interval_intersection':
            a,b,c,d = args
            invalid |= b < a or d < c
            expected = -1 if invalid else max(0,min(b,d)-max(a,c))
        else:
            start,end,modulus = args
            invalid |= modulus == 0 or start >= modulus or end >= modulus
            expected = -1 if invalid else (end-start)%modulus
        label = 'domain_rejection' if invalid else 'boundary_semantics' if any(v in (0,maximum) for v in args) else 'arithmetic_semantics'
        yield args,expected,label


def verify(source, task):
    try:
        function = compile_bounded(source,task['family'])
    except SyntaxError as error:
        return dict(passed=False,category='syntax',detail=str(error),cases=0)
    except (UnsupportedSource, RecursionError) as error:
        return dict(passed=False,category='unsupported_syntax',detail=str(error),cases=0)
    count = 0
    for args,expected,label in cases(task):
        count += 1
        try:actual = function(*args)
        except (ArithmeticError,KeyError,UnsupportedSource,TypeError,RecursionError) as error:
            return dict(passed=False,category='execution',detail=str(error),input=args,cases=count)
        if type(actual) is not int or actual != expected:
            return dict(passed=False,category=label,input=args,expected=expected,actual=actual,cases=count)
    return dict(passed=True,category='passed',cases=count,source_sha256=digest(source))


def validate_record(m, row):
    tasks = {t['id']:t for t in m['tasks']}
    if row['task'] not in tasks or row['arm'] not in ARMS or row['seed'] not in m['seeds']:
        raise ValueError('record outside declared schedule')
    if row['manifest_sha256'] != digest(canonical(m)) or row['budget'] != m['budget']:
        raise ValueError('manifest or budget mismatch')
    role = 'reference' if row['arm'] == 'independent_reference' else 'candidate'
    if row['model_sha256'] != m['models'][role]['sha256'] or row['tokenizer_sha256'] != m['models'][role]['tokenizer_sha256']:
        raise ValueError('model/tokenizer mismatch')
    usage = row['usage']
    for key,cap in [('attempts','attempts'),('tools','tools'),('output_tokens','max_new_tokens'),('input_tokens','context_tokens')]:
        if usage[key] is None and key in ('output_tokens','input_tokens'):
            continue  # Explicitly unmeasured; never substitute a cap or estimate.
        if type(usage[key]) is not int or not 0 <= usage[key] <= m['budget'][cap]:
            raise ValueError('usage exceeds common budget: '+key)
    if usage['attempts'] != 1 or type(usage['wall_seconds']) not in (int,float) or not 0 <= usage['wall_seconds'] <= m['budget']['wall_seconds']:
        raise ValueError('invalid attempt or time budget')
    context = row['retrieval']['context']
    lessons = row['retrieval']['lessons']
    if len(context.encode()) > m['budget']['lesson_bytes']:
        raise ValueError('lesson context exceeds budget')
    if row['arm'] in ('none','independent_reference') and (context or lessons):
        raise ValueError('baseline/reference must not receive lessons')
    if bool(context) != bool(lessons):
        raise ValueError('context and lesson provenance must agree')
    if len({lesson['id'] for lesson in lessons}) != len(lessons):
        raise ValueError('duplicate retrieval provenance')
    for lesson in lessons:
        if lesson['family'] not in m['training_families'] or lesson['family'] in m['held_out_families']:
            raise ValueError('retrieval family leakage')
        if lesson['source_sha256'] in m['excluded_source_sha256']:
            raise ValueError('retrieval source leakage')
    source = row.get('source','')
    if source:
        try: fingerprint=source_fingerprint(source)
        except (SyntaxError,RecursionError):fingerprint=None
        if digest(source) in m['training_source_sha256'] or fingerprint in m['training_source_fingerprints']:
            raise ValueError('generated solution duplicates training source')
    if row['status'] not in ('completed','timeout','runtime_error','cancelled'):
        raise ValueError('unknown execution status')
    return tasks[row['task']]


def evaluate(manifest, records):
    m = validate_manifest(manifest)
    outcomes = {}
    for row in records:
        task=validate_record(m,row)
        key=(row['task'],row['seed'],row['arm'])
        if key in outcomes:raise ValueError('duplicate attempt')
        if row['status'] != 'completed':
            outcome=dict(passed=False,category=row['status'],cases=0)
        elif row.get('finish_reason') == 'length':
            outcome=dict(passed=False,category='output_budget',cases=0)
        else:outcome=verify(row.get('source',''),task)
        outcomes[key]=outcome
    summary={}
    for arm in ARMS:
        rows=[v for k,v in outcomes.items() if k[2]==arm]
        summary[arm]=dict(observed=len(rows),expected=len(m['tasks'])*len(m['seeds']),
            passed=sum(r['passed'] for r in rows),failures=dict(Counter(r['category'] for r in rows if not r['passed'])))
    pairs={}
    for arm in ARMS[1:]:
        counts=Counter()
        for task,seed in itertools.product(m['tasks'],m['seeds']):
            a,b=outcomes.get((task['id'],seed,'none')),outcomes.get((task['id'],seed,arm))
            if a is None or b is None:counts['missing']+=1;continue
            counts['both_pass' if a['passed'] and b['passed'] else 'both_fail' if not a['passed'] and not b['passed'] else 'arm_only_pass' if b['passed'] else 'baseline_only_pass']+=1
        pairs[arm]=dict(counts)
    return dict(schema='lilith/evaluation-report@2',manifest_sha256=digest(canonical(m)),
        complete=len(outcomes)==len(schedule(m)),
        budget_usage_complete=bool(records) and all(r['usage']['input_tokens'] is not None and r['usage']['output_tokens'] is not None for r in records),
        arms=summary,paired_against_none=pairs,
        outcomes=[dict(task=k[0],seed=k[1],arm=k[2],**v) for k,v in sorted(outcomes.items())],
        claims=dict(statistical_superiority=False,qwen_parity=False,native_illumos=False),
        interpretation='Descriptive development results only; seeds are repeated observations, not independent task families.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--results',type=Path,help='JSON array of execution records; omit to emit schedule')
    args=parser.parse_args()
    m=json.loads(args.manifest.read_text())
    result=evaluate(m,json.loads(args.results.read_text())) if args.results else schedule(m)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':main()
