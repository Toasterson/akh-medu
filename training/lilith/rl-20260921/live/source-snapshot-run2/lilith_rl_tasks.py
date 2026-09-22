"""Public development RL tasks with interpreted, bounded Python semantics."""
import ast
from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path

GRAMMAR = ('Output only Python source defining f(a, b), with optional int/bool annotations and no defaults. '
           'Allowed: integer/bool literals, local assignments, if/else, return, comparisons, '
           'and/or/not, + - * // % & | ^ ~ << >>. No imports, calls, loops or other syntax. '
           'All intermediate integers must fit signed 128 bits; shift counts must be 0..63.')


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    split: str
    prompt: str
    reference: str
    negative_control: str
    cases: tuple


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def tasks():
    byte_values = (-256, -1, 0, 1, 2, 7, 8, 15, 16, 127, 128, 254, 255, 256, 511)
    strict_bytes = tuple(itertools.product(byte_values, repeat=2))
    # Cases contain independently computed expected outputs; never supplied to models.
    def make(id, family, split, spec, ref, wrong, inputs, oracle):
        return Task(id, family, split, spec+' '+GRAMMAR, ref, wrong,
                    tuple((a, b, oracle(a, b)) for a,b in inputs))
    return (
        make('mask-intersection-u8/1', 'permission_masks', 'train',
             'Inputs a,b are arbitrary integers. Return -1 if either is outside 0..255; otherwise return their bit-mask intersection as an integer.',
             'def f(a, b):\n    if not (0 <= a <= 255 and 0 <= b <= 255):\n        return -1\n    return a & b\n',
             'def f(a, b):\n    return a & b\n', strict_bytes,
             lambda a,b: -1 if a not in range(256) or b not in range(256) else sum(2**i for i in range(8) if a//2**i%2 and b//2**i%2)),
        make('required-flags-u8/1', 'permission_masks', 'train',
             'Inputs a,b are guaranteed integers in 0..255. Return bool: all flags in required mask b are present in available mask a. Zero required flags succeeds. No out-of-domain validation is required.',
             'def f(a, b):\n    return a & b == b\n',
             'def f(a, b):\n    return a & b != 0\n', tuple(itertools.product(range(32), repeat=2)),
             lambda a,b: {i for i in range(8) if b//2**i%2} <= {i for i in range(8) if a//2**i%2}),
        make('compose-le16/1', 'byte_order', 'train',
             'Inputs a,b are arbitrary integers. Return -1 if either is outside 0..255; otherwise decode two little-endian bytes a then b as an unsigned integer.',
             'def f(a, b):\n    if not (0 <= a <= 255 and 0 <= b <= 255):\n        return -1\n    return a + 256 * b\n',
             'def f(a, b):\n    return a + 256 * b\n', strict_bytes,
             lambda a,b: int.from_bytes(bytes((a,b)), 'little') if a in range(256) and b in range(256) else -1),
        make('select-le16-byte/1', 'byte_order', 'train',
             'Input a is guaranteed an unsigned 16-bit integer; b is guaranteed 0 or 1. Return byte b of a in little-endian order as an integer. No out-of-domain validation is required.',
             'def f(a, b):\n    return (a >> (8 * b)) & 255\n',
             'def f(a, b):\n    return (a >> (8 * (1 - b))) & 255\n', tuple(itertools.product(tuple(range(0,65536,1021)) + (0,1,255,256,257,32767,32768,65534,65535), (0,1))),
             lambda a,b: a.to_bytes(2, 'little')[b]),
        make('table-slot-offset/1', 'table_index', 'development_holdout',
             'Inputs a,b are arbitrary integers. A table has b slots, each 8 bytes. Return byte offset of slot a. Return -1 unless 0 <= b <= 255 and 0 <= a < b.',
             'def f(a, b):\n    if not (0 <= b <= 255 and 0 <= a < b):\n        return -1\n    return a * 8\n',
             'def f(a, b):\n    if 0 <= a <= b:\n        return a * 8\n    return -1\n', strict_bytes,
             lambda a,b: tuple(range(0,b*8,8))[a] if b in range(256) and a in range(b) else -1),
        make('table-slot-valid/1', 'table_index', 'development_holdout',
             'Inputs a,b are guaranteed integers with 0 <= a <= 255 and 0 <= b <= 255. Return bool: a is a valid zero-based slot index for a table with b slots. A zero-slot table has no valid index.',
             'def f(a, b):\n    return a < b\n',
             'def f(a, b):\n    return a <= b\n', tuple(itertools.product(range(32), repeat=2)),
             lambda a,b: a in tuple(range(b))),
        make('resource-unit-bytes/1', 'resource_units', 'development_holdout',
             'Inputs a,b are arbitrary integers. Return bytes occupied by a units of b bytes each. Return -1 unless 0 <= a <= 31 and 1 <= b <= 16.',
             'def f(a, b):\n    if not (0 <= a <= 31 and 1 <= b <= 16):\n        return -1\n    return a * b\n',
             'def f(a, b):\n    return a * b\n', tuple(itertools.product(range(-1,34), range(-1,19))),
             lambda a,b: sum([b]*a) if a in range(32) and b in range(1,17) else -1),
        make('complete-resource-units/1', 'resource_units', 'development_holdout',
             'Inputs a,b are guaranteed integers with 0 <= a <= 255 and 1 <= b <= 16. Return the count of complete b-byte units fitting in a bytes. Discard leftover bytes; no out-of-domain validation is required.',
             'def f(a, b):\n    return a // b\n',
             'def f(a, b):\n    return (a + b - 1) // b\n', tuple(itertools.product(range(256), range(1,17))),
             lambda a,b: len(tuple(range(b,a+1,b)))),
    )


class Rejected(ValueError):
    pass


class Interpreter:
    """No exec/eval: fixed operation/node budgets and bounded intermediate ints."""
    def __init__(self, source):
        if len(source.encode()) > 4096:
            raise Rejected('source_budget')
        try:
            self.tree = ast.parse(source)
        except (SyntaxError, ValueError, RecursionError) as error:
            raise Rejected('syntax') from error
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Assign,
            ast.Return, ast.If, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
            ast.UnaryOp, ast.BoolOp, ast.Compare, ast.Add, ast.Sub, ast.Mult,
            ast.FloorDiv, ast.Mod, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Invert,
            ast.LShift, ast.RShift, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
        nodes=list(ast.walk(self.tree))
        if len(nodes)>512 or any(not isinstance(n,allowed) for n in nodes):
            raise Rejected('grammar')
        if len(self.tree.body)!=1 or not isinstance(self.tree.body[0],ast.FunctionDef):
            raise Rejected('signature')
        fn=self.tree.body[0]; args=fn.args
        def annotation_ok(value):
            return value is None or isinstance(value,ast.Name) and value.id in ('int','bool')
        if (fn.name!='f' or [a.arg for a in args.args]!=['a','b'] or fn.decorator_list
                or not annotation_ok(fn.returns) or args.defaults or args.kw_defaults or args.kwonlyargs
                or args.posonlyargs or args.vararg or args.kwarg
                or any(not annotation_ok(a.annotation) for a in args.args)
                or sum(isinstance(n,ast.FunctionDef) for n in nodes)!=1):
            raise Rejected('signature')
        for n in nodes:
            if isinstance(n,ast.Constant): self.bound(n.value)
            if isinstance(n,ast.Assign) and (len(n.targets)!=1 or not isinstance(n.targets[0],ast.Name)):
                raise Rejected('assignment')
        self.body=fn.body

    def bound(self,value):
        if type(value) not in (int,bool) or not -(2**127)<=value<2**127:
            raise Rejected('integer_budget')
        return value

    def step(self):
        self.budget-=1
        if self.budget<0: raise Rejected('operation_budget')

    def expression(self,node):
        self.step()
        if isinstance(node,ast.Constant): return node.value
        if isinstance(node,ast.Name):
            if node.id not in self.env: raise Rejected('undefined_name')
            return self.env[node.id]
        if isinstance(node,ast.UnaryOp):
            value=self.expression(node.operand)
            if isinstance(node.op,ast.Not): return not value
            if isinstance(node.op,ast.USub): return self.bound(-value)
            if isinstance(node.op,ast.UAdd): return self.bound(+value)
            return self.bound(~value)
        if isinstance(node,ast.BoolOp):
            result=self.expression(node.values[0])
            for value in node.values[1:]:
                if isinstance(node.op,ast.And) and not result: break
                if isinstance(node.op,ast.Or) and result: break
                result=self.expression(value)
            return result
        if isinstance(node,ast.Compare):
            left=self.expression(node.left)
            for op,term in zip(node.ops,node.comparators):
                right=self.expression(term)
                valid = (left==right if isinstance(op,ast.Eq) else left!=right if isinstance(op,ast.NotEq)
                    else left<right if isinstance(op,ast.Lt) else left<=right if isinstance(op,ast.LtE)
                    else left>right if isinstance(op,ast.Gt) else left>=right)
                if not valid:return False
                left=right
            return True
        if isinstance(node,ast.BinOp):
            left,right=self.expression(node.left),self.expression(node.right);op=node.op
            if isinstance(op,(ast.LShift,ast.RShift)) and not 0<=right<=63: raise Rejected('shift_budget')
            if isinstance(op,(ast.FloorDiv,ast.Mod)) and right==0: raise Rejected('division_by_zero')
            result=(left+right if isinstance(op,ast.Add) else left-right if isinstance(op,ast.Sub)
                else left*right if isinstance(op,ast.Mult) else left//right if isinstance(op,ast.FloorDiv)
                else left%right if isinstance(op,ast.Mod) else left&right if isinstance(op,ast.BitAnd)
                else left|right if isinstance(op,ast.BitOr) else left^right if isinstance(op,ast.BitXor)
                else left<<right if isinstance(op,ast.LShift) else left>>right)
            return self.bound(result)
        raise Rejected('expression')

    def block(self,body):
        for node in body:
            self.step()
            if isinstance(node,ast.Return):return True,self.expression(node.value)
            if isinstance(node,ast.Assign):self.env[node.targets[0].id]=self.expression(node.value)
            elif isinstance(node,ast.If):
                done,value=self.block(node.body if self.expression(node.test) else node.orelse)
                if done:return done,value
        return False,None

    def __call__(self,a,b):
        self.env={'a':a,'b':b};self.budget=1024
        done,result=self.block(self.body)
        if not done:raise Rejected('missing_return')
        return result


def verify(source,task):
    result=dict(passed=False,category=None,counterexample=None,cases=0,total_cases=len(task.cases),source_sha256=digest(source))
    try: interpreter=Interpreter(source)
    except (Rejected,RecursionError) as error:
        result['category']=str(error);return result
    for a,b,expected in task.cases:
        result['cases']+=1
        try:actual=interpreter(a,b)
        except (Rejected,RecursionError) as error:
            result.update(category=str(error),counterexample=dict(inputs=[a,b],expected=expected));return result
        if type(actual) is not type(expected) or actual!=expected:
            result.update(category='contract_mismatch',counterexample=dict(inputs=[a,b],expected=expected,actual=actual));return result
    result.update(passed=True,category='passed');return result


def export(path):
    all_tasks=tasks();rows=[];training=[]
    for task in all_tasks:
        good,bad=verify(task.reference,task),verify(task.negative_control,task)
        if not good['passed'] or bad['passed']:raise AssertionError(task.id)
        rows.append(dict(id=task.id,family=task.family,split=task.split,prompt=task.prompt,
            prompt_sha256=digest(task.prompt),source_sha256=digest(task.reference),
            cases_sha256=digest(canonical(task.cases)),reference_receipt=good,negative_control_receipt=bad))
        if task.split=='train':training.append(dict(id=task.id,family=task.family,input=task.prompt,output=task.reference,
            source_sha256=digest(task.reference),origin='assistant_authored_reference',teacher_model=None))
    manifest=dict(schema='lilith/rl-tasks@1',tasks=rows,training_targets=training,
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['public development holdout, not sealed or secret', 'family split does not imply unseen mathematical concepts',
            'training families permission_masks and byte_order were already exposed',
            'prior exposed families saturation, bounded_range, error_conventions; prior public evaluation alignment, interval_intersection, modular_sequence_order',
            'Python reference checks only; no native illumos validation', 'assistant-authored references; no teacher provenance or trained improvement claimed'])
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    return manifest


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps({'tasks':len(export(args.output)['tasks'])}))
