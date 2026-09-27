"""Independent checker for the development-only bounded-capacity transfer task."""
import ast
import hashlib
import itertools


def check(source):
    source=source.strip()
    if source.startswith('```python\n') and source.endswith('```'):
        source=source[len('```python\n'):-3].strip()
    elif source.startswith('```\n') and source.endswith('```'):
        source=source[4:-3].strip()
    if len(source.encode())>4096:raise ValueError('source exceeds 4 KiB')
    tree=ast.parse(source)
    allowed=(ast.Module,ast.FunctionDef,ast.arguments,ast.arg,ast.Return,ast.If,ast.Compare,
        ast.Constant,ast.Name,ast.Load,ast.Lt,ast.Gt,ast.LtE,ast.GtE,ast.Eq,ast.NotEq,
        ast.UnaryOp,ast.USub,ast.Not,ast.BoolOp,ast.And,ast.Or,ast.BinOp,ast.Add,ast.Sub,
        ast.Assign,ast.Store,ast.IfExp)
    nodes=list(ast.walk(tree))
    if any(not isinstance(n,allowed) for n in nodes):raise ValueError('unsupported executable syntax')
    if len(tree.body)!=1 or sum(isinstance(n,ast.FunctionDef) for n in nodes)!=1:raise ValueError('one function required')
    fn=tree.body[0]
    if not isinstance(fn,ast.FunctionDef):raise ValueError('function required')
    if (fn.name!='can_reserve' or fn.decorator_list or fn.args.defaults or fn.args.kw_defaults
        or fn.args.vararg or fn.args.kwarg or fn.args.kwonlyargs or fn.args.posonlyargs
        or [a.arg for a in fn.args.args]!=['used','requested','capacity']):raise ValueError('wrong function signature')
    if any(isinstance(n,ast.Constant) and (type(n.value) not in (int,bool) or abs(n.value)>2**64) for n in nodes):raise ValueError('literal outside fixture domain')
    namespace={'__builtins__':{},'int':int,'bool':bool}
    exec(compile(tree,'<transfer-probe>','exec'),namespace)
    boundary=[-1,0,1,2,15,16,32767,65534,65535,65536]
    cases=list(itertools.product(boundary,repeat=3))+list(itertools.product(range(8),repeat=3))
    for used,requested,capacity in cases:
        expected=all(0<=v<=65535 for v in (used,requested,capacity)) and used<=capacity and used+requested<=capacity
        actual=namespace['can_reserve'](used,requested,capacity)
        if type(actual) is not bool or actual!=expected:
            raise AssertionError(f'can_reserve({used}, {requested}, {capacity}) = {actual!r}; expected {expected}')
    return dict(passed=True,cases=len(cases),source_sha256=hashlib.sha256(source.encode()).hexdigest(),source=source)
