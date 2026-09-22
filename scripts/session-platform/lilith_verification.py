"""Independent, bounded verification for the first local coding fixture."""
import ast
import hashlib


def verify_clamp(source):
    """Admit only a small pure function before evaluating the fixture's code."""
    if len(source.encode()) > 4096:
        raise ValueError('fixture source exceeds 4 KiB')
    tree = ast.parse(source)
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.If, ast.Compare, ast.Constant, ast.Name, ast.Load, ast.Lt,
               ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.UnaryOp, ast.USub,
               ast.Expr, ast.Assign, ast.Store, ast.IfExp)
    if any(not isinstance(node, allowed) for node in ast.walk(tree)):
        raise ValueError('fixture only admits a pure comparison/return function')
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError('expected exactly one function')
    fn = tree.body[0]
    if (fn.name != 'clamp_u8' or fn.decorator_list or fn.args.defaults
            or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg
            or fn.args.kwonlyargs or fn.args.posonlyargs or len(fn.args.args) != 1):
        raise ValueError('expected plain clamp_u8(value)')
    namespace = {'__builtins__': {}, 'int': int}
    exec(compile(tree, '<lilith-clamp-fixture>', 'exec'), namespace)
    values = list(range(-1024, 1025)) + [-(2 ** 63), 2 ** 63 - 1]
    for value in values:
        actual = namespace['clamp_u8'](value)
        if type(actual) is not int or actual != max(0, min(255, value)):
            raise AssertionError(f'clamp_u8({value}) returned {actual!r}')
    return {'passed': True, 'cases': len(values), 'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'scope': 'integer clamp fixture only; not general code execution or illumos validation'}



def verify_u64_range(source):
    """A bounded Python reference fixture for unsigned systems range arithmetic."""
    import itertools
    if len(source.encode()) > 4096:
        raise ValueError('fixture source exceeds 4 KiB')
    tree = ast.parse(source)
    allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return,
               ast.If, ast.Compare, ast.Constant, ast.Name, ast.Load, ast.Lt,
               ast.Gt, ast.LtE, ast.GtE, ast.Eq, ast.NotEq, ast.UnaryOp, ast.USub,
               ast.Not, ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.Add, ast.Sub,
               ast.Assign, ast.Store, ast.IfExp)
    nodes = list(ast.walk(tree))
    if any(not isinstance(node, allowed) for node in nodes):
        raise ValueError('range fixture only admits bounded integer comparisons and addition/subtraction')
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or sum(isinstance(n, ast.FunctionDef) for n in nodes) != 1):
        raise ValueError('expected exactly one function')
    fn = tree.body[0]
    if (fn.name != 'range_within_u64' or fn.decorator_list or fn.args.defaults
            or fn.args.kw_defaults or fn.args.vararg or fn.args.kwarg
            or fn.args.kwonlyargs or fn.args.posonlyargs or len(fn.args.args) != 3):
        raise ValueError('expected plain range_within_u64(start, length, limit)')
    maximum = (1 << 64) - 1
    if any(isinstance(n, ast.Constant) and
           (type(n.value) not in (int, bool) or abs(n.value) > maximum + 1) for n in nodes):
        raise ValueError('range fixture literals must be bounded integers or booleans')
    namespace = {'__builtins__': {}, 'int': int, 'bool': bool}
    exec(compile(tree, '<lilith-u64-range-fixture>', 'exec'), namespace)
    values = [-1, 0, 1, 2, 255, 256, (1 << 32)-1, 1 << 32,
              (1 << 63)-1, 1 << 63, maximum-1, maximum, maximum+1]
    cases = list(itertools.product(values, repeat=3))
    # Dense small ranges catch off-by-one errors independently of large boundaries.
    cases.extend(itertools.product(range(9), repeat=3))
    for start, length, limit in cases:
        # Python's unbounded sum is an independent oracle for the subtraction implementation.
        expected = (all(0 <= v <= maximum for v in (start, length, limit))
                    and start <= limit and start + length <= limit)
        actual = namespace['range_within_u64'](start, length, limit)
        if type(actual) is not bool or actual != expected:
            raise AssertionError(f'range_within_u64({start}, {length}, {limit}) returned {actual!r}; expected {expected}')
    return {'passed': True, 'cases': len(cases),
            'source_sha256': hashlib.sha256(source.encode()).hexdigest(),
            'scope': 'Python unsigned-64 range reference fixture; not C overflow, ABI, kernel or native illumos validation'}
