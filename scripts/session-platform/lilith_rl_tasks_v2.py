"""Versioned Lilith quality tasks and artifact/reward contract (v2)."""
import hashlib, itertools, json, re
from pathlib import Path
from lilith_rl_tasks import Task, Interpreter, Rejected, canonical, digest

GRAMMAR = ('Output Python source defining f(a, b), with optional int/bool annotations and no defaults. '
           'A single enclosing ```python fence is accepted but unnecessary. Allowed: integer/bool literals, local assignments, '
           'if/else, return, comparisons, and/or/not, + - * // % & | ^ ~ << >>. No imports, calls, loops or other syntax. '
           'All intermediate integers must fit signed 128 bits; shift counts must be 0..63.')

def _make(id, family, split, spec, ref, wrong, inputs, oracle):
    return Task(id, family, split, spec+' '+GRAMMAR, ref, wrong,
                tuple((a,b,oracle(a,b)) for a,b in inputs))

def tasks():
    edge=(-1025,-257,-256,-129,-128,-65,-64,-33,-32,-17,-16,-9,-8,-2,-1,0,1,2,7,8,15,16,31,32,63,64,127,128,255,256,1024)
    byte=(-1,0,1,2,7,8,15,16,31,63,64,127,128,254,255,256)
    return (
      _make('signed-clamp/2','bounded_normalization','train','Inputs are arbitrary integers. Return a clamped to the inclusive interval from -b to b; return -1 if b is outside 0..255.',
        'def f(a, b):\n    if not 0 <= b <= 255:\n        return -1\n    if a < -b:\n        return -b\n    if a > b:\n        return b\n    return a\n','def f(a, b):\n    if a < -b:\n        return -b\n    if a > b:\n        return b\n    return a\n',itertools.product(edge,byte),lambda a,b:-1 if b not in range(256) else max(-b,min(a,b))),
      _make('bounded-add/2','bounded_normalization','train','Inputs are arbitrary integers. If a or b is outside -128..127, return -1. Otherwise return their sum clamped to -128..127.',
        'def f(a, b):\n    if not (-128 <= a <= 127 and -128 <= b <= 127):\n        return -1\n    if a + b < -128:\n        return -128\n    if a + b > 127:\n        return 127\n    return a + b\n','def f(a, b):\n    return a + b\n',itertools.product(edge,repeat=2),lambda a,b:-1 if not(-128<=a<=127 and -128<=b<=127) else max(-128,min(127,a+b))),
      _make('align-up-power2/2','alignment','train','Inputs are arbitrary integers. Return the smallest multiple of b greater than or equal to a. Return -1 unless 0 <= a <= 65535 and b is one of 1,2,4,8,16,32,64.',
        'def f(a, b):\n    if not (0 <= a <= 65535 and 1 <= b <= 64 and b & (b - 1) == 0):\n        return -1\n    return (a + b - 1) // b * b\n','def f(a, b):\n    return a // b * b\n',itertools.product(edge+(65534,65535,65536),(0,1,2,3,4,7,8,16,32,64,65)),lambda a,b:(a+b-1)//b*b if 0<=a<=65535 and b in (1,2,4,8,16,32,64) else -1),
      _make('aligned-offset-valid/2','alignment','train','Inputs are arbitrary integers. Return bool indicating that a is in 0..65535, b is one of 1,2,4,8,16,32,64, and a is aligned to b.',
        'def f(a, b):\n    return 0 <= a <= 65535 and 1 <= b <= 64 and b & (b - 1) == 0 and a % b == 0\n','def f(a, b):\n    return a % b == 0\n',itertools.product(edge+(65535,65536),(0,1,2,3,4,7,8,16,32,64,65)),lambda a,b:0<=a<=65535 and b in (1,2,4,8,16,32,64) and a%b==0),
      _make('signed-field-extract/2','packed_fields','development_holdout','Input a is an arbitrary integer and b is a field width. Return -1 unless 0 <= a <= 65535 and 1 <= b <= 8. Otherwise interpret the low b bits of a as a signed two-complement integer.',
        'def f(a, b):\n    if not (0 <= a <= 65535 and 1 <= b <= 8):\n        return -1\n    a = a & ((1 << b) - 1)\n    if a & (1 << (b - 1)):\n        return a - (1 << b)\n    return a\n','def f(a, b):\n    return a & ((1 << b) - 1)\n',itertools.product(edge+(32767,32768,65535,65536),range(0,10)),lambda a,b:-1 if not(0<=a<=65535 and 1<=b<=8) else ((a&((1<<b)-1))-(1<<b) if a&(1<<(b-1)) else a&((1<<b)-1))),
      _make('field-fits-unsigned/2','packed_fields','development_holdout','Inputs are arbitrary integers. Return bool indicating that a fits in an unsigned field of width b, where b must be 1..16.',
        'def f(a, b):\n    return 1 <= b <= 16 and 0 <= a < (1 << b)\n','def f(a, b):\n    return 0 <= a <= (1 << b)\n',itertools.product(edge+(32767,32768,65535,65536),range(0,18)),lambda a,b:1<=b<=16 and 0<=a<1<<b),
      _make('window-end/2','range_windows','development_holdout','Inputs are arbitrary integers. a is a start offset and b is a length. Return the inclusive end offset, or -1 unless 0 <= a <= 65535, 1 <= b <= 256, and the end is at most 65535.',
        'def f(a, b):\n    if not (0 <= a <= 65535 and 1 <= b <= 256 and a + b - 1 <= 65535):\n        return -1\n    return a + b - 1\n','def f(a, b):\n    return a + b\n',itertools.product(edge+(65534,65535,65536),byte+(257,)),lambda a,b:a+b-1 if 0<=a<=65535 and 1<=b<=256 and a+b-1<=65535 else -1),
      _make('windows-overlap/2','range_windows','development_holdout','Inputs a and b are packed windows: high byte is start and low byte is positive length. Return bool indicating whether the two half-open windows overlap. Return False if either low byte is zero.',
        'def f(a, b):\n    if a & 255 == 0 or b & 255 == 0:\n        return False\n    return a >> 8 < (b >> 8) + (b & 255) and b >> 8 < (a >> 8) + (a & 255)\n','def f(a, b):\n    return a >> 8 <= (b >> 8) + (b & 255)\n',itertools.product((0,1,255,256,257,511,512,513,4097,4352,65535),repeat=2),lambda a,b:False if a&255==0 or b&255==0 else a>>8<(b>>8)+(b&255) and b>>8<(a>>8)+(a&255)),
    )

def extract_artifact(text):
    match=re.fullmatch(r'\s*```(?:python)?[ \t]*\n(.*?)\n```\s*',text,re.DOTALL)
    if match and '```' not in match.group(1): return match.group(1), 'single_markdown_fence'
    return text, 'raw_source'

def verify(text, task):
    artifact, presentation=extract_artifact(text)
    result=dict(passed=False,category=None,counterexample=None,cases=0,total_cases=len(task.cases),
                generated_text_sha256=digest(text),artifact_sha256=digest(artifact),presentation=presentation,
                formatting_compliant=presentation=='raw_source')
    try: interpreter=Interpreter(artifact)
    except (Rejected,RecursionError) as error: result['category']=str(error); return result
    for a,b,expected in task.cases:
        result['cases']+=1
        try: actual=interpreter(a,b)
        except (Rejected,RecursionError) as error: result.update(category=str(error),counterexample={'inputs':[a,b],'expected':expected}); return result
        if type(actual) is not type(expected) or actual!=expected:
            result.update(category='contract_mismatch',counterexample={'inputs':[a,b],'expected':expected,'actual':actual}); return result
    result.update(passed=True,category='passed'); return result

def export(path):
    rows=[]; training=[]
    for task in tasks():
        good,bad=verify(task.reference,task),verify(task.negative_control,task)
        if not good['passed'] or bad['passed']: raise AssertionError(task.id)
        rows.append({'id':task.id,'family':task.family,'split':task.split,'prompt':task.prompt,'prompt_sha256':digest(task.prompt),
                     'source_sha256':digest(task.reference),'cases_sha256':digest(canonical(task.cases)),'reference_receipt':good,'negative_control_receipt':bad})
        if task.split=='train': training.append({'id':task.id,'family':task.family,'input':task.prompt,'output':task.reference,'source_sha256':digest(task.reference),'origin':'assistant_authored_reference','teacher_model':None})
    base=Path(__file__).with_name('lilith_rl_tasks.py')
    manifest={'schema':'lilith/rl-tasks@2','reward_contract':'binary semantic correctness of extracted artifact; formatting_compliant is recorded but not rewarded',
              'tasks':rows,'training_targets':training,'verifier_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'interpreter_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
              'limitations':['public development holdout; frozen before model generation','finite cases are not proofs','assistant-authored references','Python only; no native illumos validation']}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('x') as f: json.dump(manifest,f,indent=2); f.write('\n')
    return manifest

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); a=p.parse_args(); print(json.dumps({'tasks':len(export(a.output)['tasks'])}))
