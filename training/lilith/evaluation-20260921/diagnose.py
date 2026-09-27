"""Post-hoc diagnostic extraction; never rewrites frozen evaluation scores."""
import ast
import json
from pathlib import Path
import sys

repo=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(repo/'scripts/session-platform'))
import lilith_eval as e
out=Path(__file__).resolve().parent/'run2'
manifest=json.loads((out/'manifest.json').read_text())
records=json.loads((out/'records.json').read_text())
report=json.loads((out/'report.json').read_text())
tasks={task['id']:task for task in manifest['tasks']}
failures=[]
for outcome in report['outcomes']:
    if outcome['passed']:continue
    row=next(r for r in records if r['task']==outcome['task'] and r['arm']==outcome['arm'] and r['seed']==outcome['seed'])
    entry=dict(task=row['task'],arm=row['arm'],seed=row['seed'],source_sha256=e.digest(row['source']),
        strict_outcome=outcome,diagnostics=[],score_unchanged=True)
    try:
        tree=ast.parse(row['source'])
        functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)]
        entry['top_level_function_count']=len(functions)
        if len(functions)>1:
            entry['diagnostics'].append(dict(finding='duplicate_function_definitions',
                evidence='The prompt and checker require exactly one function; the raw output contains multiple definitions.'))
        if any(isinstance(node,ast.Invert) for node in ast.walk(tree)):
            class NormalizeInvert(ast.NodeTransformer):
                def visit_UnaryOp(self,node):
                    node=self.generic_visit(node)
                    if isinstance(node.op,ast.Invert):
                        return ast.copy_location(ast.BinOp(left=ast.UnaryOp(op=ast.USub(),operand=node.operand),op=ast.Sub(),right=ast.Constant(value=1)),node)
                    return node
            transformed=ast.unparse(ast.fix_missing_locations(NormalizeInvert().visit(ast.parse(row['source']))))+'\n'
            entry['diagnostics'].append(dict(kind='unscored_exact_integer_identity_normalization',
                identity='~x == -x - 1 for Python integers',scored=False,
                original_source_sha256=e.digest(row['source']),transformed_source_sha256=e.digest(transformed),
                transformed_source=transformed,result=e.verify(transformed,tasks[row['task']]),
                interpretation='Artificial tiny-language constraint; this alone is not an arithmetic or illumos-coding deficit.'))
        lines=row['source'].splitlines()
        for index,function in enumerate(functions):
            source='\n'.join(lines[function.lineno-1:function.end_lineno])+'\n'
            result=e.verify(source,tasks[row['task']])
            entry['diagnostics'].append(dict(kind='isolated_function_secondary_check',function_index=index,
                scored=False,source_sha256=e.digest(source),result=result))
    except (SyntaxError,ValueError) as error:entry['diagnostics'].append(dict(parse_error=str(error)))
    if row['task']=='modular_sequence_order-8' and row['arm']!='independent_reference':
        entry['contract_diagnosis']='Observed omitted modulus-domain check: all inputs, including modulus, are explicitly restricted to0..255. A conventional8-bit ring may use modulus256, but this task rejects256. Whether that convention influenced the model is unproven.'
    entry['next_intervention']='Diagnose semantic failures separately from the artificial tiny-language constraint. Teach complete input-domain checks and one final artifact using fresh non-holdout examples; do not export these evaluation solutions. No quality-tuning recommendation solely to avoid unary complement.'
    failures.append(entry)
(out/'failure-ledger.json').write_text(json.dumps(dict(schema='lilith/failure-diagnosis@1',
    purpose='Post-hoc diagnosis only; no changed score, tuning, retry or admission.',failures=failures),indent=2)+'\n')
print(json.dumps(dict(failed_observations=len(failures),diagnostics_scored=False)))
