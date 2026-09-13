"""Coverage inventory: authored controls never satisfy independent-world quotas."""
import json
from pathlib import Path
from r0_contract import FAMILIES
from run_public_breadth import write

EVIDENCE={
'acquisition':(['public-breadth','pipeline-evidence'],'Exact structured facts only; free-text extraction and entity disambiguation missing.'),
'persistent_recall':(['R0-reference','public-breadth','pipeline-evidence'],'Effect lookup and scoped max-version recall pass cold controls; removed-current-record fallback causes a detected wrong effect. Neural reuse and broader worlds remain missing.'),
'grounded_reasoning':(['pipeline-evidence'],'Authored fixed three-edge path; variable-depth rules, counterfactuals and held-out compositions missing.'),
'knowledge_revision':(['R0-reference','public-breadth','pipeline-evidence','derived-effect'],'Explicit replacement, history and finite derived-support invalidation to action exist; credibility, temporal precedence, concurrency and broad dependency generators remain missing.'),
 'tool_choice':(['R0-reference','public-breadth'],'Direct safe effect selection and composed capability dependencies passed actual-effect controls; broader mechanisms and neural selection remain missing.'),
'multi_step_work':(['R0-reference','public-breadth'],'Small integrated reference episodes; graph-to-pipeline integration and unseen workflow combinations missing.'),
'novelty_ambiguity':(['public-breadth','pipeline-evidence'],'Diagnostic inquiry and combined feasibility/participant constraints have distinct audited structures; natural-language ambiguity and broader generators remain missing.'),
'language_transfer':(['family-controls'],'Exact EN/DE/FR/PT phrase table and one unsupported opposite-intent control; real language understanding and paraphrase generators missing.'),
'recovery_durability':(['R0-reference'],'Lost-acknowledgment recovery and current-operation authority after restart have audited finite mechanisms; concurrency, power-loss durability and broad fault generators remain open.'),
'perception_knowledge':(['family-controls','perception-effect'],'Actual local OCR clean-label to graph and fixture action exists; layouts, source spans, noisy images, multilingual recognition and broad perception generators remain missing.'),
'dojo_pressure':(['R0-reference','family-controls'],'Scripted correction, inspect action and ownership return; full eight kata, six extended forms and human-sensitive independent judgment missing.'),
'runtime_command_learning':(['R0-reference','public-breadth'],'Known typed command specifications and cold reuse; learned novel effects, skill version changes and composed recall missing.'),
}
# Later audited component/incident evidence augments the catalog, never counts.
SUPPLEMENTAL = {
 'persistent_recall': ['akh-r1-recall-authority-20260912-v1','akh-r1-stratified-incident-20260913-v2'],
 'knowledge_revision': ['akh-r1-native-temporal-cache-20260913-v2','akh-r1-stratified-incident-20260913-v2'],
 'multi_step_work': ['akh-r1-stratified-incident-20260913-v2'],
 'language_transfer': ['akh-r1-language-composition-20260913-v1','akh-r1-language-pack-20260913-v1'],
 'perception_knowledge': ['akh-r1-ocr-layouts-20260913-v1','akh-r1-control-preparation-20260913-v1'],
 'dojo_pressure': ['akh-r1-ownership-episodes-20260913-v2','akh-r1-control-preparation-20260913-v1','akh-r1-dojo-episodes-20260912-v1','akh-r1-starter-kata-20260912-v1'],
 'runtime_command_learning': ['akh-r1-command-semantics-20260912-v1','akh-r1-stratified-incident-20260913-v2'],
}
UPDATES = {
 'persistent_recall': 'Current-version witness repair and cold command reuse have bounded reference evidence. Broader sampled environments and neural retrieval remain untested.',
 'knowledge_revision': 'Native timestamp/cache repairs and sourced support changes pass reference checks; broader dependency distributions, authenticated credibility and concurrency remain open.',
 'multi_step_work': 'Six stratified complete incidents join graph support, cold command reuse and actual verified effects. Shared constructions are not yet independently qualified.',
 'language_transfer': 'Parameterized authored grammar and counterbalanced constraint compositions pass; two unqualified pack entries exist. Learned language and independent paraphrase distributions remain missing.',
 'perception_knowledge': 'Coordinate-linked clean inline/stacked OCR and matched transcripts have actual-effect controls. Layout/target confounded; noisy and multilingual evidence, independent document generators and common OCR execution adapter remain missing.',
 'dojo_pressure': 'Simulated hidden participant corrections and joint constraints pass behavioral controls; common preparation preserves private outcomes. Starter-kata and extended-dojo authored controls exist. Blinded independent participant-intent/ownership judgments and broad scenario generators remain open.',
 'runtime_command_learning': 'Structured command discovery, scoped versions and cold reuse are observed in reference controls. Learned interpretation of unseen command semantics and broader independent incidents remain missing.',
}

# Completed integration evidence remains supplemental, never automatic quota credit.
for _family, _runs in {
 'acquisition': ['akh-r1-identity-acquisition-20260913-v1'],
 'knowledge_revision': ['akh-r1-identity-acquisition-20260913-v1'],
 'multi_step_work': ['akh-r1-stateful-service-20260913-v2','akh-r1-structural-service-20260913-v1','akh-r1-generated-service-20260913-v1'],
 'perception_knowledge': ['akh-r1-perception-structure-effects-20260913-v1','akh-r1-perception-assistance-20260913-v1'],
 'runtime_command_learning': ['akh-r1-generated-service-20260913-v1'],
}.items():
 SUPPLEMENTAL.setdefault(_family, []).extend(_runs)
UPDATES.update({
 'acquisition': 'Strict labeled-record parsing, stable-ID/location joins and current-event selection have complete integration evidence. Independent document generators and learned free-text extraction remain missing.',
 'grounded_reasoning': 'Fixed and stratified graph paths plus generated service DAGs have authored controls; broad independent rule/composition distributions and neural reasoning remain untested.',
 'multi_step_work': 'Actual service sequences, distinct support mechanisms and two generated instances pass cold/correction controls. Recipe assistance is explicit; common effect dispatch and broad qualification remain missing.',
 'perception_knowledge': 'Actual table/procedure OCR drives sourced graph effects under fixed-region/schema assistance; 13 declared dependencies are bound. Independent layouts, general interpretation and common effect dispatch remain missing.',
})

ROOTS={'derived-effect':'akh-r1-derived-effect-20260912-v1','perception-effect':'akh-r1-perception-effect-20260912-v1','R0-reference':'akh-r1-R0-reference-20260912-v2','public-breadth':'akh-r1-public-breadth-20260912-v1','pipeline-evidence':'akh-r1-pipeline-evidence-20260912-v1','family-controls':'akh-r1-family-controls-20260912-v1'}

def coverage():
    from episode_pack import file_sha,build
    runs=[Path('/Users/toasty/ws/akh-medu/runs')/n for n in ['akh-r1-graph-generators-20260912-v1','akh-r1-tool-choice-20260912-v1','akh-r1-ambiguity-validation-20260912-v1','akh-r1-epoch-recovery-20260912-v1','akh-r1-recall-20260912-v1']]
    episodes=[];registry={}
    for run in runs:
        audit=json.loads((run/'artifact-audit.json').read_text());qualification=json.loads((run/'qualification.json').read_text())
        if not audit['all_passed'] or qualification['audit_sha256']!=file_sha(run/'artifact-audit.json'):raise ValueError('Qualification witness failed')
        episodes.extend(json.loads((run/'frozen-episodes.json').read_text()))
        new=json.loads((run/'frozen-registry.json').read_text())
        if registry.keys() & new.keys():raise ValueError('Duplicate generator identity')
        registry.update(new)
    pack=build(episodes,registry)
    if not pack['manifest_valid']:raise ValueError(pack['errors'])
    return dict(schema='akh-r0-family-coverage-v1',full_R0_complete=False,R1_ready=False,
      accounting='Fourteen audited distinct reference structures contribute seven development and seven validation-fixture worlds. Earlier authored controls do not fill quotas. No neural validation performed.',
      families=[dict(id=name,development_target=n,validation_target=n,
                     current_status='reference_structural_worlds' if pack['counts']['development'][name] else 'authored_control_only',
                     evidence_runs=[str(Path('/Users/toasty/ws/akh-medu/runs')/ROOTS[k]) for k in EVIDENCE[name][0]]+([str(r) for r in runs] if pack['counts']['development'][name] else [])+[str(Path('/Users/toasty/ws/akh-medu/runs')/r) for r in SUPPLEMENTAL.get(name,[])],
                     independent_pack_worlds=pack['counts']['development'][name],validation_worlds=pack['counts']['validation'][name],model_worlds=0,remaining=UPDATES.get(name,EVIDENCE[name][1])) for name,n,_ in FAMILIES],
      R0_prerequisites=['Build independent development and validation generators for all 12 families; identity-only variants do not qualify.',
        'Run integrated episodes through one public protocol with complete traces and independent effect scoring.',
        'Add real perception-to-action and complete kata/consultation coverage; retain transcript controls separately.',
        'Freeze assistance budgets, scorer and split manifests before R1 model comparisons.'])

if __name__=='__main__':write(Path(__file__).with_name('family-coverage.json'),coverage())
