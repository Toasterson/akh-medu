"""Local seed condition with provenance; no implicit operational authority."""
import hashlib,json,tomllib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
PACKS=['ontology','common-sense','computing']
def load():
    records=[]
    for name in PACKS:
        p=ROOT/'data/seeds'/name/'seed.toml';data=tomllib.loads(p.read_text());h=hashlib.sha256(p.read_bytes()).hexdigest()
        for i,t in enumerate(data['triples']):
            fact={k:t[k] for k in ['subject','predicate','object']}
            if fact['predicate'] in ['current_skill','repair_target','assigned','intent'] or fact['predicate'].startswith('r0_'):raise ValueError('Seed contains operational authority')
            records.append(dict(fact=fact,confidence=t.get('confidence',1.0),source=dict(pack=name,version=data['seed']['version'],sha256=h,row=i,path=str(p))))
    return records

def catalog():
    sources=[]
    for name in PACKS:
        p=ROOT/'data/seeds'/name/'seed.toml';data=tomllib.loads(p.read_text());sources.append(dict(id=name,kind='local_seed_pack',path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),version=data['seed']['version'],triples=len(data['triples']),status='available_local',authority='general_background_not_live_environment'))
    for name,doc,role in [('cyc','docs/ai/plans/2026-02-17-phase9-cyc-inspired.md','context, defeasible rules and truth maintenance'),('wikidata','docs/ai/plans/2026-03-27-phase35-t5-training-infrastructure.md','identity and taxonomy'),('conceptnet','docs/ai/plans/2026-02-17-phase14-bootstrapping.md','commonsense relations and prerequisites'),('wikipedia','docs/ai/plans/2026-02-17-phase14-bootstrapping.md','source documents'),('dbpedia','docs/ai/plans/2026-02-17-phase14-bootstrapping.md','identity/category fallback')]:
        sources.append(dict(id=name,kind='documented_external_source_or_design',document=doc,document_sha256=hashlib.sha256((ROOT/doc).read_bytes()).hexdigest(),role=role,status='documented_no_dataset_import_in_this_slice'))
    return dict(schema='akh-world-model-sources-v1',sources=sources,conditions=['schema_only','local_seed_packs','source_supported_operational_rules','episode_learned_rules'],required_tests=['provenance/version retention','no current authority or examiner answer in seeds','prerequisite direction','context exceptions and correction invalidate derived beliefs','matched task effect and unsupported-action comparison','cold transfer and memory cost'],external_dataset_evaluation='Pending pinned snapshots; local packs are not external dataset imports')
if __name__=='__main__':Path(__file__).with_name('world_model_sources.json').write_text(json.dumps(catalog(),indent=2)+'\n')
