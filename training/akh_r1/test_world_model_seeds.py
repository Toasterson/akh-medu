import unittest
from world_model_seeds import load,catalog
from recall_authority_client import select
class SeedTests(unittest.TestCase):
    def test_packs_have_exact_provenance(self):
        rows=load();self.assertGreater(len(rows),0)
        for r in rows:
            self.assertEqual(len(r['source']['sha256']),64);self.assertTrue(r['source']['version']);self.assertGreaterEqual(r['source']['row'],0);self.assertGreaterEqual(r['confidence'],0);self.assertLessEqual(r['confidence'],1)
    def test_no_fixture_answers(self):
        for r in load():
            self.assertFalse(set(r['fact'].values()) & {'nodealpha','nodebeta','epoch1','epoch2','east-current'})
    def test_sources_explicitly_not_imported(self):
        s=catalog()['sources'];self.assertEqual({x['id'] for x in s if x['kind']=='documented_external_source_or_design'},{'cyc','wikidata','conceptnet','wikipedia','dbpedia'})
        self.assertTrue(all(x['status']=='documented_no_dataset_import_in_this_slice' for x in s if x['kind']=='documented_external_source_or_design'))
    def test_background_cannot_supply_missing_authority(self):
        rows=load();self.assertTrue(rows);self.assertIsNone(select([dict(id='x',scope='host',effect='repair',version=1,action='apply')],dict(scope='host',effect='repair'),[]))
