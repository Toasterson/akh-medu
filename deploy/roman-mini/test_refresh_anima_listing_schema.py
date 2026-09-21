import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('refresh', Path(__file__).with_name('refresh_anima_listing_schema.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RefreshListing(unittest.TestCase):
    def fixture(self):
        grant = {'name': 'anima__list_work_items', 'read_only': False,
                 'description': 'old', 'input_schema': {'type': 'object'},
                 'destination': {'kind': 'mcp', 'service': 'anima', 'tool': 'list_work_items'}}
        config = {'akhs': {r: {'instructions': 'retained', 'tools': [copy.deepcopy(grant), {'name': 'other'}]}
                           for r in ('janus', 'minerva', 'vesta')}, 'automations': {'janus': {'enabled': True}}}
        catalog = [{'name': 'list_work_items', 'description': 'cursor listing',
                    'inputSchema': {'type': 'object', 'properties': {'page_token': {'type': 'string'}}}},
                   {'name': 'new_write_tool'}]
        return config, catalog

    def test_only_existing_listing_schemas_change(self):
        config, catalog = self.fixture()
        before = copy.deepcopy(config)
        result, roles = m.refreshed_config(config, catalog)
        self.assertEqual(config, before)
        self.assertEqual(roles, ['janus', 'minerva', 'vesta'])
        for role in roles:
            self.assertEqual(len(result['akhs'][role]['tools']), 2)
            self.assertEqual(result['akhs'][role]['tools'][1], {'name': 'other'})
            self.assertFalse(result['akhs'][role]['tools'][0]['read_only'])
            self.assertEqual(result['akhs'][role]['instructions'], 'retained')
        self.assertEqual(result['automations'], before['automations'])

    def test_missing_deployed_cursor_or_existing_grant_fails(self):
        config, catalog = self.fixture()
        catalog[0]['inputSchema']['properties'] = {}
        with self.assertRaises(ValueError): m.refreshed_config(config, catalog)
        config, catalog = self.fixture()
        config['akhs']['janus']['tools'] = []
        with self.assertRaises(ValueError): m.refreshed_config(config, catalog)


if __name__ == '__main__':
    unittest.main()
