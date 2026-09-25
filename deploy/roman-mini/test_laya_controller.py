import unittest
from unittest.mock import patch
import urllib.request

from laya_controller import CHOICES, Controller, SCHEMA_VERSION, digest, host_action


class FakeSelf:
    """Stand-in for a Controller instance: classify() only reads self.config."""
    def __init__(self, model_revision='rev-1'):
        self.config = {'model_revision': model_revision}


class ClassifyIsRetiredTests(unittest.TestCase):
    def test_makes_no_network_call_and_returns_retired_shape(self):
        snapshot = {'kind': 'work_item', 'detail': {'work_item': {'id': 'wi-1'}}}
        with patch.object(urllib.request, 'urlopen') as urlopen:
            result = Controller.classify(FakeSelf(), snapshot)
        urlopen.assert_not_called()
        self.assertEqual(result['choice'], 'investigate')
        self.assertEqual(result['classifier'], 'retired')
        self.assertIs(result['authorizes_mutation'], False)
        self.assertIs(result['input_complete'], True)
        self.assertEqual(result['schema_version'], SCHEMA_VERSION)
        self.assertEqual(result['snapshot_id'], digest(snapshot))
        self.assertEqual(result['model_revision'], 'rev-1')
        self.assertEqual(result['scores'], {})
        self.assertEqual(result['chunks'], [])
        self.assertNotIn('error', result)

    def test_choice_is_always_a_valid_choice(self):
        result = Controller.classify(FakeSelf(), {'kind': 'plan', 'detail': {}})
        self.assertIn(result['choice'], CHOICES)

    def test_model_revision_passes_through_config_unchanged(self):
        # config['model_revision'] participates in the decision id; classify()
        # must read it verbatim, never invent or normalize a value, so
        # existing decisions are not re-decided by this retirement.
        result = Controller.classify(FakeSelf(model_revision='laya-v7'), {'kind': 'plan'})
        self.assertEqual(result['model_revision'], 'laya-v7')

    def test_host_action_routes_the_retired_proposal_structurally(self):
        # This is the documented finding behind D1: with a real, non-trivial
        # work item snapshot, host_action's structural rules decide the
        # outcome, and a retired 'investigate' proposal flows through exactly
        # like the old near-uniform-score proposals always ended up doing.
        snapshot = {'kind': 'work_item', 'missing': [],
                    'detail': {'work_item': {'id': 'wi-1', 'current_phase': 'parked',
                                              'parked_reason': 'lane-stalled'}}}
        proposal = Controller.classify(FakeSelf(), snapshot)
        self.assertEqual(host_action(snapshot, proposal), ('investigate', 'janus', 'pending'))


if __name__ == '__main__':
    unittest.main()
