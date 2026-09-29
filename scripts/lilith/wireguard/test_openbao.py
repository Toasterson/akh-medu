import unittest
import copy
from dataclasses import replace
from unittest.mock import Mock

from openbao import OpenBao
from slots import SlotError
import slots
from test_slots import VariantCBase


class OpenBaoTests(unittest.TestCase):
    def client(self):
        client = OpenBao.__new__(OpenBao)
        client.paths = {'state': 'secret/data/lilith/state'}
        client.observed = {}
        return client

    def test_update_uses_observed_version_and_preserves_metadata(self):
        client = self.client()
        client._request = Mock(side_effect=[{'data': {'data': {'enabled': 'false', 'run': '-'}, 'metadata': {'version': 4}}}, {'data': {'version': 5}}])
        client.fields('state')
        client.set_fields('state', {'enabled': 'true'})
        self.assertEqual(client._request.call_args.args, ('secret/data/lilith/state', {'options': {'cas': 4}, 'data': {'enabled': 'true', 'run': '-'}}))
        self.assertEqual(client.observed['state'][1], 5)

    def test_conflict_is_not_retried_or_written_to_other_authority(self):
        client = self.client()
        client.observed = {'state': ({'enabled': 'false'}, 4)}
        client._request = Mock(side_effect=SlotError('conflict', help='retry'))
        with self.assertRaises(SlotError):
            client.set_fields('state', {'enabled': 'true'})
        self.assertEqual(client._request.call_count, 1)
        self.assertEqual(client.observed['state'][1], 4)

    def test_second_read_detects_competing_assignment(self):
        client = self.client()
        client.observed = {'state': ({'enabled': 'false'}, 4)}
        client._request = Mock(return_value={'data': {'data': {'enabled': 'true'}, 'metadata': {'version': 5}}})
        with self.assertRaises(SlotError):
            client.fields('state')
        self.assertEqual(client.observed['state'][1], 4)

    def test_unknown_item_and_new_field_fail_closed(self):
        client = self.client()
        client._request = Mock()
        with self.assertRaises(SlotError):
            client.fields('other')
        client.observed = {'state': ({'enabled': 'false'}, 4)}
        with self.assertRaises(SlotError):
            client.set_fields('state', {'private_key': 'value'})
        client._request.assert_not_called()


class InterleavedReleaseTests(VariantCBase):
    def test_competing_release_cannot_silently_overwrite_winning_keypair(self):
        documents = copy.deepcopy(self.world.items)
        versions = {key: 1 for key in documents}
        state_item = self.config.slots.state_item
        second_config = replace(self.config, slots=replace(
            self.config.slots, state_dir=self.config.slots.state_dir.with_name('other-supervisor')))
        triggered = False
        def client():
            obj = OpenBao.__new__(OpenBao)
            obj.paths = {item: item for item in documents}
            obj.observed = {}
            return obj
        first, second = client(), client()
        def transport(item, body=None):
            nonlocal triggered
            if body is None:
                return {'data': {'data': copy.deepcopy(documents[item]), 'metadata': {'version': versions[item]}}}
            if body['options']['cas'] != versions[item]:
                raise SlotError('CAS conflict', help='retry complete operation')
            documents[item] = copy.deepcopy(body['data'])
            versions[item] += 1
            written_version = versions[item]
            if item == state_item and not triggered:
                triggered = True
                slots.release(second_config, second, slot=1)
            return {'data': {'version': written_version}}
        first._request = transport
        second._request = transport
        with self.assertRaises(SlotError):
            slots.release(self.config, first, slot=1)
        self.assertTrue(triggered)
        self.assertEqual(documents[state_item]['slot1_enabled'], 'false')
        self.assertEqual(documents[state_item]['slot1_public_key'], documents[self.config.slots.item(1)]['public_key'])


if __name__ == '__main__':
    unittest.main()
