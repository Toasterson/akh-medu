import unittest
import json
from lilith_contracts import CONTRACTS
from lilith_verification import verify_u64_range
import test_lilith_coordination as fixtures

VALID = '''def range_within_u64(start: int, length: int, limit: int) -> bool:
    if start < 0 or length < 0 or limit < 0:
        return False
    if start > 18446744073709551615 or length > 18446744073709551615 or limit > 18446744073709551615:
        return False
    if start > limit:
        return False
    return length <= limit - start
'''


class RangeVerificationTest(unittest.TestCase):
    def test_unsigned_boundaries_and_dense_small_ranges(self):
        evidence = verify_u64_range(VALID)
        self.assertTrue(evidence['passed'])
        self.assertEqual(evidence['cases'], 2926)
        self.assertIn('not C overflow', evidence['scope'])

    def test_plausible_defects_are_rejected_with_counterexamples(self):
        for source in [VALID.replace('<= limit - start', '< limit - start'),
                       VALID.replace('start < 0', 'start < -1'),
                       VALID.replace('limit > 18446744073709551615', 'limit < 0'),
                       VALID.replace('return False', 'return 0')]:
            with self.subTest(source=source), self.assertRaises(AssertionError):
                verify_u64_range(source)

    def test_equivalent_implementation_without_redundant_length_check_passes(self):
        self.assertTrue(verify_u64_range(VALID.replace('length > 18446744073709551615', 'length < 0'))['passed'])

    def test_unsafe_or_unbounded_constructs_are_rejected(self):
        for body in ['return eval(start)', 'while True: pass', 'return 2 ** start',
                     'return "x" + "y"', 'import os', 'return start.__class__']:
            with self.subTest(body=body), self.assertRaises(ValueError):
                verify_u64_range('def range_within_u64(start, length, limit):\n    '+body+'\n')


# Reuse the fixture setup without inheriting/re-running the original test cases.
class ContractCoordinationTest(unittest.TestCase):
    setUp = fixtures.CoordinationTest.setUp
    make = fixtures.CoordinationTest.make
    restart = fixtures.CoordinationTest.restart

    def configure_range(self):
        contract = CONTRACTS['u64-range/1']
        self.controller.artifact = contract.artifact
        self.controller.verifier = contract.verifier
        self.controller.contract = contract.grant()
        self.api.pending[0]['prompt']['arguments']['path'] = contract.artifact
        (self.workspace/contract.artifact).write_text(VALID)

    def test_second_contract_flows_through_same_durable_controller(self):
        self.configure_range()
        row = self.controller.tick()[0]
        self.assertEqual(row['status'], 'completed')
        self.assertEqual(row['evidence']['verifier'], 'u64-range/1')
        self.assertEqual(row['evidence']['contract'], row['contract'])
        self.assertEqual(row['evidence']['cases'], 2926)
        request = row['review_request']
        payload = json.loads(request['text'].split('\n', 1)[1])
        self.assertEqual(payload['verification']['contract']['specification'], CONTRACTS['u64-range/1'].task)

    def test_model_cannot_select_another_contract_than_operator_grant(self):
        self.api.pending[0]['prompt']['arguments']['path'] = 'range_within_u64.py'
        (self.workspace/'range_within_u64.py').write_text(VALID)
        self.assertEqual(self.controller.tick()[0]['status'], 'rejected')
        self.assertFalse(self.api.children)

    def test_registered_receipt_cannot_be_attached_to_another_verifier(self):
        from lilith_coordination import Coordinator
        contract = CONTRACTS['u64-range/1']
        with self.assertRaisesRegex(ValueError, 'registered contract'):
            Coordinator(self.api, self.ledger, 'parent', 'author', self.root,
                        lambda source: {'passed': True}, artifact=contract.artifact,
                        contract=contract.grant())

    def test_restart_cannot_silently_change_verifier_version(self):
        self.api.complete_author = False
        self.configure_range()
        self.controller.tick()
        self.restart()
        contract = CONTRACTS['u64-range/1']
        self.controller.artifact = contract.artifact
        self.controller.verifier = contract.verifier
        self.controller.contract = dict(contract.grant(), verifier_sha256='changed')
        self.api.parent['turns'][0]['status'] = 'completed'
        with self.assertRaisesRegex(ValueError, 'verification contract'):
            self.controller.tick()
        self.assertFalse(self.api.children)
