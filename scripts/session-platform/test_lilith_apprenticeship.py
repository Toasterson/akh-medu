import copy
import json
from pathlib import Path
import tempfile
import unittest
from lilith_apprenticeship import Lessons, canonical, digest
from lilith_contracts import CONTRACTS


class ApprenticeshipTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'lessons.sqlite'
        self.store=Lessons(self.path);self.addCleanup(lambda:self.store.close())
        self.source='def clamp_u8(value):\n    if value < 0:\n        return 0\n    if value > 255:\n        return 255\n    return value\n'
        c=CONTRACTS['integer-clamp/1']
        self.row=dict(id='assignment',status='completed',parent='parent',parent_turn='author',
            child='teacher',review_turn='review',reviewer='ptah',review='No defect found.',report='Checks passed.',
            intent={'path':c.artifact},source=self.source,source_sha256=digest(self.source),
            evidence=c.verifier(self.source),contract=c.grant())

    def test_verified_admission_survives_cold_reopen_and_deduplicates(self):
        key=self.store.admit(self.row);self.store.admit(self.row)
        self.store.close();self.store=Lessons(self.path)
        items=self.store.active();self.assertEqual(len(items),1)
        self.assertEqual(items[0]['id'],key)
        self.assertEqual(items[0]['provenance']['teacher_turn'],'review')

    def test_failed_cancelled_tampered_or_unverified_episodes_do_not_teach(self):
        for modify in [lambda r:r.update(status='failed'),lambda r:r.update(cancellation={'reason':'stop'}),
                       lambda r:r.update(source_sha256='wrong'),lambda r:r['evidence'].update(passed=False),
                       lambda r:r.update(review=''),lambda r:r['contract'].update(verifier_sha256='old')]:
            row=copy.deepcopy(self.row);modify(row)
            with self.assertRaises(ValueError):self.store.admit(row)
        self.assertEqual(self.store.active(),[])

    def test_wrong_code_cannot_be_admitted_with_fabricated_pass_flag(self):
        row=copy.deepcopy(self.row);row['source']='def clamp_u8(value):\n    return value\n'
        row['source_sha256']=digest(row['source']);row['evidence']['source_sha256']=row['source_sha256']
        with self.assertRaises(AssertionError):self.store.admit(row)

    def test_teacher_opinion_is_not_a_verified_training_target(self):
        self.row['review']='All illumos kernels are certified.'
        self.store.admit(self.row)
        item=self.store.active()[0]
        self.assertNotIn('certified',item['text'])
        target=Path(self.tmp.name)/'candidate.jsonl';self.store.export(target,[])
        self.assertNotIn('certified',target.read_text())

    def test_revocation_survives_readmission_and_restart(self):
        key=self.store.admit(self.row);self.store.revoke(key,'wrong applicability')
        self.store.admit(self.row);self.store.close();self.store=Lessons(self.path)
        self.assertEqual(self.store.active(),[])
        with self.assertRaises(ValueError):self.store.revoke(key,'erase history')

    def test_family_exclusion_prevents_retrieval_and_export_leakage(self):
        self.store.admit(self.row)
        self.assertEqual(self.store.retrieve('text','clamp',[],None,['saturation'])['hits'],[])
        target=Path(self.tmp.name)/'empty.jsonl'
        self.assertEqual(self.store.export(target,['saturation']),0)
        self.assertEqual(target.read_text(),'')

    def test_checksum_corruption_fails_closed(self):
        key=self.store.admit(self.row)
        with self.store.db:self.store.db.execute('UPDATE lessons SET body=? WHERE id=?',('{}',key))
        with self.assertRaisesRegex(ValueError,'checksum'):self.store.active()

    def test_similarity_cannot_override_operation_applicability(self):
        self.store.admit(self.row)
        result=self.store.retrieve('text','integer saturation clamp', [['operation','capacity_check']],None)
        self.assertEqual(result['hits'],[])
        self.assertIn('applicability',result['reason'])

    def test_context_budget_never_truncates_a_lesson(self):
        self.store.admit(self.row)
        self.assertEqual(self.store.retrieve('text','clamp',[],None,max_bytes=1)['context'],'')
        self.assertEqual(self.store.retrieve('none','clamp',[],None)['context'],'')



class TransferCheckerTest(unittest.TestCase):
    def test_correct_reference_and_boundary_failures(self):
        from lilith_transfer_probe import check
        source='''def can_reserve(used, requested, capacity):
    if used < 0 or requested < 0 or capacity < 0:
        return False
    if used > 65535 or requested > 65535 or capacity > 65535:
        return False
    if used > capacity:
        return False
    return requested <= capacity - used
'''
        self.assertEqual(check(source)['cases'],1512)
        for wrong in [source.replace('<= capacity - used','< capacity - used'),
                      source.replace('used < 0','used < -1'),
                      source.replace('return False','return 0')]:
            with self.assertRaises(AssertionError):check(wrong)
        with self.assertRaises(ValueError):check('def can_reserve(used, requested, capacity):\n    return eval(used)\n')

if __name__=='__main__':unittest.main()
