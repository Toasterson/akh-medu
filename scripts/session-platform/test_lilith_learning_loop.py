import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import test_lilith_coordination as fixtures
from lilith_coordination import Coordinator, Ledger
from lilith_apprenticeship import Lessons
from lilith_contracts import CONTRACTS

class LearningLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.work=self.root/'work';self.work.mkdir()
        (self.work/'clamp_u8.py').write_text('def clamp_u8(value):\n    if value < 0:\n        return 0\n    if value > 255:\n        return 255\n    return value\n')
        self.api=fixtures.Api(self.work)
        self.ledger=Ledger(self.root/'coord.sqlite');self.addCleanup(self.ledger.close)
        self.lessons=Lessons(self.root/'lessons.sqlite');self.addCleanup(self.lessons.close)
        c=CONTRACTS['integer-clamp/1']
        self.controller=Coordinator(self.api,self.ledger,'parent','author',self.root,c.verifier,
                                    contract=c.grant(),lessons=self.lessons)

    def test_completion_learns_once_with_separate_durable_status(self):
        row=self.controller.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['learning']['status'],'admitted')
        self.controller.tick()
        self.assertEqual(len(self.lessons.active()),1)

    def test_lost_store_ack_replays_without_duplicate_lesson(self):
        original=self.lessons.admit
        def commit_then_fail(row):
            original(row)
            raise OSError('lost result after commit')
        with patch.object(self.lessons,'admit',side_effect=commit_then_fail):
            with self.assertRaises(OSError):self.controller.tick()
        self.assertNotIn('learning',self.ledger.rows('parent')[0])
        self.assertEqual(self.controller.tick()[0]['learning']['status'],'admitted')
        self.assertEqual(len(self.lessons.active()),1)

    def test_rejected_learning_does_not_relabel_completed_coding(self):
        with patch.object(self.lessons,'admit',side_effect=ValueError('incompatible teaching contract')):
            row=self.controller.tick()[0]
        self.assertEqual(row['status'],'completed')
        self.assertEqual(row['learning']['status'],'rejected')
        self.assertEqual(self.lessons.active(),[])

    def test_revoked_lesson_is_not_reported_as_newly_active(self):
        row=self.controller.tick()[0]
        self.lessons.revoke(row['learning']['lesson_id'],'withdrawn evidence')
        self.assertEqual(self.controller.tick()[0]['learning']['status'],'revoked')
        self.assertEqual(self.lessons.active(),[])

    def test_changed_store_cannot_inherit_an_admission_receipt(self):
        self.controller.tick()
        other=Lessons(self.root/'other.sqlite')
        try:
            self.controller.lessons=other
            with self.assertRaisesRegex(ValueError, 'different or unidentified'):
                self.controller.tick()
            self.assertEqual(other.active(),[])
        finally:other.close()

    def test_failed_work_never_teaches(self):
        self.api.review_status='failed'
        row=self.controller.tick()[0]
        self.assertEqual(row['status'],'failed')
        self.assertNotIn('learning',row)
        self.assertEqual(self.lessons.active(),[])

if __name__=='__main__':unittest.main()
