import unittest
from provider_failure import classify_provider_error


class Classification(unittest.TestCase):
    def test_rate_limit_does_not_claim_exhausted_quota(self):
        self.assertEqual(classify_provider_error({'message':'model returned HTTP 429; attempts=3'}),'rate_limit')
        self.assertEqual(classify_provider_error('HTTP 429 insufficient_quota'),'quota')

    def test_authentication_is_not_timed_recovery(self):
        self.assertEqual(classify_provider_error('Your refresh token was revoked'),'authentication')
        self.assertIsNone(classify_provider_error('native CI failed'))
        self.assertIsNone(classify_provider_error(None))
