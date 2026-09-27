import unittest
from solution import range_within_u64

class TestRange(unittest.TestCase):
    def test_boundaries(self):
        m = 18446744073709551615
        self.assertTrue(range_within_u64(0, 0, 0))
        self.assertTrue(range_within_u64(5, 0, 5))
        self.assertTrue(range_within_u64(m - 1, 1, m))
        self.assertFalse(range_within_u64(6, 0, 5))
        self.assertFalse(range_within_u64(m, 1, m))
        self.assertFalse(range_within_u64(-1, 1, 2))
        self.assertFalse(range_within_u64(0, m + 1, m))
