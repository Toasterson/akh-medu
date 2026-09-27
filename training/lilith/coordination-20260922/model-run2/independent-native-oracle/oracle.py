from solution import range_within_u64
from itertools import product
m = (1 << 64) - 1
values = [-1, 0, 1, 2, 3, 7, 1 << 32, m-1, m, m+1, m+2, -(1 << 65)]
cases = 0
for start, length, limit in product(values, repeat=3):
    expected = all(0 <= x <= m for x in (start, length, limit)) and 0 <= start <= start + length <= limit
    actual = range_within_u64(start, length, limit)
    assert type(actual) is bool and actual == expected, (start, length, limit, expected, actual)
    cases += 1
print('independent mathematical boundary oracle: %d cases passed' % cases)
