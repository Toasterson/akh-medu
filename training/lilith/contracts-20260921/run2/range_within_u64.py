def range_within_u64(start: int, length: int, limit: int) -> bool:
    u64_max = 18446744073709551615
    if start < 0 or start > u64_max or length < 0 or length > u64_max or limit < 0 or limit > u64_max:
        return False
    if start > limit:
        return False
    if start == limit and length == 0:
        return True
    if length == 0:
        return True
    if length <= limit - start:
        return True
    return False