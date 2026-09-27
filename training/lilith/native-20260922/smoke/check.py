def clamp_u8(value: int) -> int:
    return max(0, min(255, value))


assert [clamp_u8(value) for value in (-3, 0, 42, 300)] == [0, 0, 42, 255]
print("python-native-smoke: ok")
