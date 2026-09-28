"""BLAKE3 (default hash mode, 32-byte output) for the decontamination hook.

Uses the `blake3` package when it is installed; otherwise a pure-Python port of
the BLAKE3 reference implementation, so the dojo stays stdlib-only. The two are
checked against each other and against published vectors in the tests.
"""
from __future__ import annotations

import struct

try:  # pragma: no cover - exercised when the optional package is present
    import blake3 as _blake3_native  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    _blake3_native = None

_MASK = 0xFFFFFFFF
_IV = (0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_CHUNK_START, _CHUNK_END, _PARENT, _ROOT = 1, 2, 4, 8
_BLOCK_LEN, _CHUNK_LEN = 64, 1024


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


def _g(s: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    s[a] = (s[a] + s[b] + mx) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 7)


def _compress(cv: tuple[int, ...], m: tuple[int, ...], counter: int, block_len: int, flags: int) -> list[int]:
    s = [*cv, *_IV[:4], counter & _MASK, (counter >> 32) & _MASK, block_len, flags]
    m = list(m)
    for r in range(7):
        _g(s, 0, 4, 8, 12, m[0], m[1])
        _g(s, 1, 5, 9, 13, m[2], m[3])
        _g(s, 2, 6, 10, 14, m[4], m[5])
        _g(s, 3, 7, 11, 15, m[6], m[7])
        _g(s, 0, 5, 10, 15, m[8], m[9])
        _g(s, 1, 6, 11, 12, m[10], m[11])
        _g(s, 2, 7, 8, 13, m[12], m[13])
        _g(s, 3, 4, 9, 14, m[14], m[15])
        if r < 6:
            m = [m[i] for i in _PERM]
    for i in range(8):
        s[i] ^= s[i + 8]
        s[i + 8] ^= cv[i]
    return s


def _words(block: bytes) -> tuple[int, ...]:
    return struct.unpack("<16I", block.ljust(_BLOCK_LEN, b"\0"))


class _Output:
    def __init__(self, cv: tuple[int, ...], block: tuple[int, ...], counter: int, block_len: int, flags: int):
        self.cv, self.block, self.counter, self.block_len, self.flags = cv, block, counter, block_len, flags

    def chaining_value(self) -> tuple[int, ...]:
        return tuple(_compress(self.cv, self.block, self.counter, self.block_len, self.flags)[:8])

    def root_bytes(self) -> bytes:
        return struct.pack("<8I", *_compress(self.cv, self.block, 0, self.block_len, self.flags | _ROOT)[:8])


class _Chunk:
    def __init__(self, counter: int):
        self.cv: tuple[int, ...] = _IV
        self.counter = counter
        self.buf = b""
        self.blocks = 0

    def __len__(self) -> int:
        return _BLOCK_LEN * self.blocks + len(self.buf)

    def _start(self) -> int:
        return _CHUNK_START if self.blocks == 0 else 0

    def update(self, data: bytes) -> None:
        while data:
            if len(self.buf) == _BLOCK_LEN:
                self.cv = tuple(_compress(self.cv, _words(self.buf), self.counter, _BLOCK_LEN, self._start())[:8])
                self.blocks += 1
                self.buf = b""
            take = min(_BLOCK_LEN - len(self.buf), len(data))
            self.buf += data[:take]
            data = data[take:]

    def output(self) -> _Output:
        return _Output(self.cv, _words(self.buf), self.counter, len(self.buf), self._start() | _CHUNK_END)


def _parent(left: tuple[int, ...], right: tuple[int, ...]) -> _Output:
    return _Output(_IV, (*left, *right), 0, _BLOCK_LEN, _PARENT)


def blake3_pure(data: bytes) -> bytes:
    stack: list[tuple[int, ...]] = []
    chunk = _Chunk(0)
    while data:
        if len(chunk) == _CHUNK_LEN:
            cv = chunk.output().chaining_value()
            total = chunk.counter + 1
            while total & 1 == 0:
                cv = _parent(stack.pop(), cv).chaining_value()
                total >>= 1
            stack.append(cv)
            chunk = _Chunk(chunk.counter + 1)
        take = min(_CHUNK_LEN - len(chunk), len(data))
        chunk.update(data[:take])
        data = data[take:]
    out = chunk.output()
    for left in reversed(stack):
        out = _parent(left, out.chaining_value())
    return out.root_bytes()


def blake3_digest(data: bytes) -> bytes:
    if _blake3_native is not None:
        return _blake3_native.blake3(data).digest()
    return blake3_pure(data)


def native_available() -> bool:
    return _blake3_native is not None
