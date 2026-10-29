"""Port of @meta-sam/parser mask-codec.js for SAM 3.1.

Implements custom base85 + range coding for one_bit (!) and lossless (~) masks.
Only decoding is needed (encoding kept for canonical check in JS but not here).

Reference: /tmp/parser/package/dist/mask-codec.js (MIT, Meta)
"""
from __future__ import annotations

import numpy as np

# ponytail: direct port of JS, keep names close to source for auditability
_EXCLUDED = set(ord(c) for c in '"\\,;<>|')
_ALPHABET_CHARS = [chr(c) for c in range(0x21, 0x7F) if c not in _EXCLUDED]
_ALPHABET = "".join(_ALPHABET_CHARS)
_RADIX = 85
_PREFIX_LEN = 5
_CHAR_VALUES = [-1] * 128
for i, ch in enumerate(_ALPHABET):
    _CHAR_VALUES[ord(ch)] = i

_TOP = 1 << 24
_BOTTOM = 1 << 16
_MASK32 = 0xFFFFFFFF
_INCR = 14
_COUNT_LIMIT = 4096
_CTX_COUNT = 1 << 12
_PAD = 2
_SYMBOLS = 256
_TOTAL_IDX = 256
_SPATIAL_MODES = 5
_ZERO_INCR = 32
_ZERO_LIMIT = 16384
_OO_INCR = 56
_OO_LIMIT = 8192
_O2_STEP = 14 * 44
_O2_LIMIT = 14 * 3584
_OZ_STEP = 2 * 16
_OZ_LIMIT = 2 * 2048
_OZ_INIT = 2
_OO_INIT = 1


class MaskCodecError(ValueError):
    pass


def _digit(ch: str) -> int:
    code = ord(ch)
    if code >= len(_CHAR_VALUES):
        raise MaskCodecError(f"invalid mask char {ch!r}")
    v = _CHAR_VALUES[code]
    if v < 0 or v >= _RADIX:
        raise MaskCodecError(f"invalid mask char {ch!r}")
    return v


def _unpack(payload: str) -> bytes:
    if len(payload) < _PREFIX_LEN:
        raise MaskCodecError("mask payload missing length prefix")
    length = 0
    for i in range(_PREFIX_LEN):
        length = length * _RADIX + _digit(payload[i])
    rem = length % 4
    expected = _PREFIX_LEN + (length // 4) * 5 + (0 if rem == 0 else rem + 1)
    if len(payload) != expected:
        raise MaskCodecError(f"mask length mismatch: got {len(payload)} expected {expected} (len {length})")
    out = bytearray(length)
    src = _PREFIX_LEN
    dst = 0
    full = length - rem
    while dst < full:
        v = 0
        for i in range(5):
            v = v * _RADIX + _digit(payload[src + i])
        if v > _MASK32:
            raise MaskCodecError("out of range group")
        src += 5
        out[dst] = (v >> 24) & 0xFF
        out[dst + 1] = (v >> 16) & 0xFF
        out[dst + 2] = (v >> 8) & 0xFF
        out[dst + 3] = v & 0xFF
        dst += 4
    if rem:
        v = 0
        for i in range(5):
            ch = payload[src + i] if i < rem + 1 else chr(ord(_ALPHABET[-1]))  # pad with max char = radix-1 -> last alphabet char
            # For tail, JS uses (radix -1) as fill when i >= rem+1
            if i < rem + 1:
                v = v * _RADIX + _digit(ch)
            else:
                v = v * _RADIX + (_RADIX - 1)
        if v > _MASK32:
            raise MaskCodecError("out of range tail")
        for i in range(rem):
            out[dst + i] = (v >> (24 - i * 8)) & 0xFF
    return bytes(out)


class _RangeDecoder:
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise MaskCodecError("missing finalization")
        self._data = data
        self._pos = 0
        self._low = 0
        self._range = _MASK32
        self._code = 0
        self._scaled = 0
        for _ in range(4):
            self._code = ((self._code << 8) | self._read()) & _MASK32

    def _read(self) -> int:
        if self._pos >= len(self._data):
            raise MaskCodecError("payload ended before decode")
        v = self._data[self._pos]
        self._pos += 1
        return v

    def freq(self, total: int) -> int:
        self._scaled = self._range // total
        if self._scaled == 0:
            raise MaskCodecError("invalid coding state")
        value = ((self._code - self._low) & _MASK32) // self._scaled
        return value if value < total else total - 1

    def decode(self, cum: int, freq: int) -> None:
        self._low = (self._low + self._scaled * cum) & _MASK32
        self._range = self._scaled * freq
        while True:
            cond_a = ((self._low ^ (self._low + self._range)) & _MASK32) < _TOP
            if cond_a:
                pass
            elif self._range < _BOTTOM:
                self._range = (-self._low & _MASK32) & (_BOTTOM - 1)
            else:
                break
            self._code = ((self._code << 8) | self._read()) & _MASK32
            self._low = (self._low << 8) & _MASK32
            self._range = (self._range << 8) & _MASK32


def _context_at(padded: np.ndarray, pos: int, prior: int, earlier: int) -> int:
    # padded is 1D uint8 array
    return (
        int(padded[pos - 1])
        | (int(padded[pos - 2]) << 1)
        | (int(padded[prior - 2]) << 2)
        | (int(padded[prior - 1]) << 3)
        | (int(padded[prior]) << 4)
        | (int(padded[prior + 1]) << 5)
        | (int(padded[prior + 2]) << 6)
        | (int(padded[earlier - 2]) << 7)
        | (int(padded[earlier - 1]) << 8)
        | (int(padded[earlier]) << 9)
        | (int(padded[earlier + 1]) << 10)
        | (int(padded[earlier + 2]) << 11)
    )


def _update_counts(counts: np.ndarray, off: int, bit: int) -> None:
    counts[off + bit] += _INCR
    if int(counts[off]) + int(counts[off + 1]) >= _COUNT_LIMIT:
        counts[off] >>= 1
        if counts[off] == 0:
            counts[off] = 1
        counts[off + 1] >>= 1
        if counts[off + 1] == 0:
            counts[off + 1] = 1


def _decode_raster(payload: str, width: int, height: int) -> np.ndarray:
    # payload includes leading "!"
    packed = _unpack(payload[1:])
    dec = _RangeDecoder(packed)
    counts = np.ones(_CTX_COUNT * 2, dtype=np.uint16)
    pw = width + 2 * _PAD
    padded = np.zeros(pw * (height + _PAD), dtype=np.uint8)
    out = np.zeros(width * height, dtype=np.uint8)
    for y in range(height):
        row = (y + _PAD) * pw + _PAD
        for x in range(width):
            pos = row + x
            prior = pos - pw
            earlier = pos - 2 * pw
            ctx = _context_at(padded, pos, prior, earlier)
            off = ctx << 1
            zero = int(counts[off])
            one = int(counts[off + 1])
            v = dec.freq(zero + one)
            bit = 0 if v < zero else 1
            if bit == 0:
                dec.decode(0, zero)
            else:
                dec.decode(zero, one)
            padded[pos] = bit
            out[y * width + x] = bit
            _update_counts(counts, off, bit)
    return out


def _paeth(a: int, b: int, c: int) -> int:
    est = a + b - c
    da = abs(est - a)
    db = abs(est - b)
    dc = abs(est - c)
    if da <= db and da <= dc:
        return a
    return b if db <= dc else c


def _predict_byte(buf: np.ndarray, off: int, x: int, y: int, mode: int, w: int) -> int:
    left = int(buf[off - 1]) if x > 0 else 0
    up = int(buf[off - w]) if y > 0 else 0
    if mode == 1:
        return up
    if mode == 2:
        return left
    if mode == 3:
        return (left + up) >> 1
    ul = int(buf[off - w - 1]) if x > 0 and y > 0 else 0
    if mode == 0:
        return _paeth(left, up, ul)
    if ul >= max(left, up):
        return min(left, up)
    if ul <= min(left, up):
        return max(left, up)
    return left + up - ul


def _unzig(v: int, pred: int) -> int:
    delta = -((v + 1) >> 1) if (v & 1) else v >> 1
    return (pred + delta) & 0xFF


def _zero_ctx(resid: np.ndarray, off: int, x: int, y: int, w: int) -> int:
    left = 1 if x == 0 else (1 if resid[off - 1] == 0 else 0)
    up = 1 if y == 0 else (1 if resid[off - w] == 0 else 0)
    ul = 1 if (x == 0 or y == 0) else (1 if resid[off - w - 1] == 0 else 0)
    ur = 1 if (y == 0 or x >= w - 1) else (1 if resid[off - w + 1] == 0 else 0)
    return left | (up << 1) | (ul << 2) | (ur << 3)


def _decode_lossless(payload: str, width: int, height: int) -> np.ndarray:
    if not payload.startswith("~"):
        raise MaskCodecError("lossless must start with ~")
    packed = _unpack(payload[1:])
    if len(packed) < 5:
        raise MaskCodecError("lossless truncated")
    selector = packed[0]
    dec = _RangeDecoder(packed[1:])
    length = width * height
    out = np.zeros(length, dtype=np.uint8)
    resid = np.zeros(length, dtype=np.uint8)
    zero_cnt = np.ones(32, dtype=np.uint32)
    order_one = np.full(_SYMBOLS * _SYMBOLS, _OO_INIT, dtype=np.uint16)
    order_one_tot = np.full(_SYMBOLS, _SYMBOLS * _OO_INIT, dtype=np.int32)
    order_zero = np.full(_SYMBOLS, _OZ_INIT, dtype=np.uint16)
    order_zero_tot = _SYMBOLS * _OZ_INIT
    order_two: dict[int, np.ndarray] = {}
    prev = 0
    for off in range(length):
        x = off % width
        y = off // width
        zc = _zero_ctx(resid, off, x, y, width) << 1
        zero = int(zero_cnt[zc])
        nz = int(zero_cnt[zc + 1])
        v = dec.freq(zero + nz)
        bit = 0 if v < zero else 1
        if bit == 0:
            dec.decode(0, zero)
        else:
            dec.decode(zero, nz)
        zero_cnt[zc + bit] += _ZERO_INCR
        if int(zero_cnt[zc]) + int(zero_cnt[zc + 1]) >= _ZERO_LIMIT:
            zero_cnt[zc] >>= 1
            if zero_cnt[zc] == 0:
                zero_cnt[zc] = 1
            zero_cnt[zc + 1] >>= 1
            if zero_cnt[zc + 1] == 0:
                zero_cnt[zc + 1] = 1
        sym = 0
        if bit == 1:
            above = int(resid[off - width]) if off >= width else 0
            key = (prev << 8) | above
            sec = order_two.get(key)
            if sec is None:
                sec = np.zeros(_SYMBOLS + 1, dtype=np.uint16)
                order_two[key] = sec
            first_off = prev << 8
            excl0 = int(sec[0]) + int(order_one[first_off]) + int(order_zero[0])
            total = int(sec[_TOTAL_IDX]) + int(order_one_tot[prev]) + order_zero_tot - excl0
            tgt = dec.freq(total)
            cum = 0
            sym = 1
            freq = int(sec[1]) + int(order_one[first_off + 1]) + int(order_zero[1])
            while cum + freq <= tgt:
                cum += freq
                sym += 1
                if sym >= _SYMBOLS:
                    raise MaskCodecError("malformed lossless")
                freq = int(sec[sym]) + int(order_one[first_off + sym]) + int(order_zero[sym])
            dec.decode(cum, freq)
            sec[sym] += _O2_STEP
            sec[_TOTAL_IDX] += _O2_STEP
            if int(sec[_TOTAL_IDX]) >= _O2_LIMIT:
                tot = 0
                for vv in range(_SYMBOLS):
                    sec[vv] >>= 1
                    tot += int(sec[vv])
                sec[_TOTAL_IDX] = tot
            order_one[first_off + sym] += _OO_INCR
            order_one_tot[prev] += _OO_INCR
            if int(order_one_tot[prev]) >= _OO_LIMIT:
                tot = 0
                for vv in range(_SYMBOLS):
                    sc = int(order_one[first_off + vv]) >> 1
                    if sc == 0:
                        sc = 1
                    order_one[first_off + vv] = sc
                    tot += sc
                order_one_tot[prev] = tot
            order_zero[sym] += _OZ_STEP
            order_zero_tot += _OZ_STEP
            if order_zero_tot >= _OZ_LIMIT:
                tot = 0
                for vv in range(_SYMBOLS):
                    order_zero[vv] >>= 1
                    tot += int(order_zero[vv])
                order_zero_tot = tot
        resid[off] = sym
        out[off] = _unzig(sym, _predict_byte(out, off, x, y, int(selector), width))
        prev = sym
    return out


def decode_mask(payload: str, width: int, height: int) -> np.ndarray:
    """Decode one SAM mask payload to (H,W) uint8 0/1 array."""
    if width <= 0 or height <= 0:
        raise MaskCodecError("dimensions must be positive")
    if payload.startswith("!"):
        flat = _decode_raster(payload, width, height)
    elif payload.startswith("~"):
        flat = _decode_lossless(payload, width, height)
        flat = (flat >= 129).astype(np.uint8)  # as in JS: coverage >=129 ?1:0
    else:
        raise MaskCodecError(f"unknown encoding for payload {payload[:10]!r}")
    return flat.reshape((height, width))
