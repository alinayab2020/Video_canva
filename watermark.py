"""
watermark.py — Keyed spread-spectrum forensic video watermarking for ASCILINE.

Embeds a 10-digit identifier into the rendered ASCII/pixel stream such that it
survives *screen capture* (screenshots and screen recordings, including lossy
H.264 re-encodes of the captured footage) and remains machine-detectable with
the secret key — while staying imperceptible to viewers.

Technique (see docs/WATERMARK.md for the full treatment)
--------------------------------------------------------
  * CDMA spread spectrum (Cox et al., "Secure Spread Spectrum Watermarking
    for Multimedia", IEEE TIP 1997): every code bit is spread over a large,
    keyed random subset of cells with a keyed ±1 chip pattern, so each bit's
    energy is distributed across the whole frame and detection integrates
    thousands of weak, independent observations.
  * Carrier: ±1 step of the glyph *density rank* (the palette index plane) in
    ASCII modes — carried bit-exact by the ASCILINE adaptive codec, which by
    design never lossifies the character plane — or a ±beta luminance dither
    of the cell colour in pixel mode. The step has constant sign correlation
    with the rendered cell brightness because every bundled palette is sorted
    by visual density, independent of the viewer's font and theme.
  * Temporal differential signalling: the whole PN field is sign-alternated
    over time (blocks of `block` frames). A capturer who differences adjacent
    frames cancels the *content* (which barely changes frame-to-frame) while
    the watermark *doubles*. This is the classic low-spatiotemporal-frequency
    placement used by screen-cam-robust cinema forensic marks.
  * Hardened payload: 10 decimal digits (≈33 bits) as a 40-bit container plus
    CRC-16-CCITT, wrapped in a RS(15,7) Reed–Solomon code over GF(2^8)
    (corrects any 4 byte errors, or 8 byte erasures, or mixtures 2e+v ≤ 8).
  * Security: all spreading sequences, cell permutations and signs derive from
    a 256-bit key via HMAC-SHA256 in counter mode. Without the key the mark is
    statistically undetectable and unremovable (keyed spread spectrum); the
    CRC gate keeps the false-accept probability at the 2^-16 level per
    hypothesis, with an additional empirical-null z-score reported.

Wire usage: 10 digits → 7 bytes → RS(15,7) → 15 bytes → 120 code bits.
Each frame carries all 120 bits (G = rows*cols/120 cells per bit), so a few
seconds of footage already integrate tens of thousands of chips per bit.

This module is NumPy-only on the embed path; the capture-detection path uses
OpenCV solely for image decode/resize when available.
"""

from __future__ import annotations

import hashlib
import hmac
import struct

import numpy as np

try:  # optional, only needed for feed_image() (pixel-level capture ingest)
    import cv2 as _cv2
except ImportError:  # pragma: no cover
    _cv2 = None

# ────────────────────────────────────────────────────────────────────────────
#  GF(2^8) arithmetic + Reed–Solomon (15, 7) — fcr=0, prim polynomial 0x11d
# ────────────────────────────────────────────────────────────────────────────

_GF_EXP = [0] * 512
_GF_LOG = [0] * 256
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


def _gf_mul(x: int, y: int) -> int:
    return 0 if x == 0 or y == 0 else _GF_EXP[_GF_LOG[x] + _GF_LOG[y]]


def _gf_div(x: int, y: int) -> int:
    if y == 0:
        raise ZeroDivisionError("GF division by zero")
    return 0 if x == 0 else _GF_EXP[(_GF_LOG[x] - _GF_LOG[y]) % 255]


def _gf_poly_mul(p: list, q: list) -> list:
    out = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            out[i + j] ^= _gf_mul(p[i], q[j])
    return out


def _gf_poly_scale(p: list, x: int) -> list:
    return [_gf_mul(c, x) for c in p]


def _gf_poly_add(p: list, q: list) -> list:
    out = [0] * max(len(p), len(q))
    for i, c in enumerate(p):
        out[i + len(out) - len(p)] = c
    for i, c in enumerate(q):
        out[i + len(out) - len(q)] ^= c
    return out


def _gf_poly_eval(p: list, x: int) -> int:
    y = p[0]
    for c in p[1:]:
        y = _gf_mul(y, x) ^ c
    return y


def _gf_poly_div(dividend: list, divisor: list) -> tuple[list, list]:
    out = list(dividend)
    for i in range(len(dividend) - len(divisor) + 1):
        coef = out[i]
        if coef:
            for j in range(1, len(divisor)):
                if divisor[j]:
                    out[i + j] ^= _gf_mul(divisor[j], coef)
    sep = -(len(divisor) - 1)
    return out[:sep], out[sep:]


_RS_NSYM = 8      # 8 parity bytes → RS(15,7), distance 9
_RS_NMESS = 15
_RS_K = 7


def _rs_generator() -> list:
    g = [1]
    for i in range(_RS_NSYM):           # roots α^0 .. α^7 (fcr = 0)
        g = _gf_poly_mul(g, [1, _GF_EXP[i]])
    return g


_RS_GEN = _rs_generator()


def rs_encode(msg: bytes) -> bytes:
    """Systematic RS(15,7) encode: 7 data bytes → 15 bytes (data ++ parity)."""
    if len(msg) != _RS_K:
        raise ValueError(f"rs_encode expects {_RS_K} data bytes")
    out = list(msg) + [0] * _RS_NSYM
    for i in range(_RS_K):
        coef = out[i]
        if coef:
            for j in range(1, len(_RS_GEN)):
                out[i + j] ^= _gf_mul(_RS_GEN[j], coef)
    # The feedback loop leaves residue where the message was; restore the
    # systematic header (standard reedsolo encode construction).
    out[:_RS_K] = msg
    return bytes(out)


def _rs_calc_syndromes(msg: list) -> list:
    return [0] + [_gf_poly_eval(msg, _GF_EXP[i]) for i in range(_RS_NSYM)]


def _rs_find_errata_locator(e_pos: list) -> list:
    loc = [1]
    for p in e_pos:
        loc = _gf_poly_mul(loc, [1, _GF_EXP[p]])
    return loc


def _rs_solve_magnitudes(synd_full: list, err_pos: list, nmess: int) -> list[int] | None:
    """
    Solve the errata magnitudes directly from the syndrome equations.

    With r located errata at byte positions p_j (0 = leftmost), the (fcr=0)
    syndromes satisfy the Vandermonde system

        S_k = Σ_j e_j · X_j^k ,   X_j = α^(nmess-1-p_j) ,  k = 0 .. r-1

    Vandermonde matrices over GF(2^8) are invertible for distinct, nonzero
    X_j — guaranteed here because positions are distinct. Gauss–Jordan
    elimination over GF(2^8) is exact (no Forney/reversal convention traps)
    and a zero pivot or zero magnitude means a false Chien root → reject.
    """
    r = len(err_pos)
    if r == 0:
        return []
    if r > _RS_NSYM:
        return None
    xs = [_GF_EXP[nmess - 1 - p] for p in err_pos]
    if len(set(xs)) != r:
        return None
    # Augmented matrix rows: [X_0^k, ..., X_{r-1}^k | S_k], k = 0..r-1
    mat = [[_GF_EXP[(_GF_LOG[x] * k) % 255] if x else (1 if k == 0 else 0)
            for x in xs] + [synd_full[1 + k]] for k in range(r)]
    for col in range(r):
        piv = next((row for row in range(col, r) if mat[row][col] != 0), None)
        if piv is None:
            return None
        mat[col], mat[piv] = mat[piv], mat[col]
        inv = _gf_div(1, mat[col][col])
        mat[col] = [_gf_mul(v, inv) for v in mat[col]]
        for row in range(r):
            if row != col and mat[row][col] != 0:
                f = mat[row][col]
                mat[row] = [a ^ _gf_mul(b, f) for a, b in zip(mat[row], mat[col])]
    # Magnitudes may legitimately be 0 for erasures whose true byte was 0 —
    # the caller's residual-syndrome check is the sound uncorrectable guard.
    return [mat[k][r] for k in range(r)]


def _rs_find_error_locator(synd: list, erase_count: int = 0) -> list:
    """
    Berlekamp–Massey over the (Forney) syndromes. With erasures, the caller
    passes Forney syndromes (already shortened by `erase_count` symbols), so
    only nsym - erase_count iterations carry information.
    """
    err_loc = [1]   # Λ — the error locator polynomial being estimated
    old_loc = [1]
    # Skip the always-zero leading syndrome element (fsynd is 1 shorter per
    # erasure, so the shift stays 1 in both branches).
    synd_shift = len(synd) - (_RS_NSYM - erase_count)
    for i in range(_RS_NSYM - erase_count):
        k = i + synd_shift
        delta = synd[k]
        for j in range(1, len(err_loc)):
            delta ^= _gf_mul(err_loc[-(j + 1)], synd[k - j])
        old_loc.append(0)
        if delta:
            if len(old_loc) > len(err_loc):
                new_loc = _gf_poly_scale(old_loc, delta)
                old_loc = _gf_poly_scale(err_loc, _gf_div(1, delta))
                err_loc = new_loc
            err_loc = _gf_poly_add(err_loc, _gf_poly_scale(old_loc, delta))
    while len(err_loc) and err_loc[0] == 0:
        del err_loc[0]
    errs = len(err_loc) - 1
    if (errs - erase_count) * 2 + erase_count > _RS_NSYM:
        raise ValueError("too many errors to correct")
    return err_loc


def _rs_find_errors(err_loc: list, nmess: int) -> list | None:
    """Chien search: return error positions (0 = leftmost byte) or None."""
    errs = len(err_loc) - 1
    if errs == 0:
        return []
    err_pos = []
    for i in range(nmess):
        if _gf_poly_eval(err_loc, _GF_EXP[i]) == 0:
            err_pos.append(nmess - 1 - i)
    if len(err_pos) != errs:
        return None  # locator degree mismatch → uncorrectable
    return err_pos


def _rs_forney_syndromes(synd: list, pos: list, nmess: int) -> list:
    """Strip erasure contributions from the syndromes."""
    fsynd = list(synd[1:])
    for p in pos:
        x = _GF_EXP[nmess - 1 - p]
        for i in range(len(fsynd) - 1):
            fsynd[i] = _gf_mul(fsynd[i], x) ^ fsynd[i + 1]
        fsynd.pop()
    return [0] + fsynd


def _rs_correct_errata(msg: list, synd: list, err_pos: list) -> list:
    """
    Apply errata magnitudes solved from the syndrome Vandermonde system.
    `synd` are the FULL syndromes of `msg` (erasures already nulled), err_pos
    lists ALL errata (errors + erasures), 0 = leftmost byte. Raises
    ValueError when the located set is inconsistent (false Chien root).
    """
    order = sorted(range(len(err_pos)), key=lambda i: err_pos[i])
    positions = [err_pos[i] for i in order]
    mags = _rs_solve_magnitudes(synd, positions, len(msg))
    if mags is None:
        raise ValueError("inconsistent errata set — uncorrectable")
    out = list(msg)
    for p, m in zip(positions, mags):
        out[p] ^= m
    return out


def rs_decode(codeword: bytes, erasures: list[int] | None = None) -> bytes | None:
    """
    Decode RS(15,7); `erasures` lists byte positions (0 = first byte) treated
    as erasures. Corrects any 2·errors + erasures ≤ 8. Returns the corrected
    15-byte codeword, or None when uncorrectable.
    """
    if len(codeword) != _RS_NMESS:
        raise ValueError(f"rs_decode expects {_RS_NMESS} bytes")
    erasures = list(erasures or [])
    if len(erasures) > _RS_NSYM:
        return None
    msg = list(codeword)
    for p in erasures:
        if not 0 <= p < _RS_NMESS:
            return None
        msg[p] = 0
    synd = _rs_calc_syndromes(msg)
    if max(synd) == 0:
        return bytes(msg)  # already a valid codeword (erased bytes were 0)
    fsynd = _rs_forney_syndromes(synd, erasures, _RS_NMESS) if erasures else synd
    try:
        err_loc = _rs_find_error_locator(fsynd, len(erasures))
    except ValueError:
        return None
    err_pos = _rs_find_errors(err_loc[::-1], _RS_NMESS)
    if err_pos is None:
        return None
    full_pos = sorted(set(err_pos) | set(erasures))
    try:
        corrected = _rs_correct_errata(msg, synd, full_pos)
    except (ValueError, ZeroDivisionError):
        return None
    if max(_rs_calc_syndromes(corrected)) != 0:
        return None  # residual syndrome → miscorrection guard
    return bytes(corrected)


# ────────────────────────────────────────────────────────────────────────────
#  CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) — payload integrity gate
# ────────────────────────────────────────────────────────────────────────────

def _make_crc16_table() -> list[int]:
    table = []
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
        table.append(crc)
    return table


_CRC16_TABLE = _make_crc16_table()


def crc16_ccitt(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC16_TABLE[((crc >> 8) ^ b) & 0xFF]
    return crc


# ────────────────────────────────────────────────────────────────────────────
#  Payload: 10 digits ⇄ 7-byte container ⇄ 15-byte RS-protected codeword
# ────────────────────────────────────────────────────────────────────────────

PAYLOAD_DIGITS = 10
CODE_BYTES = _RS_NMESS          # 15
CODE_BITS = CODE_BYTES * 8      # 120


def encode_payload(digits: str) -> bytes:
    """
    Pack exactly 10 decimal digits into the 15-byte protected codeword:
        5 bytes  big-endian integer (10 digits < 2^34, top 6 bits zero)
        2 bytes  CRC-16-CCITT over the 5 data bytes
        8 bytes  RS(15,7) parity
    """
    if not isinstance(digits, str) or len(digits) != PAYLOAD_DIGITS \
            or not digits.isdigit():
        raise ValueError("watermark payload must be exactly 10 decimal digits")
    body = int(digits).to_bytes(5, "big")
    msg7 = body + struct.pack(">H", crc16_ccitt(body))
    return rs_encode(msg7)


def decode_payload(codeword: bytes, erasures: list[int] | None = None) -> str | None:
    """
    Inverse of encode_payload: RS decode (with optional byte erasures), then
    CRC-16 gate. Returns the 10-digit string, or None on failure.
    """
    msg = rs_decode(codeword, erasures)
    if msg is None:
        return None
    body, crc_lo = bytes(msg[:5]), bytes(msg[5:7])
    if crc16_ccitt(body) != struct.unpack(">H", crc_lo)[0]:
        return None
    value = int.from_bytes(body, "big")
    if value >= 10 ** 10:
        return None
    return f"{value:010d}"


def code_bit_signs(codeword: bytes) -> np.ndarray:
    """15 codeword bytes → int8[120] of ±1 (MSB first); +1 for a set bit."""
    if len(codeword) != CODE_BYTES:
        raise ValueError(f"expected {CODE_BYTES} codeword bytes")
    bits = np.unpackbits(np.frombuffer(codeword, dtype=np.uint8))
    return (bits.astype(np.int8) * 2 - 1)


# ────────────────────────────────────────────────────────────────────────────
#  Keyed spreading — HMAC-SHA256 DRBG, permutation + ±1 chip field
# ────────────────────────────────────────────────────────────────────────────

def _normalize_key(key) -> bytes:
    if isinstance(key, bytes):
        raw = key
    else:
        raw = str(key).encode("utf-8")
    if not raw:
        raise ValueError("watermark key must not be empty")
    return hashlib.sha256(b"asciline-wm-key-v1" + raw).digest()


def _drbg(master: bytes, domain: bytes, nbytes: int) -> bytes:
    """HMAC-SHA256 in counter mode as a deterministic byte stream."""
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        out += hmac.new(master, domain + struct.pack(">Q", counter),
                        hashlib.sha256).digest()
        counter += 1
    return bytes(out[:nbytes])


class MarkPlan:
    """
    Keyed cell→bit assignment and chip field for one (rows, cols) grid.

    The grid's cells are partitioned into CODE_BITS groups by a uniform random
    permutation (argsort of DRBG keys) so every code bit owns ≈G spread-out
    cells; each cell also draws a ±1 spatial chip sign. Leftover cells
    (rows*cols mod 120) carry a 0 sign and are never modulated.
    """

    __slots__ = ("rows", "cols", "signs", "bitidx")

    def __init__(self, key, rows: int, cols: int):
        if rows < 1 or cols < 1:
            raise ValueError("grid must be non-empty")
        master = _normalize_key(key)
        n = rows * cols
        g = n // CODE_BITS
        if g < 1:
            raise ValueError(
                f"grid {cols}x{rows} = {n} cells is too small to carry the "
                f"{CODE_BITS}-bit watermark (need ≥ {CODE_BITS} cells, "
                f"≥ {8 * CODE_BITS} recommended)")

        perm_keys = np.frombuffer(
            _drbg(master, b"wm-perm", n * 8), dtype=np.uint8).copy()
        # Argsort of uniform random keys ⇒ uniform random permutation.
        order = np.argsort(perm_keys.view(np.uint64), kind="stable")
        cell_of_slot = order.astype(np.int64)          # slot → cell

        sign_bytes = np.frombuffer(
            _drbg(master, b"wm-signs", n // 8 + 1), dtype=np.uint8)
        slot_signs = (np.unpackbits(sign_bytes)[:n] * 2 - 1).astype(np.int8)

        used = g * CODE_BITS
        bitidx = np.full(n, -1, dtype=np.int16)
        signs = np.zeros(n, dtype=np.int8)
        slots = np.arange(n)[:used]
        bitidx[cell_of_slot[:used]] = (slots // g).astype(np.int16)
        signs[cell_of_slot[:used]] = slot_signs[:used]

        self.rows, self.cols = rows, cols
        self.signs = signs.reshape(rows, cols)
        self.bitidx = bitidx.reshape(rows, cols)

    @property
    def cells_per_bit(self) -> int:
        return (self.rows * self.cols) // CODE_BITS


def temporal_sign(tick: int, block: int = 1) -> int:
    """
    Frame-block alternation clock: +1 for even blocks, −1 for odd blocks.
    `tick` must count *sent* frames so consecutive displayed frames always
    alternate (server frame drops must not advance the clock).
    """
    block = max(1, int(block))
    return 1 if ((tick // block) & 1) == 0 else -1


# ────────────────────────────────────────────────────────────────────────────
#  Embedder
# ────────────────────────────────────────────────────────────────────────────

class Watermarker:
    """
    Applies one watermark to one grid geometry. Rebuilt whenever rows/cols or
    the payload change; the per-frame embed itself is a few vectorised passes.
    """

    def __init__(self, digits: str, key, rows: int, cols: int,
                 block: int = 1, beta: int = 8, perceptual: bool = True,
                 cov_cap: float = 0.03,
                 gate: tuple[float, float, float] = (9.0, 6.0, 22.0)):
        self.digits = digits
        self.plan = MarkPlan(key, rows, cols)
        self.code = encode_payload(digits)
        self.bit_signs = code_bit_signs(self.code)      # int8[120], ±1
        self.block = max(1, int(block))
        self.beta = max(1, min(64, int(beta)))
        self.perceptual = bool(perceptual)
        # Perceptual shaping knobs (see _visibility_mask / GLYPH_COVERAGE):
        self.cov_cap = max(0.0, float(cov_cap))
        self.gate = tuple(float(v) for v in gate)

    def _eff_field(self, tick: int) -> np.ndarray:
        """Signed per-cell modulation for this tick: (rows, cols) int8 ±1/0."""
        bit_of = np.clip(self.plan.bitidx, 0, CODE_BITS - 1)
        eff = self.plan.signs * self.bit_signs[bit_of]
        if temporal_sign(tick, self.block) < 0:
            eff = -eff
        return eff

    @staticmethod
    def _visibility_mask(gray: np.ndarray,
                         grad_thresh: float = 9.0,
                         std_thresh: float = 6.0,
                         dark_thresh: float = 22.0) -> np.ndarray:
        """
        Human-visual-system gate: a 30 Hz ±1-rank glyph flicker is only
        imperceptible where the underlying content masks it — textured cells
        (HVS luminance-contrast masking, Watson 1993 style) or very dark
        cells (flicker of near-black is invisible on any display).

        Texture is measured per cell as max(|∇x|,|∇y|) plus 3x3 local std of
        the gray grid, with ABSOLUTE thresholds — relative (percentile)
        thresholds would still dither the flattest frame in the clip, which
        is exactly where flicker hurts. Cells failing the gate are skipped
        (erasures for the detector, which is blind to the gate anyway).

        Vectorised; ~0.2 ms on a 200x50 grid. cv2-free (box filters via
        integral-image-free cumulative sums — dependency-light hot path).
        """
        g = gray.astype(np.float32)
        # horizontal / vertical gradient magnitude
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:] = np.abs(g[:, 1:] - g[:, :-1])
        gy[1:, :] = np.abs(g[1:, :] - g[:-1, :])
        grad = np.maximum(gx, gy)
        # 3x3 local mean/std via a leading-zero integral image (edge-padded).
        gp = np.pad(g, ((1, 1), (1, 1)), mode="edge")

        def _window_sums(v):
            integ = np.zeros((v.shape[0] + 1, v.shape[1] + 1),
                             dtype=np.float64)
            integ[1:, 1:] = v.cumsum(0).cumsum(1)
            return (integ[3:, 3:] - integ[:-3, 3:]
                    - integ[3:, :-3] + integ[:-3, :-3])

        mean = _window_sums(gp) / 9.0
        sq_mean = _window_sums(gp * gp) / 9.0
        std = np.sqrt(np.maximum(sq_mean - mean * mean, 0.0))
        texture = grad + std
        return (texture > (grad_thresh + std_thresh)) | (g < dark_thresh)

    def embed_indices(self, indices: np.ndarray, tick: int, n_ramp: int,
                      gray: np.ndarray | None = None) -> None:
        """
        In-place ±1 density-rank dither of the palette-index plane (ASCII
        modes). At the ramp edges the step is *folded* back inside
        (idx==0 cannot go lower): a folded cell carries an inverted chip,
        i.e. a bounded, rare, random-sign noise source that temporal
        integration + ECC absorb — never a wrap-around.

        When `gray` (the per-cell gray grid) is given and the Watermarker
        was built with perceptual=True, the dither only lands on cells the
        flicker-mask gate passes — flat bright regions stay pixel-frozen.
        """
        if indices.shape != self.plan.signs.shape:
            raise ValueError("indices shape != plan grid")
        eff = self._eff_field(tick).astype(np.int16)
        if self.perceptual and gray is not None:
            # Flicker gate (texture/dark) AND coverage-step cap: a ±1-rank
            # transition with a large ink-coverage jump is visible even on
            # textured content, so those cells are skipped (erasures).
            allowed = self._visibility_mask(gray, *self.gate)
            if self.cov_cap > 0.0:
                plus, minus = _coverage_steps(n_ramp)
                step = np.where(eff > 0, plus[indices], minus[indices])
                allowed = allowed & (step <= self.cov_cap)
            eff = np.where(allowed, eff, 0)
        idx = indices.astype(np.int16)
        cand = idx + eff
        folded = (cand < 0) | (cand > (n_ramp - 1))
        # int8 view keeps everything branch-free
        np.copyto(indices, np.where(folded, idx - eff, cand).astype(np.uint8),
                  casting="unsafe")

    def embed_pixels(self, bgr: np.ndarray, tick: int,
                     gray: np.ndarray | None = None) -> None:
        """
        In-place ±beta luminance dither of the cell COLOUR (all channels
        shifted equally → chroma untouched).

        This is the primary carrier for BOTH pixel mode and ASCII colour
        modes: unlike a glyph-rank swap it is font-agnostic (the observed
        luminance step is ink_fraction·beta — always sign-correct, since
        coverage can only modulate magnitude, never polarity), structurally
        invisible (glyphs stay identical), and saturates as erasures-only
        (clipped cells carry no signal, never an inverted chip).

        `gray`: optional texture source for the flicker gate; defaults to
        the frame's own BT.601 luma.
        """
        if bgr.shape[:2] != self.plan.signs.shape:
            raise ValueError("frame shape != plan grid")
        eff = self._eff_field(tick).astype(np.int16) * self.beta
        if self.perceptual:
            if gray is None:
                gray = (0.299 * bgr[:, :, 2] + 0.587 * bgr[:, :, 1]
                        + 0.114 * bgr[:, :, 0])
            eff = np.where(self._visibility_mask(
                np.clip(gray, 0, 255).astype(np.uint8)),
                eff, 0).astype(np.int16)
        work = bgr.astype(np.int16)
        work += eff[:, :, None]
        np.clip(work, 0, 255, out=work)
        np.copyto(bgr, work, casting="unsafe")


# ────────────────────────────────────────────────────────────────────────────
#  Detector — blind keyed correlation from captured luma grids
# ────────────────────────────────────────────────────────────────────────────

# Palette of the ASCII engine, reused to model cell brightness from the
# character plane in "grid-domain" detection (decoded stream / unit tests).
DEFAULT_PALETTE = list(
    " `.-':_,^=;><+!rc*/z?sLTv)J7(|Fi{C}fI31tlu[neoZ5Yxjya]2ESwqkP6h9d4VpOGbUAKXHm8RD#$Bg0MNWQ%&@")

# Per-glyph ink coverage of the default palette (fraction of a 6x10 cell,
# DejaVu Sans Mono Bold at 10px — the shape the browser's bold 8px Courier
# rendering closely tracks). Only ORDER-OF-MAGNITUDE stability is needed:
# the embedder uses it to prune the rare, egregious ±1-rank transitions
# whose coverage jump would be visible (top transitions reach 9.4% of a
# cell vs a 0.45% average). For non-default palettes we fall back to the
# uniform-rank assumption 1/(n-1), which is what all ramps approximate.
GLYPH_COVERAGE = np.array([
    0.0196, 0.0579, 0.0729, 0.0699, 0.0812, 0.1263, 0.1176, 0.0881,
    0.1420, 0.1882, 0.1444, 0.1931, 0.1927, 0.1831, 0.1425, 0.1820,
    0.2020, 0.2034, 0.1605, 0.2442, 0.1890, 0.2478, 0.2365, 0.2448,
    0.2310, 0.2098, 0.2488, 0.2435, 0.2099, 0.2013, 0.2809, 0.2638,
    0.2699, 0.2460, 0.2707, 0.2753, 0.2792, 0.2810, 0.2582, 0.2611,
    0.2550, 0.2712, 0.2496, 0.2688, 0.2876, 0.2614, 0.3331, 0.2902,
    0.2754, 0.2448, 0.2793, 0.2925, 0.2914, 0.2491, 0.2837, 0.3310,
    0.2961, 0.2961, 0.3404, 0.3380, 0.3373, 0.3453, 0.3385, 0.3441,
    0.3637, 0.3026, 0.3248, 0.3398, 0.3519, 0.3281, 0.3630, 0.3559,
    0.3388, 0.3751, 0.3309, 0.3797, 0.3229, 0.3495, 0.3920, 0.3779,
    0.3737, 0.3281, 0.4122, 0.3769, 0.3529, 0.4058, 0.4068, 0.4038,
    0.3769, 0.2744, 0.3680, 0.4250,
], dtype=np.float32)


def _coverage_steps(n_ramp: int) -> tuple[np.ndarray, np.ndarray]:
    """
    (step_plus, step_minus): ABSOLUTE ink-coverage change of a ±1-rank
    transition from every rank.

    Only MACRO steps (≳2%) are font-stable and safe to gate on; micro-step
    signs (≲1%) reverse between rasterizers/hinting, so the cap must never
    descend into the micro range — selecting small steps would concentrate
    exactly the non-monotone transitions and cancel the correlation.
    """
    if n_ramp == len(GLYPH_COVERAGE):
        cov = GLYPH_COVERAGE
    else:
        cov = (np.arange(n_ramp, dtype=np.float32) + 1.0) / n_ramp
    plus = np.empty(n_ramp, dtype=np.float32)
    minus = np.empty(n_ramp, dtype=np.float32)
    plus[:-1] = cov[1:] - cov[:-1]
    plus[-1] = abs(cov[-1] - cov[-2])      # fold destination
    minus[1:] = cov[:-1] - cov[1:]
    minus[0] = abs(cov[1] - cov[0])        # fold destination
    return np.abs(plus), np.abs(minus)


class DetectionResult(dict):
    """dict subclass so `if result:` reads naturally; keys documented in decode()."""
    __getattr__ = dict.get


class WatermarkDetector:
    """
    Soft-accumulating detector. Feed it per-frame *cell-luma* grids
    (rows×cols×float32, arbitrary linear brightness units), then call
    decode(). Grids may come from feed_image() (screen-capture pixels) or be
    built directly from decoded stream frames (see luma_from_ascii_frame).
    """

    def __init__(self, key, rows: int, cols: int, block=None,
                 segment_pairs: int = 8, null_trials: int = 12):
        self.plan = MarkPlan(key, rows, cols)
        # Detection scans every candidate alternation clock: a 1/k-fps
        # capture of a block-k stream looks like a block-1 stream, so a fixed
        # clock cannot cover both. Default set covers blocks 1-6 and the
        # common fps-decimation aliases.
        if block is None:
            self.blocks = (1, 2, 3, 4, 6)
        elif isinstance(block, (tuple, list, set)):
            self.blocks = tuple(sorted({max(1, int(b)) for b in block}))
        else:
            self.blocks = (max(1, int(block)),)
        if not self.blocks:
            raise ValueError("empty block scan set")
        self.segment_pairs = max(2, int(segment_pairs))
        self.null_trials = max(4, int(null_trials))
        self._grids: list[np.ndarray] = []
        # Flat views for the bincount correlator
        self._bit_flat = np.clip(self.plan.bitidx.reshape(-1), 0, CODE_BITS - 1)
        self._sign_flat = self.plan.signs.reshape(-1).astype(np.float64)
        # Null-reference chip fields (wrong keys ⇒ detection floor, z-score)
        master = _normalize_key(key)
        self._null_masks = [
            MarkPlan(master + f"::null{i}".encode(), rows, cols)
            .signs.reshape(-1).astype(np.float64)
            for i in range(self.null_trials)
        ]

    # ── ingest ──

    def feed_luma_grid(self, grid: np.ndarray) -> None:
        g = np.asarray(grid, dtype=np.float32)
        if g.shape != (self.plan.rows, self.plan.cols):
            raise ValueError(f"luma grid shape {g.shape} != "
                             f"({self.plan.rows}, {self.plan.cols})")
        self._grids.append(g)

    def feed_image(self, bgr_image: np.ndarray, crop=None) -> None:
        """
        Ingest one captured *pixel* frame (any resolution). If `crop`
        (x, y, w, h) is given, that rectangle is the player canvas; otherwise
        the whole image is assumed to be the canvas. Rectangle is resampled
        to the cell grid — the resample itself performs the per-cell average.
        """
        if _cv2 is None:  # pragma: no cover
            raise RuntimeError("feed_image needs OpenCV (cv2)")
        img = bgr_image
        if crop is not None:
            x, y, w, h = (int(v) for v in crop)
            img = img[y:y + h, x:x + w]
        # float32 keeps sub-quantisation cell means (INTER_AREA box averages)
        gray = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY).astype(np.float32)
        cell = _cv2.resize(gray, (self.plan.cols, self.plan.rows),
                           interpolation=_cv2.INTER_AREA)
        self.feed_luma_grid(cell)

    def __len__(self) -> int:
        return len(self._grids)

    # ── scored packet assembly (one pairing phase) ──

    def _pair_fields(self, resid: np.ndarray, phase: int,
                     block: int) -> list[np.ndarray]:
        """
        Differential pair fields: F = s_a·(R[a] − R[b]) flattened, with robust
        per-pair scale normalisation (gain/AGC invariance, outlier trimming).
        """
        t_total = resid.shape[0]
        fields = []
        m = 0
        while True:
            a0 = phase + 2 * m * block
            if a0 + 2 * block > t_total:
                break
            for j in range(block):
                # s(a) depends only on (phase, j): the b-frame is always in the
                # next block, so its sign is the opposite of s(a).
                s_a = 1.0 if ((phase + j) // block) % 2 == 0 else -1.0
                fa, fb = resid[a0 + j], resid[a0 + block + j]
                d = s_a * (fa - fb).reshape(-1)
                scale = np.median(np.abs(d)) * 1.4826
                if scale < 1.0:
                    scale = 1.0
                fields.append((d / scale).astype(np.float64))
            m += 1
        if len(fields) >= 4:
            # Per-cell reliability (maximum-ratio combining): cells whose
            # differential energy is huge across time are motion-dominated —
            # they contribute noise, so their chip terms are downweighted by
            # the inverse of their MAD. The watermark Δ is nearly constant
            # per cell, so this essentially equalises cell reliabilities.
            stack_f = np.abs(np.stack(fields))          # (pairs, cells)
            cell_scale = np.median(stack_f, axis=0) * 1.4826
            anchor = float(np.median(cell_scale))
            if not np.isfinite(anchor) or anchor < 1e-6:
                # Over half the cells carry no differential energy at all
                # (static/quantised content, zero-padded sync borders):
                # anchor on the active cells only, else plain unit scale.
                pos = cell_scale[cell_scale > 1e-6]
                anchor = float(np.median(pos)) if pos.size else 1.0
            lo, hi = max(anchor, 1e-6), max(anchor * 4.0, 1e-6)
            cell_scale = np.clip(cell_scale, lo, hi)    # bound the boost
            fields = [f / cell_scale for f in fields]
        return fields

    def _segments_to_llr(self, segments: list[np.ndarray],
                         sign_flat: np.ndarray) -> np.ndarray:
        """Project one accumulated cell-field onto every bit's chip space."""
        w = np.zeros(self._bit_flat.shape[0], dtype=np.float64)
        for seg in segments:
            w += seg
        w *= sign_flat
        return np.bincount(self._bit_flat, weights=w,
                           minlength=CODE_BITS)[:CODE_BITS]

    @staticmethod
    def _em_align(seg_llrs: list[np.ndarray]) -> np.ndarray | None:
        """
        Blind segment-sign alignment by expectation–maximisation.

        A capture that drops/duplicates frames flips the alternation parity
        for every segment after the slip; each segment therefore carries the
        watermark as ±llr with an unknown sign. EM jointly recovers the soft
        bit vector b and the segment signs s_i:  s_i = sign(⟨b, seg_i⟩),
        b ← sign(Σ s_i·seg_i) — iterated to a (local) fixed point. Trying
        both global signs at the CRC gate disambiguates the fixed point.
        """
        if not seg_llrs:
            return None
        stack = np.stack(seg_llrs)
        b = np.sign(stack.sum(axis=0))
        b[b == 0] = 1.0
        signs = np.ones(len(stack))
        for _ in range(25):
            dots = stack @ b
            signs = np.where(dots >= 0.0, 1.0, -1.0)
            total = (signs[:, None] * stack).sum(axis=0)
            bn = np.sign(total)
            bn[bn == 0] = 1.0
            if np.array_equal(bn, b):
                break
            b = bn
        return signs @ stack

    def _score(self, llr: np.ndarray) -> float:
        return float(np.abs(llr).sum())

    def _decode_llr(self, llr: np.ndarray,
                    erasures: tuple[int, ...] = (0, 2, 4, 6, 8),
                    ) -> tuple[str | None, int, float]:
        """
        Hard-limit + erasure sweep → RS → CRC. Returns (digits|None,
        corrected_bytes, raw_bit_error_rate_estimate).
        """
        for sign_flip in (1.0, -1.0):
            soft = sign_flip * llr
            hard = np.where(soft >= 0, 1, 0).astype(np.uint8)
            weights = np.abs(soft).reshape(CODE_BYTES, 8).sum(axis=1)
            weak_order = np.argsort(weights)
            for n_era in erasures:
                packet = np.packbits(hard).tobytes()
                digits = decode_payload(packet, list(weak_order[:n_era]))
                if digits is not None:
                    # BER estimate vs the decoded, re-encoded codeword
                    truth = np.unpackbits(np.frombuffer(encode_payload(digits),
                                                        dtype=np.uint8))
                    ber = float(np.mean(truth != hard))
                    return digits, n_era, ber
        return None, -1, 1.0

    # ── public result ──

    def decode(self, max_frames: int | None = None,
               erasures: tuple[int, ...] | None = None) -> DetectionResult:
        """
        Attempt recovery from the frames fed so far.
        `erasures`: RS byte-erasure sweep to try (default (0, 2, 4, 6, 8);
        geometry-scan probes use the cheaper (0, 4)).
        Returns a dict with:
          ok          bool       True iff digits survived RS+CRC
          digits      str|None   the 10-digit payload
          bit_errors  float      raw bit error rate before ECC (post-AGC)
          z           float      empirical detection z-score vs wrong-key null
          confidence  float      z / 8.0 clamped to [0, 1] (8σ = decisive)
          frames_used int        luma grids consumed
          pairs       int        differential pairs integrated
          phase       int        winning pairing phase (0/1)
          erasures    int        RS byte-erasures needed by the winner
        """
        era_sweep = erasures if erasures is not None else (0, 2, 4, 6, 8)
        grids = self._grids if max_frames is None else self._grids[:max_frames]
        min_block = min(self.blocks)
        if len(grids) < 2 * min_block + 1:
            return DetectionResult(ok=False, digits=None, bit_errors=1.0, z=0.0,
                                   confidence=0.0, frames_used=len(grids),
                                   pairs=0, phase=-1, erasures=-1, block=-1)
        stack = np.stack(grids)
        resid = stack - stack.mean(axis=0, keepdims=True)
        s = self.segment_pairs

        def phase_pipeline(phase: int, block: int, sign_flat: np.ndarray):
            """pair fields → per-segment LLRs → EM-aligned total LLR."""
            fields = self._pair_fields(resid, phase, block)
            if not fields:
                return None, 0
            seg_llrs = []
            for i in range(0, len(fields), s):
                seg_llrs.append(self._segments_to_llr(fields[i:i + s],
                                                      sign_flat))
            return self._em_align(seg_llrs), len(fields)

        best = None
        for block in self.blocks:
            for phase in (0, 1):
                llr, n_pairs = phase_pipeline(phase, block, self._sign_flat)
                if llr is None:
                    continue
                digits, n_era, ber = self._decode_llr(llr, era_sweep)
                cand = dict(digits=digits, ber=ber, erasures=n_era,
                            phase=phase, pairs=n_pairs, llr=llr, block=block)
                if best is None or self._score(llr) > self._score(best["llr"]):
                    best = cand
                if digits is not None:
                    best = cand       # CRC-passing hypothesis wins outright
                    break
            if best is not None and best["digits"] is not None:
                break

        # Empirical null: identical pipeline, wrong-key chip fields.
        null_stats = []
        if best is not None:
            for mask in self._null_masks:
                nullllr, _ = phase_pipeline(best["phase"], best["block"], mask)
                if nullllr is not None:
                    null_stats.append(self._score(nullllr))
        mu = float(np.mean(null_stats)) if null_stats else 0.0
        sd = max(float(np.std(null_stats)) if null_stats else 1.0, 1e-9)

        score = self._score(best["llr"]) if best else 0.0
        z = (score - mu) / sd
        ok = bool(best and best["digits"] is not None)
        return DetectionResult(
            ok=ok,
            digits=best["digits"] if ok else None,
            bit_errors=float(best["ber"]) if best else 1.0,
            z=float(z),
            confidence=float(min(1.0, max(0.0, z / 8.0))),
            frames_used=len(grids),
            pairs=best["pairs"] if best else 0,
            phase=best["phase"] if best else -1,
            erasures=best["erasures"] if best else -1,
            block=best["block"] if best else -1,
        )


# ────────────────────────────────────────────────────────────────────────────
#  Grid-domain helper — model screen brightness from a decoded stream frame
# ────────────────────────────────────────────────────────────────────────────

_PALETTE_RANK = {ord(c): i for i, c in enumerate(DEFAULT_PALETTE)}
_RANK_LUT = np.full(256, -1, dtype=np.int16)
for _ch, _rk in _PALETTE_RANK.items():
    _RANK_LUT[_ch] = _rk


def luma_from_ascii_frame(frame: np.ndarray) -> np.ndarray:
    """
    Model the on-screen mean brightness of each cell of a decoded ASCII-color
    frame (rows, cols, 4) [char, R, G, B]: rank(char)·BT.601(R,G,B).

    Proportional (not absolute) brightness is all the differential correlator
    needs — density ranks are uniform steps in every bundled palette, so the
    model matches any viewer font/theme up to a global scale.
    """
    chars = frame[:, :, 0]
    rank = _RANK_LUT[chars]
    if (rank < 0).any():
        raise ValueError("frame contains characters outside the default palette")
    r = frame[:, :, 1].astype(np.float32)
    g = frame[:, :, 2].astype(np.float32)
    b = frame[:, :, 3].astype(np.float32)
    tint = 0.299 * r + 0.587 * g + 0.114 * b
    return (rank.astype(np.float32) + 1.0) * (tint + 1.0)


def luma_from_pixel_frame(frame: np.ndarray) -> np.ndarray:
    """BT.601 cell luma of a decoded pixel-mode frame (rows, cols, 3) BGR."""
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


# ────────────────────────────────────────────────────────────────────────────
#  Geometry re-synchronisation — recover from cropped / reframed captures
# ────────────────────────────────────────────────────────────────────────────

def detect_with_sync(images, key, rows: int, cols: int,
                     search_frames: int = 36,
                     scales=None,
                     max_offset: int = 5,
                     block=None, base_crop=None,
                     probe_budget: int = 96,
                     progress=None) -> DetectionResult:
    """
    Detection under unknown canvas geometry — e.g. the capturer cropped a
    few percent of the player border, or the screenshot caught UI/chrome
    around the player.

    Strategy: blind coarse-to-fine search over the canvas→grid fit —
    uniform scale first (CRC pass = instant accept), then a pattern search
    refining scale, then translation in cell units, then independent x/y
    (anisotropic) scale for aspect-changing crops. Every stage is gated by
    the payload CRC-16 (false-accept ≤ 2^-16 per probe), so the search
    exits as soon as it touches adequate geometry; the winning hypothesis is
    then re-confirmed on ALL frames with the full ECC sweep.

    Fit rects extending beyond the captured canvas (border-crop case) are
    zero-padded before the per-cell box resample: a constant fill cancels
    exactly in the detector's differential pair fields, so the surviving
    interior cells alone drive the correlation. With G ≈ cells-bit⁻¹ ≫ 1,
    losing ~14% of cells to the crop costs only ~0.7 dB of margin.

    `images`: iterable of BGR pixel frames (any resolution).
    `scales`: coarse scale scan set (default covers border crops up to
              ~15 %/side and surround-captures down to a 55 % player fill).
    `max_offset`: translation search radius in grid cells.
    `base_crop`: optional (x, y, w, h) canvas rectangle the scan refines.
    `probe_budget`: cap on scanned hypotheses (wrong-key safety valve).
    `progress`: optional callable(str) invoked per probe (CLI tracing).
    Returns DetectionResult extended with geometry keys
    {scale_x, scale_y, dx, dy, probes}.

    Measured coverage (120x40 grid, 48-120 frames — see
    experiments/wm_screen_capture.py):
      * centred border crops up to ~15 %/side (scale 1.19 found at 8 %),
      * surround captures (player filling ~55 %+ of the frame),
      * aspect-changing crops ~5 % on one axis,
      * single-sided strips up to ~3 % (translation walk).
    Deeper one-sided strips sit on a flat, CRC-rejecting correlation
    plateau (registration needs scale ±2 % AND offset ±½ cell
    simultaneously, and the removed chips carry no reference): they are a
    documented limit, not a false-negative risk — the CRC gate never
    accepts a wrong hypothesis.
    """
    if _cv2 is None:  # pragma: no cover
        raise RuntimeError("detect_with_sync needs OpenCV (cv2)")
    frames = list(images)
    if not frames:
        return DetectionResult(ok=False, digits=None, bit_errors=1.0, z=0.0,
                               confidence=0.0, frames_used=0, pairs=0,
                               phase=-1, erasures=-1, block=-1)
    if scales is None:
        # interleaved around 1.0 so identity geometry exits on probe #1
        scales = (1.00, 1.06, 0.94, 1.125, 0.885, 1.19, 0.83, 1.27, 0.78,
                  1.36, 0.72, 1.45)
    REFINE_MIN_Z = 3.0    # null fluctuation ~N(0,1); asymmetric crops can
    # hide under the coarse (dx=dy=0) scan until offsets refine them
    tell = progress if callable(progress) else (lambda _m: None)

    h0, w0 = frames[0].shape[:2]
    bx, by, bw, bh = base_crop if base_crop else (0, 0, w0, h0)
    cell_w, cell_h = bw / cols, bh / rows

    def grids_for(sx, sy, dx, dy, limit):
        # fit rect: scale about the base-rect centre, then translate (dx, dy
        # in grid-cell units); out-of-canvas area → zero pad (see docstring)
        cw, ch = bw * sx, bh * sy
        rx0 = bx + (bw - cw) / 2 + dx * cell_w * sx
        ry0 = by + (bh - ch) / 2 + dy * cell_h * sy
        xi0, yi0 = int(np.floor(rx0)), int(np.floor(ry0))
        xi1, yi1 = int(np.ceil(rx0 + cw)), int(np.ceil(ry0 + ch))
        if xi1 - xi0 < cols // 2 or yi1 - yi0 < rows // 2:
            return None
        pl, pt = max(0, -xi0), max(0, -yi0)
        pr, pb = max(0, xi1 - w0), max(0, yi1 - h0)
        out = []
        for fr in (frames if limit is None else frames[:limit]):
            gray = _cv2.cvtColor(fr, _cv2.COLOR_BGR2GRAY).astype(np.float32)
            if pl or pt or pr or pb:
                gray = np.pad(gray, ((pt, pb), (pl, pr)))
            roi = gray[yi0 + pt: yi1 + pt, xi0 + pl: xi1 + pl]
            out.append(_cv2.resize(roi, (cols, rows),
                                   interpolation=_cv2.INTER_AREA))
        return out

    spent = 0

    def probe(sx, sy, dx, dy):
        """Quick scored hypothesis on `search_frames` frames, or None when
        the geometry is infeasible / the budget is spent."""
        nonlocal spent
        if spent >= probe_budget:
            return None
        grids = grids_for(sx, sy, dx, dy, search_frames)
        if grids is None:
            return None
        spent += 1
        det = WatermarkDetector(key, rows, cols, block=block, null_trials=4)
        for g in grids:
            det.feed_luma_grid(g)
        r = det.decode(erasures=(0, 4))     # cheap ECC sweep for scanning
        tell(f"[sync] probe sx={sx:.3f} sy={sy:.3f} dx={dx:+.1f} dy={dy:+.1f}"
             f" -> z={r.z:7.1f} ok={r.ok}")
        return r

    def better(cand, cur):
        """(ok, z) lexicographic: a CRC pass beats any raw score."""
        return cand is not None and (cur is None
                                     or (cand.ok, cand.z) > (cur[0].ok,
                                                             cur[0].z))

    # Each entry: (DetectionResult, (sx, sy, dx, dy))
    best = None

    def consider(r, geo):
        nonlocal best
        if better(r, best):
            best = (r, geo)

    # ── stage 1: coarse isotropic scale scan at the canvas centre ──
    for s in scales:
        r = probe(s, s, 0.0, 0.0)
        consider(r, (s, s, 0.0, 0.0))
        if r is not None and r.ok:
            break

    # ── stage 1b: coarse translation scan at s=1 (single-sided strips are
    #    invisible to the centred scan: a uniform cell shift dephases every
    #    cell at once. A strip of ≲3 % still leaves a z≥3 gradient here —
    #    scale error only smears correlation gradually until ~±2-4 % — and
    #    the pattern search below refines scale+offsets jointly from it).
    if best is None or not best[0].ok:
        r0 = probe(1.0, 1.0, 0.0, 0.0) if all(
            abs(s - 1.0) > 1e-9 for s in scales) else None
        if r0 is not None:
            consider(r0, (1.0, 1.0, 0.0, 0.0))
        half = min(float(max_offset), 4.0)
        coarse_off = (-half, -half / 2, half / 2, half)
        for dy_c in coarse_off:
            if best is not None and best[0].ok:
                break
            for dx_c in coarse_off:
                r = probe(1.0, 1.0, dx_c, dy_c)
                consider(r, (1.0, 1.0, dx_c, dy_c))
                if r is not None and (r.ok or spent >= probe_budget):
                    break

    # (Deeper geometry than stages 1b+2-4 reach is a measured plateau, not
    #  a tuning problem — see the docstring's coverage list.)
    if best is None:    # every coarse hypothesis was infeasible geometry
        return DetectionResult(ok=False, digits=None, bit_errors=1.0, z=0.0,
                               confidence=0.0, frames_used=len(frames),
                               pairs=0, phase=-1, erasures=-1, block=-1,
                               scale=1.0, scale_x=1.0, scale_y=1.0,
                               dx=0.0, dy=0.0, probes=spent)
    sx = sy = 1.0
    dx = dy = 0.0
    _, (sx, sy, dx, dy) = best

    # ── stages 2-4: pattern-search refinement (only if the coarse scan saw
    #    *something* — a flat null landscape means the mark/geometry family
    #    is absent and refinement would just burn the budget) ──
    if best is not None and not best[0].ok and best[0].z >= REFINE_MIN_Z:
        # stage 2: scale walk
        step = 0.04
        while step >= 0.004 and not best[0].ok:
            improved = False
            for cand in (sx - step, sx + step):
                if not (0.60 <= cand <= 1.60):
                    continue
                r = probe(cand, cand, dx, dy)
                if better(r, best):
                    best = (r, (cand, cand, dx, dy))
                    sx = sy = cand
                    improved = True
                if best[0].ok:
                    break
            if not improved:
                step /= 2.0
        # stage 3: translation walk in grid cells (coarse step, then fine)
        for step in (3.0, 1.0):
            while not best[0].ok:
                improved = False
                for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step),
                                 (step, step), (-step, -step),
                                 (step, -step), (-step, step)):
                    nx = float(np.clip(dx + ddx, -max_offset, max_offset))
                    ny = float(np.clip(dy + ddy, -max_offset, max_offset))
                    if (nx, ny) == (dx, dy):
                        continue
                    r = probe(sx, sy, nx, ny)
                    if better(r, best):
                        best = (r, (sx, sy, nx, ny))
                        dx, dy = nx, ny
                        improved = True
                    if best[0].ok:
                        break
                if not improved:
                    break
        # stage 4: anisotropic polish (aspect-changing crops)
        for step in (0.03, 0.015):
            while not best[0].ok:
                improved = False
                for dsx, dsy in ((step, 0.0), (-step, 0.0),
                                 (0.0, step), (0.0, -step)):
                    nsx, nsy = sx + dsx, sy + dsy
                    if not (0.60 <= nsx <= 1.60 and 0.60 <= nsy <= 1.60):
                        continue
                    r = probe(nsx, nsy, dx, dy)
                    if better(r, best):
                        best = (r, (nsx, nsy, dx, dy))
                        sx, sy = nsx, nsy
                        improved = True
                    if best[0].ok:
                        break
                if not improved:
                    break

    # ── final full-accuracy confirmation at the winning geometry ──
    r, (sx, sy, dx, dy) = best
    det = WatermarkDetector(key, rows, cols, block=block)
    for g in grids_for(sx, sy, dx, dy, None):
        det.feed_luma_grid(g)
    final = det.decode()
    final.update(scale_x=round(sx, 4), scale_y=round(sy, 4),
                 scale=round((sx + sy) / 2, 4), dx=dx, dy=dy, probes=spent)
    if not final.ok and r.ok:
        # Cheap probe passed the CRC but the full decode didn't (extremely
        # marginal geometry): keep the stronger evidence.
        r.update(scale_x=round(sx, 4), scale_y=round(sy, 4),
                 scale=round((sx + sy) / 2, 4), dx=dx, dy=dy, probes=spent)
        return r
    return final
