"""
Equivalence tests for the frame-pipeline hot-path optimizations.

The live server, the static compiler and the vector generator all build the
ASCII colour framebuffer ([char, R, G, B] per cell) from decoded frames. That
assembly was micro-optimized to:
  * a precomputed 256-entry gray->index LUT (replacing per-pixel mul/div/clip)
  * a fused BGR->RGB reversal + bit-drop quantization written straight into the
    framebuffer via np.bitwise_and(..., out=...)

Both must be numerically identical to the original expressions. These tests pin
that equivalence exhaustively (every input byte, every qb value) so future
refactors cannot silently drift.

    pytest test/test_pipeline_mapping.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascii_video_player2 import AsciiMapper, MODE_QUANTIZE_BITS
from stream_server import FILTER_PALETTES


def test_index_lut_matches_original_formula_all_palettes():
    palettes = [AsciiMapper.DEFAULT_PALETTE, *FILTER_PALETTES.values(),
                ["@"], list("abc")]  # incl. degenerate 1-char palettes
    gray = np.arange(256, dtype=np.uint8)
    for pal in palettes:
        mapper = AsciiMapper(palette=pal)
        lut = mapper.index_lut()
        old = (gray.astype(np.uint16) * (mapper._n - 1)) // 255
        np.clip(old, 0, mapper._n - 1, out=old)
        assert lut.dtype == np.uint8
        assert np.array_equal(lut, old.astype(np.uint8))
        # max reachable index is exactly n-1 → downstream clip is provably moot
        assert lut.max() == (mapper._n - 1) if mapper._n > 1 else lut.max() == 0


def test_index_lut_is_cached():
    mapper = AsciiMapper()
    assert mapper.index_lut() is mapper.index_lut()


def test_quantize_mask_matches_shift_roundtrip():
    """x & mask === (x >> qb) << qb for every byte, every qb used (and beyond)."""
    xs = np.arange(256, dtype=np.uint8)
    for qb in range(0, 8):
        mask = np.uint8((0xFF << qb) & 0xFF)
        fused = np.bitwise_and(xs, mask)
        shifted = (xs >> qb) << qb
        assert np.array_equal(fused, shifted), f"qb={qb} mismatch"


def test_mode_quantize_bits_table():
    """The shared mode table must keep the documented color depths."""
    assert MODE_QUANTIZE_BITS == {6: 0, 5: 2, 4: 3, 3: 5, 2: 6}


def test_fused_framebuffer_assembly_matches_reference():
    """Full cell assembly: char plane + quantized RGB, fused vs reference."""
    rng = np.random.default_rng(2024)
    rows, cols = 37, 53
    mapper = AsciiMapper()
    char_byte_lut = np.array([ord(c) for c in mapper._lut], dtype=np.uint8)
    gray = rng.integers(0, 256, (rows, cols), dtype=np.uint8)
    bgr = rng.integers(0, 256, (rows, cols, 3), dtype=np.uint8)

    for qb in (0, 2, 3, 5, 6):
        qb_mask = np.uint8((0xFF << qb) & 0xFF) if qb else None

        # Reference: original expression tree
        indices = (gray.astype(np.uint16) * (mapper._n - 1)) // 255
        np.clip(indices, 0, mapper._n - 1, out=indices)
        ref = np.empty((rows, cols, 4), dtype=np.uint8)
        ref[:, :, 0] = char_byte_lut[indices]
        rgb = bgr[:, :, ::-1]
        if qb > 0:
            rgb = (rgb >> qb) << qb
        ref[:, :, 1:] = rgb

        # Fused: shipped hot path
        out = np.empty((rows, cols, 4), dtype=np.uint8)
        out[:, :, 0] = char_byte_lut[mapper.index_lut()[gray]]
        if qb_mask is not None:
            np.bitwise_and(bgr[:, :, ::-1], qb_mask, out=out[:, :, 1:])
        else:
            out[:, :, 1:] = bgr[:, :, ::-1]

        assert np.array_equal(out, ref), f"qb={qb} assembly mismatch"
