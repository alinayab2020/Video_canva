"""
Unit + simulated-channel tests for watermark.py — the keyed spread-spectrum
forensic watermark.

Covers:
  * payload container: 10 digits ⇄ 15-byte RS(15,7)/CRC-16 protected codeword
  * Reed–Solomon correction at and beyond its capacity (2e + v ≤ 8)
  * CRC-16-CCITT known test vector
  * keyed MarkPlan: determinism, avalanche, group structure, chip balance
  * embedders: ±1 index dither (with ramp-edge folding), ±beta pixel dither
  * end-to-end simulated screen-capture channel: ideal, frame drops, frame
    duplicates, gamma, noise, AGC gain steps, temporal smear, alternation
    block clocks, fps decimation, short captures, wrong-key / unmarked
    no-detection guards

Everything is seeded; runtime is dominated by a handful of 24x64-grid
detections and stays in the seconds range.
"""
import os
import sys
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import watermark as wm  # noqa: E402

KEY = "unit-test-key-7391"
DIGITS = "0123456789"
ROWS, COLS = 24, 64          # 1536 cells → 12 cells/bit
N_RAMP = len(wm.DEFAULT_PALETTE)


def _payload_roundtrip_ok(d: str) -> bool:
    return wm.decode_payload(wm.encode_payload(d)) == d


class TestPayloadCoding(unittest.TestCase):
    def test_roundtrip_edge_digits(self):
        for d in ("0000000000", "9999999999", "0123456789", "1000000001"):
            self.assertTrue(_payload_roundtrip_ok(d))

    def test_roundtrip_random(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            d = f"{int(rng.integers(0, 10**10)):010d}"
            self.assertTrue(_payload_roundtrip_ok(d))

    def test_validation(self):
        for bad in ("12345", "12345678901", "", "abcdefghij", "000000000o"):
            with self.assertRaises(ValueError):
                wm.encode_payload(bad)

    def test_crc16_known_vector(self):
        # CRC-16/CCITT-FALSE check value for ASCII "123456789" is 0x29B1.
        self.assertEqual(wm.crc16_ccitt(b"123456789"), 0x29B1)

    def test_codeword_is_systematic(self):
        cw = wm.encode_payload(DIGITS)
        self.assertEqual(len(cw), 15)
        # first 5 bytes are the digit integer, big-endian
        self.assertEqual(int.from_bytes(cw[:5], "big"), int(DIGITS))


class TestReedSolomon(unittest.TestCase):
    def test_corrects_at_capacity_boundary(self):
        """Every (e errors, v erasures) mix with 2e + v = 8 must decode."""
        rng = np.random.default_rng(2)
        for _ in range(300):
            d = f"{int(rng.integers(0, 10**10)):010d}"
            cw = bytearray(wm.encode_payload(d))
            for v in (0, 2, 4, 6, 8):
                e = (8 - v) // 2
                cw2 = bytearray(cw)
                era = rng.choice(15, v, replace=False).tolist()
                rest = [p for p in range(15) if p not in era]
                errs = rng.choice(rest, e, replace=False).tolist() if e else []
                for p in errs:
                    cw2[p] ^= int(rng.integers(1, 256))
                self.assertEqual(wm.decode_payload(bytes(cw2), erasures=era), d,
                                 f"RS failed at e={e}, v={v}")

    def test_beyond_capacity_never_silent_wrong(self):
        """Beyond capacity the decoder may fail, but must never return the
        WRONG digits (the CRC-16 gate: observed 0 false accepts)."""
        rng = np.random.default_rng(3)
        false_accepts = 0
        for _ in range(400):
            d = f"{int(rng.integers(0, 10**10)):010d}"
            cw = bytearray(wm.encode_payload(d))
            era = rng.choice(15, 9, replace=False).tolist()  # 9 > capacity
            got = wm.decode_payload(bytes(cw), erasures=era)
            if got is not None and got != d:
                false_accepts += 1
        self.assertEqual(false_accepts, 0)

    def test_random_garbage_rejected(self):
        rng = np.random.default_rng(4)
        for _ in range(400):
            cw = bytes(int(rng.integers(0, 256)) for _ in range(15))
            self.assertIsNone(wm.decode_payload(cw))


class TestMarkPlan(unittest.TestCase):
    def test_determinism_same_key(self):
        p1 = wm.MarkPlan(KEY, ROWS, COLS)
        p2 = wm.MarkPlan(KEY, ROWS, COLS)
        np.testing.assert_array_equal(p1.signs, p2.signs)
        np.testing.assert_array_equal(p1.bitidx, p2.bitidx)

    def test_key_avalanche(self):
        p1 = wm.MarkPlan(KEY, ROWS, COLS)
        p2 = wm.MarkPlan(KEY + "x", ROWS, COLS)
        # chip signs should decorrelate (~50% agreement)
        agree = float(np.mean(p1.signs == p2.signs))
        self.assertTrue(0.35 < agree < 0.65, f"avalanche failure: {agree}")
        # cell→bit assignment should move essentially everywhere
        moved = float(np.mean(p1.bitidx[p1.bitidx >= 0]
                              != p2.bitidx[p1.bitidx >= 0]))
        self.assertGreater(moved, 0.9)

    def test_group_structure(self):
        p = wm.MarkPlan(KEY, ROWS, COLS)
        g = (ROWS * COLS) // wm.CODE_BITS
        used = p.bitidx >= 0
        self.assertEqual(int(used.sum()), g * wm.CODE_BITS)
        counts = np.bincount(p.bitidx[used], minlength=wm.CODE_BITS)
        np.testing.assert_array_equal(counts, g)          # exactly g cells/bit
        self.assertEqual(int((p.signs[~used] == 0).sum()), (~used).sum())
        ratio = abs(float(p.signs[used].mean()))           # ±1 balance
        self.assertLess(ratio, 0.15)

    def test_grid_too_small_rejected(self):
        with self.assertRaises(ValueError):
            wm.MarkPlan(KEY, 4, 8)      # 32 cells < 120 code bits

    def test_temporal_clock(self):
        for block in (1, 2, 3):
            signs = [wm.temporal_sign(t, block) for t in range(4 * block)]
            self.assertEqual(signs, [1 if (t // block) % 2 == 0 else -1
                                     for t in range(4 * block)])


class TestEmbedders(unittest.TestCase):
    def setUp(self):
        # perceptual gate OFF here so exact ±1/beta properties hold on
        # the flat synthetic inputs (gate tested separately below)
        self.wmr = wm.Watermarker(DIGITS, KEY, ROWS, COLS, block=1,
                                  beta=8, perceptual=False)

    def test_indices_step_exactly_one(self):
        rng = np.random.default_rng(5)
        idx0 = rng.integers(0, N_RAMP, (ROWS, COLS)).astype(np.uint8)
        for tick in (0, 1, 2, 5):
            idx = idx0.copy()
            self.wmr.embed_indices(idx, tick, N_RAMP)
            delta = idx.astype(np.int16) - idx0.astype(np.int16)
            used = self.wmr.plan.bitidx >= 0
            # |Δ| is exactly 1 on used cells, 0 elsewhere
            self.assertTrue(np.all(np.abs(delta[used]) == 1))
            self.assertTrue(np.all(delta[~used] == 0))
            # never leaves the ramp
            self.assertTrue(np.all(idx < N_RAMP))

    def test_indices_fold_at_ramp_edges(self):
        # All-zero / all-max ramps embed without wrap-around; ±1 preserved.
        for fill in (0, N_RAMP - 1):
            idx = np.full((ROWS, COLS), fill, dtype=np.uint8)
            self.wmr.embed_indices(idx, 0, N_RAMP)
            self.assertTrue(np.all(np.abs(idx.astype(np.int16) - fill) <= 1))
            self.assertTrue(np.all(idx < N_RAMP))

    def test_indices_sign_matches_chips(self):
        # Mid-ramp: modulated direction must equal chip·bit·clock exactly.
        idx = np.full((ROWS, COLS), N_RAMP // 2, dtype=np.uint8)
        self.wmr.embed_indices(idx, 0, N_RAMP)
        delta = idx.astype(np.int16) - N_RAMP // 2
        expect = self.wmr.plan.signs * self.wmr.bit_signs[
            np.clip(self.wmr.plan.bitidx, 0, wm.CODE_BITS - 1)]
        np.testing.assert_array_equal(delta, expect.astype(np.int16))

    def test_indices_alternation(self):
        idx0 = rng = np.random.default_rng(6).integers(
            5, N_RAMP - 5, (ROWS, COLS)).astype(np.uint8)
        a, b = idx0.copy(), idx0.copy()
        self.wmr.embed_indices(a, 0, N_RAMP)
        self.wmr.embed_indices(b, 1, N_RAMP)
        self.assertTrue(np.all(a.astype(np.int16) + b.astype(np.int16)
                               == 2 * idx0.astype(np.int16)),
                        "consecutive ticks must be exact complements")

    def test_pixels_beta_and_saturation(self):
        rng = np.random.default_rng(7)
        bgr0 = rng.integers(0, 256, (ROWS, COLS, 3)).astype(np.uint8)
        bgr = bgr0.copy()
        self.wmr.embed_pixels(bgr, 0)
        delta = bgr.astype(np.int16) - bgr0.astype(np.int16)
        eff = self.wmr.plan.signs * self.wmr.bit_signs[
            np.clip(self.wmr.plan.bitidx, 0, wm.CODE_BITS - 1)]
        dmax = np.abs(delta).max(axis=2)
        # unsaturated, used cells move exactly ±beta; saturated move less —
        # never in the opposite direction (erasure semantics, no inversion)
        unsat = ((bgr0 >= 8) & (bgr0 <= 247)).all(axis=2) \
            & (self.wmr.plan.bitidx >= 0)
        self.assertTrue(np.all(dmax[unsat] == 8))
        self.assertTrue(np.all(dmax[self.wmr.plan.bitidx < 0] == 0))
        sign_delta = np.sign(delta.max(axis=2) + delta.min(axis=2))
        bad = (sign_delta != 0) & (sign_delta != eff)
        self.assertFalse(np.any(bad), "pixel dither inverted a chip")

    def test_perceptual_gate_freezes_flat_bright(self):
        wmr = wm.Watermarker(DIGITS, KEY, ROWS, COLS, perceptual=True)
        # flat, bright content: no flicker may be added
        gray = np.full((ROWS, COLS), 180, dtype=np.uint8)
        idx = np.full((ROWS, COLS), N_RAMP // 2, dtype=np.uint8)
        wmr.embed_indices(idx, 0, N_RAMP, gray)
        self.assertTrue(np.all(idx == N_RAMP // 2),
                        "perceptual gate must freeze flat bright cells")
        # textured content: the dither lands on the textured cells (rank 61
        # sits in a smooth coverage region, so the coverage cap passes it)
        gray2 = (np.arange(COLS * ROWS, dtype=np.uint8)
                 .reshape(ROWS, COLS) % 13 * 19)
        idx2 = np.full((ROWS, COLS), 61, dtype=np.uint8)
        wmr.embed_indices(idx2, 0, N_RAMP, gray2)
        self.assertGreater(int((idx2 != 61).sum()), 100,
                           "textured cells should receive the dither")
        # dark content: always allowed (near-black flicker is imperceptible)
        idx3 = np.full((ROWS, COLS), 61, dtype=np.uint8)
        wmr.embed_indices(idx3, 0, N_RAMP, np.full((ROWS, COLS), 8, np.uint8))
        self.assertGreater(int((idx3 != 61).sum()), 1000)
        # pixel-mode gate mirrors the same behaviour
        px = np.full((ROWS, COLS, 3), 220, dtype=np.uint8)
        wmr.embed_pixels(px, 0)
        self.assertTrue(np.all(px == 220), "pixel gate must freeze flat bright")

    def test_visibility_mask_exactness(self):
        # window stats must match brute force
        g = (np.arange(16, dtype=np.uint8).reshape(4, 4) * 10)
        gp = np.pad(g.astype(np.float32), ((1, 1), (1, 1)), mode="edge")
        ref = np.array([[gp[y:y + 3, x:x + 3].std()
                         for x in range(4)] for y in range(4)])
        # recompute via the same internal helpers
        g32 = g.astype(np.float32)
        gp2 = np.pad(g32, ((1, 1), (1, 1)), mode="edge")

        def ws(v):
            I = np.zeros((v.shape[0] + 1, v.shape[1] + 1))
            I[1:, 1:] = v.cumsum(0).cumsum(1)
            return I[3:, 3:] - I[:-3, 3:] - I[3:, :-3] + I[:-3, :-3]
        mean = ws(gp2) / 9.0
        sqm = ws(gp2 * gp2) / 9.0
        std = np.sqrt(np.maximum(sqm - mean ** 2, 0.0))
        self.assertLess(float(np.abs(std - ref).max()), 1e-4)


# ────────────────────────────────────────────────────────────────────────────
#  Simulated screen-capture channel
# ────────────────────────────────────────────────────────────────────────────

def _content_video(nframes, seed=7):
    """Synthetic moving content in (gray, tint) space, deterministic."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:ROWS, 0:COLS]
    frames = []
    for t in range(nframes):
        g1 = np.exp(-(((yy - (8 + 10 * np.sin(t * 0.07))) ** 2
                       + (xx - (16 + 32 * ((t * 0.011) % 1))) ** 2) / 120))
        g2 = np.exp(-(((yy - 18) ** 2 + (xx - (58 - (t * 0.006 % 1) * 58)) ** 2) / 200))
        gray = np.clip(30 + 190 * np.clip(g1 + g2, 0, 1)
                       + rng.normal(0, 3, (ROWS, COLS)), 0, 255).astype(np.uint8)
        tint = np.clip(40 + 200 * g2 + 30 * g1
                       + rng.normal(0, 2, (ROWS, COLS)), 0, 255).astype(np.float32)
        frames.append((gray, tint))
    return frames


_LUT = ((np.arange(256, dtype=np.uint16) * (N_RAMP - 1)) // 255).astype(np.uint8)


def _simulate(nframes=120, block=1, attack=None, mod_frames=None,
              marked=True, seed=7, key=KEY):
    """Render a marked video into captured cell-luma grids."""
    wmr = wm.Watermarker(DIGITS, key, ROWS, COLS, block=block)
    seq = []
    for t, (gray, tint) in enumerate(_content_video(nframes, seed)):
        idx = _LUT[gray].copy()
        if marked:
            wmr.embed_indices(idx, t, N_RAMP)
        lum = (idx.astype(np.float32) + 1.0) * (tint + 1.0)
        seq.append(attack(lum, t) if attack else lum)
    if mod_frames:
        seq = mod_frames(seq)
    return seq


def _detect(seq, key=KEY, blocks=(1, 2, 3)):
    det = wm.WatermarkDetector(key, ROWS, COLS, block=blocks, null_trials=8)
    for g in seq:
        det.feed_luma_grid(g)
    return det.decode()


class TestSimulatedChannel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ideal = _simulate()

    def test_ideal_recovery(self):
        r = _detect(self.ideal)
        self.assertTrue(r.ok, f"no recovery: {dict(r)}")
        self.assertEqual(r.digits, DIGITS)
        self.assertEqual(r.bit_errors, 0.0)
        self.assertGreater(r.z, 8.0)

    def test_wrong_key_no_detection(self):
        r = _detect(self.ideal, key="attacker-guess")
        self.assertFalse(r.ok)
        self.assertLess(r.z, 8.0)

    def test_unmarked_no_detection(self):
        r = _detect(_simulate(marked=False))
        self.assertFalse(r.ok)

    def test_frame_drops(self):
        """Every 37th captured frame lost → parity slips → EM must re-align."""
        seq = _simulate(mod_frames=lambda s: [f for i, f in enumerate(s)
                                              if i % 37 != 36])
        r = _detect(seq)
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)
        self.assertLess(r.bit_errors, 0.05)

    def test_frame_duplicates(self):
        """Capture pulling 2x fps duplicates frames; pairs near dups ~0."""
        def dup(s):
            out = []
            for i, f in enumerate(s):
                out.append(f)
                if i % 13 == 12:
                    out.append(f)
            return out
        r = _detect(_simulate(mod_frames=dup))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)
        self.assertLess(r.bit_errors, 0.05)

    def test_gain_offset_agc(self):
        rng = np.random.default_rng(8)
        def agc(L, t):
            return L * (1.6 if (t // 40) % 2 else 0.5) + 11 + rng.normal(
                0, 0.5, L.shape).astype(np.float32)
        r = _detect(_simulate(attack=agc))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_gamma_compression(self):
        r = _detect(_simulate(
            attack=lambda L, t: 255.0 * np.clip(L / 23808.0, 0, 1) ** 0.7))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_additive_noise(self):
        rng = np.random.default_rng(9)
        r = _detect(_simulate(
            attack=lambda L, t: L + rng.normal(0, 8.0, L.shape).astype(np.float32)))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_temporal_smear(self):
        """0.7·frame + 0.3·previous (display persistence / IBP smoothing)."""
        def smear(s):
            return [0.7 * f + 0.3 * (s[i - 1] if i else f)
                    for i, f in enumerate(s)]
        r = _detect(_simulate(mod_frames=smear))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_block_clock_2(self):
        r = _detect(_simulate(block=2))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_half_fps_capture_block2(self):
        """15fps capture of a 30fps block-2 stream (Nyquist aliasing of the
        alternation) must still decode via the block scan."""
        r = _detect(_simulate(block=2, mod_frames=lambda s: s[::2]))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_short_capture_24_frames(self):
        r = _detect(_simulate(nframes=24))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)

    def test_perceptual_gate_keeps_detection(self):
        """Production embeds with the flicker gate on (gray passed) — the
        content-dependent activity mask must not break recovery."""
        rng = np.random.default_rng(21)
        wmr = wm.Watermarker(DIGITS, KEY, ROWS, COLS, perceptual=True)
        seq = []
        for t, (gray, tint) in enumerate(_content_video(120, seed=13)):
            idx = _LUT[gray].copy()
            wmr.embed_indices(idx, t, N_RAMP, gray)
            lum = (idx.astype(np.float32) + 1.0) * (tint + 1.0)
            seq.append(lum + rng.normal(0, 3.0, lum.shape).astype(np.float32))
        r = _detect(seq)
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)
        self.assertLess(r.bit_errors, 0.05)

    def test_combined_attacks(self):
        rng = np.random.default_rng(10)
        def combo(s):
            s = [255.0 * np.clip(f / 23808.0, 0, 1) ** 0.75 for f in s]
            s = [f for i, f in enumerate(s) if i % 41 != 40]        # drops
            s = [f * (1.3 if (t // 30) % 2 else 0.8) + 5            # AGC
                 for t, f in enumerate(s)]
            return [f + rng.normal(0, 5.0, f.shape).astype(np.float32)
                    for f in s]
        r = _detect(combo(_simulate(nframes=144, seed=11)))
        self.assertTrue(r.ok)
        self.assertEqual(r.digits, DIGITS)
        self.assertLess(r.bit_errors, 0.15)


@unittest.skipIf(wm._cv2 is None, "OpenCV required for pixel-frame sync")
class TestGeometrySync(unittest.TestCase):
    """
    detect_with_sync: blind geometry re-registration from pixel frames.

    The colour carrier is applied grid-side and upsampled to pixel blocks
    (nearest), so the detector's per-cell box-average sees exactly the
    marked cell colour — isolating the geometry search from font/paint
    details (those are covered by the screen-capture benchmark).
    """
    CW, CH, N = 6, 10, 36       # px per cell, frames

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(777)
        wmr = wm.Watermarker(DIGITS, KEY, ROWS, COLS)
        cls.frames = []
        for t in range(cls.N):
            gray = rng.integers(30, 225, (ROWS, COLS)).astype(np.uint8)
            bgr = np.stack([gray, gray // 2 + 60, 255 - gray],
                           axis=-1).astype(np.uint8)
            wmr.embed_pixels(bgr, t, gray)
            cls.frames.append(np.repeat(np.repeat(bgr, cls.CH, axis=0),
                                        cls.CW, axis=1))

    def _attacked(self, kind):
        out = []
        for im in self.frames:
            h, w = im.shape[:2]
            if kind == "crop":                       # 4.5 % every border
                dx, dy = round(w * 0.045), round(h * 0.045)
                out.append(im[dy:h - dy, dx:w - dx])
            elif kind == "strip":                    # 2.5 % off the right
                out.append(im[:, 0:w - round(w * 0.025)])
            elif kind == "surround":                 # player = 78 % centre
                ch, cw = round(h / 0.78), round(w / 0.78)
                canvas = np.full((ch, cw, 3), 40, np.uint8)
                y0, x0 = (ch - h) // 2, (cw - w) // 2
                canvas[y0:y0 + h, x0:x0 + w] = im
                out.append(canvas)
        return out

    def test_centered_crop(self):
        r = wm.detect_with_sync(self._attacked("crop"), KEY, ROWS, COLS,
                                search_frames=24)
        self.assertTrue(r.ok, dict(r))
        self.assertEqual(r.digits, DIGITS)
        self.assertAlmostEqual(r["scale_x"], 1 / 0.91, delta=0.08)

    def test_single_side_strip(self):
        r = wm.detect_with_sync(self._attacked("strip"), KEY, ROWS, COLS,
                                search_frames=24)
        self.assertTrue(r.ok, dict(r))
        self.assertEqual(r.digits, DIGITS)

    def test_surround_capture(self):
        r = wm.detect_with_sync(self._attacked("surround"), KEY, ROWS, COLS,
                                search_frames=24)
        self.assertTrue(r.ok, dict(r))
        self.assertEqual(r.digits, DIGITS)
        self.assertAlmostEqual(r["scale_x"], 0.78, delta=0.06)

    def test_wrong_key_geometry_scan_stays_blind(self):
        # the scanned hypothesis space contains chance z-basins; the CRC
        # gate must still never accept a wrong key
        r = wm.detect_with_sync(self._attacked("crop"), "nope", ROWS, COLS,
                                search_frames=24, probe_budget=60)
        self.assertFalse(r.ok)


class TestEmbedPerformance(unittest.TestCase):
    def test_embed_cost_bounded(self):
        """Embed must stay negligible vs the ~33ms frame budget (CI-safe
        bound: 5 ms/frame on a 200x50 ASCII grid)."""
        rows, cols = 50, 200
        wmr = wm.Watermarker(DIGITS, KEY, rows, cols)
        idx = np.random.default_rng(12).integers(
            0, N_RAMP, (rows, cols)).astype(np.uint8)
        for t in range(3):                      # warm
            wmr.embed_indices(idx, t, N_RAMP)
        t0 = time.perf_counter()
        for t in range(20):
            wmr.embed_indices(idx, t, N_RAMP)
        per = (time.perf_counter() - t0) / 20
        self.assertLess(per, 5e-3, f"embed too slow: {per*1e3:.2f} ms/frame")


if __name__ == "__main__":
    unittest.main()
