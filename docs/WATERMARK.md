# Forensic Watermarking — Design, Security and Robustness

ASCILINE can invisibly burn a **10-digit identifier** into every stream and
compiled `.ascf` file. The mark is designed to survive the one channel no
container-level DRM can touch: **a camera or software pointing at the
screen** — screenshots, screen recordings (including lossy H.264
re-encodes of the captured footage), zoomed players and sloppy crops —
while remaining statistically invisible to viewers and **undetectable and
unremovable without the secret key**.

This document is the full engineering treatment: threat model, the math of
the scheme, operational usage, measured robustness, and honest limits.

---

## 1. Threat model

| Adversary capability | In scope? |
| :------------------- | :-------: |
| Full-quality screenshot / screen recording of the player | ✅ |
| Re-encoding the capture (H.264 CRF 23–35, JPEG q50–70, phone-upload chains) | ✅ |
| Digital zoom, arbitrary display scaling, brightness/contrast/gamma, AGC | ✅ |
| Dropping/duplicating frames, fps decimation (30 → 30/k fps) | ✅ |
| Cropping: centred border crops (≤15 %/side), surround captures, aspect crops, ≤3 % single-sided strips, crop-then-stretch | ✅ |
| Deep single-sided strips (>3 %), rotation/keystoning, frame-averaging with an aligned second copy | ⚠️ documented limits |
| **Removing or reading the mark without the key** | ❌ computationally infeasible by design |

The security never rests on “the attacker can’t see it”: the mark is a
keyed spread-spectrum signal. Without the key it is statistically
indistinguishable from content noise (§4), and every positive detection is
guarded by a CRC-16 (false-accept ≤ 2⁻¹⁶ per tested hypothesis) plus an
empirical wrong-key z-score (§6).

## 2. What gets embedded

```
10 decimal digits  →  33-bit info
└─► 5-byte container (4 B digits BCD + 1 B) ─► +CRC-16-CCITT  (7 B)
    └─► RS(15, 7) over GF(2⁸)  (fcr=0, prim 0x11d)
        └─► 15 bytes = 120 code bits, soft-LLR detected, erased up to 8 bytes
```

* **CRC-16-CCITT** — the acceptance gate. A random 120-bit vector passes
  with probability 2⁻¹⁶; mis-keys, mis-registrations and null hypotheses
  are rejected here, never by eyeballing a score.
* **RS(15, 7)** corrects any mix `2·errors + erasures ≤ 8` codeword bytes.
  Detection is soft: bits are accumulated as log-likelihoods, the |LLR|
  per byte selects erasure candidates, and a sweep of erasure counts
  {0, 2, 4, 6, 8} × both global signs is tried against the CRC —
  bit-error rates up to ~25 % still decode.

## 3. The carrier (why it is invisible *and* robust)

Each of the 120 code bits is spread by a keyed CDMA chip pattern
(Cox et al., *Secure Spread Spectrum Watermarking for Multimedia*, IEEE
TIP 1997) over `G = rows·cols/120` grid cells (G = 40 on a 120×40 grid),
with a keyed ±1 sign per cell. Every frame carries all 120 bits; a few
seconds of footage integrate tens of thousands of chips per bit.

The chip modulates the **cell colour luminance by ±β (default β = 8)**
— glyphs stay *structurally identical*; only their tint shifts:

* **Font-agnostic polarity.** The detector observes `ΔL ≈ ink_fraction ·
  β · sign`. Ink coverage can only *scale* the magnitude, never flip the
  sign — so detection works regardless of the viewer's font/theme. (A
  glyph-rank swap, the obvious ASCII-native carrier, fails this: it is
  font-fragile and visibly flickers. It remains only for monochrome
  `-m 1`, where colours don't exist.)
* **Saturation = erasure, never inversion.** Cells clipped at 0/255
  carry no signal; they cannot inject counter-chips.
* **Perceptual flicker gate.** A 30 Hz ±8 flicker is only masked where
  the HVS can't resolve it: textured cells (luminance-contrast masking)
  or very dark ones. The gate (|∇| + 3×3 σ > threshold, or luma < 22)
  freezes flat bright regions pixel-exactly. Measured cost of the gate:
  flat-card PSNR = **99 dB** (i.e. *zero* change — the gate turns the
  mark off there), video PSNR **38 dB / SSIM 0.9964** vs unmarked.

Signalling is **temporally differential**: the whole PN field flips sign
every `block` frames (default 1). A capturer who subtracts adjacent
frames cancels the content (which barely moves at 30 fps) and *doubles*
the mark — the classic low-spatiotemporal placement of screen-cam-robust
cinema forensic marks. Detection pairs frames `(s·(R[a] − R[b]))`,
normalises per-cell by MAD (motion-heavy cells are down-weighted,
maximum-ratio combining), projects onto chips, and re-aligns the segment
parities blindly with EM (frames dropped/duplicated by the capture flip
the alternation parity for whole segments).

## 4. Keyed security

All spreading sequences, cell→bit permutations and signs derive from a
256-bit key via **HMAC-SHA256 in counter mode** (`MarkPlan`). Without the
key:

* the mark is a zero-mean ±8/255 perturbation decorrelated across cells
  and frames — to anyone, including the watermark holder's *other*
  customers (each key is an independent universe);
* removal requires subtracting a pattern you cannot know; blind
  averaging attacks need frame-accurate alignment *and* content
  cancellation, and the differential signalling turns naive averaging
  into preservaton of the mark (it flips, it doesn't smear);
* detection with a wrong key is empirically z ≈ 0–3σ (null), and the CRC
  gate rejects at ≤ 2⁻¹⁶ per hypothesis. Measured over the full
  18-attack benchmark battery: **0 wrong-key detections**, including
  through the full geometry-scan space, which contains chance
  correlation basins (z up to ~30 are seen and CRC-rejected).

## 5. Geometry: blind re-registration from the capture

Screen captures rarely frame the player exactly. `detect_with_sync`
recovers the canvas→grid fit blindly:

1. **Zero-pad exact resampling.** Any fit rectangle may extend beyond the
   captured canvas; out-of-canvas area is zero-padded before the per-cell
   box-average. A constant fill cancels *exactly* in the differential
   pair fields, so surviving interior cells alone drive the correlation —
   losing ~14 % of cells to a crop costs ≈0.7 dB of margin.
2. **Coarse→fine CRC-gated search.** Uniform scale scan (identity first;
   covers crops to ~15 %/side and surrounds down to a 55 % player fill),
   then a translation mesh at s = 1 (single-sided strips), then a pattern
   search refining scale → cell offsets → anisotropic (aspect-changing)
   scale. *Every* stage is CRC-gated: the search exits the moment a
   hypothesis decodes, then re-confirms on all frames with the full ECC
   sweep. A probe budget caps the wrong-key cost.
3. **Honest plateau.** Registration coherence needs scale within ≈±2 %
   *and* offset within ≈±½ cell simultaneously. Deeper one-sided strips
   (>3 % of the frame, i.e. >~3 cells of pure shift with no reference
   left from the removed chips) sit on a flat, always-CRC-rejecting
   plateau — measured, documented, and not a false-negative *risk* (the
   worst case is “not detected”, never “wrong digits”).

## 6. Temporal registration: fps decimation & frame slips

The detector scans alternation clocks `block ∈ {1, 2, 3, 4, 6}` × pairing
phases {0, 1}. A 1/k-fps capture of a block-k stream *reads as* a block-1
stream, so this set covers both the embed clock and the fps-decimation
aliases, and the winning block/phase/erasure counts are reported for
audit.

Fundamental (Nyquist) limits, by design of any alternated mark:
block = 1 embed + exactly-½-rate capture aliases the mark to DC
(unrecoverable *in that capture* — capture at any other rate, or embed
with `--watermark-block 2` when 15 fps rips are expected). The server
advances the clock on *sent* frames only, so server-side backpressure
drops never desync the alternation.

## 7. Usage

### Embed — live server

```bash
./ascil video.mp4 --watermark 0123456789 --watermark-key "$KEY"
# or keep the key out of the process list:
export ASCILINE_WM_KEY="…"
./ascil video.mp4 --watermark 0123456789
```

Options: `--watermark-block N` (default 1; use 2 for 15 fps captures),
`--watermark-beta 1-64` (default 8). The server prints only the key
fingerprint `sha256(key)[:10]` — never the key.

Works in every pipeline: `--pixel`, ASCII modes 1–6, with the adaptive
codec on or off (the char plane is bit-exact; the colour plane dither
rides above quantisation — see tolerances in §8).

### Embed — offline compiler

```bash
python compiler.py video.mp4 out.ascf --mode 4 \
    --watermark 0123456789 --watermark-key "$KEY"
```

(`--profile` quality tiers are rejected with `--watermark`: their
colour-plane tolerance can exceed the dither amplitude.)

### Detect — from a capture

```bash
# capture = screenshot folder, video file, anything OpenCV reads
python tools/watermark_detect.py cap.mp4 --rows 40 --cols 120 \
    --key "$KEY" --frames 240

# geometry scan for cropped/zoomed/surround captures
python tools/watermark_detect.py cap.png --rows 40 --cols 120 \
    --key "$KEY" --sync -v          # -v traces every probe

# if you know the player rect in the capture (analyst hint):
python tools/watermark_detect.py cap.mp4 --rows 40 --cols 120 \
    --key "$KEY" --sync --crop 60,120,966,400
```

Exit code is 0 *only* on a CRC-passing decode; the report carries
`digits`, z-score, bit-error rate, winning block/phase/erasures and the
recovered geometry (`scale_x/scale_y/dx/dy`, probe count). `--json` for
pipelines.

## 8. Measured robustness

`experiments/wm_screen_capture.py` — fully deterministic: synthetic movie
→ production embedder → browser-faithful pixel render (8 px bold mono
glyphs, per-cell tint, #050505 bg) → attack → blind keyed detection.
120 frames (4 s), 120×40 grid, 720×400 px render.

| Capture attack | Recovered | BER | z |
| :------------- | :-------: | --: | -: |
| lossless screenshot (PNG) | ✅ | 0.000 | 92.0 |
| screen zoom 50 % / 75 % / 150 % | ✅ | 0.000 | 91–93 |
| H.264 screen-recording CRF 23 / 28 / 35 | ✅ | 0.000 | 65–88 |
| JPEG q=70 / q=50 | ✅ | 0.000 | 91–92 |
| sensor noise σ = 6/255 | ✅ | 0.000 | 89.7 |
| bright ×1.12, contrast ×0.88, γ 1.25 | ✅ | 0.000 | 94.2 |
| 3.5 % border crop | ✅ | 0.000 | 64.0 |
| 8 % border crop | ✅ | 0.000 | 72.5 |
| 2.5 % single-sided strip | ✅ | 0.017 | 16.4 |
| player @ 80 % of capture (UI around) | ✅ | 0.025 | 23.5 |
| 5 %+2 % aspect-changing crop | ✅ | 0.017 | 15.9 |
| 3.5 % crop + stretch-back | ✅ | 0.000 | 51.3 |
| phone-upload chain (zoom 0.85 → CRF 32 → JPEG q60 → noise) | ✅ | 0.000 | 66.7 |
| **wrong key on every attack above** | **blind, 0 leaks** | — | ≈0–3 |

Known limit (not counted, reported by the bench to prove the CRC gate
never false-accepts): deep single-sided strips (6 %+5 %) — flat
correlation plateau, z ≈ 1.4, no detection.

### Practical guidance

* **β vs codec tolerance.** With the adaptive codec's `--quality` tiers,
  colour-plane delta tolerance can swallow a ±8 dither; keep `lossless`
  for watermarked streams, or set `--watermark-beta > 2·tolerance + 4`
  (default β = 8 pairs with the lossless default).
* **β vs stealth.** β = 8 on glyph *tints* is PSNR-38 dB invisible on
  moving content and gated off entirely on flat cards. Raise to 12–16
  for hostile re-encode margins; the gate still holds flicker down.
* **Frame budget.** 120 frames (4 s) already yields z ≥ 15 on the worst
  measured attack; 30–60 frames suffice for clean captures. Detection
  integrates linearly in chips, so z grows ~√frames.
* **Grid size.** Larger grids raise G = rows·cols/120 (more cells per
  bit); everything else scales with it.

## 9. What this is not

* Not DRM: a capturing viewer can *re-watch* content; the mark *identifies
  the leak source* after the fact. That is the entire point of forensic
  marking (same model as cinema screen-cam watermarks).
* Not a proof of viewing: presence of the digits in a capture proves the
  capture came from a stream keyed to that ID; it says nothing about who
  held the camera.
* Not rotation/keystoning-proof (phone pointed at a slanted screen):
  that needs perspective feature registration, a deliberate next step
  (keyed pilot tones are the standard answer; see §5's plateau).

## References

* Cox, Kilian, Leighton, Shamoon — *Secure Spread Spectrum Watermarking
  for Multimedia*, IEEE Trans. Image Processing, 1997.
* Perry/Macq — differential (informed-detection) placement strategies in
  screen-cam forensic marking practice.
* Reed & Solomon (1960); errors-and-erasures decoding via
  Berlekamp–Massey with a GF(2⁸) Vandermonde magnitude solve (this repo,
  `watermark.py`).
* Watson (1993) luminance-contrast masking model for the flicker gate.
