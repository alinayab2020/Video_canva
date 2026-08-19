#!/usr/bin/env python3
"""
wm_screen_capture.py — screen-capture robustness benchmark for the ASCILINE
forensic watermark.

Pipeline (everything offline, deterministic):
  1. synthesise a short "movie" on an ASCII grid (rich motion, dark/bright
     scenes, high-frequency detail),
  2. embed the 10-digit mark with watermark.Watermarker (ASCII mode),
  3. render the stream to ACTUAL PIXELS exactly like the browser client
     (8px bold monospace glyphs tinted per cell on #050505),
  4. run a battery of capture attacks: pure screenshot, rescale (screen
     zoom), H.264 re-encode (screen recording), JPEG re-compression,
     noise, brightness/contrast/gamma, border crop + rescale, and a
     combined worst-case "phone-upload" chain,
  5. attempt recovery from the atttacked pixels (blind, keyed detector)
     and report digits/BER/z under every attack,
  6. quantify imperceptibility: PSNR/SSIM between marked and unmarked
     renders.

Usage:
    python experiments/wm_screen_capture.py [--frames 120] [--json]
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from PIL import Image, ImageDraw, ImageFont

import watermark as wm

KEY = "bench-key-8842"
DIGITS = "0123456789"
COLS, ROWS = 120, 40
CELL_W, CELL_H = 6, 10            # ≈ browser 8px bold monospace cell
BG = (5, 5, 5)
FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

_LUT = ((np.arange(256, dtype=np.uint16)
         * (len(wm.DEFAULT_PALETTE) - 1)) // 255).astype(np.uint8)


# ── synthetic movie ─────────────────────────────────────────────────────────

def make_movie(nframes, seed=19):
    """
    Realistic texture: synthesise at 8x pixel resolution (like a 960x320
    source) with hard edges + fractal noise, then downscale to the cell grid
    — exactly what VideoDecoder.cv2.resize feeds the mapper in production.
    """
    rng = np.random.default_rng(seed)
    UP = 8
    H, W = ROWS * UP, COLS * UP
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # static fractal background texture (3 octaves of upscaled white noise)
    tex = np.zeros((H, W), np.float32)
    for oct_ in range(3):
        s = 8 * (2 ** oct_)
        small = rng.normal(0, 1, (H // s + 2, W // s + 2)).astype(np.float32)
        tex += (0.5 ** oct_) * cv2.resize(small, (W, H),
                                          interpolation=cv2.INTER_CUBIC)
    text_col = np.dstack([tex * 90 + 120, tex * 70 + 110, tex * 60 + 100])
    frames = []
    for t in range(nframes):
        u = t / nframes
        layer = text_col.copy()
        # hard-edged moving objects (~<=1 cell/frame — typical for 30 fps)
        x1 = int((0.5 + 0.35 * np.sin(2 * np.pi * u * 1.0)) * W)
        y1 = int((0.3 + 0.22 * np.cos(2 * np.pi * u * 1.5)) * H)
        cv2.circle(layer, (x1, y1), H // 5, (230, 90, 40), -1)
        cv2.rectangle(layer, (int((u * 0.22 % 1.0) * W), H // 3),
                      (int((u * 0.22 % 1.0) * W) + W // 6, H // 3 + H // 8),
                      (40, 190, 210), -1)
        wave = 0.5 + 0.5 * np.sin((xx / 34.0) + t * 0.35) * np.cos((yy / 27.0) - t * 0.22)
        layer += np.dstack([wave * 30, wave * 18, wave * 40])
        layer += rng.normal(0, 4, layer.shape)
        bgr_hi = np.clip(layer, 0, 255).astype(np.uint8)
        # third act: dark scene; second act: bright
        if u > 0.66:
            bgr_hi = (bgr_hi.astype(np.float32) * 0.3).astype(np.uint8)
        elif u > 0.4:
            bgr_hi = np.clip(bgr_hi.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)
        bgr = cv2.resize(bgr_hi, (COLS, ROWS), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        frames.append((gray, bgr))
    return frames


def render_frames(movie, mark):
    """Movie → list of (idx, bgr) grid frames, optionally watermarked.

    Production carrier (ASCII colour modes 2-6): ±beta luminance dither on
    the cell colour — glyphs stay identical, only the tint luminance shifts.
    """
    # perceptual=True → identical to the production server/compiler path
    wmr = wm.Watermarker(DIGITS, KEY, ROWS, COLS) if mark else None
    out = []
    for t, (gray, bgr) in enumerate(movie):
        idx = _LUT[gray].copy()
        bgr_m = bgr.copy()
        if wmr is not None:
            wmr.embed_pixels(bgr_m, t, gray)
        out.append((idx, bgr_m))
    return out


def paint(frames, font):
    """Render grid frames to pixel images (list of HxWx3 RGB uint8)."""
    imgs = []
    for idx, bgr in frames:
        img = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H), BG)
        draw = ImageDraw.Draw(img)
        lut = wm.DEFAULT_PALETTE
        for y in range(ROWS):
            row = idx[y]
            bgr_row = bgr[y]
            for x in range(COLS):
                c = lut[row[x]]
                if c == " ":
                    continue
                bb = bgr_row[x]
                draw.text((x * CELL_W, y * CELL_H - 2), c, font=font,
                          fill=(int(bb[2]), int(bb[1]), int(bb[0])))
        imgs.append(np.asarray(img, dtype=np.uint8))
    return imgs


# ── attacks ─────────────────────────────────────────────────────────────────

def atk_identity(imgs):
    return imgs


def atk_resize(factor):
    def f(imgs):
        h, w = imgs[0].shape[:2]
        return [cv2.resize(im, (max(1, round(w * factor)),
                                max(1, round(h * factor))),
                           interpolation=cv2.INTER_AREA)
                for im in imgs]
    return f


def atk_jpeg(quality):
    def f(imgs):
        out = []
        for im in imgs:
            enc = cv2.imencode(".jpg", cv2.cvtColor(im, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, quality])[1]
            out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR),
                                    cv2.COLOR_BGR2RGB))
        return out
    return f


def atk_h264(crf):
    def f(imgs):
        h, w = imgs[0].shape[:2]
        with tempfile.TemporaryDirectory() as td:
            mp4 = os.path.join(td, "cap.mp4")
            proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", "30", "-i", "-",
                 "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                 "-pix_fmt", "yuv420p", "-loglevel", "error", mp4],
                stdin=subprocess.PIPE)
            for im in imgs:
                proc.stdin.write(np.ascontiguousarray(im).tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg x264 failed")
            cap = cv2.VideoCapture(mp4)
            out = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                out.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            return out
    return f


def atk_noise(sigma):
    def f(imgs):
        rng = np.random.default_rng(23)
        return [np.clip(im.astype(np.float32)
                        + rng.normal(0, sigma, im.shape), 0, 255).astype(np.uint8)
                for im in imgs]
    return f


def atk_bc_gamma(bright=1.12, contrast=0.88, gamma=1.25):
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 * contrast
                  + bright * 18 - 10, 0, 255).astype(np.uint8)
    return lambda imgs: [lut[im] for im in imgs]


def atk_crop(frac=0.035):
    """Screenshot cropped around the player (no rescale back)."""
    def f(imgs):
        h, w = imgs[0].shape[:2]
        dx, dy = round(w * frac), round(h * frac)
        return [im[dy:h - dy, dx:w - dx] for im in imgs]
    return f

def atk_crop_rescale(frac=0.035):
    """Inner crop stretched back to canvas size (destructive zoom)."""
    def f(imgs):
        h, w = imgs[0].shape[:2]
        dx, dy = round(w * frac), round(h * frac)
        return [cv2.resize(im[dy:h - dy, dx:w - dx], (w, h),
                           interpolation=cv2.INTER_AREA) for im in imgs]
    return f


def atk_strip(frac_r=0.025, frac_b=0.0):
    """Single-sided strip cut (sloppy screenshot trimmed past the edge)."""
    def f(imgs):
        h, w = imgs[0].shape[:2]
        return [im[0:h - round(h * frac_b), 0:w - round(w * frac_r)]
                for im in imgs]
    return f


def atk_surround(fill=0.80):
    """Player occupies the centre `fill` of the capture (UI/chrome around)."""
    def f(imgs):
        h, w = imgs[0].shape[:2]
        ch, cw = round(h / fill), round(w / fill)
        out = []
        for im in imgs:
            canvas = np.full((ch, cw, 3), 40, np.uint8)
            y0, x0 = (ch - h) // 2, (cw - w) // 2
            canvas[y0:y0 + h, x0:x0 + w] = im
            out.append(canvas)
        return out
    return f


def atk_aniso(fx=0.05, fy=0.02):
    """Different crop fractions per axis (aspect change)."""
    def f(imgs):
        h, w = imgs[0].shape[:2]
        dx, dy = round(w * fx), round(h * fy)
        return [im[dy:h - dy, dx:w - dx] for im in imgs]
    return f


def atk_phone_upload(imgs):
    """Worst-case chain: slight zoom-out → H.264 CRF 32 → JPEG q60 → noise."""
    return atk_noise(2.0)(atk_jpeg(60)(atk_h264(32)(atk_resize(0.85)(imgs))))


ATTACKS = [
    ("screenshot (lossless PNG)", atk_identity),
    ("screen zoom 75%", atk_resize(0.75)),
    ("screen zoom 50%", atk_resize(0.50)),
    ("screen zoom 150%", atk_resize(1.5)),
    ("H.264 screen-rec CRF 23", atk_h264(23)),
    ("H.264 screen-rec CRF 28", atk_h264(28)),
    ("H.264 screen-rec CRF 35", atk_h264(35)),
    ("JPEG q=70", atk_jpeg(70)),
    ("JPEG q=50", atk_jpeg(50)),
    ("sensor noise σ=6", atk_noise(6.0)),
    ("bright/contrast/gamma", atk_bc_gamma()),
    ("3.5% screenshot crop", atk_crop(0.035)),
    ("8% screenshot crop", atk_crop(0.08)),
    ("2.5% single-side strip", atk_strip(0.025)),
    ("player @ 80% of capture", atk_surround(0.80)),
    ("5%+2% aspect crop", atk_aniso(0.05, 0.02)),
    ("3.5% crop + stretch-back", atk_crop_rescale(0.035)),
    ("phone-upload chain", atk_phone_upload),
]

# attacks that need the geometry re-sync scan
SYNC_ATTACKS = {"3.5% screenshot crop", "8% screenshot crop",
                "2.5% single-side strip", "player @ 80% of capture",
                "5%+2% aspect crop", "3.5% crop + stretch-back"}

# known-limitation probe (reported, not counted): deep single-sided strips
# sit on a flat correlation plateau — registration needs scale ±2 % AND
# offset ±½ cell simultaneously, and the removed chips carry no reference
# (see the detect_with_sync docstring). Reported to prove the CRC gate
# never false-accepts out there.
LIMIT_ATTACKS = [("6%+5% single-side strips", atk_strip(0.06, 0.05))]


# ── metrics ─────────────────────────────────────────────────────────────────

def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """Compact global SSIM on luminance (uniform 7x7 windows)."""
    x = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    k = (7, 7)
    mx, my = cv2.blur(x, k), cv2.blur(y, k)
    xx, yy = cv2.blur(x * x, k), cv2.blur(y * y, k)
    xy = cv2.blur(x * y, k)
    vx, vy = xx - mx * mx, yy - my * my
    cxy = xy - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    m = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1)
                                                 * (vx + vy + c2))
    return float(m.mean())


# ── driver ──────────────────────────────────────────────────────────────────

def detect_pixels(imgs, cols=COLS, rows=ROWS, key=KEY):
    det = wm.WatermarkDetector(key, rows, cols)
    for im in imgs:
        det.feed_image(cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    return det.decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    for p in FONT_PATHS:
        if os.path.exists(p):
            font = ImageFont.truetype(p, 10)
            break
    else:
        raise SystemExit("no DejaVu Mono font found")

    print(f"[*] movie: {args.frames} frames on a {COLS}x{ROWS} grid, "
          f"rendered {COLS * CELL_W}x{ROWS * CELL_H}px")
    movie = make_movie(args.frames)
    marked = paint(render_frames(movie, True), font)
    clean = paint(render_frames(movie, False), font)

    # Worst-case flicker case: a static, flat, bright test card.
    flat = [(np.full((ROWS, COLS), 190, np.uint8),
             np.stack([np.full((ROWS, COLS), v, np.uint8) for v in (60, 150, 200)], 2))
            for _ in range(4)]
    fm, fc = paint(render_frames(flat, True), font), paint(render_frames(flat, False), font)
    flat_psnr = min(psnr(a, b) for a, b in zip(fm, fc))

    mid = args.frames // 2
    ps = [psnr(a, b) for a, b in zip(marked, clean)]
    ss = [ssim(a, b) for a, b in zip(marked[mid:mid + 8], clean[mid:mid + 8])]
    print(f"[*] imperceptibility: video PSNR {np.mean(ps):.1f} dB "
          f"(min {np.min(ps):.1f}), SSIM {np.mean(ss):.4f}, "
          f"flat-card PSNR {flat_psnr:.1f} dB")

    rows_out = []
    for name, fn in ATTACKS:
        atk_imgs = fn(marked)
        if name in SYNC_ATTACKS:
            bgrs = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in atk_imgs]
            r = wm.detect_with_sync(bgrs, KEY, ROWS, COLS)
            rw = wm.detect_with_sync(bgrs, "decoy-key", ROWS, COLS)
        else:
            r = detect_pixels(atk_imgs)
            rw = detect_pixels(atk_imgs, key="decoy-key")  # wrong-key guard
        row = dict(attack=name, ok=bool(r.ok), digits=r.digits,
                   ber=round(r.bit_errors, 4), z=round(r.z, 1),
                   pairs=r.pairs, block=r.block,
                   wrong_key_ok=bool(rw.ok))
        rows_out.append(row)
        geo = (f"  [fit s=({r['scale_x']},{r['scale_y']}) "
               f"d=({r['dx']:+},{r['dy']:+})]"
               if "scale_x" in r and name in SYNC_ATTACKS else "")
        print(f"  {'PASS' if row['ok'] else 'FAIL':4s}  {name:28s} "
              f"digits={row['digits']} ber={row['ber']:.3f} "
              f"z={row['z']:6.1f}  wrongkey={'LEAK!' if rw.ok else 'blind'}"
              f"{geo}")

    if args.json:
        print(json.dumps(rows_out, indent=2))
    for name, fn in LIMIT_ATTACKS:
        atk_imgs = fn(marked)
        bgrs = [cv2.cvtColor(im, cv2.COLOR_RGB2BGR) for im in atk_imgs]
        r = wm.detect_with_sync(bgrs, KEY, ROWS, COLS)
        print(f"  {'(info)' :5s} {name:28s} digits={r.digits} z={r.z:6.1f} "
              f"[known geometric limit]")

    hard_fail = [r for r in rows_out if not r["ok"]]
    wrong_leak = [r for r in rows_out if r["wrong_key_ok"]]
    print(f"[*] {len(rows_out) - len(hard_fail)}/{len(rows_out)} attacks "
          f"recovered; wrong-key detections: {len(wrong_leak)}")
    return 1 if (hard_fail or wrong_leak) else 0


if __name__ == "__main__":
    sys.exit(main())
