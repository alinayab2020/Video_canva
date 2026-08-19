#!/usr/bin/env python3
"""
watermark_detect.py — recover the ASCILINE forensic watermark from a screen
capture (screenshot image or screen recording video) of the player.

The detector is blind (no original content needed) but keyed: you must supply
the same key that embedded the mark. Geometry of the rendered ASCII grid
(cols × rows) must match the embedding configuration — it is printed at
server/compiler startup and part of every INIT message.

Usage:
    python tools/watermark_detect.py capture.png  --cols 200 --rows 50 --key SECRET
    python tools/watermark_detect.py screencast.mp4 --cols 200 --rows 50 \\
        --key SECRET --crop 60,120,966,400 --frames 240 --json

Exit code: 0 when the 10-digit payload was recovered and CRC-verified,
1 otherwise. --json prints the full DetectionResult dict.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import cv2
except ImportError:
    cv2 = None

from watermark import WatermarkDetector  # noqa: E402

_IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


def _iter_capture_frames(path: str, crop, max_frames: int):
    """Yield (possibly cropped) BGR frames from an image or video file."""
    if cv2 is None:
        sys.exit("OpenCV (cv2) is required for capture ingestion: "
                 "pip install opencv-python")
    if path.lower().endswith(_IMAGE_EXT):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            sys.exit(f"cannot read image: {path}")
        if crop:
            x, y, w, h = crop
            img = img[y:y + h, x:x + w]
        yield img
        return
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or max_frames
    # Subsample uniformly when the clip is longer than max_frames.
    step = max(1, int(np.ceil(total / max_frames))) if max_frames else 1
    idx, kept = 0, 0
    while max_frames is None or kept < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            if crop:
                x, y, w, h = crop
                frame = frame[y:y + h, x:x + w]
            yield frame
            kept += 1
        idx += 1
    cap.release()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recover the ASCILINE 10-digit forensic watermark from "
                    "a screen capture of the player.")
    ap.add_argument("capture", help="screenshot image or screen recording")
    ap.add_argument("--cols", type=int, required=True,
                    help="ASCII grid columns used at embedding (from INIT)")
    ap.add_argument("--rows", type=int, required=True,
                    help="ASCII grid rows used at embedding (from INIT)")
    ap.add_argument("--key", default=None,
                    help="watermark key (default: env ASCILINE_WM_KEY)")
    ap.add_argument("--block", type=int, default=0,
                    help="alternation clock if known (default: scan 1,2,3,4,6)")
    ap.add_argument("--crop", default=None, metavar="X,Y,W,H",
                    help="rectangle of the player canvas in the capture "
                         "(default: whole frame)")
    ap.add_argument("--frames", type=int, default=240,
                    help="max frames to analyse (default 240; ~8 s at 30 fps)")
    ap.add_argument("--sync", action="store_true",
                    help="geometry re-sync scan (unknown canvas crop/scale; "
                         "slower — use when the capture is cropped/resized)")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="trace every geometry hypothesis the sync scan "
                         "probes (with --sync)")
    ap.add_argument("--json", action="store_true",
                    help="print the full result as JSON")
    args = ap.parse_args()

    key = args.key or os.environ.get("ASCILINE_WM_KEY")
    if not key:
        ap.error("--key or env ASCILINE_WM_KEY is required")
    crop = None
    if args.crop:
        try:
            crop = tuple(int(v) for v in args.crop.split(","))
            assert len(crop) == 4
        except (ValueError, AssertionError):
            ap.error("--crop must be X,Y,W,H")

    if args.sync:
        from watermark import detect_with_sync
        frames = list(_iter_capture_frames(args.capture, crop, args.frames))
        result = detect_with_sync(
            frames, key, args.rows, args.cols,
            block=(args.block,) if args.block else None,
            progress=print if args.verbose else None)
    else:
        det = WatermarkDetector(
            key, args.rows, args.cols,
            block=(args.block,) if args.block else None)
        fed = 0
        for frame in _iter_capture_frames(args.capture, crop, args.frames):
            det.feed_image(frame)      # resize-to-grid == per-cell averaging
            fed += 1
        result = det.decode()

    if args.json:
        import json
        print(json.dumps(dict(result), indent=2, default=str))
    else:
        if result.ok:
            geo = (f"  geometry=sx {result.get('scale_x', 1.0):.3f} "
                   f"sy {result.get('scale_y', 1.0):.3f} "
                   f"dx {result.get('dx', 0):+} dy {result.get('dy', 0):+} "
                   f"({result.get('probes', 0)} probes)"
                   if "scale" in result else "")
            print(f"[DETECTED] digits={result.digits}  "
                  f"z={result.z:.1f}σ  bit_errors={result.bit_errors:.3f}  "
                  f"frames={result.frames_used} pairs={result.pairs} "
                  f"block={result.block} phase={result.phase} "
                  f"erasures={result.erasures}{geo}")
        else:
            print(f"[NOT DETECTED] frames={result.frames_used} "
                  f"z={result.z:.1f}σ — wrong key/geometry, too few frames, "
                  f"or destructive post-processing")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
