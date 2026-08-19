"""
Generate cross-language test vectors: encode real frames with codec.py exactly
as the server would, and dump both the adaptive messages and the ground-truth
raw framebuffers so codec.js (Node) can decode and verify byte-for-byte.

Output dir layout (experiments/vectors/<name>/):
  meta.json   {cellBytes, nframes, rows, cols}
  adaptive.bin  concat of [4B len][message] ...   (what the server would send)
  truth.bin     concat of [4B len][framebuffer] ... (legacy raw bodies)
"""
import os, sys, json, struct
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ascii_video_player2 import VideoDecoder, AsciiMapper, MODE_QUANTIZE_BITS
from codec import encode_frame

def gen(path, name, mode, pixel, cols=200, rows=80, limit=90, tol=0):
    # Same mapping the live server/compiler use: proportional gray->index LUT
    # and the shared mode->quantization table (single source of truth).
    mapper = AsciiMapper(); qb = MODE_QUANTIZE_BITS.get(mode,0)
    lut = np.array([ord(c) for c in mapper._lut], np.uint8)
    idx_lut = mapper.index_lut()
    qb_mask = np.uint8((0xFF << qb) & 0xFF) if qb else None
    dec = VideoDecoder(path, cols, rows, skip_gray=pixel)
    outdir = os.path.join("experiments/vectors", name); os.makedirs(outdir, exist_ok=True)
    fa = open(os.path.join(outdir,"adaptive.bin"),"wb")
    ft = open(os.path.join(outdir,"truth.bin"),"wb")
    prev = None; n = 0; raw_total = adapt_total = 0
    for gray, bgr in dec:
        if pixel:
            frame = np.ascontiguousarray(bgr)               # (rows,cols,3) BGR
        else:
            idx = idx_lut[gray]
            frame = np.empty((rows,cols,4),np.uint8); frame[:,:,0]=lut[idx]
            if qb_mask is not None:
                np.bitwise_and(bgr[:,:,::-1], qb_mask, out=frame[:,:,1:])
            else:
                frame[:,:,1:] = bgr[:,:,::-1]
        msg, prev = encode_frame(frame, prev, n, tolerance=tol)
        # Truth = the encoder's intended frame (prev/shown), which for lossy is
        # the bounded approximation the client must reconstruct exactly.
        body = prev.tobytes()
        fa.write(struct.pack(">I", len(msg))); fa.write(msg)
        ft.write(struct.pack(">I", len(body))); ft.write(body)
        raw_total += 4 + len(body); adapt_total += len(msg)
        n += 1
        if n >= limit: break
    dec.release(); fa.close(); ft.close()
    cell = 3 if pixel else 4
    json.dump({"cellBytes":cell,"nframes":n,"rows":rows,"cols":cols,
               "legacyBytes":raw_total,"adaptiveBytes":adapt_total},
              open(os.path.join(outdir,"meta.json"),"w"))
    print(f"{name:28} {n} frames  legacy={raw_total/1024:7.0f}KB  "
          f"adaptive={adapt_total/1024:6.0f}KB ({adapt_total/raw_total:5.1%})")

print("Generating test vectors (Python encoder):\n")
# lossless (must decode bit-exact to the true frame)
gen("videos/bars.mp4",   "bars_color_m5",   mode=5, pixel=False)
gen("videos/test.mp4",   "test_color_m5",   mode=5, pixel=False)
gen("videos/mandel.mp4", "mandel_color_m3", mode=3, pixel=False)
gen("videos/bars.mp4",   "bars_pixel",      mode=5, pixel=True)
gen("videos/test.mp4",   "test_pixel",      mode=5, pixel=True)
# lossy (must decode bit-exact to the encoder's bounded approximation)
gen("videos/test.mp4",   "test_color_T8",   mode=5, pixel=False, tol=8)
gen("videos/mandel.mp4", "mandel_color_T8", mode=3, pixel=False, tol=8)
gen("videos/test.mp4",   "test_pixel_T8",   mode=5, pixel=True,  tol=8)
