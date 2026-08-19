"""
End-to-end compiler test: encode a real video with compiler.py, then decode
every frame of the resulting .ascf with the SHIPPED browser decoder
(codec.js, running under Node). This pins the whole compile-side pipeline —
mapping, quantization, codec tags, container framing — against the code the
user's browser actually runs.

Skips cleanly when Node.js is unavailable (the Python-only suite still covers
the encoder itself via the vector suite in experiments/).

    pytest test/test_compiler_e2e.py
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DECODE_JS = r"""
const fs = require('fs');
const codec = require(process.env.CODEC_JS);
(async () => {
  const buf = fs.readFileSync(process.env.ASCF_PATH);
  const cols = buf.readUInt16BE(10), rows = buf.readUInt16BE(12);
  const total = buf.readUInt32BE(14);
  const pixel = buf.readUInt8(9) === 1;
  const dec = codec.makeDecoder(pixel ? 3 : 4);
  let off = 18, n = 0;
  const tags = {};
  while (off + 4 <= buf.length) {
    const len = buf.readUInt32BE(off); off += 4;
    const msg = buf.subarray(off, off + len); off += len;
    tags[msg[4]] = (tags[msg[4]] || 0) + 1;
    const { frameIndex, frame } = await dec.decode(msg);
    if (frameIndex !== n) throw new Error(`order ${frameIndex} != ${n}`);
    const want = rows * cols * (pixel ? 3 : 4);
    if (frame.length !== want) throw new Error(`size ${frame.length} != ${want}`);
    n++;
  }
  console.log(JSON.stringify({ n, total, tags }));
})();
"""


def _make_clip(path, frames=32, w=96, h=64, fps=12.0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        return False
    rng = np.random.default_rng(5)
    base = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    for i in range(frames):
        img = base.copy()
        img[:, i % w:(i % w) + 8] = (255, 255, 255)  # moving bar
        vw.write(img)
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


@unittest.skipUnless(shutil.which("node"), "node required to decode with shipped JS codec")
class TestCompilerE2E(unittest.TestCase):
    def _compile_and_decode(self, extra_args):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.avi")
            self.assertTrue(_make_clip(clip), "could not create test clip")
            cmd = [sys.executable, os.path.join(ROOT, "compiler.py"), clip,
                   "--cols", "48", "--rows", "12", "--out", "out"] + extra_args
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
            # compiler writes into <repo>/static_player/out.ascf
            ascf = os.path.join(ROOT, "static_player", "out.ascf")
            try:
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertTrue(os.path.exists(ascf), res.stdout + res.stderr)
                with open(ascf, "rb") as f:
                    hdr = f.read(18)
                self.assertEqual(hdr[:4], b"ASC2")
                env = dict(os.environ,
                           CODEC_JS=os.path.join(ROOT, "codec.js"), ASCF_PATH=ascf)
                node = subprocess.run(["node", "-e", DECODE_JS],
                                      capture_output=True, text=True, env=env)
                self.assertEqual(node.returncode, 0, node.stderr)
                report = node.stdout.strip().splitlines()[-1]
                import json
                info = json.loads(report)
                self.assertGreater(info["n"], 0)
                self.assertEqual(info["n"], info["total"], "frame count mismatch")
                return info
            finally:
                for suffix in (".ascf", ".mp3"):
                    p = os.path.join(ROOT, "static_player", "out" + suffix)
                    if os.path.exists(p):
                        os.remove(p)

    def test_ascii_color_mode(self):
        self._compile_and_decode(["--mode", "4"])

    def test_pixel_mode(self):
        self._compile_and_decode(["--pixel", "--quantize", "2"])

    def test_lossy_tolerance_mode(self):
        self._compile_and_decode(["--mode", "6", "--tolerance", "8"])


if __name__ == "__main__":
    unittest.main()
