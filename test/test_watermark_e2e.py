"""
Forensic-watermark end-to-end tests across the whole stack.

  * compiler.py --watermark → .ascf → decode every frame with the SHIPPED
    browser codec (codec.js under Node) → detector recovers the 10 digits
    from the decoded cell grid, wrong key stays blind.
  * live uvicorn server with --watermark → adaptive WebSocket stream →
    Node collector (shipped codec.js) → same recovery, proving the mark
    survives the server produce path, the adaptive codec and the wire.

Skips cleanly when Node or cv2 is unavailable.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import watermark as wm  # noqa: E402

try:
    import cv2
except ImportError:
    cv2 = None

KEY = "e2e-secret"
DIGITS = "8081828384"
COLS, ROWS = 80, 30

DUMP_JS = r"""
const fs = require('fs');
const codec = require(process.env.CODEC_JS);
(async () => {
  const buf = fs.readFileSync(process.env.ASCF_PATH);
  const cols = buf.readUInt16BE(10), rows = buf.readUInt16BE(12);
  const pixel = buf.readUInt8(9) === 1;
  const dec = codec.makeDecoder(pixel ? 3 : 4);
  let off = 18, n = 0;
  const frames = [];
  while (off + 4 <= buf.length) {
    const len = buf.readUInt32BE(off); off += 4;
    const msg = buf.subarray(off, off + len); off += len;
    const { frameIndex, frame } = await dec.decode(msg);
    if (frameIndex !== n) throw new Error(`order ${frameIndex} != ${n}`);
    frames.push(Buffer.from(frame)); n++;
  }
  fs.writeFileSync(process.env.OUT_BIN, Buffer.concat(frames));
  console.log(JSON.stringify({ n }));
})();
"""

COLLECT_JS = r"""
/** Connect to the live server, decode `MAX` adaptive frames, dump them.
 *
 * Raw RFC-6455 client over net/crypto only — CI runs Node 20, where the
 * global WebSocket is still flag-gated, and no npm deps are installed.
 * Handles the unmasked server->client frame grammar (16/64-bit lengths,
 * TCP fragmentation) and answers pings; client frames are masked per spec.
 */
const fs = require('fs');
const net = require('net');
const crypto = require('crypto');
const codec = require(process.env.CODEC_JS);
// node -e shifts argv: the port is the first all-digit argv entry.
const PORT = process.argv.find(a => /^\d+$/.test(a));
const MAX = 120;

function sendFrame(sock, op, payload) {
  const len = payload.length;
  const mask = crypto.randomBytes(4);
  let head;
  if (len < 126) {
    head = Buffer.from([0x80 | op, 0x80 | len]);
  } else if (len < 65536) {
    head = Buffer.alloc(4);
    head[0] = 0x80 | op; head[1] = 0x80 | 126; head.writeUInt16BE(len, 2);
  } else {
    head = Buffer.alloc(10);
    head[0] = 0x80 | op; head[1] = 0x80 | 127;
    head.writeBigUInt64BE(BigInt(len), 2);
  }
  const masked = Buffer.from(payload);
  for (let i = 0; i < masked.length; i++) masked[i] ^= mask[i & 3];
  sock.write(Buffer.concat([head, mask, masked]));
}

function wsConnect(port, path, onText, onBinary) {
  return new Promise((resolve, reject) => {
    const sock = net.connect(port, '127.0.0.1', () => {
      sock.write(
        `GET ${path} HTTP/1.1\r\nHost: 127.0.0.1:${port}\r\n` +
        'Upgrade: websocket\r\nConnection: Upgrade\r\n' +
        `Sec-WebSocket-Key: ${crypto.randomBytes(16).toString('base64')}\r\n` +
        'Sec-WebSocket-Version: 13\r\n\r\n');
    });
    let buf = Buffer.alloc(0);
    let handshakeDone = false;
    const fail = (e) => { try { sock.destroy(); } catch (_) {} reject(e); };
    sock.on('error', fail);
    sock.on('data', (d) => {
      buf = Buffer.concat([buf, d]);
      if (!handshakeDone) {
        const idx = buf.indexOf('\r\n\r\n');
        if (idx < 0) return;
        const head = buf.slice(0, idx).toString();
        if (!head.startsWith('HTTP/1.1 101')) {
          return fail(new Error('handshake: ' + head.split('\r\n')[0]));
        }
        buf = buf.slice(idx + 4);
        handshakeDone = true;
        resolve({ close: () => sendFrame(sock, 0x8, Buffer.alloc(0)) });
      }
      while (true) {
        if (buf.length < 2) return;
        const fin = buf[0] & 0x80, op = buf[0] & 0x0f;
        let len = buf[1] & 0x7f, off = 2;
        if (len === 126) {
          if (buf.length < 4) return;
          len = buf.readUInt16BE(2); off = 4;
        } else if (len === 127) {
          if (buf.length < 10) return;
          len = Number(buf.readBigUInt64BE(2)); off = 10;
        }
        if (buf.length < off + len) return;
        const payload = buf.slice(off, off + len);
        buf = buf.slice(off + len);
        if (op === 0x9) { sendFrame(sock, 0xA, payload); continue; }  // ping
        if (!fin || op === 0x8) return;      // no fragmentation expected
        if (op === 0x1) onText(payload.toString());
        else if (op === 0x2) onBinary(payload);
      }
    });
  });
}

(async () => {
  let rows = 0, cols = 0, decoder = null;
  const frames = [];
  await new Promise((resolve, reject) => {
    const kill = setTimeout(() => reject(new Error('timeout collecting')),
                            60000);
    let ws = null;
    wsConnect(PORT, '/ws?codec=adaptive',
      (t) => {
        if (t.startsWith('INIT:')) {
          const p = t.split(':');
          cols = parseInt(p[3]); rows = parseInt(p[4]);
          decoder = codec.makeDecoder(parseInt(p[5]) === 1 ? 3 : 4);
        }
      },
      async (b) => {
        const { frame } = await decoder.decode(b);
        frames.push(Buffer.from(frame));
        if (frames.length >= MAX) {
          clearTimeout(kill); ws.close(); resolve();
        }
      })
      .then((w) => { ws = w; }, reject);
  });
  fs.writeFileSync(process.env.OUT_BIN, Buffer.concat(frames));
  console.log(JSON.stringify({ n: frames.length, rows, cols }));
  process.exit(0);
})();
"""


def _make_clip(path, frames=150, w=160, h=96, fps=12.0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        return False
    rng = np.random.default_rng(17)
    base = rng.integers(20, 200, (h, w, 3), dtype=np.uint8)
    for i in range(frames):
        img = base.copy()
        img[:, (i * 2) % w:((i * 2) % w) + 14] = (235, 220, 205)
        cv2.circle(img, (w // 2, h // 2), 6 + (i % 18), (70, 190, 250), -1)
        vw.write(img)
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


def _detect_frames_bin(path, rows, cols, cell_bytes=4, key=KEY, pixel=False):
    buf = np.fromfile(path, dtype=np.uint8)
    per = rows * cols * cell_bytes
    n = buf.size // per
    assert n > 0, "no frames dumped"
    fr = buf.reshape(n, rows, cols, cell_bytes)
    det = wm.WatermarkDetector(key, rows, cols)
    for i in range(n):
        lum = (wm.luma_from_pixel_frame(fr[i]) if pixel
               else wm.luma_from_ascii_frame(fr[i]))
        det.feed_luma_grid(lum)
    return det.decode(), n


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@unittest.skipUnless(shutil.which("node"), "node required")
@unittest.skipUnless(cv2 is not None, "cv2 required")
class TestWatermarkCompileE2E(unittest.TestCase):
    def test_compile_ascii_then_detect(self):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.avi")
            self.assertTrue(_make_clip(clip))
            cmd = [sys.executable, os.path.join(ROOT, "compiler.py"), clip,
                   "--cols", str(COLS), "--rows", str(ROWS), "--mode", "4",
                   "--watermark", DIGITS, "--watermark-key", KEY,
                   "--out", "wme2e"]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
            ascf = os.path.join(ROOT, "static_player", "wme2e.ascf")
            try:
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                self.assertTrue(os.path.exists(ascf))
                out_bin = os.path.join(td, "frames.bin")
                env = dict(os.environ,
                           CODEC_JS=os.path.join(ROOT, "codec.js"),
                           ASCF_PATH=ascf, OUT_BIN=out_bin)
                node = subprocess.run(["node", "-e", DUMP_JS],
                                      capture_output=True, text=True, env=env)
                self.assertEqual(node.returncode, 0, node.stderr)

                result, n = _detect_frames_bin(out_bin, ROWS, COLS)
                self.assertTrue(result.ok, dict(result))
                self.assertEqual(result.digits, DIGITS)
                self.assertGreaterEqual(result.z, 8.0)

                blind, _ = _detect_frames_bin(out_bin, ROWS, COLS, key="wrong")
                self.assertFalse(blind.ok)
            finally:
                for suffix in (".ascf", ".mp3"):
                    p = os.path.join(ROOT, "static_player", "wme2e" + suffix)
                    if os.path.exists(p):
                        os.remove(p)

    def test_compile_pixel_then_detect(self):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "clip.avi")
            self.assertTrue(_make_clip(clip))
            cmd = [sys.executable, os.path.join(ROOT, "compiler.py"), clip,
                   "--cols", "96", "--rows", "54", "--pixel", "--quantize", "1",
                   "--watermark", DIGITS, "--watermark-key", KEY,
                   "--watermark-beta", "12", "--out", "wme2epx"]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=td)
            ascf = os.path.join(ROOT, "static_player", "wme2epx.ascf")
            try:
                self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
                out_bin = os.path.join(td, "px.bin")
                env = dict(os.environ,
                           CODEC_JS=os.path.join(ROOT, "codec.js"),
                           ASCF_PATH=ascf, OUT_BIN=out_bin)
                node = subprocess.run(["node", "-e", DUMP_JS],
                                      capture_output=True, text=True, env=env)
                self.assertEqual(node.returncode, 0, node.stderr)
                result, n = _detect_frames_bin(out_bin, 54, 96,
                                               cell_bytes=3, pixel=True)
                self.assertTrue(result.ok, dict(result))
                self.assertEqual(result.digits, DIGITS)
            finally:
                for suffix in (".ascf", ".mp3"):
                    p = os.path.join(ROOT, "static_player", "wme2epx" + suffix)
                    if os.path.exists(p):
                        os.remove(p)


@unittest.skipUnless(shutil.which("node"), "node required")
@unittest.skipUnless(cv2 is not None, "cv2 required")
class TestWatermarkLiveServer(unittest.TestCase):
    def test_live_stream_detect(self):
        with tempfile.TemporaryDirectory() as td:
            clip = os.path.join(td, "live.avi")
            self.assertTrue(_make_clip(clip, frames=150))
            port = _free_port()
            server = subprocess.Popen(
                [sys.executable, os.path.join(ROOT, "stream_server.py"), clip,
                 "--port", str(port), "--mode", "4", "--cols", str(COLS),
                 "--rows", str(ROWS),
                 "--watermark", DIGITS, "--watermark-key", KEY],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL)
            try:
                # wait for HTTP to come up
                import urllib.request
                deadline = time.time() + 30
                up = False
                while time.time() < deadline:
                    try:
                        urllib.request.urlopen(
                            f"http://127.0.0.1:{port}/healthz", timeout=1)
                        up = True
                        break
                    except Exception:
                        time.sleep(0.3)
                self.assertTrue(up, "server did not start")

                out_bin = os.path.join(td, "live.bin")
                env = dict(os.environ,
                           CODEC_JS=os.path.join(ROOT, "codec.js"),
                           OUT_BIN=out_bin)
                node = subprocess.run(["node", "-e", COLLECT_JS, str(port)],
                                      capture_output=True, text=True, env=env,
                                      timeout=90)
                self.assertEqual(node.returncode, 0, node.stderr)
                info = json.loads(node.stdout.strip().splitlines()[-1])
                self.assertGreaterEqual(info["n"], 90)

                result, n = _detect_frames_bin(out_bin, info["rows"],
                                               info["cols"])
                self.assertTrue(result.ok, dict(result))
                self.assertEqual(result.digits, DIGITS)
                self.assertGreaterEqual(result.z, 8.0)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server.kill()


if __name__ == "__main__":
    unittest.main()
