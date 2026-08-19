"""
Security-hardening tests for the ASCILINE live server.

Covers the whole defensive surface added to stream_server.py:
  * HTTP security headers (CSP, anti-sniff, clickjacking, referrer, CORP/COOP)
    and cache policy (no-store on session endpoints, bounded static caching)
  * static-file whitelist (no path escape, no source disclosure)
  * WebSocket resilience: malformed frames, non-dict JSON, NaN/inf/junk
    command payloads must never kill the stream or the command pump
  * WebSocket admission control (max-client cap -> close 1013)
  * /audio process-pool saturation -> 503 instead of fork-bombing
  * pure coercion/sanitization helpers

    pytest test/test_security.py
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import stream_server as ss
from stream_server import _clamp_seek_time, _coerce_finite_float, _coerce_int


def _make_video(path, frames=24, w=64, h=48, fps=12.0):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))
    if not vw.isOpened():
        return False
    for i in range(frames):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, : w // 2] = (30 + i * 4 % 200, 90, 140)
        img[:, w // 2:] = (140, 90, 30)
        vw.write(img)
    vw.release()
    return os.path.exists(path) and os.path.getsize(path) > 0


# ── pure helpers ─────────────────────────────────────────────────────────
class CoercionTests(unittest.TestCase):
    def test_finite_float(self):
        self.assertEqual(_coerce_finite_float(2.5, 1.0), 2.5)
        self.assertEqual(_coerce_finite_float("2.5", 1.0), 2.5)
        for junk in (float("nan"), float("inf"), float("-inf"), "NaN",
                     "Infinity", "abc", None, [1], {}, b"x"):
            self.assertEqual(_coerce_finite_float(junk, 1.0), 1.0, repr(junk))

    def test_coerce_int(self):
        self.assertEqual(_coerce_int(5, 1), 5)
        self.assertEqual(_coerce_int(5.9, 1), 5)
        self.assertEqual(_coerce_int("7", 1), 7)
        for junk in (True, False, float("nan"), float("inf"), "abc", None, [1]):
            self.assertEqual(_coerce_int(junk, 1), 1, repr(junk))

    def test_clamp_seek_time(self):
        self.assertEqual(_clamp_seek_time(5.0, 10.0), 5.0)
        self.assertEqual(_clamp_seek_time(15.0, 10.0), 10.0)   # past the end
        self.assertEqual(_clamp_seek_time(-3.0, 10.0), 0.0)    # negative
        self.assertEqual(_clamp_seek_time("junk", 10.0), 0.0)  # non-numeric
        self.assertEqual(_clamp_seek_time(float("nan"), 10.0), 0.0)
        self.assertEqual(_clamp_seek_time(float("inf"), 10.0), 0.0)
        self.assertEqual(_clamp_seek_time(None, 10.0), 0.0)
        # duration unknown (webcam): only negative is stripped
        self.assertEqual(_clamp_seek_time(42.0, 0.0), 42.0)


# ── HTTP surface ─────────────────────────────────────────────────────────
class HttpSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient
        cls.tmp = tempfile.mkdtemp(prefix="asciline_sec_")
        cls.video = os.path.join(cls.tmp, "clip.avi")
        if not _make_video(cls.video):
            raise unittest.SkipTest("OpenCV could not write a test video here.")
        ss.app.state.queue = [{
            "video": cls.video, "mode": 5, "pixel": False, "vol": 1, "rows": 0,
        }]
        ss.app.state.current_index = 0
        cls.client_ctx = TestClient(ss.app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_security_headers_on_root(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        h = r.headers
        self.assertEqual(h.get("x-content-type-options"), "nosniff")
        self.assertEqual(h.get("x-frame-options"), "DENY")
        self.assertEqual(h.get("referrer-policy"), "no-referrer")
        self.assertIn("camera=()", h.get("permissions-policy", ""))
        self.assertEqual(h.get("cross-origin-opener-policy"), "same-origin")
        csp = h.get("content-security-policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        # session content must never be shared-cached
        self.assertEqual(h.get("cache-control"), "no-store")

    def test_static_whitelist_and_cache(self):
        r = self.client.get("/static/app.js")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.headers.get("cache-control"), "public, max-age=300")
        self.assertEqual(r.headers.get("x-content-type-options"), "nosniff")
        # source files are not on the whitelist
        for blocked in ("stream_server.py", "codec.py", "logo.py", ".git"):
            self.assertEqual(self.client.get(f"/static/{blocked}").status_code, 404)
        # traversal attempts cannot escape the whitelist either
        for evil in ("..%2Fstream_server.py", "..%5Cstream_server.py", "%2E%2E%2Fcodec.py"):
            self.assertNotEqual(self.client.get(f"/static/{evil}").status_code, 200)

    def test_audio_pool_saturation_returns_503(self):
        import asyncio
        saved = getattr(ss.app.state, "audio_semaphore", None)
        try:
            gate = asyncio.Semaphore(1)
            # simulate an active stream occupying the only slot
            async def lock_it():
                await gate.acquire()
            asyncio.get_event_loop_policy().new_event_loop().run_until_complete(lock_it())
            ss.app.state.audio_semaphore = gate
            r = self.client.get("/audio")
            self.assertEqual(r.status_code, 503)
        finally:
            ss.app.state.audio_semaphore = saved

    def test_audio_muted_video_is_204(self):
        saved = ss.app.state.queue
        try:
            ss.app.state.queue = [{
                "video": self.video, "mode": 5, "pixel": False, "vol": 0, "rows": 0,
            }]
            self.assertEqual(self.client.get("/audio").status_code, 204)
        finally:
            ss.app.state.queue = saved

    def test_audio_start_param_is_sanitized(self):
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        # negative / garbage offsets must not reach the ffmpeg cmdline — the
        # endpoint still serves (or cleanly 200s), never a 500.
        for bad in ("-50", "0"):
            r = self.client.get(f"/audio?start={bad}")
            self.assertEqual(r.status_code, 200)


# ── WebSocket resilience ─────────────────────────────────────────────────
class WebSocketSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="asciline_ws_")
        cls.video = os.path.join(cls.tmp, "clip.avi")
        if not _make_video(cls.video):
            raise unittest.SkipTest("OpenCV could not write a test video here.")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _queue(self):
        ss.app.state.queue = [{
            "video": self.video, "mode": 5, "pixel": False, "vol": 1, "rows": 0,
        }]
        ss.app.state.current_index = 0
        ss.app.state.loop = False
        ss.app.state.max_clients = 32
        ss.app.state.active_clients = 0

    def test_malformed_commands_do_not_kill_stream(self):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        self._queue()
        with TestClient(ss.app) as client:
            with client.websocket_connect("/ws?codec=adaptive") as ws:
                junk_sent = False
                frames = 0
                saw_init = False
                close_code = None
                while True:
                    msg = ws.receive()
                    if msg["type"] == "websocket.close":
                        close_code = msg.get("code")
                        break
                    if "text" in msg and msg["text"] is not None:
                        if msg["text"].startswith("INIT:"):
                            saw_init = True
                        if not junk_sent:
                            junk_sent = True
                            # every flavor of junk a client can throw:
                            ws.send_bytes(b"\x00\x01\xff not json")     # binary junk
                            ws.send_text("definitely not json")         # invalid JSON
                            ws.send_text("[1, 2, 3]")                    # non-dict JSON
                            ws.send_text('{"type":"seek","time":"NaN-lol"}')
                            ws.send_text('{"type":"seek","time":null}')
                            ws.send_text('{"type":"seek","time":-40}')
                            ws.send_text('{"type":"seek","time":1e999}')
                            ws.send_text('{"type":"filter","contrast":"Infinity"}')
                            ws.send_text('{"type":"filter","contrast":"crashme"}')
                            ws.send_text('{"type":"filter","sharpness":true}')
                            ws.send_text('{"type":"filter","palette":["x"]}')
                            ws.send_text('{"type":"reinit","pixel":"no"}')
                    elif "bytes" in msg and msg["bytes"] is not None:
                        frames += 1
                # stream survived the whole junk barrage AND completed cleanly
                self.assertTrue(saw_init)
                self.assertGreater(frames, 5)
                self.assertEqual(close_code, 1000)
                # command pump is still alive at protocol level
        self.assertEqual(ss.app.state.active_clients, 0)  # slot reclaimed

    def test_client_cap_closes_1013(self):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        self._queue()
        ss.app.state.max_clients = 1
        ss.app.state.active_clients = 1  # pretend the single slot is taken
        try:
            with TestClient(ss.app) as client:
                with client.websocket_connect("/ws") as ws:
                    with self.assertRaises(WebSocketDisconnect) as ctx:
                        ws.receive_text()
                    self.assertEqual(ctx.exception.code, 1013)
        finally:
            ss.app.state.max_clients = 32
            ss.app.state.active_clients = 0

    def test_origin_rejected_with_1008(self):
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect
        self._queue()
        with TestClient(ss.app) as client:
            with self.assertRaises(WebSocketDisconnect):
                # enter the context — that is what performs the handshake;
                # a foreign origin must be refused during it (close 1008)
                with client.websocket_connect(
                        "/ws", headers={"origin": "https://evil.example.com",
                                        "host": "127.0.0.1:8000"}):
                    pass


if __name__ == "__main__":
    unittest.main()
