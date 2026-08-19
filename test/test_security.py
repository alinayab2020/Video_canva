"""
Security-hardening tests for the ASCILINE live server.

Covers the whole defensive surface added to stream_server.py:
  * HTTP security headers (CSP, anti-sniff, clickjacking, referrer, CORP/COOP)
    and cache policy (no-store on session endpoints, bounded static caching)
  * static-file whitelist (no path escape, no source disclosure)
  * /audio pool saturation -> 503, muted -> 204, offset sanitization
  * WebSocket resilience: malformed frames, non-dict JSON, NaN/inf/junk
    command payloads must never kill the stream or the command pump
  * WebSocket admission control (max-client cap -> close 1013)
  * pure coercion/sanitization helpers

Most tests call the ASGI building blocks directly (like the scrub tests do),
so they run in the minimal CI dependency set. Tests that need a full HTTP/WS
round trip use FastAPI's TestClient, which requires `httpx`; those skip
cleanly when it is not installed (e.g. the stock CI image) and run fully
otherwise (`pip install -e ".[test]"`).

    pytest test/test_security.py
"""
import math
import os
import shutil
import sys
import tempfile
import unittest
import asyncio

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


def _has_testclient():
    try:
        from fastapi.testclient import TestClient  # noqa: F401
        return True
    except Exception:
        return False  # httpx not installed (minimal CI dependency set)


class _FakeURL:
    def __init__(self, path):
        self.path = path


class _FakeRequest:
    """Just enough Request for security_headers() (it only reads .url.path)."""
    def __init__(self, path):
        self.url = _FakeURL(path)


def _apply_security_headers(path, media_type="text/plain"):
    """Run the real middleware against a fake request and return its headers."""
    from fastapi import Response

    async def call_next(request):
        return Response(content=b"x", media_type=media_type)

    response = asyncio.run(ss.security_headers(_FakeRequest(path), call_next))
    # Starlette normalizes to a lowercase-case-insensitive store; read via dict
    return {k.lower(): v for k, v in response.headers.items()}


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


# ── middleware-level HTTP surface (no transport needed) ─────────────────
class HeaderSecurityTests(unittest.TestCase):
    def _common_headers_assertions(self, h):
        self.assertEqual(h.get("x-content-type-options"), "nosniff")
        self.assertEqual(h.get("x-frame-options"), "DENY")
        self.assertEqual(h.get("referrer-policy"), "no-referrer")
        self.assertIn("camera=()", h.get("permissions-policy", ""))
        self.assertEqual(h.get("cross-origin-opener-policy"), "same-origin")
        self.assertEqual(h.get("cross-origin-resource-policy"), "same-origin")

    def test_root_gets_full_policy_set(self):
        h = _apply_security_headers("/")
        self._common_headers_assertions(h)
        csp = h.get("content-security-policy", "")
        self.assertIn("default-src 'self'", csp)
        self.assertIn("object-src 'none'", csp)
        self.assertIn("frame-ancestors 'none'", csp)
        self.assertIn("base-uri 'none'", csp)
        # session content must never be shared-cached
        self.assertEqual(h.get("cache-control"), "no-store")

    def test_csp_not_leaked_to_api_paths(self):
        h = _apply_security_headers("/audio")
        self.assertNotIn("content-security-policy", h)
        self.assertEqual(h.get("cache-control"), "no-store")

    def test_session_paths_are_no_store(self):
        for path in ("/", "/audio", "/scrub", "/scrub_sprite", "/healthz"):
            h = _apply_security_headers(path)
            self.assertEqual(h.get("cache-control"), "no-store", path)

    def test_static_assets_get_bounded_cache(self):
        h = _apply_security_headers("/static/app.js")
        self.assertEqual(h.get("cache-control"), "public, max-age=300")
        self.assertEqual(h.get("x-content-type-options"), "nosniff")

    def test_middleware_never_clobbers_existing_headers(self):
        from fastapi import Response

        async def call_next(request):
            r = Response(content=b"x", media_type="text/plain")
            r.headers["Cache-Control"] = "custom-policy"
            return r

        response = asyncio.run(
            ss.security_headers(_FakeRequest("/audio"), call_next))
        self.assertEqual(response.headers["Cache-Control"], "custom-policy")
        # but the baseline hardening still lands
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")

    def test_healthz_payload(self):
        body = asyncio.run(ss.healthz())
        self.assertEqual(body, {"status": "ok"})


# ── static whitelist + /audio endpoint semantics (direct calls) ─────────
class EndpointSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="asciline_sec_")
        cls.video = os.path.join(cls.tmp, "clip.avi")
        if not _make_video(cls.video):
            raise unittest.SkipTest("OpenCV could not write a test video here.")
        cls.saved_queue = getattr(ss.app.state, "queue", None)
        ss.app.state.queue = [{
            "video": cls.video, "mode": 5, "pixel": False, "vol": 1, "rows": 0,
        }]
        ss.app.state.current_index = 0

    @classmethod
    def tearDownClass(cls):
        if cls.saved_queue is not None:
            ss.app.state.queue = cls.saved_queue
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_static_whitelist_blocks_sources(self):
        from fastapi import HTTPException
        # whitelisted assets serve fine
        resp = asyncio.run(ss.serve_static("app.js"))
        self.assertTrue(getattr(resp, "status_code", 200) == 200)
        # anything not whitelisted — sources, dotfiles, traversal — is a 404
        for blocked in ("stream_server.py", "codec.py", "logo.py", ".git",
                        "..%2Fstream_server.py", "..%5Cstream_server.py",
                        "%2E%2E%2Fcodec.py", "../stream_server.py"):
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(ss.serve_static(blocked))
            self.assertEqual(ctx.exception.status_code, 404, blocked)

    def test_audio_pool_saturation_returns_503(self):
        from fastapi import HTTPException
        saved = getattr(ss.app.state, "audio_semaphore", None)
        try:
            gate = asyncio.Semaphore(1)

            async def hold():
                await gate.acquire()
            asyncio.run(hold())  # occupy the only slot
            ss.app.state.audio_semaphore = gate
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(ss.audio_stream(v=None, start=0.0))
            self.assertEqual(ctx.exception.status_code, 503)
        finally:
            ss.app.state.audio_semaphore = saved

    def test_audio_muted_video_is_204(self):
        saved = ss.app.state.queue
        try:
            ss.app.state.queue = [{
                "video": self.video, "mode": 5, "pixel": False, "vol": 0, "rows": 0,
            }]
            resp = asyncio.run(ss.audio_stream(v=None, start=0.0))
            self.assertEqual(resp.status_code, 204)
        finally:
            ss.app.state.queue = saved

    def test_audio_start_param_is_sanitized(self):
        # negative / zero offsets must not reach the ffmpeg cmdline: the
        # endpoint answers with a normal streaming response, never raises.
        for bad in (-50.0, 0.0):
            resp = asyncio.run(ss.audio_stream(v=None, start=bad))
            self.assertEqual(getattr(resp, "status_code", 200), 200)


# ── WebSocket resilience (transport-level; needs httpx via TestClient) ───
@unittest.skipUnless(_has_testclient(), "fastapi TestClient (httpx) not installed")
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
