# Security Policy

ASCILINE is a self-hosted media streaming server written in Python (FastAPI) +
vanilla JS. This document describes the threat model the code is hardened
against, the defenses that ship enabled by default, and how to deploy it
beyond a trusted LAN.

## Supported Versions

Security fixes land on `main`; there are no stable back-port branches. Run the
latest release.

## Threat Model

ASCILINE is designed for localhost and trusted-LAN use. When bound to
`--host 0.0.0.0`, anyone who can reach the port can watch the stream and send
WebSocket commands. The defenses below assume a **hostile network**: a client
may send malformed, hostile or flooding traffic, but the playlists, video
files and CLI flags are operator-controlled (they are *not* network input).

What is defended, by design:

| Attack surface | Defense |
| :------------- | :------ |
| Cross-site WebSocket hijacking | Origin allowlist on `/ws` (localhost / same-host only), close 1008 on violation |
| Client-driven memory floods | `ws_max_size` bound (1 MiB) on inbound frames → close 1009; bounded per-connection send queues; max-client admission control (close 1013) |
| Command-channel abuse | Per-message isolation (a bad frame can never kill the pump), NaN/Infinity/junk-proof coercion of every numeric field, seek times clamped to the video, token-bucket rate limiting on expensive seek/reinit commands |
| Process-pool exhaustion (`/audio` spawns ffmpeg per request) | Global semaphore cap → 503 at saturation; argv-array invocation only (no shell, so no injection); offset sanity-clamped before reaching the command line |
| Thumbnail CPU exhaustion (`/scrub` decodes videos) | Same-path build de-duplication + global concurrency cap; build failures degrade to "unavailable", never a 500 |
| Path traversal / source disclosure | `/static/*` serves an explicit file whitelist only; no dynamic file resolution from request data |
| Frontend injection | Strict CSP (`default-src 'self'`, no inline scripts, `object-src 'none'`, `frame-ancestors 'none'`), `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, tight `Permissions-Policy`, COOP/CORP `same-origin` |
| Cache confusion | `no-store` on all session/dynamic endpoints; bounded 5-minute caching on the whitelisted static assets only |
| Container escape impact | Docker image runs as a dedicated non-root user; compose drops all capabilities and sets `no-new-privileges` |
| Poisoned downloads | yt-dlp downloads are probed and normalized through subprocesses with hard timeouts; a stuck source can never hang the server |

Out of scope (by design): authentication and authorization. ASCILINE has no
accounts. **Do not expose the port to the public internet directly.** For
remote access, put it behind a reverse proxy that terminates TLS and adds
authentication (e.g. nginx + Basic Auth, Caddy, or an identity-aware proxy),
and connect the browser via `https://`/`wss://` — the frontend negotiates
`wss:` automatically from the page scheme.

## Reporting a Vulnerability

Please report vulnerabilities privately to
[asciline.engine@gmail.com](mailto:asciline.engine@gmail.com) with a
reproduction (request payload or script), the affected version/commit, and
impact assessment. Do not open public issues for unpatched vulnerabilities.
We aim to acknowledge within 7 days.
