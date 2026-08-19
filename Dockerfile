# ASCILINE live server — Python + ffmpeg, no host dependency install needed.
FROM python:3.11-slim

WORKDIR /app

# ffmpeg/ffprobe for audio + thumbnails; ca-certificates for yt-dlp HTTPS.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Container has no display — headless OpenCV is the lighter drop-in (see README).
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python || true \
    && pip install --no-cache-dir opencv-python-headless

COPY . .

# Default playback folder; mount host videos here via compose.
# Runtime user: the server must NOT run as root. A vulnerability anywhere in
# the stack (dependency, decoder, ffmpeg) then meets an unprivileged account
# whose writable surface is limited to /app (the bind-mounted cache dir).
RUN mkdir -p videos \
    && groupadd --system asciline \
    && useradd --system --gid asciline --home-dir /app asciline \
    && chown -R asciline:asciline /app

# Python must not try to write bytecode caches under a non-root account.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER asciline

EXPOSE 8000

# Must bind 0.0.0.0 so port mapping reaches the host browser.
# --folder videos: drop files into the mounted ./videos directory.
# NOTE: the container binds the unprivileged port 8000, which the asciline
# user is allowed to do (no capabilities required).
CMD ["python", "stream_server.py", "--folder", "videos", "--host", "0.0.0.0", "--port", "8000"]
