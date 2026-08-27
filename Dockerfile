FROM python:3.12-slim

# ffmpeg is all yt-dlp needs to extract audio - no ML deps here.
# Local Whisper (faster-whisper/ctranslate2, ~1.5GB) is intentionally NOT
# installed: it doesn't fit Render's free 512MB, and it's a fallback of last
# resort behind Groq + YouTube captions anyway (app.HOSTED disables it).
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" yt-dlp

COPY app.py index.html ./

ENV PORT=8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
