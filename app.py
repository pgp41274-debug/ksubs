# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi", "uvicorn", "faster-whisper", "yt-dlp", "imageio-ffmpeg", "deno"]
# ///
"""Korean YouTube -> bilingual subtitles, watched in place. All local, no paid API."""
from __future__ import annotations

import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import threading
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

# Windows consoles default to cp1252, which raises on Korean titles and cue text
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

HERE = Path(__file__).parent
MODEL = os.environ.get("KSUBS_MODEL", "large-v3-turbo")
# set when deployed (Render etc): skip local Whisper (too heavy for a free 512MB
# host) and gate every request behind a shared password so a public URL can't
# spend someone else's Groq key
HOSTED = bool(os.environ.get("KSUBS_PASSWORD"))
KSUBS_PASSWORD = os.environ.get("KSUBS_PASSWORD", "")
DEVICE = os.environ.get("KSUBS_DEVICE", "cpu")  # set KSUBS_DEVICE=cuda to try the MX450
BATCH = 40
WORKERS = 4  # parallel claude calls; translation is the bottleneck once captions are free

# Groq runs the same Whisper weights on LPU hardware at ~220x realtime, free tier,
# no card: 28,800 audio-seconds/day. Key: https://console.groq.com/keys
def _user_env(name: str) -> str:
    """Windows only. A shell opened before `setx` ran keeps the old environment
    forever, so fall back to the persistent user value in the registry."""
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            return str(winreg.QueryValueEx(k, name)[0])
    except Exception:
        return ""


# comma-separate several keys to fail over between orgs you actually hold:
#   setx GROQ_API_KEY "gsk_personal,gsk_work"
_raw_keys = os.environ.get("GROQ_API_KEY") or _user_env("GROQ_API_KEY")
KEY_SOURCE = "environment" if os.environ.get("GROQ_API_KEY") else "saved user setting"
GROQ_KEYS = [k.strip() for k in _raw_keys.split(",") if k.strip()]
GROQ_KEY = GROQ_KEYS[0] if GROQ_KEYS else ""
GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = os.environ.get("KSUBS_GROQ_MODEL", "whisper-large-v3-turbo")
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_CHAT_MODEL = os.environ.get("KSUBS_CHAT_MODEL", "qwen/qwen3.8-27b")
GROQ_MAX_MB = 25
# claude gives better Korean slang but needs `claude` + /login; groq needs only the key
TRANSLATOR = os.environ.get("KSUBS_TRANSLATOR", "groq" if GROQ_KEY else "claude")
YDL = {"quiet": True, "no_warnings": True, "noprogress": True,
       "js_runtimes": {"deno": {}, "node": {}}}  # node is installed; without it YT extraction degrades

_cookie_file = None


def cookie_opts() -> dict:
    """YouTube bot-checks datacenter IPs (Render, AWS, ...) on ordinary videos,
    not just age-restricted ones - the fix is cookies from a real signed-in
    session. Set YTDLP_COOKIES to the contents of a Netscape-format cookies.txt
    (e.g. from the 'Get cookies.txt' browser extension). Applies to every
    request when set - there's no browser on the server for the local
    'cookiesfrombrowser' checkbox to read from."""
    global _cookie_file
    raw = os.environ.get("YTDLP_COOKIES", "")
    if not raw:
        return {}
    if _cookie_file is None:
        f = Path(tempfile.gettempdir()) / "ksubs_cookies.txt"
        f.write_text(raw, encoding="utf-8")
        _cookie_file = str(f)
    return {
        "cookiefile": _cookie_file,
        # once cookies are set, YouTube's default (tv_downgraded) client is
        # broken server-side - this pair is yt-dlp's documented workaround
        "extractor_args": {"youtube": {"player_client": ["default", "web_embedded"]}},
        # even with that, most clients now need YouTube's "n" JS challenge
        # solved or every format gets silently dropped ("page needs to be
        # reloaded"). This downloads yt-dlp's own solver script from
        # github.com/yt-dlp/ejs on first use and caches it - it's yt-dlp's own
        # official release, not arbitrary code, but it is a runtime fetch+run.
        "remote_components": {"ejs:github"},
    }
CLAUDE = shutil.which("claude") or "claude"

# ---------------------------------------------------------------- srt


def fmt_ts(t: float) -> str:
    ms = int(round(max(t, 0.0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def build_srt(cues: list[dict]) -> str:
    """English only - Korean doubles the on-screen height. A cue that failed to
    translate falls back to its Korean so you never get a blank subtitle."""
    blocks = []
    for i, c in enumerate(cues, 1):
        text = c.get("en") or c["ko"]
        blocks.append(f"{i}\n{fmt_ts(c['start'])} --> {fmt_ts(c['end'])}\n{text}")
    return "\n\n".join(blocks) + "\n"


# ---------------------------------------------------------------- translation

PROMPT = """You are translating Korean subtitles from a hidden-camera prank video into natural, colloquial English.

Rules:
- Output EXACTLY one line per input line, with the SAME number, in the SAME order.
- Format every line as: N|||English text
- Never merge, split, reorder, add or omit lines. A line that is just laughter or a filler sound still gets its own numbered output line.
- These are friends joking around. Use casual spoken English, not formal prose. Keep the joke intact rather than translating word-for-word.
- Korean drops subjects constantly. Use the surrounding lines to work out who is doing what.
- No commentary, no preamble, no code fences. Only the N|||text lines.

Lines:
{lines}"""

LINE_RE = re.compile(r"^\s*(\d+)\s*\|\|\|\s*(.*)$")


def _claude(prompt: str) -> str:
    p = subprocess.run(
        [CLAUDE, "-p", "--output-format", "text"],
        input=prompt,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    out = (p.stdout or "").strip()
    if p.returncode != 0 or "Not logged in" in out:
        raise RuntimeError(f"claude failed: {(out or p.stderr or '')[:200]}")
    return out


# quota, auth and overload errors are worth trying the next key for; a 400 is
# our own bad request and retrying it anywhere is pointless
GROQ_FAILOVER = {401, 403, 429, 500, 502, 503, 504}
_key_idx = 0  # stick to whichever key last worked, so an exhausted one isn't retried


def _http_send(url: str, data: bytes, headers: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def groq_request(url: str, data: bytes, content_type: str, timeout: int = 180,
                 send=_http_send, log=print) -> dict:
    global _key_idx
    if not GROQ_KEYS:
        raise RuntimeError("no GROQ_API_KEY set")
    last = "no attempt made"
    for attempt in range(len(GROQ_KEYS)):
        i = (_key_idx + attempt) % len(GROQ_KEYS)
        headers = {"Authorization": f"Bearer {GROQ_KEYS[i]}",
                   "Content-Type": content_type,
                   "User-Agent": "ksubs/1.0"}
        try:
            out = send(url, data, headers, timeout)
            _key_idx = i  # stick here for next time
            return out
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
            if e.code not in GROQ_FAILOVER:
                break
            if attempt + 1 < len(GROQ_KEYS):
                log(f"  key {i + 1} gave {e.code}, trying key {i + 2}")
            elif e.code == 429:
                time.sleep(8)  # every key is rate-limited: wait out the window
    raise RuntimeError(f"Groq failed on {len(GROQ_KEYS)} key(s) - {last}")


def _groq_chat(prompt: str) -> str:
    body = json.dumps({
        "model": GROQ_CHAT_MODEL, "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    data = groq_request(GROQ_CHAT_URL, body, "application/json", timeout=180)
    return data["choices"][0]["message"]["content"]


def translator():
    return _groq_chat if TRANSLATOR == "groq" else _claude


def _parse(reply: str, want: set[int]) -> dict[int, str]:
    got = {}
    for line in reply.splitlines():
        m = LINE_RE.match(line)
        if m:
            i = int(m.group(1))
            if i in want and m.group(2).strip():
                got[i] = m.group(2).strip()
    return got


def translate_batch(batch: list[tuple[int, str]], call=None, log=print) -> dict[int, str]:
    """Return idx -> English. A missing idx means that cue failed; caller keeps Korean."""
    call = call or translator()
    got: dict[int, str] = {}
    remaining = list(batch)
    for attempt in (1, 2, 3):
        body = "\n".join(f"{i}|||{ko}" for i, ko in remaining)
        try:
            got.update(_parse(call(PROMPT.format(lines=body)), {i for i, _ in remaining}))
        except Exception as e:
            log(f"  attempt {attempt} errored: {e}")
        # only chase the lines still missing - a 1-line retry almost always lands,
        # where resending all 40 just rolls the dice on a different 39
        remaining = [(i, ko) for i, ko in remaining if i not in got]
        if not remaining:
            break
        if attempt < 3:
            log(f"  {len(remaining)}/{len(batch)} line(s) missing, retrying just those")
    if remaining:
        log(f"  gave up on {len(remaining)} line(s), keeping Korean")
    return got


def translate(cues: list[dict], job: dict | None = None, call=None, log=print,
              workers=WORKERS) -> int:
    """Fill c['en'] in place. Cues are never dropped, merged or reordered, so
    timestamps survive untouched no matter what the model returns."""
    call = call or translator()
    batches = [(s, cues[s : s + BATCH]) for s in range(0, len(cues), BATCH)]
    done = 0

    def run(item):
        start, chunk = item
        got = translate_batch(
            [(start + n, c["ko"]) for n, c in enumerate(chunk)], call, log
        )
        return start, chunk, got

    def apply(start, chunk, got):
        nonlocal done
        for n, c in enumerate(chunk):
            c["en"] = got.get(start + n, "")
        done += len(chunk)
        if job is not None:
            job["percent"] = int(done / len(cues) * 100)
            job["cues"] = cues

    # first batch alone: a broken translator (usually "not logged in") should
    # fail in 30 seconds, not after grinding through every batch
    start, chunk, got = run(batches[0])
    if not got:
        raise RuntimeError(
            "translator returned nothing - check it with:  uv run app.py --check"
            "  (the Korean-only .srt is still downloadable)"
        )
    apply(start, chunk, got)

    # the rest in parallel - batches are independent and merged back by index
    if len(batches) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for start, chunk, got in ex.map(run, batches[1:]):
                apply(start, chunk, got)
    return sum(1 for c in cues if not c["en"])


# ---------------------------------------------------------------- youtube captions

SRV1_RE = re.compile(r'<text start="([\d.]+)" dur="([\d.]+)">(.*?)</text>', re.S)


def _pick_track(track, exts=("srv1", "vtt", "json3")):
    for ext in exts:
        for f in track or ():
            if f.get("ext") == ext and f.get("url"):
                return ext, f["url"]
    return None, None


def parse_srv1(raw: str) -> list[dict]:
    cues = []
    for start, dur, text in SRV1_RE.findall(raw):
        # YouTube double-escapes: "&amp;#39;" -> "&#39;" -> "'", so unescape twice
        t = html.unescape(html.unescape(re.sub(r"<[^>]+>", "", text))).strip()
        if t:
            cues.append({"start": float(start), "end": float(start) + float(dur),
                         "ko": t, "en": ""})
    # auto-captions roll: each line's dur runs past the next line's start, which
    # would make two cues live at once. Clamp every cue to where the next begins.
    for a, b in zip(cues, cues[1:]):
        a["end"] = min(a["end"], b["start"])
    return [c for c in cues if c["end"] > c["start"]]


def merge_cues(cues, max_gap=0.8, max_dur=5.5, max_chars=42) -> list[dict]:
    """YouTube breaks lines mid-sentence. Glue them back into readable cues -
    it also gives the translator whole thoughts instead of fragments."""
    out: list[dict] = []
    for c in cues:
        if out:
            p = out[-1]
            if (c["start"] - p["end"] <= max_gap
                    and c["end"] - p["start"] <= max_dur
                    and len(p["ko"]) + len(c["ko"]) + 1 <= max_chars):
                p["ko"] += " " + c["ko"]
                p["end"] = c["end"]
                continue
        out.append(dict(c))
    return out


def collapse_repeats(cues: list[dict]) -> list[dict]:
    """Whisper loops the same short line over non-speech (screaming, music), and the
    Groq API takes no vad_filter. Collapse consecutive identical text into one cue
    spanning the whole run - one steady subtitle instead of 19 flickering ones."""
    out: list[dict] = []
    for c in cues:
        if out and out[-1]["ko"] == c["ko"]:
            out[-1]["end"] = c["end"]
            continue
        out.append(dict(c))
    return out


def fetch_captions(info: dict, log=print):
    """Korean subtitles straight from YouTube: seconds instead of minutes.
    Returns (cues, source) or (None, None) if this video has no Korean track."""
    for tracks, label in ((info.get("subtitles") or {}, "creator Korean subtitles"),
                          (info.get("automatic_captions") or {}, "YouTube auto-captions")):
        ext, url = _pick_track(tracks.get("ko"))
        if not url:
            continue
        try:
            raw = urllib.request.urlopen(url, timeout=60).read().decode("utf-8", "replace")
        except Exception as e:
            log(f"  {label} fetch failed: {e}")
            continue
        cues = parse_srv1(raw) if ext == "srv1" else []
        if cues:
            log(f"  got {len(cues)} segments from {label}")
            return cues, label
    return None, None


# ---------------------------------------------------------------- groq whisper


def _multipart(fields: dict, filename: str, data: bytes):
    b = uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    out.append(
        f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n".encode()
    )
    out.append(data)
    out.append(f"\r\n--{b}--\r\n".encode())
    return b, b"".join(out)


def transcribe_groq(path: Path, log=print) -> list[dict]:
    """Same Whisper weights as a local run, but ~220x realtime on Groq's LPUs."""
    size_mb = path.stat().st_size / 1e6
    if size_mb > GROQ_MAX_MB:
        raise RuntimeError(f"audio is {size_mb:.0f} MB, over Groq's {GROQ_MAX_MB} MB limit")
    boundary, body = _multipart(
        {"model": GROQ_MODEL, "language": "ko",
         "response_format": "verbose_json", "temperature": "0"},
        path.name, path.read_bytes(),
    )
    # without a User-Agent, Cloudflare in front of Groq replies 403 code 1010
    data = groq_request(GROQ_URL, body, f"multipart/form-data; boundary={boundary}",
                        timeout=600, log=log)
    cues = []
    for s in data.get("segments", []):
        t = (s.get("text") or "").strip()
        if t and s.get("no_speech_prob", 0) <= 0.6:
            cues.append({"start": s["start"], "end": s["end"], "ko": t, "en": ""})
    log(f"  Groq returned {len(cues)} segments from {size_mb:.1f} MB")
    return cues


# ---------------------------------------------------------------- audio + whisper


def fetch_audio(url: str, outdir: Path, job: dict, cookies: bool = False, fmt: str = "wav"):
    import imageio_ffmpeg
    import yt_dlp

    def hook(d):
        if d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate")
            if tot:
                job["percent"] = int(d.get("downloaded_bytes", 0) / tot * 100)

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(outdir / "audio.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        # YouTube extraction is degraded without a JS runtime; node is already installed
        "js_runtimes": {"deno": {}, "node": {}},
        # bundled static binary - no system ffmpeg needed, so this also runs on a
        # bare Render container with nothing but Python installed
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        # audio only, never the video; 16 kHz mono is what whisper wants anyway.
        # mp3 32k keeps a 30-min video ~7 MB, well under Groq's 25 MB limit.
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": fmt}],
        "postprocessor_args": {
            "extractaudio": ["-ar", "16000", "-ac", "1"]
            + ([] if fmt == "wav" else ["-b:a", "32k"])
        },
        **cookie_opts(),
    }
    if cookies:
        opts["cookiesfrombrowser"] = ("chrome",)
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(url, download=True)
    return outdir / f"audio.{fmt}", info.get("title", ""), info.get("id", "")


_model = None


def load_model(log=print):
    global _model
    if HOSTED:
        raise RuntimeError(
            "local Whisper isn't available on this deployment "
            "(Groq and YouTube captions both missed on this video - try another)"
        )
    if _model is None:
        from faster_whisper import WhisperModel

        ct = "int8_float16" if DEVICE == "cuda" else "int8"
        log(f"loading {MODEL} on {DEVICE}/{ct} (first run downloads ~1.6 GB)")
        _model = WhisperModel(MODEL, device=DEVICE, compute_type=ct, cpu_threads=8)
    return _model


def transcribe(wav: Path, job: dict, log=print) -> list[dict]:
    segments, info = load_model(log).transcribe(
        str(wav),
        language="ko",  # pinned: auto-detect drifts to Japanese on noisy Korean
        task="transcribe",  # turbo cannot translate; Claude does that step
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=400),
        condition_on_previous_text=False,  # stops hallucination loops
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )
    total = info.duration or 0
    cues: list[dict] = []
    for s in segments:
        text = s.text.strip()
        if not text or s.no_speech_prob > 0.6:
            continue
        cues.append({"start": s.start, "end": s.end, "ko": text, "en": ""})
        job["cues"] = cues
        if total:
            job["percent"] = min(99, int(s.end / total * 100))
    return cues


# ---------------------------------------------------------------- job runner

# ponytail: in-memory jobs, no persistence - add sqlite only if you want history
JOBS: dict[str, dict] = {}


def run_job(job_id: str, url: str, cookies: bool, force_whisper: bool = False):
    job = JOBS[job_id]

    def log(msg):
        print(f"[{job_id[:6]}] {msg}", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="ksubs-"))
    try:
        import yt_dlp

        job.update(stage="captions", percent=0)
        opts = dict(YDL, skip_download=True, **cookie_opts())
        if cookies:
            opts["cookiesfrombrowser"] = ("chrome",)
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(url, download=False)
        job.update(title=info.get("title", ""), video_id=info.get("id", ""))
        log(f"video: {info.get('title', '')}")

        cues, source = None, None

        # 1. Groq: real Whisper large-v3-turbo at ~220x realtime, free tier.
        if GROQ_KEY and not force_whisper:
            try:
                job.update(stage="audio", percent=0)
                mp3, _, _ = fetch_audio(url, tmp, job, cookies, fmt="mp3")
                job.update(stage="groq", percent=0)
                cues = transcribe_groq(mp3, log)
                source = f"Groq {GROQ_MODEL}"
            except Exception as e:
                log(f"Groq failed ({e}) - falling back")
                cues = None

        # 2. YouTube's own Korean captions: seconds, no key needed.
        if not cues and not force_whisper:
            job.update(stage="captions", percent=0)
            cues, source = fetch_captions(info, log)
            if cues:
                cues = merge_cues(cues)

        # 3. Local Whisper: same quality as Groq but ~1x video length on this CPU.
        if not cues:
            log("Whisper requested" if force_whisper else "no captions - using local Whisper")
            job.update(stage="audio", percent=0)
            wav, _, _ = fetch_audio(url, tmp, job, cookies, fmt="wav")
            job.update(stage="transcribe", percent=0)
            cues = transcribe(wav, job, log)
            source = "local Whisper"
        before = len(cues)
        cues = collapse_repeats(cues)
        if len(cues) != before:
            log(f"  collapsed {before - len(cues)} repeated cues (hallucination loops)")
        job["cues"] = cues
        log(f"{len(cues)} cues from {source}")
        job["source"] = source
        if not cues:
            raise RuntimeError("no Korean speech or captions found in this video")

        job.update(stage="translate", percent=0)
        missing = translate(cues, job, log=log)
        job.update(stage="done", percent=100, untranslated=missing)
        log(f"done ({missing} cues left untranslated)")
    except Exception as e:
        job.update(stage="error", error=str(e))
        log(f"FAILED: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- web

# injected into index.html only when hosted (KSUBS_PASSWORD set). Prompts once,
# caches in this browser's localStorage, and patches fetch() so every POST /jobs
# carries it - index.html itself needs no changes for the hosted case.
INLINE_PASSWORD_SCRIPT = """
<script>
(function () {
  const KEY = 'ksubs_pw';
  const realFetch = window.fetch;
  window.fetch = function (url, opts) {
    if (typeof url === 'string' && url.startsWith('/jobs') && (!opts || !opts.method || opts.method === 'POST')) {
      let pw = localStorage.getItem(KEY);
      if (!pw) { pw = prompt('Password:') || ''; localStorage.setItem(KEY, pw); }
      opts = opts || {};
      opts.headers = Object.assign({}, opts.headers, {'X-Ksubs-Password': pw});
      return realFetch(url, opts).then(r => {
        if (r.status === 401) { localStorage.removeItem(KEY); alert('Wrong password - try again.'); }
        return r;
      });
    }
    return realFetch(url, opts);
  };
})();
</script>
"""

app = FastAPI()


def require_password(request: Request):
    if HOSTED and request.headers.get("x-ksubs-password") != KSUBS_PASSWORD:
        raise HTTPException(401, "wrong or missing password")


@app.get("/")
def index():
    page = (HERE / "index.html").read_text(encoding="utf-8")
    if HOSTED:
        page = page.replace("</main>", INLINE_PASSWORD_SCRIPT + "</main>")
    return HTMLResponse(page)


@app.post("/jobs")
def create_job(body: dict, _=Depends(require_password)):
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "stage": "captions",
        "percent": 0,
        "cues": [],
        "title": "",
        "video_id": "",
        "source": "",
        "error": "",
        "untranslated": 0,
    }
    threading.Thread(
        target=run_job,
        args=(job_id, body["url"], bool(body.get("cookies")), bool(body.get("whisper"))),
        daemon=True,
    ).start()
    return {"id": job_id}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    return JOBS.get(job_id, {"stage": "error", "error": "unknown job"})


@app.get("/jobs/{job_id}/srt")
def get_srt(job_id: str):
    job = JOBS.get(job_id, {})
    title = (job.get("title") or job_id)[:80]
    # HTTP headers are latin-1 only, so a Korean title needs an ASCII fallback
    # plus the real name in RFC 5987 form
    ascii_name = re.sub(r"[^A-Za-z0-9._ -]", "_", title).strip("_ ") or "subtitles"
    utf8_name = urllib.parse.quote(f"{title}.srt")
    return PlainTextResponse(
        build_srt(job.get("cues", [])),
        headers={
            "Content-Disposition": (
                f'attachment; filename="{ascii_name}.srt"; '
                f"filename*=UTF-8''{utf8_name}"
            )
        },
        media_type="application/x-subrip",
    )


def check():
    """uv run app.py --check - verify the pipeline before burning a whole video."""
    if GROQ_KEY:
        n = len(GROQ_KEYS)
        print(f"transcribe : Groq {GROQ_MODEL}  (~200x realtime)")
        print(f"keys       : {n} key{'s' if n > 1 else ''}"
              + (" - fails over automatically" if n > 1 else ""))
    else:
        print("transcribe : GROQ_API_KEY not set - falling back to YouTube captions.")
        print("             Free key, no card: https://console.groq.com/keys")
        print("             PowerShell, persists across windows:")
        print('               setx GROQ_API_KEY "gsk_..."   (then open a NEW window)')
    if TRANSLATOR == "groq":
        print(f"translate  : Groq {GROQ_CHAT_MODEL}")
    else:
        print(f"translate  : claude cli {CLAUDE}")
        print("             not logged in? run  claude  then type  /login  inside it")
    try:
        got = translate_batch([(1, "야 이거 진짜 대박이다"), (2, "차라리 그냥 집에 가자")])
    except Exception as e:
        print(f"FAIL: {e}")
        return 1
    if len(got) == 2:
        for k, v in sorted(got.items()):
            print(f"OK  {k}: {v}")
        return 0
    print(f"FAIL: got {len(got)}/2 lines back")
    return 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        raise SystemExit(check())
    import uvicorn

    if GROQ_KEYS:
        n = len(GROQ_KEYS)
        print(f"transcribe : Groq {GROQ_MODEL} - {n} key{'s' if n > 1 else ''}"
              f" from {KEY_SOURCE}")
        print(f"translate  : Groq {GROQ_CHAT_MODEL}")
    else:
        print("transcribe : NO GROQ KEY - will use YouTube auto-captions (lower quality)")
        print('             fix: setx GROQ_API_KEY "gsk_..."   then restart this')
        print(f"translate  : {TRANSLATOR}")
    print("open http://localhost:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
