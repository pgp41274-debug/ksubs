# /// script
# requires-python = ">=3.11"
# dependencies = ["fastapi"]
# ///
"""Subtitle sync integrity. Run: uv run test_srt.py

The one thing that must never break: a bad translation reply must not shift
subtitles onto the wrong timestamps. Everything else is cosmetic.
"""
import app


def ids_in(prompt):
    """Pull the cue numbers back out of a prompt, ignoring the instructions."""
    out = []
    for line in prompt.splitlines():
        if "|||" in line:
            head = line.split("|||")[0].strip()
            if head.isdigit():
                out.append(int(head))
    return out


def make_cues(n):
    return [
        {"start": i * 2.0, "end": i * 2.0 + 1.5, "ko": f"한국어{i}", "en": ""}
        for i in range(n)
    ]


# ---- timestamps ----------------------------------------------------------

assert app.fmt_ts(0) == "00:00:00,000"
assert app.fmt_ts(3722.5) == "01:02:02,500"
assert app.fmt_ts(59.999) == "00:00:59,999"
assert app.fmt_ts(-1) == "00:00:00,000"
print("ok  timestamps")

# ---- bilingual layout ----------------------------------------------------

srt = app.build_srt([{"start": 842.12, "end": 844.88, "ko": "야 대박", "en": "Dude, insane"}])
assert srt == "1\n00:14:02,120 --> 00:14:04,880\nDude, insane\n", repr(srt)
assert "야 대박" not in srt, "Korean must not appear when a translation exists"

# an untranslated cue falls back to Korean rather than emitting a blank subtitle
solo = app.build_srt([{"start": 0, "end": 1, "ko": "야", "en": ""}])
assert solo == "1\n00:00:00,000 --> 00:00:01,000\n야\n", repr(solo)
print("ok  english-only srt, korean fallback when untranslated")

# ---- parsing a messy reply ----------------------------------------------

reply = "Sure, here you go:\n```\n0|||Hello\n1|||Bye\n```\n99|||not asked for"
got = app._parse(reply, {0, 1})
assert got == {0: "Hello", 1: "Bye"}, got
print("ok  parse ignores preamble, fences and unasked ids")

# ---- THE CHECK: a dropped line must not desync the rest ------------------

cues = make_cues(40)
before = [(c["start"], c["end"]) for c in cues]
calls = []


def drops_one(prompt):
    ids = ids_in(prompt)
    calls.append(len(ids))
    return "\n".join(f"{i}|||EN{i}" for i in ids[:-1])  # loses the last cue, every time


missing = app.translate(cues, call=drops_one, log=lambda *a: None)

assert len(cues) == 40, f"cue count changed: {len(cues)}"
assert [(c["start"], c["end"]) for c in cues] == before, "timestamps moved"
assert missing == 1, missing
assert cues[39]["en"] == "", "dropped cue should have no English"
assert cues[38]["en"] == "EN38", "surviving cues must keep their own translation"
assert all(c["en"] == f"EN{i}" for i, c in enumerate(cues[:39])), "translations shifted"
assert calls == [40, 1, 1], f"retries must resend ONLY the missing line, sent {calls}"
assert app.build_srt(cues).count(" --> ") == 40
print("ok  dropped line falls back to Korean, nothing shifts")

# ---- a flaky first attempt recovers on retry -----------------------------

cues = make_cues(5)
state = {"n": 0}


def flaky(prompt):
    state["n"] += 1
    if state["n"] == 1:
        return "sorry, I could not do that"
    return "\n".join(f"{i}|||EN{i}" for i in ids_in(prompt))


assert app.translate(cues, call=flaky, log=lambda *a: None) == 0
assert [c["en"] for c in cues] == [f"EN{i}" for i in range(5)]
print("ok  retry recovers a failed batch")

# ---- a dead translator fails fast instead of grinding every batch --------

cues = make_cues(3)
before = [(c["start"], c["end"]) for c in cues]


def always_raises(prompt):
    raise RuntimeError("claude failed: Not logged in")


try:
    app.translate(cues, call=always_raises, log=lambda *a: None)
    raise AssertionError("a dead translator should raise on the first batch")
except RuntimeError as e:
    assert "--check" in str(e), e
assert [(c["start"], c["end"]) for c in cues] == before, "timestamps moved"
assert app.build_srt(cues).count(" --> ") == 3, "Korean-only srt must still build"
print("ok  dead translator fails fast, Korean transcript survives")

# ---- but a LATER batch failing only degrades that batch ------------------

cues = make_cues(app.BATCH + 5)
state = {"n": 0}


def dies_after_first_batch(prompt):
    state["n"] += 1
    if state["n"] >= 2:  # first batch (1 call) ok, then every later call fails
        raise RuntimeError("transient")
    return "\n".join(f"{i}|||EN{i}" for i in ids_in(prompt))


missing = app.translate(cues, call=dies_after_first_batch, log=lambda *a: None)
assert missing == 5, missing
assert all(c["en"] == f"EN{i}" for i, c in enumerate(cues[: app.BATCH])), "batch 1 lost"
assert all(c["en"] == "" for c in cues[app.BATCH :]), "batch 2 should be empty"
assert app.build_srt(cues).count(" --> ") == app.BATCH + 5
print("ok  a later batch failing degrades only itself")

# ---- batching boundary ---------------------------------------------------

cues = make_cues(app.BATCH * 2 + 7)
sizes = []


def perfect(prompt):
    ids = ids_in(prompt)
    sizes.append(len(ids))
    return "\n".join(f"{i}|||EN{i}" for i in ids)


assert app.translate(cues, call=perfect, log=lambda *a: None) == 0
assert sorted(sizes) == [7, app.BATCH, app.BATCH], sizes  # parallel: order not fixed
assert all(c["en"] == f"EN{i}" for i, c in enumerate(cues)), "indices wrong across batches"
print("ok  batching keeps global indices (parallel)")

# ---- youtube caption parsing --------------------------------------------

srv1 = (
    '<?xml version="1.0" encoding="utf-8"?><transcript>'
    '<text start="0.0" dur="1.7">내가 어제 배우는 여자 동생들</text>'
    '<text start="1.7" dur="1.1">많았다고 했잖아</text>'
    '<text start="12.8" dur="2.0">근데 반면에 나는 &amp;quot;막&amp;quot; 우악스럽고</text>'
    "</transcript>"
)
parsed = app.parse_srv1(srv1)
assert len(parsed) == 3, parsed
assert parsed[0]["start"] == 0.0 and round(parsed[0]["end"], 2) == 1.7
assert "&" not in parsed[2]["ko"], parsed[2]["ko"]  # entities decoded
print("ok  srv1 captions parse with entities decoded")

# real auto-captions roll: dur runs past the next line's start, so two cues
# would be live at once and the player would show the wrong one
rolling = app.parse_srv1(
    '<text start="0.0" dur="4.6">가</text>'
    '<text start="2.8" dur="5.0">나</text>'
    '<text start="6.5" dur="5.2">다</text>'
)
assert [round(c["end"], 1) for c in rolling] == [2.8, 6.5, 11.7], rolling
assert all(a["end"] <= b["start"] for a, b in zip(rolling, rolling[1:])), "still overlapping"
print("ok  rolling caption overlap clamped")

# ---- merging youtube's mid-sentence breaks -------------------------------

merged = app.merge_cues(parsed)
assert len(merged) == 2, [c["ko"] for c in merged]
assert merged[0]["ko"] == "내가 어제 배우는 여자 동생들 많았다고 했잖아"
assert merged[0]["start"] == 0.0 and round(merged[0]["end"], 2) == 2.8, merged[0]
assert merged[1]["start"] == 12.8, "a 10s gap must not be merged"
print("ok  choppy caption lines merge back into sentences")

# merging must never invent, lose or reorder time
long_run = [{"start": i * 0.5, "end": i * 0.5 + 0.5, "ko": "가", "en": ""} for i in range(50)]
m = app.merge_cues(long_run)
assert m[0]["start"] == 0.0
assert m[-1]["end"] == long_run[-1]["end"], "merge lost the tail"
assert all(m[i]["end"] <= m[i + 1]["start"] for i in range(len(m) - 1)), "merged cues overlap"
assert all(c["end"] - c["start"] <= 5.5 + 1e-9 for c in m), "merged cue too long to read"
print("ok  merging preserves span, order and readable length")

# ---- hallucination loops collapse into one steady cue --------------------

loop = [{"start": float(i), "end": float(i + 1), "ko": "아!", "en": ""} for i in range(19)]
loop = [{"start": -1.0, "end": 0.0, "ko": "시작", "en": ""}] + loop
loop.append({"start": 19.0, "end": 20.0, "ko": "끝", "en": ""})
col = app.collapse_repeats(loop)
assert len(col) == 3, [c["ko"] for c in col]
assert col[1]["ko"] == "아!" and col[1]["start"] == 0.0 and col[1]["end"] == 19.0, col[1]
assert col[0]["ko"] == "시작" and col[2]["ko"] == "끝", "surrounding cues must survive"
# a repeat that is NOT consecutive must be left alone
sep = app.collapse_repeats([
    {"start": 0, "end": 1, "ko": "네", "en": ""},
    {"start": 5, "end": 6, "ko": "아니", "en": ""},
    {"start": 9, "end": 10, "ko": "네", "en": ""},
])
assert len(sep) == 3, "non-consecutive repeats are real speech, keep them"
print("ok  hallucination loops collapse, real repeats survive")

# ---- multi-key failover --------------------------------------------------

import io
import urllib.error


def http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, io.BytesIO(b'{"error":"x"}'))


app.GROQ_KEYS = ["KEY_A", "KEY_B"]
app._key_idx = 0
seen = []


def send_a_exhausted(url, data, headers, timeout):
    key = headers["Authorization"].split()[-1]
    seen.append(key)
    if key == "KEY_A":
        raise http_error(429)
    return {"ok": True}


assert app.groq_request("u", b"", "application/json", send=send_a_exhausted,
                        log=lambda *a: None) == {"ok": True}
assert seen == ["KEY_A", "KEY_B"], seen
# it must now stick to B rather than burning a request on the exhausted A
seen.clear()
app.groq_request("u", b"", "application/json", send=send_a_exhausted, log=lambda *a: None)
assert seen == ["KEY_B"], f"should stick to the working key, tried {seen}"
print("ok  exhausted key fails over and the working key sticks")

# a 400 is our own bad request - do not burn the other key on it
app._key_idx = 0
tried = []


def always_400(url, data, headers, timeout):
    tried.append(headers["Authorization"])
    raise http_error(400)


try:
    app.groq_request("u", b"", "application/json", send=always_400, log=lambda *a: None)
    raise AssertionError("should have raised")
except RuntimeError as e:
    assert "400" in str(e), e
assert len(tried) == 1, f"a 400 must not rotate keys, tried {len(tried)}"
print("ok  non-quota errors do not waste the backup key")

# ---- friendly_error rewrites known YouTube failures -----------------------

# exact strings pulled from real Render logs, not paraphrased
bot_check = Exception(
    "ERROR: [youtube] xj1NBNnlk4o: Sign in to confirm you’re not a bot. "
    "Use --cookies-from-browser or --cookies for the authentication."
)
msg = app.friendly_error(bot_check)
assert "YTDLP_COOKIES" in msg and "dashboard.render.com" in msg, msg
assert "Sign in to confirm" not in msg, "raw yt-dlp text should be replaced, not appended"

reload_err = Exception("ERROR: [youtube] xj1NBNnlk4o: The page needs to be reloaded.")
msg = app.friendly_error(reload_err)
assert "YTDLP_COOKIES" in msg, msg

# an unrelated error must pass through untouched - this isn't a catch-all
other = Exception("ERROR: [youtube] xj1NBNnlk4o: Video unavailable")
assert app.friendly_error(other) == str(other)
print("ok  friendly_error rewrites known YouTube failures, leaves others alone")

print("\nall passed")
