# ============================
# PART 1/5 — Core Setup, Sidebar API Key, Clients, FFmpeg, Text Utilities
# ============================
# app.py — UAPpress Documentary TTS Studio (MVP)
# INVESTIGATIVE MODE (Watergate-style)
# ------------------------------------------------------------
# REQUIREMENTS (requirements.txt):
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9
#
# DESIGN GOALS:
# - Audio-first investigative documentaries
# - No book-style narration
# - No chapter roadmaps
# - Continuity-safe, repetition-safe
# - Streamlit Cloud compatible
# - ✅ API key entered manually in sidebar per run (public GitHub repo safe)

from __future__ import annotations

import io
import os
import re
import json
import time
import zipfile
import tempfile
import subprocess
from typing import List, Dict, Tuple, Optional

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio")
st.caption("Investigative, audio-first documentaries. No hype. No roadmap narration.")


# ----------------------------
# Sidebar — OpenAI API Key (manual per run)
# ----------------------------
with st.sidebar:
    st.header("🔐 Keys (not saved)")
    st.caption("Enter your OpenAI API key each run. It is kept only in session memory.")

    # Keep only in session_state (never in repo / secrets)
    st.session_state.setdefault("OPENAI_API_KEY_INPUT", "")

    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        key="OPENAI_API_KEY_INPUT",
        placeholder="sk-...",
        help="Stored only in this session (st.session_state). Not written to disk.",
    )

    st.divider()
    st.header("⚙️ Models")
    SCRIPT_MODEL = st.text_input("Script model", value="gpt-4o-mini")
    TTS_MODEL = st.text_input("TTS model", value="gpt-4o-mini-tts")
    TTS_VOICE = st.text_input("TTS voice", value="onyx")

    st.divider()
    st.header("🎧 Default assets")
    DEFAULT_MUSIC_PATH = st.text_input(
        "Default music path",
        value="/mnt/data/dark-ambient-soundscape-music-409350.mp3",
        help="Used if you don't upload music.",
    )


# ----------------------------
# OpenAI client (requires sidebar key)
# ----------------------------
api_key = (api_key_input or "").strip()
if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

client = OpenAI(api_key=api_key)


# ----------------------------
# FFmpeg
# ----------------------------
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# ----------------------------
# Investigative system role
# ----------------------------
INVESTIGATIVE_SYSTEM = """
You are a serious investigative documentary writer in the tradition of Watergate-era reporting,
the Pentagon Papers, and Frontline.

This is NOT a book, lecture, or explainer.
This is NOT a roadmap-style narration.

Write as if the listener is already inside the story.
Never announce structure. Never preview chapters.
Every line should feel like evidence or consequence.
""".strip()


# ----------------------------
# Shell helpers
# ----------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def run_ffmpeg(cmd: List[str]) -> None:
    code, _, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err[-4000:]}")


# ----------------------------
# Filename + text utilities
# ----------------------------
def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", (text or "")).strip()
    text = re.sub(r"\s+", "_", text)
    return (text.lower()[:80] or "episode")


def chunk_text(text: str, max_chars: int = 3200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    parts = text.split("\n\n")
    chunks: List[str] = []
    buf: List[str] = []
    length = 0

    for p in parts:
        p = p.strip()
        if not p:
            continue
        add = len(p) + (2 if buf else 0)
        if buf and (length + add) > max_chars:
            chunks.append("\n\n".join(buf))
            buf, length = [p], len(p)
        else:
            buf.append(p)
            length += add

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def sanitize_for_tts(text: str) -> str:
    """Remove invisible Unicode control chars that can break TTS."""
    if not text:
        return ""
    t = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_tts_directives(text: str) -> str:
    """Remove accidental style/voice directives from model output."""
    if not text:
        return ""
    t = text
    t = re.sub(r"(?im)^\s*VOICE\s*DIRECTION.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*PACE\s*:.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*STYLE\s*:.*$\n?", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_meta_narration(text: str) -> str:
    """
    🚫 Anti-book / Anti-roadmap cleaner (CRITICAL PATCH)
    Removes roadmap narration: 'In the next chapter...', 'We will explore...', etc.
    """
    if not text:
        return ""

    t = text.strip()
    patterns = [
        r"(?im)^\s*in the next (chapter|section)[^.\n]*[.\n]\s*",
        r"(?im)^\s*in chapter\s*\d+[^.\n]*[.\n]\s*",
        r"(?im)^\s*this chapter[^.\n]*[.\n]\s*",
        r"(?im)^\s*as we'?ll see[^.\n]*[.\n]\s*",
        r"(?im)^\s*as discussed earlier[^.\n]*[.\n]\s*",
        r"(?im)^\s*we will[^.\n]*[.\n]\s*",
        r"(?im)^\s*we('?re| are) going to[^.\n]*[.\n]\s*",
    ]
    for p in patterns:
        t = re.sub(p, "", t)

    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

# ============================
# PART 2/5 — OpenAI Helpers + JSON Extraction + Audio Engine (FFmpeg + TTS)
# ============================

# ----------------------------
# JSON + OpenAI helpers
# ----------------------------
def extract_json_object(text: str) -> dict:
    """
    Extracts a JSON object from a model response.
    Accepts either:
      - a fenced ```json { ... } ``` block, or
      - a raw { ... } object somewhere in the text
    """
    text = (text or "").strip()

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m2 = re.search(r"(\{.*\})", text, re.S)
    if not m2:
        raise ValueError("Could not find JSON object in response.")
    return json.loads(m2.group(1))


def safe_chat(prompt: str, temperature: float = 0.4, tries: int = 2) -> str:
    """
    Small retry wrapper around chat.completions.
    Keeps INVESTIGATIVE_SYSTEM from Part 1 as the system message.
    """
    last_err: Optional[Exception] = None
    for _ in range(max(1, tries)):
        try:
            r = client.chat.completions.create(
                model=SCRIPT_MODEL,
                messages=[
                    {"role": "system", "content": INVESTIGATIVE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"Chat request failed: {last_err}")


def json_outline_from_model(prompt: str) -> dict:
    """
    Gets an outline JSON response reliably by retrying with stricter instruction if needed.
    """
    content = safe_chat(prompt, temperature=0.25, tries=2)
    try:
        return extract_json_object(content)
    except Exception:
        stricter = prompt + "\n\nIMPORTANT: Return ONLY valid JSON."
        content2 = safe_chat(stricter, temperature=0.15, tries=2)
        return extract_json_object(content2)


# ----------------------------
# Audio helpers (FFmpeg)
# ----------------------------
def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    """
    Concatenate WAV files with ffmpeg concat demuxer (no re-encode).
    """
    if not wav_paths:
        raise ValueError("concat_wavs: wav_paths is empty")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            # Escape single quotes for ffmpeg concat list format
            escaped = wp.replace("'", "'\\''")
            f.write("file '{}'\n".format(escaped))
        list_path = f.name

    try:
        run_ffmpeg([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_wav])
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    run_ffmpeg([FFMPEG, "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", bitrate, mp3_path])


def get_audio_duration_seconds(path: str) -> float:
    """
    Reads duration from ffmpeg stderr (fast + no extra deps).
    """
    _, _, err = run_cmd([FFMPEG, "-i", path])
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        return 0.0
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def mix_music_under_voice(
    voice_wav: str,
    music_path: str,
    out_wav: str,
    music_db: int = -24,
    fade_s: int = 6,
) -> None:
    """
    Mix a looping music bed under the voice, with fades + loudnorm.
    Output: PCM WAV (safe for later mp3 encoding).
    """
    dur = max(0.0, get_audio_duration_seconds(voice_wav))
    fade_out_start = max(0.0, dur - fade_s)

    filter_complex = (
        f"[1:a]volume={music_db}dB,"
        f"afade=t=in:st=0:d={fade_s},"
        f"afade=t=out:st={fade_out_start}:d={fade_s}[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
    )

    run_ffmpeg([
        FFMPEG, "-y",
        "-i", voice_wav,
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        out_wav,
    ])


# ----------------------------
# TTS (OpenAI) -> WAV
# ----------------------------
def tts_to_wav(
    text: str,
    delivery_instructions: str,
    speed: float,
    out_wav: str,
) -> None:
    """
    Generate WAV narration via OpenAI TTS and concatenate chunks.
    NOTE:
      - OpenAI TTS supports chunking by input length; we do that here.
      - delivery_instructions + speed are reserved for future wiring; kept
        in signature so Part 4/5 can use them without refactors.
    """
    chunks = chunk_text(text)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, 1):
            payload = strip_tts_directives(sanitize_for_tts(ch))

            # Keeping it simple + stable: no hidden params that may vary by SDK/version
            r = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                response_format="wav",
                input=payload,
            )

            part_path = os.path.join(td, f"part_{i:02d}.wav")
            with open(part_path, "wb") as f:
                f.write(r.read())

            wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)

# ============================
# PART 3/5 — Continuity Engine + Streamlit State (Single Source of Truth)
# ============================

# ----------------------------
# Generic continuity + anti-repetition (episode-agnostic)
# ----------------------------
STOPWORDS = set("""
a an the and or but if then else when while of in on at to for from by with as is are was were be been being
this that these those it its it's he she they them his her their we our you your i me my
not no do does did done can could should would may might will just than into over under about
""".split())


def last_paragraph(text: str, max_chars: int = 900) -> str:
    """Return last paragraph for continuity handoff."""
    if not text:
        return ""
    t = re.sub(r"\n{3,}", "\n\n", text).strip()
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]
    if not parts:
        return ""
    lp = parts[-1]
    return lp[-max_chars:] if len(lp) > max_chars else lp


def normalize_token(t: str) -> str:
    return re.sub(r"[^a-z0-9\-']", "", (t or "").lower()).strip("-'")


def extract_keyphrases(text: str, max_phrases: int = 24) -> List[str]:
    """
    Lightweight 'already introduced' extractor:
    - Capitalized multi-word phrases (names/places/orgs)
    - Repeated distinctive tokens (frequency-based)

    No external libs. Works across topics.
    """
    if not text:
        return []

    # Proper-ish noun phrases: "Walter Haut", "Roswell Army Air Field"
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text)
    caps = [c.strip() for c in caps if len(c.strip()) >= 4]

    # Distinctive repeated tokens
    tokens = [normalize_token(x) for x in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text)]
    tokens = [t for t in tokens if t and t not in STOPWORDS and not t.isdigit()]

    freq: Dict[str, int] = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1

    top_tokens = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:18]
    top_tokens = [t for t, n in top_tokens if n >= 2]

    seen = set()
    out: List[str] = []
    for p in caps + top_tokens:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
        if len(out) >= max_phrases:
            break
    return out


def build_continuity_rules(introduced: List[str], central_question: str) -> str:
    """
    Continuity rules that stop repetition WITHOUT banning names.
    Key rule: introduce once (full name + role), then later: last name only, no redefinition.
    """
    introduced_block = ", ".join(introduced) if introduced else "(none detected yet)"
    return f"""
CONTINUITY RULE (NON-NEGOTIABLE):
This episode is linear. The listener remembers prior segments.

DO NOT:
- restate the premise, background, or the initial incident
- reintroduce people/places/organizations already mentioned
- repeat previously established dates/locations
- summarize earlier chapters
- reset the timeline

NAMING RULE (CRITICAL):
- First appearance of a person in the entire episode: FULL NAME + ROLE + why they matter (one tight sentence).
- After first appearance: use LAST NAME ONLY.
- Do NOT repeat roles, titles, or background later.

Already introduced (auto-extracted from prior text):
{introduced_block}

INVESTIGATIVE RULES:
- Every chapter must add NEW evidence/angle/consequence.
- Include at least one REVERSAL (what the listener must reconsider).
- Keep the Central Question alive; do NOT answer it early.

Central Question:
{central_question}
""".strip()


# ----------------------------
# Streamlit state (single source of truth)
# ----------------------------
def ensure_state() -> None:
    """
    Initializes required session_state keys exactly once.
    This prevents Streamlit widget/state sync errors.
    """
    st.session_state.setdefault("outline", [])
    st.session_state.setdefault("chapter_count", 0)
    st.session_state.setdefault("built", None)

    # Episode defaults
    st.session_state.setdefault("topic", "Roswell UFO Incident")
    st.session_state.setdefault(
        "central_question",
        "Why did the Army Air Forces announce a 'flying disc'—and then reverse itself within hours? What changed, and who needed the story changed?"
    )

    # Used to detect whether outline params changed
    st.session_state.setdefault("last_outline_params", None)


ensure_state()


def text_key(kind: str, idx: int = 0) -> str:
    """Uniform naming for session_state text fields."""
    return f"text::{kind}::{idx}"


def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> str:
    """
    Ensures the key exists BEFORE any widget uses it.
    Prevents StreamlitAPIException from assigning widget-owned keys after creation.
    """
    k = text_key(kind, idx)
    st.session_state.setdefault(k, default or "")
    return k


def get_text(kind: str, idx: int = 0) -> str:
    return st.session_state.get(text_key(kind, idx), "")


def reset_script_text_fields(chapter_count: int) -> None:
    """
    Create/refresh text keys for intro/chapters/outro in a predictable way.
    """
    ensure_text_key("intro", 0, "")
    ensure_text_key("outro", 0, "")
    for i in range(1, chapter_count + 1):
        ensure_text_key("chapter", i, "")

# ============================
# PART 4/5 — Prompt Builders + Episode Setup UI (Outline + Intro/Outro/Chapter Prompts)
# ============================

# ----------------------------
# Prompt builders (Investigative Mode Only)
# ----------------------------
def prompt_outline(
    topic: str,
    total_minutes: int,
    chapter_minutes: int,
    global_style: str,
    episode_notes: str,
    central_question: str,
) -> str:
    chapters = max(6, total_minutes // chapter_minutes)
    return f"""
Create an outline for an audio-only investigative documentary (Watergate-style).

TOPIC:
{topic}

CENTRAL QUESTION (NON-NEGOTIABLE):
{central_question}

TOTAL LENGTH:
{total_minutes} minutes

CHAPTER TARGET:
~{chapter_minutes} minutes each (so about {chapters} chapters)

GLOBAL STYLE (must follow):
{global_style}

EPISODE-SPECIFIC NOTES:
{episode_notes}

REQUIREMENTS:
- Create exactly {chapters} chapters
- Each chapter must include:
  - title (short, cinematic but not hype)
  - target_minutes (integer)
  - beats (6–10 bullet beats; each beat is a single sentence)
- Each chapter must force a reversal:
  - what was believed
  - what disrupted it
  - why it mattered (institutional/human stakes)
  - what new uncertainty it created
- Beats should be chronological when possible
- Separate: contemporaneous facts vs official statements vs later testimony vs disputed claims
- Avoid recap beats and structural narration
- Ensure key participants appear early in the outline

Return ONLY valid JSON with this exact shape:
{{
  "chapters": [
    {{
      "title": "string",
      "target_minutes": 10,
      "beats": ["string", "string"]
    }}
  ]
}}
""".strip()


def prompt_intro(
    topic: str,
    global_style: str,
    episode_notes: str,
    central_question: str,
) -> str:
    return f"""
Write an INTRO for an audio-only investigative documentary (Watergate-style).

TOPIC:
{topic}

CENTRAL QUESTION (state explicitly near the end):
{central_question}

STYLE (must follow):
{global_style}

NOTES:
{episode_notes}

INTRO REQUIREMENTS:
- 60–90 seconds spoken
- Start inside a real moment of doubt (statement, reversal, decision, phone call, briefing, memo, headline)
- Establish stakes: credibility, secrecy, institutional self-protection
- Do NOT announce structure or chapters

MANDATORY: KEY PLAYERS (ONE-TIME WHO’S WHO)
- Include a tight key-players rundown in 2–4 sentences.
- Each player gets: NAME + ROLE + why they matter (7–12 words).
- Keep it brisk and intriguing; do NOT do biography.
- After this, assume the listener remembers these individuals.
- Later references should use LAST NAMES only, with no redefinition.

Then:
- State the Central Question explicitly near the end.

Sponsor mention (premium, compliant):
- This episode is sponsored by OPA Nutrition
- Mention they make premium wellness supplements (focus, clarity, energy, resilience, long-term health)
- Mention the website: opanutrition.com
- No disease claims

Engagement CTA:
- Ask listeners to subscribe
- Ask them to comment where they’re listening from
- Ask them to share what you believe happened (open-ended)

Return ONLY the intro narration text. No headings.
""".strip()


def prompt_outro(
    topic: str,
    global_style: str,
    episode_notes: str,
    central_question: str,
) -> str:
    return f"""
Write an OUTRO for an audio-only investigative documentary (Watergate-style).

TOPIC:
{topic}

CENTRAL QUESTION (revisit, do not resolve cleanly):
{central_question}

STYLE (must follow):
{global_style}

NOTES:
{episode_notes}

OUTRO REQUIREMENTS:
- 60–90 seconds spoken
- State what can be known vs what cannot be verified
- Emphasize institutional consequence (trust, credibility, precedent), not sensational mystery
- Do NOT announce structure or chapters
- End with a clean, documentary final line (not cheesy)

Sponsor mention again:
- Sponsored by OPA Nutrition
- Mention premium wellness supplements that support focus, clarity, and daily performance
- Mention: opanutrition.com

Engagement CTA:
- Ask where they’re listening from
- Ask what case/topic to cover next
- Ask to like + subscribe

Return ONLY the outro narration text. No headings.
""".strip()


def prompt_chapter(
    topic: str,
    chapter_title: str,
    target_minutes: int,
    beats: List[str],
    global_style: str,
    episode_notes: str,
    central_question: str,
    prev_chapter_tail: str,
    introduced_phrases: List[str],
    chapter_index: int,
) -> str:
    words = int(target_minutes * 145)
    beats_block = "\n".join([f"- {b}" for b in beats]) if beats else "- (No beats provided)"
    continuity = build_continuity_rules(introduced_phrases, central_question)

    cast_requirement = ""
    if chapter_index == 1:
        cast_requirement = """
MANDATORY (CHAPTER 1 ONLY): KEY PLAYERS LOCK-IN
- In the first 20–30 seconds, identify the key players by FULL NAME + ROLE once (tight).
- After this chapter, use LAST NAMES only with no role repetition.
- No biographies. No recap. Just roles and why they matter.
""".strip()

    forbidden = """
FORBIDDEN META-NARRATION (NON-NEGOTIABLE):
Do NOT say or imply:
- "In the next chapter" / "In the next section"
- "In Chapter X"
- "This chapter"
- "We will" / "We’re going to"
- "As we’ll see"
- "As discussed earlier"

Instead, end with a DOCUMENTARY HOOK:
- 1–2 sentences that raise a question, reveal a new tension, or introduce a new document/witness
- It must feel like a natural cut, not a roadmap
""".strip()

    return f"""
Write CHAPTER {chapter_index} of an audio-only investigative documentary (Watergate-style).

TOPIC:
{topic}

CHAPTER TITLE:
{chapter_title}

TARGET LENGTH:
~{words} words (aim for {target_minutes} minutes at ~145 wpm)

STYLE (must follow):
{global_style}

EPISODE NOTES:
{episode_notes}

{continuity}

{cast_requirement}

HANDOFF (begin immediately after this; do NOT restate earlier context):
{prev_chapter_tail if prev_chapter_tail else "[No previous segment text available — start in medias res without repeating the premise]"}

MUST COVER THESE BEATS (new material only):
{beats_block}

MANDATORY CHAPTER SHAPE:
1) 1–2 sentences continuing from the handoff (no reset)
2) New evidence/statement/action
3) Why it mattered (institutional/human stakes)
4) Reversal (what changes in understanding)
5) Documentary hook ending (no roadmap)

{forbidden}

OUTPUT:
Return ONLY narration text. No headings, no bullet points.
""".strip()


# ----------------------------
# UI — Episode setup (Outline builder)
# ----------------------------
st.header("1️⃣ Episode Setup")

colA, colB = st.columns([2, 1])

with colA:
    topic = st.text_input("Episode topic", value=st.session_state.topic)
    st.session_state.topic = topic

with colB:
    total_minutes = st.slider("Total length (minutes)", 30, 180, 90, 5)
    chapter_minutes = st.slider("Minutes per chapter", 5, 20, 10, 1)

central_question = st.text_area(
    "Central Question (required)",
    value=st.session_state.central_question,
    height=90,
    help="Make it specific and high-stakes. Example: Why did the official story reverse within hours—and who needed it to?",
)
st.session_state.central_question = central_question

if not central_question.strip():
    st.warning("Add a Central Question before generating the outline.")

global_style = st.text_area(
    "Global style instructions",
    value=(
        "Calm, authoritative male narrator.\n"
        "Serious investigative documentary tone.\n"
        "Measured pacing with subtle pauses.\n"
        "Clarity first. No hype, no jokes, no sensationalism.\n"
        "Label uncertainty clearly: verified facts vs disputed claims vs theories.\n"
        "Audio-only narration; do not reference visuals."
    ),
    height=150,
)

episode_notes = st.text_area(
    "Episode-specific notes",
    value=(
        "Focus on the timeline, key witnesses, institutional response, and how the official narrative evolved. "
        "Separate contemporaneous facts from later testimony and disputed claims. Avoid invented details.\n\n"
        "If known, list key players to identify once early (Name + Role):"
    ),
    height=140,
)

outline_btn = st.button("Generate Chapter Outline", type="primary")

if outline_btn:
    if not central_question.strip():
        st.error("Central Question is required.")
        st.stop()

    with st.spinner("Creating outline…"):
        data = json_outline_from_model(
            prompt_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes, central_question)
        )
        chapters_list = data.get("chapters", [])

        normalized = []
        for ch in chapters_list:
            title = str(ch.get("title", "")).strip() or "Untitled Chapter"
            try:
                tmin = int(ch.get("target_minutes", chapter_minutes))
            except Exception:
                tmin = int(chapter_minutes)

            beats = ch.get("beats", [])
            if not isinstance(beats, list):
                beats = []
            beats = [str(b).strip() for b in beats if str(b).strip()]
            normalized.append({"title": title, "target_minutes": tmin, "beats": beats[:12]})

        st.session_state.outline = normalized
        st.session_state.chapter_count = len(normalized)
        st.session_state.built = None

        # IMPORTANT: create text keys BEFORE widgets exist in Part 5
        reset_script_text_fields(st.session_state.chapter_count)

        st.success("Outline generated.")

if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")

# ============================
# PART 5/5 — Scripts UI + Audio Build + ZIP Packaging + Downloads
# ============================

# ----------------------------
# Scripts (Intro + Chapters + Outro)
# ----------------------------
st.header("2️⃣ Scripts")

# Ensure keys exist BEFORE widgets are created (safe even if chapter_count is 0)
reset_script_text_fields(st.session_state.get("chapter_count", 0))


# ----------------------------
# Master Script Paste (Strict Template → Fill Boxes)
# ----------------------------
def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


_CHAPTER_RE = re.compile(r"(?i)^\s*chapter\s+(\d+)\s*(?:[:\-–—]\s*(.*))?\s*$")


def detect_chapters_and_titles(master_text: str) -> Dict[int, str]:
    """
    Detect CHAPTER N headings and optional titles from the heading line.

    Examples supported:
      CHAPTER 1
      CHAPTER 1: The Calm Before
      CHAPTER 1 - The Calm Before
      Chapter 1 — The Calm Before

    Returns: {1: "The Calm Before", 2: "", ...}
    """
    txt = (master_text or "").replace("\r\n", "\n").replace("\r", "\n")
    titles: Dict[int, str] = {}
    for line in txt.split("\n"):
        m = _CHAPTER_RE.match(line or "")
        if not m:
            continue
        try:
            n = int(m.group(1))
        except Exception:
            continue
        title = (m.group(2) or "").strip()
        titles[n] = title
    return titles


def parse_master_script(master_text: str, expected_chapters: int) -> Dict[str, object]:
    """
    Parse a single master script into intro / chapters / outro using strict headings.

    Headings (case-insensitive):
      INTRO
      CHAPTER 1[: Title]
      ...
      OUTRO

    CRITICAL:
      - Heading lines are NOT included in returned narration text.
      - Only the body under each heading is returned.
    """
    txt = (master_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")

    intro_idx = None
    outro_idx = None
    chapter_idx: Dict[int, int] = {}

    for i, raw in enumerate(lines):
        h = _normalize_line(raw).upper()
        if h == "INTRO":
            intro_idx = i
            continue
        if h == "OUTRO":
            outro_idx = i
            continue
        m = _CHAPTER_RE.match(raw or "")
        if m:
            try:
                n = int(m.group(1))
                chapter_idx[n] = i
            except Exception:
                pass

    # Collect all heading positions, sorted
    positions: List[int] = []
    if intro_idx is not None:
        positions.append(intro_idx)
    if outro_idx is not None:
        positions.append(outro_idx)
    for _, idx in chapter_idx.items():
        positions.append(idx)
    positions = sorted(set(positions))

    # Map each heading line index -> next heading line index (or end)
    next_pos: Dict[int, int] = {}
    for k in range(len(positions)):
        cur = positions[k]
        nxt = positions[k + 1] if (k + 1) < len(positions) else len(lines)
        next_pos[cur] = nxt

    def slice_block(start_i: Optional[int]) -> str:
        if start_i is None:
            return ""
        end_i = next_pos.get(start_i, len(lines))
        body = "\n".join(lines[start_i + 1 : end_i]).strip()  # EXCLUDES heading line
        body = re.sub(r"\n{3,}", "\n\n", body).strip()
        return body

    parsed: Dict[str, object] = {"intro": "", "outro": "", "chapters": {}}
    parsed["intro"] = slice_block(intro_idx)
    parsed["outro"] = slice_block(outro_idx)

    chapters_out: Dict[int, str] = {}
    for n in range(1, expected_chapters + 1):
        chapters_out[n] = slice_block(chapter_idx.get(n))
    parsed["chapters"] = chapters_out
    return parsed


st.subheader("📌 Paste Full Script (INTRO + CHAPTER 1–N + OUTRO)")

st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")

master_cols = st.columns([3, 1])
with master_cols[0]:
    st.text_area(
        "Master script",
        key="MASTER_SCRIPT_TEXT",
        height=220,
        placeholder=(
            "Use this strict template:\n\n"
            "INTRO\n"
            "<intro narration...>\n\n"
            "CHAPTER 1: Optional Title\n"
            "<chapter 1 narration...>\n\n"
            "CHAPTER 2\n"
            "<chapter 2 narration...>\n\n"
            "...\n\n"
            "OUTRO\n"
            "<outro narration...>\n"
        ),
        help=(
            "Headings (INTRO / CHAPTER N / OUTRO) are STRUCTURE ONLY.\n"
            "They will NOT be inserted into narration boxes and will NOT be spoken in MP3."
        ),
    )

with master_cols[1]:
    fill_btn = st.button(
        "Fill Boxes from Paste",
        type="primary",
        disabled=not st.session_state.get("MASTER_SCRIPT_TEXT", "").strip(),
    )
    clear_paste_btn = st.button("Clear Paste", type="secondary")

if clear_paste_btn:
    st.session_state["MASTER_SCRIPT_TEXT"] = ""
    st.session_state.built = None
    st.rerun()


if fill_btn:
    raw_master = st.session_state.get("MASTER_SCRIPT_TEXT", "")
    if not raw_master.strip():
        st.error("Paste a master script first.")
        st.stop()

    # 1) Detect how many chapters exist from headings
    chapter_titles_map = detect_chapters_and_titles(raw_master)
    detected_n = max(chapter_titles_map.keys()) if chapter_titles_map else 0

    if detected_n <= 0:
        st.error("No CHAPTER headings found. Use 'CHAPTER 1', 'CHAPTER 2', etc.")
        st.stop()

    # 2) Set chapter_count and create the session keys for boxes
    st.session_state.chapter_count = detected_n
    reset_script_text_fields(detected_n)

    # 3) Build a lightweight, derived outline for existing UI + audio filename logic
    #    (This is NOT a generated outline; it's extracted from your headings.)
    derived_outline = []
    for i in range(1, detected_n + 1):
        title = (chapter_titles_map.get(i) or "").strip()
        derived_outline.append(
            {
                "title": title if title else f"Chapter {i}",
                "target_minutes": 10,   # harmless default; only shown as UI hint
                "beats": [],            # empty because we are not generating an outline
            }
        )
    st.session_state.outline = derived_outline

    # 4) Parse sections and fill boxes (HEADINGS STRIPPED)
    parsed = parse_master_script(raw_master, detected_n)

    st.session_state[text_key("intro", 0)] = (parsed.get("intro") or "").strip()
    chapters_map: Dict[int, str] = parsed.get("chapters", {}) or {}
    for i in range(1, detected_n + 1):
        st.session_state[text_key("chapter", i)] = (chapters_map.get(i) or "").strip()
    st.session_state[text_key("outro", 0)] = (parsed.get("outro") or "").strip()

    st.session_state.built = None

    # 5) Helpful warnings (you said you’ll keep a strict template; this just flags blanks)
    missing = []
    if not st.session_state[text_key("intro", 0)].strip():
        missing.append("INTRO")
    for i in range(1, detected_n + 1):
        if not st.session_state[text_key("chapter", i)].strip():
            missing.append(f"CHAPTER {i}")
    if not st.session_state[text_key("outro", 0)].strip():
        missing.append("OUTRO")

    if missing:
        st.warning("Filled what I could, but these sections were empty/missing: " + ", ".join(missing))
    else:
        st.success(f"Boxes filled: Intro + {detected_n} chapters + Outro (headings stripped).")

st.divider()

# ----------------------------
# Clear all script boxes (manual reset)
# ----------------------------
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.caption("Paste → Fill Boxes above.")
with c2:
    clear_all = st.button("Clear Chapters")
with c3:
    st.caption("Headings are structure-only and will NOT be spoken in MP3.")

if clear_all:
    st.session_state[text_key("intro", 0)] = ""
    for i in range(1, st.session_state.get("chapter_count", 0) + 1):
        st.session_state[text_key("chapter", i)] = ""
    st.session_state[text_key("outro", 0)] = ""
    st.session_state.built = None
    st.success("Intro, chapters, and outro cleared.")


def has_any_script() -> bool:
    if get_text("intro", 0).strip():
        return True
    for i in range(1, st.session_state.get("chapter_count", 0) + 1):
        if get_text("chapter", i).strip():
            return True
    if get_text("outro", 0).strip():
        return True
    return False


# ----------------------------
# Intro box
# ----------------------------
st.subheader("Intro")
st.text_area(
    "Intro text",
    key=text_key("intro", 0),
    height=220,
    placeholder="Paste via Master Script above, then click “Fill Boxes from Paste”…",
)

st.divider()

# ----------------------------
# Chapter boxes (no generation; paste fills; manual edits allowed)
# ----------------------------
chapter_count = st.session_state.get("chapter_count", 0)
outline = st.session_state.get("outline", [])

if chapter_count > 0:
    for i in range(1, chapter_count + 1):
        # Title from derived outline if present
        title = ""
        try:
            title = str(outline[i - 1].get("title", "")).strip()
        except Exception:
            title = ""
        label = f"Chapter {i}: {title}" if title else f"Chapter {i}"

        with st.expander(label, expanded=(i == 1)):
            st.text_area(
                f"Chapter {i} text",
                key=text_key("chapter", i),
                height=320,
                placeholder="Filled from Master Script paste above (headings stripped). You can edit here.",
            )
else:
    st.caption("Paste your master script above and click “Fill Boxes from Paste” to generate the boxes.")

st.divider()

# ----------------------------
# Outro box
# ----------------------------
st.subheader("Outro")
st.text_area(
    "Outro text",
    key=text_key("outro", 0),
    height=220,
    placeholder="Paste via Master Script above, then click “Fill Boxes from Paste”…",
)

# ----------------------------
# Audio build
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

