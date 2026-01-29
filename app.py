# ============================
# PART 1/4 — Core Setup, Sidebar, Clients, FFmpeg, Text + Script Parsing Utilities
# ============================
# app.py — UAPpress Documentary TTS Studio
# MODE: Script → Audio
# OPTIONAL: Master Prompt → Full Script → Auto-fill → Audio
#
# REQUIREMENTS:
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9

from __future__ import annotations

import io
import os
import re
import json
import time
import zipfile
import hashlib
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary TTS Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio")
st.caption("Credibility-first investigative documentaries. Script → Audio.")


# ----------------------------
# FFmpeg
# ----------------------------
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr


def run_ffmpeg(cmd: List[str]) -> None:
    code, _, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err[-4000:]}")


# ----------------------------
# 🔒 MASTER PROMPT — SYSTEM ROLE (STYLE ONLY)
# ----------------------------
MASTER_ROLE_PROMPT = """
You are the most successful long-form documentary creator in history.

You have 7+ billion subscribers across platforms.

Your documentaries are trusted by scientists, skeptics, journalists,
military professionals, and serious long-form audiences.

Your work is cited, debated, and archived.

You are not a hype creator.
You are a precision investigator, narrative architect, and editorial surgeon.

Your reputation is built on credibility, restraint,
intellectual honesty, and depth over speed.

You do not optimize for short content.
You optimize for retention, trust, and long-form authority.

IMPORTANT:
This role defines investigative standards and narrative discipline only.
Do NOT reference or acknowledge this role inside the script itself.
""".strip()


# ----------------------------
# 🔒 MASTER PROMPT — RULES + OUTPUT CONTRACT
# ----------------------------
MASTER_SCRIPT_CONTRACT = """
OBJECTIVE:
Generate a complete long-form, audio-only investigative documentary script
in one response.

NON-NEGOTIABLE RULES:
- ZERO repetition across chapters
- STRICT continuity (timeline only moves forward)
- Key players introduced ONCE (full name + role), then last name only
- No recap language
- No roadmap narration
- Clear evidentiary discipline (facts vs testimony vs official vs later claims)

MANDATORY STRUCTURE (EXACT):
INTRO
<text>

CHAPTER 1: <Title>
<text>

CHAPTER 2: <Title>
<text>

...

OUTRO
<text>

ABSOLUTELY FORBIDDEN:
- "In the next chapter"
- "This chapter"
- "As mentioned earlier"
- "We will / We're going to"

INTRO REQUIREMENT:
- Include a tight Key Players section (2–4 sentences)
- Each player: FULL NAME + ROLE + why they matter (7–12 words)
- Never reintroduce them later

OPA NUTRITION:
- Mention in INTRO and OUTRO
- "OPA Nutrition"
- "premium wellness supplements"
- "opanutrition.com"
- No disease claims

AUDIENCE CTA (INTRO + OUTRO):
- Ask to subscribe
- Ask where they’re listening from
- Ask what topic to cover next

LENGTH TARGETS:
- Total: ~7,000–13,000 words
- Chapters: no chapter under ~900 words
""".strip()


# ----------------------------
# Sidebar — API Key + High-Level Settings
# ----------------------------
with st.sidebar:
    st.header("🔐 API Key (not saved)")
    st.session_state.setdefault("OPENAI_API_KEY_INPUT", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        key="OPENAI_API_KEY_INPUT",
        placeholder="sk-...",
        help="Stored only in session memory.",
    )

    st.divider()
    st.header("⚙️ Models")
    st.session_state.setdefault("SCRIPT_MODEL", "gpt-4o-mini")
    st.session_state.setdefault("TTS_MODEL", "gpt-4o-mini-tts")
    st.session_state.setdefault("TTS_VOICE", "onyx")

    st.text_input("Script model", key="SCRIPT_MODEL")
    st.text_input("TTS model", key="TTS_MODEL")
    st.text_input("TTS voice", key="TTS_VOICE")

    st.divider()
    st.header("🧠 Script Generation")
    st.session_state.setdefault("ENABLE_SCRIPT_GEN", True)
    st.checkbox("Enable Master Prompt → Script", key="ENABLE_SCRIPT_GEN")

    st.caption("Uses the locked investigative Master Prompt + strict template.")


# ----------------------------
# OpenAI client
# ----------------------------
api_key = (api_key_input or "").strip()
if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

client = OpenAI(api_key=api_key)


# ----------------------------
# Text utilities
# ----------------------------
def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", (text or "")).strip()
    text = re.sub(r"\s+", "_", text)
    return (text.lower()[:80] or "episode")


def sanitize_for_tts(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ----------------------------
# Streamlit state (single source of truth)
# ----------------------------
def text_key(kind: str, idx: int = 0) -> str:
    return f"text::{kind}::{idx}"


def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> None:
    st.session_state.setdefault(text_key(kind, idx), default or "")


def reset_script_text_fields(chapter_count: int) -> None:
    ensure_text_key("intro", 0, "")
    ensure_text_key("outro", 0, "")
    for i in range(1, chapter_count + 1):
        ensure_text_key("chapter", i, "")


def ensure_state() -> None:
    st.session_state.setdefault("episode_title", "Untitled Episode")
    st.session_state.setdefault("MASTER_PROMPT_INPUT", "")
    st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")
    st.session_state.setdefault("chapter_count", 0)
    reset_script_text_fields(st.session_state["chapter_count"])


ensure_state()


# ----------------------------
# Master Script parsing (STRICT headings)
# ----------------------------
_CHAPTER_RE = re.compile(r"(?i)^\s*chapter\s+(\d+)\s*(?:[:\-–—]\s*(.*))?\s*$")


def parse_master_script(master_text: str, expected_chapters: int) -> Dict[str, object]:
    """
    Parses INTRO / CHAPTER N / OUTRO.
    Heading lines are NOT included in narration output.
    """
    txt = (master_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")

    intro_i, outro_i = None, None
    chapter_i: Dict[int, int] = {}

    for idx, line in enumerate(lines):
        u = line.strip().upper()
        if u == "INTRO":
            intro_i = idx
        elif u == "OUTRO":
            outro_i = idx
        else:
            m = _CHAPTER_RE.match(line)
            if m:
                chapter_i[int(m.group(1))] = idx

    markers = sorted([i for i in [intro_i, outro_i, *chapter_i.values()] if i is not None])
    bounds = {markers[i]: markers[i + 1] if i + 1 < len(markers) else len(lines) for i in range(len(markers))}

    def block(start: Optional[int]) -> str:
        if start is None:
            return ""
        return "\n".join(lines[start + 1 : bounds[start]]).strip()

    return {
        "intro": block(intro_i),
        "outro": block(outro_i),
        "chapters": {n: block(i) for n, i in chapter_i.items()},
    }

# ============================
# PART 2/4 — OpenAI Script Generator + TTS Engine + FFmpeg Audio Ops + Cache + ZIP Helpers
# ============================

# ----------------------------
# Script generation (Master Prompt → STRICT template master script)
# ----------------------------
def _build_generation_prompt(*, topic: str, master_prompt: str, chapters: int) -> str:
    """
    Builds the user prompt that pairs with:
      - MASTER_ROLE_PROMPT (system)
      - MASTER_SCRIPT_CONTRACT (system)
    """
    topic = (topic or "").strip()
    master_prompt = (master_prompt or "").strip()

    # IMPORTANT: keep this as plain text. No markdown.
    return f"""
TOPIC:
{topic}

CHAPTER COUNT:
{chapters}

MASTER PROMPT (follow closely; do NOT narrate this block as meta):
{master_prompt}

FINAL INSTRUCTION:
Generate the entire documentary script now.
Return ONLY the script text in the strict template required by the contract.
""".strip()


def generate_master_script_one_shot(
    *,
    topic: str,
    master_prompt: str,
    chapters: int,
    model: str,
    temperature: float = 0.55,
    tries: int = 2,
) -> str:
    """
    One-shot generation:
      - Uses MASTER_ROLE_PROMPT + MASTER_SCRIPT_CONTRACT as system messages
      - Forces strict template (INTRO / CHAPTER N / OUTRO)
      - Returns raw text (to be parsed by parse_master_script in Part 1)

    Retries once with stricter "ONLY TEMPLATE" instruction if needed.
    """
    chapters = max(3, int(chapters or 0))
    user_prompt = _build_generation_prompt(topic=topic, master_prompt=master_prompt, chapters=chapters)

    last_err: Optional[Exception] = None
    for attempt in range(max(1, tries)):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": MASTER_ROLE_PROMPT},
                    {"role": "system", "content": MASTER_SCRIPT_CONTRACT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if attempt == 0 else max(0.2, temperature - 0.25),
            )
            txt = (r.choices[0].message.content or "").strip()

            # Quick sanity checks (parser-safe)
            # Must contain INTRO and OUTRO headings
            if "INTRO" not in txt.upper() or "OUTRO" not in txt.upper():
                raise ValueError("Model output missing INTRO/OUTRO headings.")

            # Encourage the strict template if it drifted (retry)
            if attempt == 0:
                # If it contains obvious analysis headers, force retry
                bad_markers = ["HIGH-LEVEL ASSESSMENT", "FINAL VERDICT", "WHAT WORKS", "PROPOSED IMPROVEMENTS"]
                if any(b in txt.upper() for b in bad_markers):
                    raise ValueError("Model output included editorial sections (not allowed).")

            return txt
        except Exception as e:
            last_err = e
            time.sleep(0.6)

            # Tighten on retry
            user_prompt = user_prompt + "\n\nSTRICT REMINDER: Output ONLY the template headings + narration text. No commentary."
            continue

    raise RuntimeError(f"Script generation failed: {last_err}")


# ----------------------------
# TTS config + caching
# ----------------------------
@dataclass
class TTSConfig:
    model: str
    voice: str
    speed: float = 1.0  # reserved
    enable_cache: bool = True
    cache_dir: str = ".uappress_tts_cache"


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _tts_cache_path(cfg: TTSConfig, text: str) -> str:
    # Cache key includes model + voice + text. (Speed reserved but not applied yet.)
    key = _sha1(f"model={cfg.model}|voice={cfg.voice}|text={text}")
    _ensure_dir(cfg.cache_dir)
    return os.path.join(cfg.cache_dir, f"tts_{key}.wav")


# ----------------------------
# Chunking + cleanup (TTS-stable)
# ----------------------------
def chunk_text(text: str, max_chars: int = 2800) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", (text or "")).strip()
    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    length = 0

    for p in parts:
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


def strip_tts_directives(text: str) -> str:
    """
    Remove accidental style/voice directives that might slip into pasted/generated scripts.
    Keeps narration clean for TTS.
    """
    if not text:
        return ""
    t = text
    t = re.sub(r"(?im)^\s*VOICE\s*DIRECTION.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*PACE\s*:.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*STYLE\s*:.*$\n?", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ----------------------------
# FFmpeg audio ops
# ----------------------------
def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    """Concatenate WAVs via concat demuxer (stream copy)."""
    if not wav_paths:
        raise ValueError("concat_wavs: wav_paths is empty")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            escaped = wp.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")
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
    Loop music bed under voice, fade in/out, then loudnorm.
    Output WAV for later MP3 encoding.
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
# OpenAI TTS → WAV (chunked, cached)
# ----------------------------
def tts_to_wav(*, text: str, out_wav: str, cfg: TTSConfig) -> None:
    """
    Generates narration WAV by chunking and concatenating.
    Uses per-chunk caching to speed up reruns.
    """
    cleaned = strip_tts_directives(sanitize_for_tts(text or ""))
    if not cleaned.strip():
        # Tiny silence to avoid downstream failures
        run_ffmpeg([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.1", out_wav])
        return

    chunks = chunk_text(cleaned, max_chars=2800)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, 1):
            payload = strip_tts_directives(sanitize_for_tts(ch))

            cache_path = _tts_cache_path(cfg, payload) if cfg.enable_cache else ""
            if cfg.enable_cache and cache_path and os.path.exists(cache_path) and os.path.getsize(cache_path) > 2000:
                wav_parts.append(cache_path)
                continue

            r = client.audio.speech.create(
                model=cfg.model,
                voice=cfg.voice,
                response_format="wav",
                input=payload,
            )

            part_path = os.path.join(td, f"part_{i:02d}.wav")
            with open(part_path, "wb") as f:
                f.write(r.read())

            if cfg.enable_cache and cache_path:
                try:
                    _ensure_dir(cfg.cache_dir)
                    with open(cache_path, "wb") as f2, open(part_path, "rb") as f1:
                        f2.write(f1.read())
                    wav_parts.append(cache_path)
                except Exception:
                    wav_parts.append(part_path)
            else:
                wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)


# ----------------------------
# Packaging helpers
# ----------------------------
def make_zip_bytes(files: List[Tuple[str, str]]) -> bytes:
    """
    files: [(abs_path, arcname), ...]
    Returns zip as bytes.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for abs_path, arcname in files:
            if not abs_path or not os.path.exists(abs_path):
                continue
            z.write(abs_path, arcname=arcname)
    buf.seek(0)
    return buf.read()


def section_texts() -> List[Tuple[str, str]]:
    """
    Playback order: intro -> chapters -> outro
    Returns list of (section_id, text).
    """
    out: List[Tuple[str, str]] = []
    out.append(("00_intro", st.session_state.get(text_key("intro", 0), "")))

    n = int(st.session_state.get("chapter_count", 0))
    for i in range(1, n + 1):
        out.append((f"{i:02d}_chapter_{i}", st.session_state.get(text_key("chapter", i), "")))

    out.append(("99_outro", st.session_state.get(text_key("outro", 0), "")))
    return out


def has_any_script() -> bool:
    for _, t in section_texts():
        if (t or "").strip():
            return True
    return False

# ============================
# PART 3/4 — Script Generation UI + Parsing + Box Population
# ============================

st.header("1️⃣ Script Creation")

# ----------------------------
# Episode metadata
# ----------------------------
st.session_state.setdefault("episode_title", "Untitled Episode")
st.session_state.setdefault("DEFAULT_CHAPTERS", 8)

colA, colB = st.columns([3, 1])
with colA:
    st.text_input("Episode title", key="episode_title")
with colB:
    st.number_input(
        "Chapter count",
        min_value=3,
        max_value=20,
        step=1,
        key="DEFAULT_CHAPTERS",
    )

st.divider()

# ----------------------------
# Master Prompt input
# ----------------------------
st.subheader("🧠 Master Prompt (Investigative Brief)")

st.text_area(
    "Paste your investigative brief / case focus here",
    key="MASTER_PROMPT_INPUT",
    height=220,
    placeholder=(
        "Example:\n"
        "Focus on the Roswell Army Air Field press release, the rapid retraction, "
        "the role of General Ramey, and how the official explanation evolved.\n\n"
        "Emphasize institutional pressure, information control, and credibility risk."
    ),
    help="This is NOT narrated. It instructs the investigative direction only.",
)

# ----------------------------
# Generate Script Button
# ----------------------------
gen_col1, gen_col2 = st.columns([1, 3])

with gen_col1:
    generate_script_btn = st.button(
        "Generate Full Script",
        type="primary",
        disabled=not st.session_state.get("ENABLE_SCRIPT_GEN", True),
    )

with gen_col2:
    st.caption(
        "One-shot generation using the locked Master Prompt.\n"
        "Produces INTRO / CHAPTERS / OUTRO in a strict template."
    )

# ----------------------------
# Script generation execution
# ----------------------------
if generate_script_btn:
    if not st.session_state.get("MASTER_PROMPT_INPUT", "").strip():
        st.error("Master Prompt is required.")
        st.stop()

    with st.spinner("Generating long-form investigative script…"):
        try:
            raw_script = generate_master_script_one_shot(
                topic=st.session_state.get("episode_title", ""),
                master_prompt=st.session_state.get("MASTER_PROMPT_INPUT", ""),
                chapters=int(st.session_state.get("DEFAULT_CHAPTERS", 8)),
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
            )

            # Store raw master script
            st.session_state["MASTER_SCRIPT_TEXT"] = raw_script

            # Detect chapter count from headings
            chapter_map = detect_chapters_and_titles(raw_script)
            detected_n = max(chapter_map.keys()) if chapter_map else 0

            if detected_n <= 0:
                raise ValueError("No CHAPTER headings detected in generated script.")

            # Reset script boxes safely
            st.session_state["chapter_count"] = detected_n
            reset_script_text_fields(detected_n)

            # Parse script and populate boxes
            parsed = parse_master_script(raw_script, detected_n)

            st.session_state[text_key("intro", 0)] = parsed.get("intro", "").strip()
            for i in range(1, detected_n + 1):
                st.session_state[text_key("chapter", i)] = (
                    parsed.get("chapters", {}).get(i, "").strip()
                )
            st.session_state[text_key("outro", 0)] = parsed.get("outro", "").strip()

            st.success(f"Script generated and loaded ({detected_n} chapters).")

        except Exception as e:
            st.error(str(e))

st.divider()

# ----------------------------
# Script review & edit UI
# ----------------------------
st.header("2️⃣ Script Review (Editable)")

# Intro
st.subheader("INTRO")
st.text_area(
    "Intro narration",
    key=text_key("intro", 0),
    height=220,
)

st.divider()

# Chapters
chapter_count = int(st.session_state.get("chapter_count", 0))
if chapter_count > 0:
    for i in range(1, chapter_count + 1):
        with st.expander(f"CHAPTER {i}", expanded=(i == 1)):
            st.text_area(
                f"Chapter {i} narration",
                key=text_key("chapter", i),
                height=320,
            )
else:
    st.info("No chapters loaded yet.")

st.divider()

# Outro
st.subheader("OUTRO")
st.text_area(
    "Outro narration",
    key=text_key("outro", 0),
    height=220,
)

# ----------------------------
# Safety / reset controls
# ----------------------------
st.divider()
reset_col1, reset_col2 = st.columns([1, 3])

with reset_col1:
    clear_script_btn = st.button("Clear Script")

with reset_col2:
    st.caption("Clears Intro, Chapters, and Outro (does not affect audio builds).")

if clear_script_btn:
    st.session_state[text_key("intro", 0)] = ""
    for i in range(1, chapter_count + 1):
        st.session_state[text_key("chapter", i)] = ""
    st.session_state[text_key("outro", 0)] = ""
    st.session_state["chapter_count"] = 0
    st.success("Script cleared.")

# ============================
# PART 4/4 — Audio Build (Per-Section + Full Episode) + Downloads + Final ZIP
# ============================

st.header("3️⃣ Create Audio (Onyx + Music)")

# ----------------------------
# Workdir (Streamlit-safe)
# ----------------------------
def _workdir() -> str:
    st.session_state.setdefault("WORKDIR", tempfile.mkdtemp(prefix="uappress_tts_"))
    return st.session_state["WORKDIR"]


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")


def episode_slug() -> str:
    title = st.session_state.get("episode_title", "Untitled Episode")
    return clean_filename(title)


def section_output_paths(section_id: str, episode_slug_: str) -> Dict[str, str]:
    wd = _workdir()
    base = f"{episode_slug_}__{section_id}"
    return {
        "voice_wav": os.path.join(wd, f"{base}__voice.wav"),
        "mixed_wav": os.path.join(wd, f"{base}__mix.wav"),
        "mp3": os.path.join(wd, f"{base}.mp3"),
        "txt": os.path.join(wd, f"{base}.txt"),
    }


# ----------------------------
# Music upload / selection
# ----------------------------
music_file = st.file_uploader(
    "Optional music bed (mp3/wav/m4a). If omitted, Default music path is used.",
    type=["mp3", "wav", "m4a"],
)

music_path = ""
if music_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{music_file.name}") as tf:
        tf.write(music_file.read())
        music_path = tf.name
else:
    music_path = st.session_state.get("DEFAULT_MUSIC_PATH", "") or ""

use_music = st.checkbox(
    "Mix music under voice",
    value=True,
    help="If enabled, a music bed is mixed under narration with fades + loudnorm.",
)

music_db_i = int(st.session_state.get("MUSIC_DB", -24))
fade_s_i = int(st.session_state.get("MUSIC_FADE_S", 6))


# ----------------------------
# TTS config
# ----------------------------
cfg = TTSConfig(
    model=st.session_state.get("TTS_MODEL", "gpt-4o-mini-tts"),
    voice=st.session_state.get("TTS_VOICE", "onyx"),
    speed=float(st.session_state.get("TTS_SPEED", 1.0)),
    enable_cache=bool(st.session_state.get("ENABLE_TTS_CACHE", True)),
    cache_dir=str(st.session_state.get("CACHE_DIR", ".uappress_tts_cache")),
)

st.caption(f"TTS: model={cfg.model} | voice={cfg.voice} | cache={'on' if cfg.enable_cache else 'off'}")


# ----------------------------
# Build one section
# ----------------------------
def build_one_section(
    *,
    section_id: str,
    text: str,
    cfg: TTSConfig,
    mix_music: bool,
    music_path: str,
    music_db: int,
    fade_s: int,
) -> Dict[str, str]:
    """
    Produces:
      - voice_wav
      - (optional) mixed_wav
      - mp3 (from whichever wav is final)
      - txt (section text)
    Returns dict of output paths.
    """
    slug = episode_slug()
    paths = section_output_paths(section_id, slug)

    cleaned = strip_tts_directives(sanitize_for_tts(text or ""))
    write_text_file(paths["txt"], cleaned)

    # 1) TTS → voice WAV
    tts_to_wav(text=cleaned, out_wav=paths["voice_wav"], cfg=cfg)

    # 2) Optional music mix → WAV
    final_wav = paths["voice_wav"]
    if mix_music:
        if not music_path or not os.path.exists(music_path):
            raise FileNotFoundError(
                "Music bed path not found. Upload music or set a valid Default music path in the sidebar."
            )
        mix_music_under_voice(
            voice_wav=paths["voice_wav"],
            music_path=music_path,
            out_wav=paths["mixed_wav"],
            music_db=music_db,
            fade_s=fade_s,
        )
        final_wav = paths["mixed_wav"]

    # 3) WAV → MP3
    wav_to_mp3(final_wav, paths["mp3"], bitrate="192k")

    return paths


# ----------------------------
# Build controls
# ----------------------------
st.subheader("🎙️ Build audio for each section")

if not has_any_script():
    st.info("Generate or paste a script first (Intro/Chapters/Outro) before building audio.")
else:
    st.session_state.setdefault("built_sections", {})  # {section_id: {paths...}}
    built_sections: Dict[str, Dict[str, str]] = st.session_state.get("built_sections", {}) or {}

    sec_list = section_texts()
    ordered_ids = [sid for sid, _ in sec_list]

    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        build_all_btn = st.button("Build ALL Sections", type="primary")
    with b2:
        clear_built_btn = st.button("Clear Built Audio")
    with b3:
        st.caption("Tip: Build one section first to QC voice + tone before building all.")

    if clear_built_btn:
        wd = _workdir()
        try:
            for name in os.listdir(wd):
                _safe_remove(os.path.join(wd, name))
        except Exception:
            pass
        st.session_state["built_sections"] = {}
        st.session_state["last_build"] = None
        st.success("Cleared built audio from workdir.")

    # Build selected section
    st.caption("Optional: build one section at a time (fast QC)")
    pick = st.selectbox(
        "Choose a section to build",
        options=[sid for sid, _ in sec_list],
        index=0,
    )
    build_one_btn = st.button("Build Selected Section")

    if build_one_btn:
        sec_text_map = {sid: txt for sid, txt in sec_list}
        sec_text = sec_text_map.get(pick, "")
        if not (sec_text or "").strip():
            st.error(f"Selected section '{pick}' is empty.")
        else:
            with st.spinner(f"Building {pick}…"):
                try:
                    out_paths = build_one_section(
                        section_id=pick,
                        text=sec_text,
                        cfg=cfg,
                        mix_music=bool(use_music),
                        music_path=music_path,
                        music_db=music_db_i,
                        fade_s=fade_s_i,
                    )
                    st.session_state["built_sections"][pick] = out_paths
                    st.success(f"Built: {pick}")
                except Exception as e:
                    st.error(str(e))

    # Build all sections
    if build_all_btn:
        with st.spinner("Building all sections…"):
            built = {}
            errors = []
            for sid, txt in sec_list:
                if not (txt or "").strip():
                    continue
                try:
                    out_paths = build_one_section(
                        section_id=sid,
                        text=txt,
                        cfg=cfg,
                        mix_music=bool(use_music),
                        music_path=music_path,
                        music_db=music_db_i,
                        fade_s=fade_s_i,
                    )
                    built[sid] = out_paths
                except Exception as e:
                    errors.append((sid, str(e)))

            st.session_state["built_sections"].update(built)
            st.session_state["last_build"] = {
                "ts": time.time(),
                "episode": st.session_state.get("episode_title", "Untitled Episode"),
                "sections_built": sorted(list(built.keys())),
                "errors": errors,
                "mix_music": bool(use_music),
                "music_path": music_path,
                "tts_model": cfg.model,
                "tts_voice": cfg.voice,
                "cache": cfg.enable_cache,
            }

            if errors:
                st.warning("Some sections failed:\n" + "\n".join([f"- {sid}: {msg}" for sid, msg in errors]))
            else:
                st.success("All available sections built.")


    # ----------------------------
    # Display built sections + downloads
    # ----------------------------
    built_sections = st.session_state.get("built_sections", {}) or {}
    if built_sections:
        st.subheader("✅ Built Sections")
        for sid in ordered_ids:
            if sid not in built_sections:
                continue

            paths = built_sections[sid]
            mp3_path = paths.get("mp3", "")
            txt_path = paths.get("txt", "")

            with st.expander(f"{sid} — downloads & preview", expanded=(sid == "00_intro")):
                colx, coly, colz = st.columns([1, 1, 2])

                with colx:
                    if mp3_path and os.path.exists(mp3_path):
                        with open(mp3_path, "rb") as f:
                            st.download_button(
                                label=f"Download {sid}.mp3",
                                data=f.read(),
                                file_name=os.path.basename(mp3_path),
                                mime="audio/mpeg",
                                key=f"dl_mp3_{sid}",
                            )

                with coly:
                    if txt_path and os.path.exists(txt_path):
                        with open(txt_path, "rb") as f:
                            st.download_button(
                                label=f"Download {sid}.txt",
                                data=f.read(),
                                file_name=os.path.basename(txt_path),
                                mime="text/plain",
                                key=f"dl_txt_{sid}",
                            )

                with colz:
                    if mp3_path and os.path.exists(mp3_path):
                        st.audio(mp3_path)

        # ZIP download of all section MP3s (+ txt)
        st.subheader("📦 Download Sections ZIP")
        zip_sections_btn = st.button("Build ZIP of section MP3s + TXT")
        if zip_sections_btn:
            slug = episode_slug()
            files: List[Tuple[str, str]] = []

            for sid, paths in built_sections.items():
                mp3_path = paths.get("mp3", "")
                txt_path = paths.get("txt", "")
                if mp3_path and os.path.exists(mp3_path):
                    files.append((mp3_path, os.path.basename(mp3_path)))
                if txt_path and os.path.exists(txt_path):
                    files.append((txt_path, os.path.basename(txt_path)))

            meta_path = os.path.join(_workdir(), f"{slug}__build_meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(st.session_state.get("last_build", {}), f, indent=2)
            files.append((meta_path, os.path.basename(meta_path)))

            zip_bytes = make_zip_bytes(files)
            st.download_button(
                "Download sections ZIP",
                data=zip_bytes,
                file_name=f"{slug}__sections.zip",
                mime="application/zip",
            )

    # ----------------------------
    # Build FULL episode MP3 by concatenating section MP3s
    # ----------------------------
    st.divider()
    st.subheader("🎬 Build FULL Episode MP3 (from built sections)")

    ep_slug = episode_slug()
    full_mp3_path = os.path.join(_workdir(), f"{ep_slug}__FULL.mp3")

    def concat_mp3s(mp3_paths: List[str], out_mp3: str) -> None:
        if not mp3_paths:
            raise ValueError("concat_mp3s: mp3_paths is empty")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            for p in mp3_paths:
                escaped = p.replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")
            list_path = f.name

        try:
            run_ffmpeg([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_mp3])
        finally:
            try:
                os.remove(list_path)
            except Exception:
                pass

    def build_full_episode_from_built_sections(
        *,
        built_sections: Dict[str, Dict[str, str]],
        ordered_section_ids: List[str],
        out_mp3: str,
    ) -> None:
        mp3s = []
        for sid in ordered_section_ids:
            paths = built_sections.get(sid) or {}
            p = paths.get("mp3", "")
            if p and os.path.exists(p):
                mp3s.append(p)

        if not mp3s:
            raise ValueError("No built section MP3s found to concatenate.")

        concat_mp3s(mp3s, out_mp3)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        build_full_btn = st.button("Build FULL Episode MP3", type="primary", disabled=not bool(built_sections))
    with col2:
        dl_full_btn = st.button("Prepare FULL Episode Download", disabled=not os.path.exists(full_mp3_path))
    with col3:
        st.caption("Full episode is built by concatenating your already-built section MP3s in order.")

    if build_full_btn:
        with st.spinner("Building full episode MP3…"):
            try:
                build_full_episode_from_built_sections(
                    built_sections=built_sections,
                    ordered_section_ids=ordered_ids,
                    out_mp3=full_mp3_path,
                )
                st.success("Built FULL episode MP3.")
                st.session_state.setdefault("last_build", {})
                st.session_state["last_build"]["full_episode_mp3"] = os.path.basename(full_mp3_path)
            except Exception as e:
                st.error(str(e))

    if os.path.exists(full_mp3_path):
        st.audio(full_mp3_path)

    if dl_full_btn and os.path.exists(full_mp3_path):
        with open(full_mp3_path, "rb") as f:
            st.download_button(
                "Download FULL Episode MP3",
                data=f.read(),
                file_name=os.path.basename(full_mp3_path),
                mime="audio/mpeg",
            )

    # ----------------------------
    # Final delivery ZIP (full + sections + txt + meta + master script)
    # ----------------------------
    st.divider()
    st.subheader("📦 Final Delivery ZIP (FULL MP3 + Sections + Text)")

    final_zip_btn = st.button("Build FINAL ZIP (everything)")
    if final_zip_btn:
        slug = episode_slug()
        files: List[Tuple[str, str]] = []

        # Full MP3
        if os.path.exists(full_mp3_path):
            files.append((full_mp3_path, os.path.basename(full_mp3_path)))

        # Section MP3/TXT
        for sid, paths in (built_sections or {}).items():
            mp3_path = paths.get("mp3", "")
            txt_path = paths.get("txt", "")
            if mp3_path and os.path.exists(mp3_path):
                files.append((mp3_path, os.path.basename(mp3_path)))
            if txt_path and os.path.exists(txt_path):
                files.append((txt_path, os.path.basename(txt_path)))

        # Build meta
        meta_path = os.path.join(_workdir(), f"{slug}__build_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.get("last_build", {}), f, indent=2)
        files.append((meta_path, os.path.basename(meta_path)))

        # Raw master script (for archive)
        raw_path = os.path.join(_workdir(), f"{slug}__MASTER_SCRIPT.txt")
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write((st.session_state.get("MASTER_SCRIPT_TEXT", "") or "").strip() + "\n")
        files.append((raw_path, os.path.basename(raw_path)))

        zip_bytes = make_zip_bytes(files)
        st.download_button(
            "Download FINAL ZIP",
            data=zip_bytes,
            file_name=f"{slug}__FINAL.zip",
            mime="application/zip",
        )

# Debug panel
with st.expander("🔎 Build Metadata (debug)", expanded=False):
    st.json(st.session_state.get("last_build", {}))
    st.caption(f"Workdir: {_workdir()}")
