# ============================
# PART 1/4 — Core Setup, Sidebar, Clients, FFmpeg, Text + Script Parsing Utilities
# ============================
# app.py — UAPpress Documentary TTS Studio
# MODE: Script → Audio
# OPTIONAL: Super Master Prompt → Full Script → Auto-fill → Audio
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


# =============================================================================
# 🔒 SUPER MASTER PROMPT v3.1 — PRESTIGE UFO / UAP INVESTIGATIVE DOCUMENTARY SYSTEM
# (Universal | Fact-Anchored | Character-Driven | Witness-Centered | Audio-First | High-Retention)
#
# NOTE:
# - Keep ROLE separate from CONTRACT so you can pass ROLE as system message and
#   CONTRACT as user message (or concatenate for a single user message).
# - This prompt is optimized for OpenAI TTS output: cadence, paragraph rhythm, and modular chapters.
# =============================================================================

# ----------------------------
# 🔒 SUPER MASTER PROMPT — ROLE (STYLE ONLY, NEVER ACKNOWLEDGED IN SCRIPT)
# ----------------------------
SUPER_ROLE_PROMPT = """
ROLE & AUTHORITY
(STYLE-SETTING ONLY — NEVER ACKNOWLEDGED IN SCRIPT)

You are the most trusted long-form investigative documentary creator in the field of unidentified aerial phenomena.

Your work is consumed by:
• Scientists
• Military professionals
• Intelligence analysts
• Journalists
• Skeptical long-form audiences

Your documentaries are cited, archived, debated, replayed, and scrutinized.

You are not:
• A believer
• A debunker
• A theorist
• A mystery narrator

You are a forensic storyteller reconstructing human decisions under uncertainty.

IMPORTANT:
This role defines investigative standards and narrative discipline only.
Do NOT reference or acknowledge this role inside the script itself.
""".strip()


# ----------------------------
# 🔒 SUPER MASTER PROMPT — RULES + OUTPUT CONTRACT (v3.1)
# ----------------------------
SUPER_SCRIPT_CONTRACT = r"""
🎯 OBJECTIVE (NON-NEGOTIABLE)

Generate a complete, audio-only, long-form investigative documentary script.

This script will be:
• Narrated entirely using OpenAI Text-to-Speech
• Exported as MP3 chapters
• Converted into MP4 documentary video

Write for listening endurance, not silent reading.
Target runtime: 45–120 minutes narrated.

🧠 CORE STORY LAW (ABSOLUTE)

This documentary is not about UFOs.

It is about:
• People trained to follow procedure
• Moments when procedure proves insufficient
• Decisions made with careers, credibility, and responsibility at stake
• How individuals and institutions respond differently to uncertainty

The phenomenon matters only because it forces people to choose, act, and live with consequences.

🧭 FACTUAL & TEMPORAL ANCHORING (HARD REQUIREMENT)

At the earliest appropriate moment, the script MUST:
• State the specific date or date range
• Clarify time-of-day ambiguity, especially when events cross midnight
• Distinguish between:
  – Calendar date
  – Operational shift timing
  – Later recollection phrasing

If events span multiple nights:
• Treat each night as a separate operational window
• Identify who was present each time

Forbidden vagueness:
• “Around Christmas”
• “During the holidays”
• “Late December” without dates

Precision is mandatory, even when uncertainty exists.

🧍 CHARACTER CENTRALITY LAW (NON-NEGOTIABLE)

Named individuals are not background texture.

They are:
• Decision-makers
• Witnesses
• Record-keepers
• Consequence-bearers

Rules:
• Characters must be present in scenes
• Their choices, hesitations, and constraints must be explicit
• Their professional or social risk must be clear
• Their later consequences must be tracked over time

Forbidden:
• Mythologizing
• Hero or villain framing
• Psychological speculation not supported by record

Characters are defined by what they did, documented, or refused to do.

🔥 NARRATIVE PRESSURE LAW (MANDATORY)

Every paragraph must contain at least one human pressure point:
• A decision made without full information
• A reporting threshold crossed
• A procedural rule strained or bypassed
• A professional or social risk introduced
• A consequence that cannot be undone

If a paragraph contains no pressure: Cut it.

🎙️ WITNESS TESTIMONY & HUMAN REACTION ENGINE (MANDATORY)

Witness testimony is evidence of perception, reaction, and decision-making, not proof of explanation.

Each witness scene must include at least three:
• Initial perception (before interpretation)
• Immediate reaction (shown through action or hesitation)
• Contextual constraint (rank, crowd, duty, fear, credibility risk)
• Decision point (what they did instead of speculating)
• Aftereffect (doubt, silence, consequence, persistence)

Temporal discipline:
• During the event
• Immediately after
• Years later
Later recollections require greater restraint.

Quote discipline:
• Use short verbatim fragments only
• Never stack quotes
• Every quote must trigger a reaction, decision, or consequence

🎧 AUDIO-FIRST / OPENAI TTS ENGINE (MANDATORY)

Assume the script will be narrated entirely by AI TTS.
Write for spoken clarity, not visual formatting.

Breath & cadence:
• Average sentence length: 12–22 words
• Long sentences must include natural pause points
• Avoid stacked subordinate clauses
If it cannot be spoken comfortably in one breath, split it.

Paragraph rhythm:
• Paragraphs: 3–5 sentences max
• One audible beat or action per paragraph
• No paragraph may contain more than one decision or revelation
Paragraph breaks must sound intentional.

Transitions:
• Scene changes must be embedded in language and audible without visuals
Forbidden: “Cut to”, “Fade”, visual-only cues

No stage directions:
Never write: “Pause”, “Emphasis”, “Dramatic”
Use rhythm and sentence length for weight.

Chapter modularity:
Each chapter must function as a standalone MP3.
Begin with immediate orientation. End on a clean narrative stop.
Avoid mid-thought endings.

📢 MANDATORY PRESERVATIONS

OPA NUTRITION (HARD LOCK)
INTRO — verbatim:
“This episode is sponsored by OPA Nutrition, makers of premium wellness supplements designed to support focus, clarity, energy, resilience, and long-term health. Explore their offerings at opanutrition.com.”

OUTRO — verbatim:
“This episode was sponsored by OPA Nutrition. For premium wellness supplements designed to support focus, clarity, energy, resilience, and long-term health, visit opanutrition.com.”

Do not paraphrase. Do not shorten. No disease claims anywhere.

AUDIENCE ENGAGEMENT (INTRO + OUTRO)
Must explicitly ask listeners to:
• Subscribe
• Comment where they’re listening from
• Suggest future cases/topics/documents to investigate
Tone: measured, human, unforced.

🚨 STRUCTURAL LAWS (ABSOLUTE)

ZERO REPETITION
No recaps. No reintroductions. No recycled phrasing.
No “as mentioned earlier”, “once again”, or timeline resets.

STRICT CONTINUITY
Time moves forward only.
Each chapter begins exactly where the previous chapter ends.
Assume perfect listener memory.

KEY PLAYERS LOCK-IN (INTRO ONLY)
Early in INTRO, include a tight Key Players section:
• 2–4 sentences total
• Each: FULL NAME + ROLE + why they matter (7–12 words)
Afterward: LAST NAMES ONLY. No reintroductions.

🧾 EVIDENTIARY DISCIPLINE (CRITICAL)

Clearly distinguish between:
• Documented fact
• Firsthand testimony
• Official explanation
• Later reinterpretation
• Speculation

Never blur categories. Never imply certainty.

🎭 TONE & DELIVERY

Calm. Controlled. Human. Precise.
Authority comes from clarity about people under pressure, not detachment.

⏱️ LENGTH REQUIREMENTS

Total script: 7,000–13,000 words
Chapters: 900–1,300 words each
No filler or transition-only chapters.

🚫 FORBIDDEN META LANGUAGE

Never write:
• “In this chapter…”
• “In the next chapter…”
• “We will… / We’re going to…”
• “As discussed earlier…”
• “You won’t believe…”

📄 MODEL OUTPUT FORMAT (MANDATORY)

Return ONLY the script text in plain text.
No commentary. No analysis. No bullet points. No markdown.

Use exactly this structure and headings:

INTRO
<Intro narration>

CHAPTER 1: <Title>
<Chapter narration>

CHAPTER 2: <Title>
<Chapter narration>

CHAPTER 3: <Title>
<Chapter narration>

(continue sequentially as needed…)

OUTRO
<Outro narration>

✅ FINAL COMMAND

Generate the complete long-form investigative documentary script in one response.
""".strip()

# ----------------------------
# Convenience: one combined prompt (if you prefer single user message)
# ----------------------------
SUPER_PROMPT_FULL = f"{SUPER_ROLE_PROMPT}\n\n{SUPER_SCRIPT_CONTRACT}".strip()

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

def detect_chapters_and_titles(master_text: str) -> Dict[int, str]:
    """
    Returns {chapter_number: chapter_title_or_empty}
    Used ONLY to detect how many chapters exist.
    """
    chapters: Dict[int, str] = {}
    if not master_text:
        return chapters

    for line in master_text.splitlines():
        m = _CHAPTER_RE.match(line)
        if m:
            n = int(m.group(1))
            title = (m.group(2) or "").strip()
            chapters[n] = title

    return chapters

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
# PART 2/4 — OpenAI Script Generator + TTS Engine + FFmpeg Audio Ops + Cache + ZIP Helpers (FIXED)
# ============================
# Drop this block BETWEEN Part 1 and Part 3.
#
# Fixes included:
# ✅ Defines generate_master_script_one_shot (your Part 3 calls this)
# ✅ Defines get_media_duration_seconds (your Part 4 expects this)
# ✅ Provides TTSConfig + caching + chunking + tts_to_wav
# ✅ Keeps your SUPER_ROLE_PROMPT + SUPER_SCRIPT_CONTRACT usage
# ✅ Robust ffmpeg concat list escaping
# ✅ make_zip_bytes helper
#
# NOTE: This part assumes Part 1 already defined:
# - client (OpenAI)
# - st, os, re, io, time, zipfile, tempfile, hashlib, dataclass, List, Tuple, Optional
# - FFMPEG, run_ffmpeg, run_cmd
# - sanitize_for_tts, clean_filename, text_key
#
# ============================

# ----------------------------
# Script generation (Outline/Treatment -> Full Script)
# ----------------------------
def _build_generation_prompt(*, topic: str, master_prompt: str, chapters: int) -> str:
    """
    Builds the USER prompt. System behavior is controlled by:
      - SUPER_ROLE_PROMPT
      - SUPER_SCRIPT_CONTRACT

    IMPORTANT:
    - master_prompt can be your outline/treatment text
    - Keep plain text (no markdown)
    """
    topic = (topic or "").strip()
    master_prompt = (master_prompt or "").strip()
    chapters = max(3, int(chapters or 0))

    return f"""
DOCUMENTARY TITLE:
{topic}

CHAPTER COUNT:
{chapters}

OUTLINE / TREATMENT (DO NOT NARRATE THIS AS META):
{master_prompt}

FINAL INSTRUCTION:
Generate the complete long-form investigative documentary script now.
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
      - Uses SUPER_ROLE_PROMPT + SUPER_SCRIPT_CONTRACT as system messages
      - Forces strict template (INTRO / CHAPTER N / OUTRO)
      - Returns raw text (to be parsed by parse_master_script in Part 1)
    """
    chapters = max(3, int(chapters or 0))
    user_prompt = _build_generation_prompt(topic=topic, master_prompt=master_prompt, chapters=chapters)

    last_err: Optional[Exception] = None
    for attempt in range(max(1, int(tries or 1))):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SUPER_ROLE_PROMPT},
                    {"role": "system", "content": SUPER_SCRIPT_CONTRACT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if attempt == 0 else max(0.2, float(temperature) - 0.25),
            )

            txt = (r.choices[0].message.content or "").strip()
            u = txt.upper()

            # Parser-safe sanity checks
            if "INTRO" not in u or "OUTRO" not in u:
                raise ValueError("Model output missing INTRO/OUTRO headings.")
            if "CHAPTER" not in u:
                raise ValueError("Model output missing CHAPTER headings.")

            # Reject common drift modes
            banned = [
                "HIGH-LEVEL ASSESSMENT",
                "FINAL VERDICT",
                "WHAT WORKS",
                "PROPOSED IMPROVEMENTS",
                "SUMMARY:",
                "CONCLUSION:",
            ]
            if attempt == 0 and any(b in u for b in banned):
                raise ValueError("Model output included editorial sections (not allowed).")

            return txt

        except Exception as e:
            last_err = e
            time.sleep(0.6)
            user_prompt += "\n\nSTRICT REMINDER: Output ONLY the required template headings + narration text. No commentary."
            continue

    raise RuntimeError(f"Script generation failed: {last_err}")


# ----------------------------
# TTS config + caching
# ----------------------------
@dataclass
class TTSConfig:
    model: str
    voice: str
    speed: float = 1.0  # reserved (not applied)
    enable_cache: bool = True
    cache_dir: str = ".uappress_tts_cache"


def _ensure_dir2(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _sha1(s: str) -> str:
    return hashlib.sha1((s or "").encode("utf-8")).hexdigest()


def _tts_cache_path(cfg: TTSConfig, text: str) -> str:
    key = _sha1(f"model={cfg.model}|voice={cfg.voice}|text={text}")
    _ensure_dir2(cfg.cache_dir)
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
# Media duration (for Part 4 UI)
# ----------------------------
def get_media_duration_seconds(path: str) -> float:
    """
    Uses ffmpeg stderr Duration line. Works on mp3/wav/m4a/mp4.
    """
    try:
        _, _, err = run_cmd([FFMPEG, "-i", path])
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
        if not m:
            return 0.0
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    except Exception:
        return 0.0


# ----------------------------
# FFmpeg audio ops
# ----------------------------
def _ff_concat_list_escape(p: str) -> str:
    # ffmpeg concat list uses single quotes; escape single quote safely
    return (p or "").replace("'", r"'\''")


def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    """
    Concatenate WAVs via concat demuxer (stream copy).
    REQUIREMENT: WAV formats must match. (Part 4 enforces unified WAV.)
    """
    if not wav_paths:
        raise ValueError("concat_wavs: wav_paths is empty")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for wp in wav_paths:
            f.write(f"file '{_ff_concat_list_escape(wp)}'\n")
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
    dur = max(0.0, float(get_media_duration_seconds(voice_wav)))
    fade_out_start = max(0.0, dur - float(fade_s))

    filter_complex = (
        f"[1:a]volume={int(music_db)}dB,"
        f"afade=t=in:st=0:d={int(fade_s)},"
        f"afade=t=out:st={fade_out_start}:d={int(fade_s)}[m];"
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
                    _ensure_dir2(cfg.cache_dir)
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

# ============================
# PART 3/4 — Outline → Full Script Pipeline (STREAMLIT-SAFE)
# ============================

st.header("1️⃣ Script Creation")
import hashlib

# ----------------------------
# Session defaults
# ----------------------------
st.session_state.setdefault("episode_title", "Untitled Episode")
st.session_state.setdefault("DEFAULT_CHAPTERS", 8)

st.session_state.setdefault("OUTLINE_TEXT_UI", "")
st.session_state.setdefault("OUTLINE_TEXT_SRC", "")
st.session_state.setdefault("_PENDING_OUTLINE_SYNC", False)

st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")
st.session_state.setdefault("chapter_count", 0)

# ----------------------------
# Episode metadata
# ----------------------------
colA, colB = st.columns([3, 1])
with colA:
    st.text_input("Episode Title", key="episode_title")
with colB:
    st.number_input("Chapter Count", min_value=3, max_value=20, step=1, key="DEFAULT_CHAPTERS")

st.divider()
st.subheader("🧠 Outline / Treatment")

# ----------------------------
# Preload outline BEFORE widget
# ----------------------------
if st.session_state.get("_PENDING_OUTLINE_SYNC"):
    st.session_state["_PENDING_OUTLINE_SYNC"] = False
    st.session_state["OUTLINE_TEXT_UI"] = st.session_state.get("OUTLINE_TEXT_SRC", "")

# ----------------------------
# Outline editor
# ----------------------------
st.text_area(
    "Outline (editable)",
    key="OUTLINE_TEXT_UI",
    height=320,
)

# Always mirror UI → SRC (safe)
st.session_state["OUTLINE_TEXT_SRC"] = (st.session_state["OUTLINE_TEXT_UI"] or "").strip()

outline_ok = bool(st.session_state["OUTLINE_TEXT_SRC"])
title_ok = bool(st.session_state["episode_title"].strip())

c1, c2 = st.columns(2)
with c1:
    gen_outline = st.button("Generate Outline", type="primary", disabled=not title_ok)
with c2:
    gen_script = st.button("Generate Full Script", disabled=not outline_ok)

st.divider()

# ----------------------------
# Generate OUTLINE
# ----------------------------
if gen_outline:
    with st.spinner("Generating outline…"):
        try:
            outline_prompt = f"""
DOCUMENTARY TITLE:
{st.session_state['episode_title']}

CHAPTER COUNT:
{int(st.session_state['DEFAULT_CHAPTERS'])}

TASK:
Generate an OUTLINE ONLY.
NO narration prose.
INTRO / CHAPTERS / OUTRO only.
""".strip()

            r = client.chat.completions.create(
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SUPER_ROLE_PROMPT},
                    {"role": "system", "content": SUPER_SCRIPT_CONTRACT},
                    {"role": "user", "content": outline_prompt},
                ],
                temperature=0.35,
            )

            outline = (r.choices[0].message.content or "").strip()
            if "INTRO" not in outline.upper():
                raise ValueError("Invalid outline structure.")

            st.session_state["OUTLINE_TEXT_SRC"] = outline
            st.session_state["_PENDING_OUTLINE_SYNC"] = True
            st.rerun()

        except Exception as e:
            st.error(str(e))

# ----------------------------
# Generate FULL SCRIPT (FIXED CALL)
# ----------------------------
if gen_script:
    with st.spinner("Generating full script…"):
        try:
            raw = generate_master_script_one_shot(
                topic=st.session_state["episode_title"],
                master_prompt=st.session_state["OUTLINE_TEXT_SRC"],  # ✅ FIX
                chapters=int(st.session_state["DEFAULT_CHAPTERS"]),
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
            )

            st.session_state["MASTER_SCRIPT_TEXT"] = raw

            chapter_map = detect_chapters_and_titles(raw)
            n = max(chapter_map.keys()) if chapter_map else 0
            if n <= 0:
                raise ValueError("No chapters detected.")

            st.session_state["chapter_count"] = n
            reset_script_text_fields(n)

            parsed = parse_master_script(raw, n)
            st.session_state[text_key("intro", 0)] = parsed.get("intro", "").strip()
            for i in range(1, n + 1):
                st.session_state[text_key("chapter", i)] = parsed["chapters"].get(i, "").strip()
            st.session_state[text_key("outro", 0)] = parsed.get("outro", "").strip()

            st.success(f"Script generated: Intro + {n} chapters + Outro")

# ----------------------------
# Script review & edit (YOU WERE MISSING THIS)
# ----------------------------
st.header("2️⃣ Script Review (Editable)")

# Intro
st.subheader("INTRO")
st.text_area(
    "Intro narration",
    key=text_key("intro", 0),
    height=240,
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
                height=340,
            )
else:
    st.info("No chapters loaded yet.")

st.divider()

# Outro
st.subheader("OUTRO")
st.text_area(
    "Outro narration",
    key=text_key("outro", 0),
    height=240,
)

# ----------------------------
# Reset controls
# ----------------------------
st.divider()
c1, c2, c3 = st.columns([1, 1, 2])

with c1:
    if st.button("Clear Script"):
        reset_script_text_fields(st.session_state.get("chapter_count", 0))
        st.session_state["chapter_count"] = 0
        st.session_state["MASTER_SCRIPT_TEXT"] = ""
        st.success("Script cleared.")

with c2:
    if st.button("Clear Outline"):
        st.session_state["OUTLINE_TEXT_SRC"] = ""
        st.session_state["_PENDING_OUTLINE_SYNC"] = True
        st.rerun()

with c3:
    st.caption("Clearing text does not affect any already-generated audio.")
        
        except Exception as e:
            st.error(str(e))
            
# ============================
# PART 4/4 — Audio Build (Per-Section + Full Episode) + QC + Downloads + Final ZIP — PATCHED
# ============================
# Goals of this patch:
# ✅ Build ALL or build SELECTED sections (Intro-only QC is fast)
# ✅ Per-section QC: audio player + duration + download
# ✅ Robust FULL episode build (re-encode WAVs → concat WAV → encode MP3) to avoid mp3 concat/copy glitches
# ✅ Optional "Force rebuild" per run (ignores cache + overwrites outputs)
# ✅ Music upload stored into WORKDIR (persists across reruns; avoids temp file disappearing)
# ✅ Final ZIP is clean: section mp3 + section txt + optional FULL mp3 + master script txt

st.header("3️⃣ Create Audio (Onyx + Music)")

import tempfile
import os
from typing import Dict, List, Tuple

# ----------------------------
# Workdir (Streamlit-safe)
# ----------------------------
def _workdir() -> str:
    st.session_state.setdefault("WORKDIR", tempfile.mkdtemp(prefix="uappress_tts_"))
    return st.session_state["WORKDIR"]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def write_text_file(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
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
        "final_wav": os.path.join(wd, f"{base}__final.wav"),  # unified 48k stereo WAV for concat safety
        "mp3": os.path.join(wd, f"{base}.mp3"),
        "txt": os.path.join(wd, f"{base}.txt"),
    }

# ----------------------------
# Duration helper (ffprobe via imageio_ffmpeg bundled ffmpeg)
# ----------------------------
def _duration_seconds(media_path: str) -> float:
    try:
        return float(get_media_duration_seconds(media_path))
    except Exception:
        return 0.0

# ----------------------------
# Music upload / selection (persist inside WORKDIR)
# ----------------------------
st.subheader("🎵 Music Bed (Optional)")

music_file = st.file_uploader(
    "Optional music bed (mp3/wav/m4a)",
    type=["mp3", "wav", "m4a"],
)

# Persist chosen/uploaded music path in session_state
st.session_state.setdefault("MUSIC_PATH", "")
music_path = ""

if music_file is not None:
    # Save upload into WORKDIR to survive reruns
    wd = _workdir()
    _ensure_dir(wd)
    save_name = clean_filename(music_file.name)
    persisted = os.path.join(wd, f"__music__{save_name}")
    with open(persisted, "wb") as f:
        f.write(music_file.read())
    st.session_state["MUSIC_PATH"] = persisted
    music_path = persisted
else:
    # Fall back to last uploaded or a configured default
    music_path = st.session_state.get("MUSIC_PATH", "") or st.session_state.get("DEFAULT_MUSIC_PATH", "") or ""

use_music = st.checkbox(
    "Mix music under voice",
    value=True,
    help="If enabled, a music bed is mixed under narration with fades + loudnorm.",
)

music_db_i = int(st.session_state.get("MUSIC_DB", -24))
fade_s_i = int(st.session_state.get("MUSIC_FADE_S", 6))

if use_music:
    if music_path and os.path.exists(music_path):
        st.caption(f"Music bed loaded: {os.path.basename(music_path)}")
    else:
        st.warning("Music mixing enabled, but no music file is loaded. Upload one (or set DEFAULT_MUSIC_PATH).")

st.divider()

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
    force: bool = False,
) -> Dict[str, str]:
    """
    Produces:
      - voice WAV
      - optional mixed WAV
      - final_wav (unified WAV for safe concat)
      - MP3
      - TXT (clean narration)
    """
    slug = episode_slug()
    paths = section_output_paths(section_id, slug)

    cleaned = strip_tts_directives(sanitize_for_tts(text or ""))
    write_text_file(paths["txt"], cleaned)

    # Force rebuild: remove outputs (cache behavior still depends on your tts_to_wav implementation;
    # this guarantees files are overwritten even if cache is on)
    if force:
        for k in ["voice_wav", "mixed_wav", "final_wav", "mp3"]:
            _safe_remove(paths[k])

    # 1) TTS → voice WAV
    tts_to_wav(text=cleaned, out_wav=paths["voice_wav"], cfg=cfg)

    # 2) Optional music mix
    intermediate_wav = paths["voice_wav"]
    if mix_music:
        if not music_path or not os.path.exists(music_path):
            raise FileNotFoundError("Music bed not found (upload one or disable mixing).")
        mix_music_under_voice(
            voice_wav=paths["voice_wav"],
            music_path=music_path,
            out_wav=paths["mixed_wav"],
            music_db=music_db,
            fade_s=fade_s,
        )
        intermediate_wav = paths["mixed_wav"]

    # 3) Re-encode to a UNIFIED WAV for concat reliability (48kHz, 2ch, pcm_s16le)
    run_ffmpeg([
        FFMPEG, "-y",
        "-i", intermediate_wav,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        paths["final_wav"]
    ])

    # 4) Unified WAV → MP3
    wav_to_mp3(paths["final_wav"], paths["mp3"], bitrate="192k")

    return paths

# ----------------------------
# Section list helper
# ----------------------------
def section_texts() -> List[Tuple[str, str]]:
    """
    Playback order: intro → chapters → outro
    """
    out: List[Tuple[str, str]] = []
    out.append(("00_intro", st.session_state.get(text_key("intro", 0), "")))

    n = int(st.session_state.get("chapter_count", 0))
    for i in range(1, n + 1):
        out.append((f"{i:02d}_chapter_{i}", st.session_state.get(text_key("chapter", i), "")))

    out.append(("99_outro", st.session_state.get(text_key("outro", 0), "")))
    return out

def has_any_script() -> bool:
    return any((t or "").strip() for _, t in section_texts())

# ----------------------------
# Build controls
# ----------------------------
st.subheader("🎙️ Build Audio Sections")

if not has_any_script():
    st.info("Generate a script first (Intro / Chapters / Outro) before building audio.")
else:
    st.session_state.setdefault("built_sections", {})
    built_sections: Dict[str, Dict[str, str]] = st.session_state["built_sections"]

    sections = section_texts()
    ordered_ids = [sid for sid, _ in sections]

    # Build mode controls
    ids_with_text = [sid for sid, txt in sections if (txt or "").strip()]
    default_select = ["00_intro"] if "00_intro" in ids_with_text else (ids_with_text[:1] if ids_with_text else [])
    selected_ids = st.multiselect(
        "Select sections to build",
        options=ids_with_text,
        default=default_select,
        help="Tip: build Intro first to QC voice + pacing before building everything.",
    )

    force_rebuild = st.checkbox(
        "Force rebuild (overwrite outputs)",
        value=False,
        help="If checked, deletes section outputs first and rebuilds them.",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        build_selected_btn = st.button("Build SELECTED", type="primary", disabled=not bool(selected_ids))
    with c2:
        build_all_btn = st.button("Build ALL Sections")
    with c3:
        clear_built_btn = st.button("Clear Built Audio")

    if clear_built_btn:
        wd = _workdir()
        for name in os.listdir(wd):
            # Don't delete persisted music unless it's one of our generated artifacts
            if name.startswith("__music__"):
                continue
            _safe_remove(os.path.join(wd, name))
        st.session_state["built_sections"] = {}
        st.success("Cleared built audio (kept persisted music bed).")

    def _build_ids(to_build: List[str]):
        with st.spinner("Building audio…"):
            prog = st.progress(0)
            total = max(len(to_build), 1)
            built = 0
            for sid, txt in sections:
                if sid not in to_build:
                    continue
                if not (txt or "").strip():
                    continue

                out_paths = build_one_section(
                    section_id=sid,
                    text=txt,
                    cfg=cfg,
                    mix_music=use_music,
                    music_path=music_path,
                    music_db=music_db_i,
                    fade_s=fade_s_i,
                    force=force_rebuild,
                )
                built_sections[sid] = out_paths
                built += 1
                prog.progress(min(built / total, 1.0))
            st.success("Build complete.")

    if build_selected_btn:
        _build_ids(selected_ids)

    if build_all_btn:
        _build_ids(ids_with_text)

    # ----------------------------
    # Per-section QC + downloads
    # ----------------------------
    st.divider()
    st.subheader("🔎 Section QC (Playback + Downloads)")

    any_built = False
    for sid in ordered_ids:
        paths = built_sections.get(sid) or {}
        mp3_path = paths.get("mp3", "")
        txt_path = paths.get("txt", "")
        if mp3_path and os.path.exists(mp3_path):
            any_built = True
            dur = _duration_seconds(mp3_path)
            with st.expander(f"{sid}  —  {dur:.1f}s", expanded=(sid == "00_intro")):
                st.audio(mp3_path)

                colx, coly = st.columns([1, 1])
                with colx:
                    with open(mp3_path, "rb") as f:
                        st.download_button(
                            f"Download {sid} MP3",
                            data=f.read(),
                            file_name=os.path.basename(mp3_path),
                            mime="audio/mpeg",
                        )
                with coly:
                    if txt_path and os.path.exists(txt_path):
                        with open(txt_path, "rb") as f:
                            st.download_button(
                                f"Download {sid} TXT",
                                data=f.read(),
                                file_name=os.path.basename(txt_path),
                                mime="text/plain",
                            )

    if not any_built:
        st.info("No built sections yet. Build Intro first to verify voice + pacing.")

# ----------------------------
# Build FULL episode MP3 (robust concat via WAV)
# ----------------------------
st.divider()
st.subheader("🎬 Build FULL Episode MP3")

ep_slug = episode_slug()
full_mp3_path = os.path.join(_workdir(), f"{ep_slug}__FULL.mp3")
full_wav_path = os.path.join(_workdir(), f"{ep_slug}__FULL.wav")

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    # concat demuxer (WAV) is reliable if formats match (we enforce unified WAV per section)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in wav_paths:
            # ffmpeg concat list expects escaping; safest is to wrap and replace single quotes
            safe_p = p.replace("'", r"'\''")
            f.write(f"file '{safe_p}'\n")
        list_path = f.name

    try:
        run_ffmpeg([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_wav])
    finally:
        _safe_remove(list_path)

def build_full_episode(built_sections: Dict[str, Dict[str, str]], ordered_ids: List[str]) -> None:
    wavs: List[str] = []
    for sid in ordered_ids:
        p = (built_sections.get(sid) or {}).get("final_wav")
        if p and os.path.exists(p):
            wavs.append(p)

    if not wavs:
        raise ValueError("No built section WAVs found. Build sections first.")

    # Build full wav then encode mp3 (uniform, predictable)
    _safe_remove(full_wav_path)
    _safe_remove(full_mp3_path)

    concat_wavs(wavs, full_wav_path)
    wav_to_mp3(full_wav_path, full_mp3_path, bitrate="192k")

if st.button("Build FULL Episode MP3", type="primary"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    sections = section_texts()
    ordered_ids = [sid for sid, _ in sections]
    with st.spinner("Building full episode…"):
        try:
            build_full_episode(built_sections, ordered_ids)
            st.success("Full episode MP3 built.")
        except Exception as e:
            st.error(str(e))

if os.path.exists(full_mp3_path):
    dur = _duration_seconds(full_mp3_path)
    st.caption(f"FULL episode duration: {dur/60.0:.1f} min")
    st.audio(full_mp3_path)
    with open(full_mp3_path, "rb") as f:
        st.download_button(
            "Download FULL Episode MP3",
            data=f.read(),
            file_name=os.path.basename(full_mp3_path),
            mime="audio/mpeg",
        )

# ----------------------------
# Final ZIP (clean + deterministic)
# ----------------------------
st.divider()
st.subheader("📦 Final Delivery ZIP")

include_full_in_zip = st.checkbox(
    "Include FULL episode MP3 in ZIP (if built)",
    value=True,
)

if st.button("Build FINAL ZIP"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    files: List[Tuple[str, str]] = []

    # Optional FULL
    if include_full_in_zip and os.path.exists(full_mp3_path):
        files.append((full_mp3_path, os.path.basename(full_mp3_path)))

    # Section MP3 + TXT (only if exists)
    for sid, paths in built_sections.items():
        mp3p = paths.get("mp3")
        txtp = paths.get("txt")
        if mp3p and os.path.exists(mp3p):
            files.append((mp3p, os.path.basename(mp3p)))
        if txtp and os.path.exists(txtp):
            files.append((txtp, os.path.basename(txtp)))

    # Master script
    raw_path = os.path.join(_workdir(), f"{ep_slug}__MASTER_SCRIPT.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write((st.session_state.get("MASTER_SCRIPT_TEXT", "") or "").strip() + "\n")
    files.append((raw_path, os.path.basename(raw_path)))

    # Outline (nice to ship)
    outline_path = os.path.join(_workdir(), f"{ep_slug}__OUTLINE.txt")
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write((st.session_state.get("OUTLINE_TEXT_SRC", "") or "").strip() + "\n")
    files.append((outline_path, os.path.basename(outline_path)))

    zip_bytes = make_zip_bytes(files)
    st.download_button(
        "Download FINAL ZIP",
        data=zip_bytes,
        file_name=f"{ep_slug}__FINAL.zip",
        mime="application/zip",
    )

