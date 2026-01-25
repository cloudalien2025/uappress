# app.py — UAPpress Documentary TTS Studio (MVP)
# ------------------------------------------------------------
# REQUIREMENTS:
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9

import io
import os
import re
import json
import time
import zipfile
import tempfile
import subprocess
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Generate documentary scripts and produce long-form narrated audio with music.")


# ------------------------------------------------------------
# OpenAI
# ------------------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEFAULT_MUSIC_PATH = "/mnt/data/dark-ambient-soundscape-music-409350.mp3"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def run_ffmpeg(cmd: List[str]) -> None:
    code, _, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(err[-4000:])

def chunk_text(text: str, max_chars: int = 3200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text]

    parts = text.split("\n\n")
    chunks, buf, size = [], [], 0
    for p in parts:
        add = len(p) + (2 if buf else 0)
        if buf and size + add > max_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [p], len(p)
        else:
            buf.append(p)
            size += add
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def sanitize_for_tts(text: str) -> str:
    """
    Removes invisible Unicode control characters that cause TTS glitches.
    """
    if not text:
        return ""

    t = text

    # Zero-width & directional Unicode characters (invisible but real)
    t = re.sub(
        r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]",
        "",
        t,
    )

    # Normalize spacing
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t.strip()

def concat_audio_reencode(paths: List[str], out_wav: str) -> None:
    cmd = [FFMPEG, "-y"]
    for p in paths:
        cmd += ["-i", p]

    cmd += [
        "-filter_complex", f"concat=n={len(paths)}:v=0:a=1[a]",
        "-map", "[a]",
        "-ar", "44100",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(cmd)

def wav_to_mp3(wav: str, mp3: str) -> None:
    run_ffmpeg([
        FFMPEG, "-y",
        "-i", wav,
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        mp3
    ])

def tts_to_wav(text: str, instructions: str, speed: float, out_wav: str) -> None:
    chunks = chunk_text(text)
    wavs = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, 1):
            payload = (
                "[INSTRUCTIONS — DO NOT READ]\n"
                f"{instructions}\n"
                f"Speed: {speed:.2f}\n"
                "[/INSTRUCTIONS]\n\n"
                f"{ch}"
            )

            r = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                response_format="wav",
                input=payload,
            )

            path = os.path.join(td, f"part_{i}.wav")
            with open(path, "wb") as f:
                f.write(r.read())
            wavs.append(path)

        concat_audio_reencode(wavs, out_wav)


# ------------------------------------------------------------
# Streamlit state
# ------------------------------------------------------------
def ensure_state():
    st.session_state.setdefault("outline", [])
    st.session_state.setdefault("chapter_count", 0)
    st.session_state.setdefault("topic", "Roswell UFO Incident")
    st.session_state.setdefault("built", None)

ensure_state()

def text_key(kind: str, idx: int = 0) -> str:
    return f"text::{kind}::{idx}"


# ------------------------------------------------------------
# UI — Episode setup
# ------------------------------------------------------------
st.header("1️⃣ Episode Setup")

topic = st.text_input("Episode topic", st.session_state.topic)
st.session_state.topic = topic

if st.button("Generate Outline"):
    st.session_state.outline = [
        {"title": "Background", "target_minutes": 10},
        {"title": "Initial Reports", "target_minutes": 10},
        {"title": "Military Response", "target_minutes": 10},
        {"title": "Public Reaction", "target_minutes": 10},
        {"title": "Theories and Speculation", "target_minutes": 10},
        {"title": "Modern Investigations", "target_minutes": 10},
    ]
    st.session_state.chapter_count = len(st.session_state.outline)
    for i in range(1, st.session_state.chapter_count + 1):
        st.session_state.setdefault(text_key("chapter", i), "")

# ------------------------------------------------------------
# Scripts
# ------------------------------------------------------------
st.header("2️⃣ Scripts")

for i, ch in enumerate(st.session_state.outline, 1):
    with st.expander(f"Chapter {i}: {ch['title']}"):
        st.text_area(
            f"Chapter {i} text",
            key=text_key("chapter", i),
            height=250
        )

# ------------------------------------------------------------
# Audio
# ------------------------------------------------------------
st.header("3️⃣ Generate Audio")

tts_instructions = st.text_area(
    "Narration style",
    "Calm, authoritative documentary narration. No hype.",
)

speed = st.slider("Narration speed", 0.9, 1.1, 1.0)

if st.button("Generate Audio"):
    with tempfile.TemporaryDirectory() as td:
        mixed = []
        for i in range(1, st.session_state.chapter_count + 1):
            raw = st.session_state.get(text_key("chapter", i), "")
            clean = sanitize_for_tts(raw)

            voice_wav = os.path.join(td, f"ch{i}.wav")
            tts_to_wav(clean, tts_instructions, speed, voice_wav)
            mixed.append(voice_wav)

        final_wav = os.path.join(td, "full.wav")
        final_mp3 = os.path.join(td, "full.mp3")

        concat_audio_reencode(mixed, final_wav)
        wav_to_mp3(final_wav, final_mp3)

        with open(final_mp3, "rb") as f:
            st.session_state.built = f.read()

        st.success("Audio generated.")

# ------------------------------------------------------------
# Download
# ------------------------------------------------------------
st.header("4️⃣ Download")

if st.session_state.built:
    st.audio(st.session_state.built)
    st.download_button(
        "⬇️ Download MP3",
        st.session_state.built,
        file_name="uappress_episode.mp3",
        mime="audio/mpeg",
    )
