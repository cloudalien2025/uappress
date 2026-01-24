import io
import os
import re
import json
import zipfile
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI

from pydub import AudioSegment
import imageio_ffmpeg

# -------------------------------------------------
# Page setup
# -------------------------------------------------
st.set_page_config(
    page_title="UAPpress Documentary Studio",
    layout="wide"
)

st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Audio-first, long-form documentary creation with AI narration + ambient music")

# -------------------------------------------------
# FFmpeg setup (required for Streamlit Cloud)
# -------------------------------------------------
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

# -------------------------------------------------
# OpenAI client
# -------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"

DEFAULT_MUSIC_PATH = "/mnt/data/dark-ambient-soundscape-music-409350.mp3"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text.lower()[:80]

def chunk_text(text: str, max_chars: int = 3200) -> List[str]:
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= max_chars:
        return [text]

    parts = text.split("\n\n")
    chunks, buf, length = [], [], 0

    for p in parts:
        p = p.strip()
        if not p:
            continue
        add = len(p) + (2 if buf else 0)
        if buf and length + add > max_chars:
            chunks.append("\n\n".join(buf))
            buf, length = [p], len(p)
        else:
            buf.append(p)
            length += add

    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def generate_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes):
    chapters = max(6, total_minutes // chapter_minutes)
    prompt = f"""
Create a documentary outline for an audio-only YouTube episode.

Topic: {topic}

Global style:
{global_style}

Episode notes:
{episode_notes}

Create {chapters} chapters.
Each chapter must include:
- title
- target_minutes
- 5–7 bullet beats

Return ONLY valid JSON:
{{"chapters":[{{"title":"","target_minutes":10,"beats":[""]}}]}}
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return json.loads(r.choices[0].message.content)["chapters"]

def generate_chapter_script(topic, chapter, global_style, episode_notes):
    words = int(chapter["target_minutes"] * 140)
    beats = "\n".join([f"- {b}" for b in chapter["beats"]])

    prompt = f"""
Write a documentary narration.

Topic: {topic}
Chapter: {chapter["title"]}
Length: ~{words} words

Style:
{global_style}

Notes:
{episode_notes}

Follow these beats:
{beats}

Rules:
- Calm, investigative tone
- Audio-only narration
- No hype or sensationalism
- End with a subtle forward hook

Return ONLY narration text.
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6
    )
    return r.choices[0].message.content.strip()

def tts_generate(text, instructions, speed):
    audio = AudioSegment.silent(duration=0)
    for chunk in chunk_text(text):
        payload = f"[Delivery: {instructions}. Pace {speed:.2f}x]\n\n{chunk}"
        r = client.audio.speech.create(
            model=TTS_MODEL,
            voice=TTS_VOICE,
            format="wav",
            input=payload
        )
        seg = AudioSegment.from_file(io.BytesIO(r.read()), format="wav")
        audio += seg
    return audio

def mix_music(voice, music, music_db, fade_ms):
    if len(music) < len(voice):
        loops = len(voice) // len(music) + 1
        music = music * loops
    music = music[:len(voice)]
    music = music + music_db
    music = music.fade_in(fade_ms).fade_out(fade_ms)
    return music.overlay(voice)

# -------------------------------------------------
# Session state
# -------------------------------------------------
for key in ["outline", "scripts", "audio"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# -------------------------------------------------
# UI — Episode setup
# -------------------------------------------------
st.header("1️⃣ Episode Setup")

topic = st.text_input("Episode topic", "Roswell UFO Incident")
total_minutes = st.slider("Total length (minutes)", 30, 120, 90, 5)
chapter_minutes = st.slider("Minutes per chapter", 5, 15, 10)

global_style = st.text_area(
    "Global style instructions",
    value=(
        "Calm, authoritative male narrator.\n"
        "Serious investigative documentary tone.\n"
        "Measured pacing with subtle pauses.\n"
        "No hype, no jokes, no sensationalism."
    ),
    height=120
)

episode_notes = st.text_area(
    "Episode-specific notes",
    value=(
        "Focus on the historical timeline, key witnesses, "
        "military response, and how the narrative evolved."
    ),
    height=120
)

# -------------------------------------------------
# Outline
# -------------------------------------------------
if st.button("Generate Chapter Outline"):
    st.session_state.outline = generate_outline(
        topic, total_minutes, chapter_minutes, global_style, episode_notes
    )
    st.session_state.scripts = {}
    st.session_state.audio = {}

if st.session_state.outline:
    st.subheader("📑 Chapter Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")

# -------------------------------------------------
# Scripts
# -------------------------------------------------
st.header("2️⃣ Chapter Scripts")

if st.session_state.outline and st.button("Generate All Scripts"):
    for i, ch in enumerate(st.session_state.outline, 1):
        st.session_state.scripts[i] = generate_chapter_script(
            topic, ch, global_style, episode_notes
        )

for i, text in st.session_state.scripts.items():
    with st.expander(f"Chapter {i} Script", expanded=(i == 1)):
        st.session_state.scripts[i] = st.text_area(
            f"Chapter {i}", value=text, height=250
        )

# -------------------------------------------------
# Audio
# -------------------------------------------------
st.header("3️⃣ Create Audio")

tts_instructions = st.text_area(
    "TTS delivery instructions",
    value=(
        "Calm, authoritative, restrained documentary delivery. "
        "Measured pace, neutral emotion."
    ),
    height=100
)

speed = st.slider("Narration speed", 0.85, 1.10, 1.00, 0.01)
music_db = st.slider("Music volume (dB)", -35, -10, -24)
fade_s = st.slider("Music fade (seconds)", 2, 12, 7)

music_file = st.file_uploader("Upload background music (optional)", type=["mp3", "wav"])
music = None

if music_file:
    music = AudioSegment.from_file(io.BytesIO(music_file.read()))
elif os.path.exists(DEFAULT_MUSIC_PATH):
    music = AudioSegment.from_file(DEFAULT_MUSIC_PATH)

if st.button("Generate Audio"):
    if not music:
        st.error("No background music available.")
    else:
        full_audio = AudioSegment.silent(duration=0)
        for i, text in st.session_state.scripts.items():
            voice = tts_generate(text, tts_instructions, speed)
            mixed = mix_music(voice, music, music_db, fade_s * 1000)
            st.session_state.audio[i] = mixed
            full_audio += mixed
        st.session_state.audio["full"] = full_audio
        st.success("Audio created!")

# -------------------------------------------------
# Export
# -------------------------------------------------
st.header("4️⃣ Export")

if "full" in st.session_state.audio:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for k, audio in st.session_state.audio.items():
            buf = io.BytesIO()
            audio.export(buf, format="mp3")
            name = "full_episode.mp3" if k == "full" else f"chapter_{k}.mp3"
            z.writestr(name, buf.getvalue())

    st.download_button(
        "⬇️ Download ZIP (All Audio)",
        zip_buffer.getvalue(),
        file_name=f"{clean_filename(topic)}_audio.zip",
        mime="application/zip"
    )

