import io
import os
import re
import json
import zipfile
import tempfile
import subprocess
from typing import List, Dict, Optional

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Generate a chaptered documentary script, then create Onyx narration with looped ambient music + fades.")

# ----------------------------
# OpenAI client
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"

# If you want a default track locally, keep this.
DEFAULT_MUSIC_PATH = "/mnt/data/dark-ambient-soundscape-music-409350.mp3"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------
# Helpers
# ----------------------------
def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text.lower()[:80] or "episode"

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
        if buf and (length + add) > max_chars:
            chunks.append("\n\n".join(buf))
            buf, length = [p], len(p)
        else:
            buf.append(p)
            length += add
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks

def run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[-2000:]}")

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    # ffmpeg concat demuxer needs a file list
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            safe_wp = wp.replace("'", "'\\''")
            f.write(f"file '{safe_wp}'\n")
        list_path = f.name

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        out_wav
    ]

    try:
        run_ffmpeg(cmd)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass
            
def mix_music_under_voice(
    voice_wav: str,
    music_path: str,
    out_wav: str,
    music_db: int = -24,
    fade_s: int = 7
) -> None:
    """
    Loops music to match voice duration, applies volume + fades, then mixes under voice.
    """
    # volume in linear multiplier: dB -> factor = 10^(dB/20)
    # But ffmpeg volume filter supports dB directly: volume=-24dB
    # Fade: afade t=in/out
    filter_complex = (
        f"[1:a]volume={music_db}dB,"
        f"afade=t=in:st=0:d={fade_s},"
        f"afade=t=out:st=0:d={fade_s},"
        f"aloop=loop=-1:size=2e+09,"
        f"atrim=0:1e+09[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
    )

    cmd = [
        FFMPEG, "-y",
        "-i", voice_wav,
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        out_wav
    ]
    run_ffmpeg(cmd)

def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    cmd = [
        FFMPEG, "-y",
        "-i", wav_path,
        "-c:a", "libmp3lame",
        "-b:a", "192k",
        mp3_path
    ]
    run_ffmpeg(cmd)

def extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from a model response.
    Handles:
      - pure JSON
      - ```json ... ```
      - extra commentary around JSON
    """
    text = text.strip()

    # 1) If fenced code block exists, pull the inside
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        text = m.group(1).strip()

    # 2) Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3) Fallback: grab the first {...} block (best-effort)
    m2 = re.search(r"(\{.*\})", text, re.S)
    if not m2:
        raise ValueError("Could not find JSON object in response.")
    return json.loads(m2.group(1))


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
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )

    content = (r.choices[0].message.content or "").strip()
    data = extract_json_object(content)

    chapters_list = data.get("chapters")
    if not isinstance(chapters_list, list) or not chapters_list:
        raise ValueError("Outline JSON missing 'chapters' list.")

    # Normalize fields to avoid downstream KeyErrors
    normalized = []
    for ch in chapters_list:
        title = str(ch.get("title", "")).strip() or "Untitled Chapter"
        target = ch.get("target_minutes", chapter_minutes)
        try:
            target = int(target)
        except Exception:
            target = int(chapter_minutes)

        beats = ch.get("beats", [])
        if not isinstance(beats, list):
            beats = []
        beats = [str(b).strip() for b in beats if str(b).strip()]

        normalized.append({
            "title": title,
            "target_minutes": target,
            "beats": beats[:8]  # keep it tidy
        })

    return normalized

def generate_chapter_script(topic, chapter, global_style, episode_notes):
    words = int(chapter["target_minutes"] * 140)
    beats = "\n".join([f"- {b}" for b in chapter["beats"]])

    prompt = f"""
Write a documentary narration chapter.

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
- Audio-only narration (no visual references)
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

def tts_to_wav(text: str, instructions: str, speed: float, out_wav: str) -> None:
    """
    Generates multiple short WAV chunks via OpenAI TTS, then concatenates into one WAV.
    """
    chunks = chunk_text(text, max_chars=3200)
    wav_parts = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, start=1):
            payload = f"[Delivery: {instructions}. Pace {speed:.2f}x]\n\n{ch}"
            r = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                format="wav",
                input=payload
            )
            part_path = os.path.join(td, f"part_{i}.wav")
            with open(part_path, "wb") as f:
                f.write(r.read())
            wav_parts.append(part_path)

        # concat into out_wav
        concat_wavs(wav_parts, out_wav)

# ----------------------------
# Session state
# ----------------------------
if "outline" not in st.session_state:
    st.session_state.outline = None
if "scripts" not in st.session_state:
    st.session_state.scripts = {}  # idx -> text
if "built" not in st.session_state:
    st.session_state.built = None  # dict with mp3 bytes, etc.

# ----------------------------
# UI — Episode setup
# ----------------------------
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
        "Focus on the historical timeline, key witnesses, military response, "
        "official statements, and how the narrative evolved. Separate facts from theories."
    ),
    height=120
)

if st.button("Generate Chapter Outline", type="primary"):
    st.session_state.outline = generate_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes)
    st.session_state.scripts = {}
    st.session_state.built = None

if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")

# ----------------------------
# Scripts
# ----------------------------
st.header("2️⃣ Chapter Scripts")

if st.session_state.outline and st.button("Generate All Scripts"):
    with st.spinner("Writing chapters..."):
        for i, ch in enumerate(st.session_state.outline, 1):
            st.session_state.scripts[i] = generate_chapter_script(topic, ch, global_style, episode_notes)
    st.success("Scripts generated. Expand chapters to edit.")

if st.session_state.outline:
    for i, ch in enumerate(st.session_state.outline, 1):
        with st.expander(f"Chapter {i}: {ch['title']}", expanded=(i == 1)):
            st.session_state.scripts[i] = st.text_area(
                f"Chapter {i} text",
                value=st.session_state.scripts.get(i, ""),
                height=260,
                key=f"ch_{i}"
            )

# ----------------------------
# Audio
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

tts_instructions = st.text_area(
    "TTS delivery instructions",
    value="Calm, authoritative, restrained documentary delivery. Measured pace, subtle pauses. No exaggerated emotion.",
    height=90
)
speed = st.slider("Narration speed (guidance)", 0.85, 1.10, 1.00, 0.01)

music_db = st.slider("Music volume (dB)", -35, -10, -24)
fade_s = st.slider("Music fade in/out (seconds)", 2, 12, 7)

music_file = st.file_uploader("Upload background music (MP3/WAV)", type=["mp3", "wav"])
music_path = None

if music_file:
    # Save to temp later during build
    pass
elif os.path.exists(DEFAULT_MUSIC_PATH):
    music_path = DEFAULT_MUSIC_PATH

build = st.button("Generate Audio + Export", type="primary", disabled=not st.session_state.scripts)

if build:
    if not st.session_state.scripts:
        st.error("Generate scripts first.")
    else:
        with tempfile.TemporaryDirectory() as td:
            # Write music to temp if uploaded
            if music_file:
                music_path = os.path.join(td, "music_upload.mp3")
                with open(music_path, "wb") as f:
                    f.write(music_file.read())

            if not music_path:
                st.error("Please upload a music file (or set a default music path).")
                st.stop()

            chapter_mp3s: Dict[int, bytes] = {}
            chapter_titles: Dict[int, str] = {}
            full_wav_paths: List[str] = []

            st.info("Generating TTS chapter audio and mixing music… (this can take a bit for long chapters)")
            progress = st.progress(0)
            total_chapters = len(st.session_state.scripts)

            for idx in range(1, total_chapters + 1):
                text = st.session_state.scripts.get(idx, "").strip()
                if not text:
                    continue

                # Find chapter title
                title = st.session_state.outline[idx-1]["title"] if st.session_state.outline else f"Chapter {idx}"
                chapter_titles[idx] = title

                raw_wav = os.path.join(td, f"ch_{idx}_voice.wav")
                mixed_wav = os.path.join(td, f"ch_{idx}_mixed.wav")
                out_mp3 = os.path.join(td, f"chapter_{idx}.mp3")

                # TTS → wav
                tts_to_wav(text, tts_instructions, speed, raw_wav)

                # Mix music under voice → wav
                mix_music_under_voice(raw_wav, music_path, mixed_wav, music_db=music_db, fade_s=fade_s)

                # Convert to mp3 for download
                wav_to_mp3(mixed_wav, out_mp3)

                with open(out_mp3, "rb") as f:
                    chapter_mp3s[idx] = f.read()

                full_wav_paths.append(mixed_wav)

                progress.progress(min(1.0, idx / total_chapters))

            # Build full episode mp3 by concatenating mixed WAVs then encoding MP3
            full_wav = os.path.join(td, "full_episode.wav")
            full_mp3 = os.path.join(td, "full_episode.mp3")

            concat_wavs(full_wav_paths, full_wav)
            wav_to_mp3(full_wav, full_mp3)

            with open(full_mp3, "rb") as f:
                full_mp3_bytes = f.read()

            # Create ZIP with scripts + audio
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # scripts
                z.writestr("scripts/episode_topic.txt", topic)
                for idx in sorted(st.session_state.scripts.keys()):
                    title = chapter_titles.get(idx, f"Chapter {idx}")
                    z.writestr(f"scripts/chapter_{idx:02d}_{clean_filename(title)}.txt", st.session_state.scripts[idx])

                # audio
                for idx in sorted(chapter_mp3s.keys()):
                    title = chapter_titles.get(idx, f"Chapter {idx}")
                    z.writestr(f"audio/chapter_{idx:02d}_{clean_filename(title)}.mp3", chapter_mp3s[idx])

                z.writestr("audio/full_episode.mp3", full_mp3_bytes)

            zip_buf.seek(0)

            st.session_state.built = {
                "zip": zip_buf.getvalue(),
                "full_mp3": full_mp3_bytes
            }

            st.success("Done! Download your audio below.")

# ----------------------------
# Downloads
# ----------------------------
st.header("4️⃣ Downloads")

if st.session_state.built:
    st.download_button(
        "⬇️ Download ZIP (scripts + chapter MP3s + full episode MP3)",
        data=st.session_state.built["zip"],
        file_name=f"{clean_filename(topic)}_uappress_pack.zip",
        mime="application/zip"
    )

    st.audio(st.session_state.built["full_mp3"], format="audio/mp3")
    st.download_button(
        "⬇️ Download Full Episode MP3",
        data=st.session_state.built["full_mp3"],
        file_name=f"{clean_filename(topic)}_full_episode.mp3",
        mime="audio/mpeg"
    )
else:
    st.caption("After you generate audio, your download buttons will appear here.")
