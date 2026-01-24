import io
import os
import re
import json
import zipfile
import tempfile
import subprocess
import time
import wave
from typing import List, Dict, Any, Optional

import requests
import streamlit as st
import imageio_ffmpeg

# ============================================================
# UAPpress Documentary Studio (MVP) — Intro + Chapters + Outro
# - Script generation (outline + chapters + intro/outro)
# - TTS via OpenAI REST (no Python SDK headaches)
# - Loop ambient music under voice + fades + loudness normalize
# - Exports: full episode MP3 + chapter MP3s + scripts ZIP
# ============================================================

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Generate a documentary script (Intro + Chapters + Outro), then create Onyx narration with looped ambient music + fades.")

# ----------------------------
# Secrets / API
# ----------------------------
API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets (Settings → Secrets).")
    st.stop()

OPENAI_BASE = "https://api.openai.com/v1"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# Models
SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"

# Default music path (optional local)
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
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[-2500:]}")

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            safe_wp = wp.replace("'", "'\\''")
            f.write(f"file '{safe_wp}'\n")
        list_path = f.name

    cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_wav]
    try:
        run_ffmpeg(cmd)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass

def wav_duration_seconds(path: str) -> float:
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        return frames / float(rate)

def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    cmd = [FFMPEG, "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "192k", mp3_path]
    run_ffmpeg(cmd)

def mix_music_under_voice(
    voice_wav: str,
    music_path: str,
    out_wav: str,
    music_db: int = -24,
    fade_s: int = 7
) -> None:
    """
    Loops music to match voice duration, mixes under voice, then fades + normalizes.
    """
    dur = wav_duration_seconds(voice_wav)
    fade_out_start = max(0.0, dur - float(fade_s))

    filter_complex = (
        f"[1:a]volume={music_db}dB,aloop=loop=-1:size=2000000000,atrim=0:{dur:.3f}[m];"
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2,"
        f"afade=t=in:st=0:d={fade_s},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_s},"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
    )

    cmd = [
        FFMPEG, "-y",
        "-i", voice_wav,
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        out_wav
    ]
    run_ffmpeg(cmd)

def extract_json_object(text: str) -> dict:
    """
    Robustly extract a JSON object from a model response:
    - pure JSON
    - ```json ... ```
    - extra commentary around JSON
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

def openai_chat(prompt: str, temperature: float = 0.4) -> str:
    """
    Uses Chat Completions endpoint via REST.
    """
    url = f"{OPENAI_BASE}/chat/completions"
    payload = {
        "model": SCRIPT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"]["content"] or "").strip()

def openai_tts_wav_bytes(text: str) -> bytes:
    """
    Uses Audio Speech endpoint via REST.
    NOTE: This avoids Python SDK signature issues entirely.
    """
    url = f"{OPENAI_BASE}/audio/speech"
    payload = {
        "model": TTS_MODEL,
        "voice": TTS_VOICE,
        "input": text,
        "format": "wav",
    }
    r = requests.post(url, headers=HEADERS, json=payload, timeout=300)
    r.raise_for_status()
    return r.content

def tts_to_wav(text: str, out_wav: str) -> None:
    """
    Generates multiple short WAV chunks via OpenAI TTS, then concatenates into one WAV.
    """
    chunks = chunk_text(text, max_chars=3200)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, start=1):
            part_path = os.path.join(td, f"part_{i}.wav")
            b = openai_tts_wav_bytes(ch)
            with open(part_path, "wb") as f:
                f.write(b)
            wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)

# ----------------------------
# Script builders
# ----------------------------
def build_outline_prompt(topic: str, total_minutes: int, chapter_minutes: int, global_style: str, episode_notes: str) -> str:
    chapters = max(6, int(round(total_minutes / max(1, chapter_minutes))))
    return f"""
Create a serious investigative documentary OUTLINE for an audio-only YouTube episode.

Topic: {topic}
Target total length: {total_minutes} minutes
Target chapter length: {chapter_minutes} minutes
Number of chapters: {chapters}

Global style:
{global_style}

Episode notes:
{episode_notes}

Return ONLY valid JSON with EXACTLY this shape (no extra keys, no commentary):

{{
  "chapters": [
    {{
      "title": "string",
      "target_minutes": {chapter_minutes},
      "beats": ["string", "string", "string", "string", "string"]
    }}
  ]
}}
"""

def generate_outline(topic: str, total_minutes: int, chapter_minutes: int, global_style: str, episode_notes: str) -> List[Dict[str, Any]]:
    prompt = build_outline_prompt(topic, total_minutes, chapter_minutes, global_style, episode_notes)

    last_err = None
    for attempt in range(1, 4):
        try:
            content = openai_chat(prompt, temperature=0.2)
            data = extract_json_object(content)
            chapters_list = data.get("chapters")
            if not isinstance(chapters_list, list) or not chapters_list:
                raise ValueError("Outline JSON missing 'chapters' list.")

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
                    "beats": beats[:8]
                })
            return normalized
        except Exception as e:
            last_err = e
            time.sleep(0.6)
            # tighten instructions on retries
            prompt = prompt + "\n\nIMPORTANT: Output MUST be valid JSON only. No markdown. No backticks. No explanation."
    raise RuntimeError(f"Failed to generate outline after retries: {last_err}")

def generate_intro(topic: str, global_style: str, episode_notes: str) -> str:
    return openai_chat(f"""
Write an INTRO for a serious investigative audio documentary.

Topic: {topic}

Global style:
{global_style}

Episode notes:
{episode_notes}

Hard requirements:
- 60–90 seconds of spoken narration
- Cold open hook in the first 2–3 sentences
- State what the listener will learn (without spoilers)
- Include sponsor segment (natural, not cheesy) that says:
  "This documentary is sponsored by OPA Nutrition."
  Mention OPA Nutrition makes premium supplements that support:
  - focus and clarity
  - daily energy and resilience
  - general wellness
  Mention: opanutrition.com
- Engagement CTA: ask listeners to subscribe, and to comment where they're listening from (city/state/country)
- NO visual references
Return ONLY the intro narration text.
""", temperature=0.6)

def generate_outro(topic: str, global_style: str, episode_notes: str) -> str:
    return openai_chat(f"""
Write an OUTRO for a serious investigative audio documentary.

Topic: {topic}

Global style:
{global_style}

Episode notes:
{episode_notes}

Hard requirements:
- 45–75 seconds of spoken narration
- Summarize the lasting questions / significance (no hype)
- Sponsor mention again (short and clean):
  "This documentary was sponsored by OPA Nutrition."
  Mention opanutrition.com
- Engagement CTA: ask them to like, subscribe, and comment where they're listening from + what case to cover next
- End on a memorable final line
Return ONLY the outro narration text.
""", temperature=0.6)

def generate_chapter_script(topic: str, chapter: Dict[str, Any], global_style: str, episode_notes: str) -> str:
    words = int(chapter["target_minutes"] * 150)  # slightly denser for doc pacing
    beats = "\n".join([f"- {b}" for b in chapter.get("beats", [])])

    return openai_chat(f"""
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
- Serious, investigative, grounded tone
- Audio-only narration (no visual references)
- Use specific names/titles/roles when commonly known for the case (no wild invention)
- Separate facts vs allegations vs later claims (signal uncertainty clearly)
- End with a subtle forward hook to the next chapter

Return ONLY narration text.
""", temperature=0.65)

# ----------------------------
# Session state
# ----------------------------
if "outline" not in st.session_state:
    st.session_state.outline = None
if "intro" not in st.session_state:
    st.session_state.intro = ""
if "outro" not in st.session_state:
    st.session_state.outro = ""
if "scripts" not in st.session_state:
    st.session_state.scripts = {}  # chapter_index -> text
if "built" not in st.session_state:
    st.session_state.built = None

# ----------------------------
# UI — Episode setup
# ----------------------------
st.header("1️⃣ Episode Setup")

left, right = st.columns([2, 1], vertical_alignment="top")

with left:
    topic = st.text_input("Episode topic", "Roswell UFO Incident")
    total_minutes = st.slider("Total length (minutes)", 30, 120, 90, 5)
    chapter_minutes = st.slider("Minutes per chapter", 5, 15, 10)

with right:
    st.markdown("**Narration voice**")
    st.write(f"Model: `{TTS_MODEL}`")
    st.write(f"Voice: `{TTS_VOICE}`")

global_style = st.text_area(
    "Global style instructions",
    value=(
        "Calm, authoritative male narrator.\n"
        "Serious investigative documentary tone.\n"
        "Measured pacing with subtle pauses.\n"
        "Cinematic clarity — vivid but restrained.\n"
        "No hype, no jokes, no sensationalism.\n"
        "Signal uncertainty cleanly: what we know, what we don't, and why it matters."
    ),
    height=160
)

episode_notes = st.text_area(
    "Episode-specific notes",
    value=(
        "Prioritize the historical timeline, primary accounts, named officials, and how the story changed over time.\n"
        "Separate contemporaneous reporting from later retellings.\n"
        "Include key locations, dates, and the military/public information response.\n"
        "Treat claims carefully: label them as claims and note sourcing when possible."
    ),
    height=160
)

btn_row = st.columns([1, 1, 1, 2])
with btn_row[0]:
    if st.button("Generate Outline", type="primary"):
        st.session_state.outline = generate_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes)
        st.session_state.scripts = {}
        st.session_state.intro = ""
        st.session_state.outro = ""
        st.session_state.built = None

with btn_row[1]:
    if st.button("Generate Intro"):
        st.session_state.intro = generate_intro(topic, global_style, episode_notes)
        st.session_state.built = None

with btn_row[2]:
    if st.button("Generate Outro"):
        st.session_state.outro = generate_outro(topic, global_style, episode_notes)
        st.session_state.built = None

if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")

st.divider()

# ----------------------------
# Scripts
# ----------------------------
st.header("2️⃣ Script Editor (Intro + Chapters + Outro)")

# Intro editor
st.subheader("Intro")
st.session_state.intro = st.text_area(
    "Intro text",
    value=st.session_state.intro,
    height=220,
    key="intro_text"
)

# Chapter generation controls
if st.session_state.outline:
    gen_cols = st.columns([1, 1, 2, 2])
    with gen_cols[0]:
        if st.button("Generate All Chapters"):
            with st.spinner("Writing chapter scripts..."):
                for i, ch in enumerate(st.session_state.outline, 1):
                    st.session_state.scripts[i] = generate_chapter_script(topic, ch, global_style, episode_notes)
            st.success("Chapters generated. Expand to edit.")
            st.session_state.built = None

    with gen_cols[1]:
        if st.button("Clear Chapters"):
            st.session_state.scripts = {}
            st.session_state.built = None

# Chapter editors
if st.session_state.outline:
    for i, ch in enumerate(st.session_state.outline, 1):
        with st.expander(f"Chapter {i}: {ch['title']}", expanded=(i == 1)):
            c1, c2 = st.columns([1, 2], vertical_alignment="top")
            with c1:
                if st.button(f"Generate Chapter {i}", key=f"gen_ch_{i}"):
                    st.session_state.scripts[i] = generate_chapter_script(topic, ch, global_style, episode_notes)
                    st.session_state.built = None
                st.caption(f"Target: ~{ch['target_minutes']} min")

            with c2:
                st.session_state.scripts[i] = st.text_area(
                    f"Chapter {i} text",
                    value=st.session_state.scripts.get(i, ""),
                    height=260,
                    key=f"ch_text_{i}",
                    placeholder="Generate this chapter to populate text here…"
                )

# Outro editor
st.subheader("Outro")
st.session_state.outro = st.text_area(
    "Outro text",
    value=st.session_state.outro,
    height=220,
    key="outro_text"
)

st.divider()

# ----------------------------
# Audio
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

st.caption("Tip: Start with 1–2 chapters to validate output + timing before generating the full 90 minutes.")

music_file = st.file_uploader("Upload background music (MP3/WAV)", type=["mp3", "wav"])
music_path = None

if (not music_file) and os.path.exists(DEFAULT_MUSIC_PATH):
    st.info("Using default ambient track already present on the server.")
    music_path = DEFAULT_MUSIC_PATH

music_db = st.slider("Music volume (dB)", -35, -10, -24)
fade_s = st.slider("Music fade in/out (seconds)", 2, 12, 7)

# Build script order: intro -> chapters -> outro
def collect_script_sections() -> List[Dict[str, str]]:
    sections: List[Dict[str, str]] = []
    intro = (st.session_state.intro or "").strip()
    if intro:
        sections.append({"title": "Intro", "text": intro})

    # chapters
    if st.session_state.outline:
        for i, ch in enumerate(st.session_state.outline, 1):
            t = (st.session_state.scripts.get(i) or "").strip()
            if t:
                sections.append({"title": f"Chapter {i} — {ch['title']}", "text": t})

    outro = (st.session_state.outro or "").strip()
    if outro:
        sections.append({"title": "Outro", "text": outro})

    return sections

sections = collect_script_sections()
can_build = len(sections) > 0

if not can_build:
    st.warning("Generate or paste at least an Intro, one Chapter, or an Outro before building audio.")

build = st.button("Generate Audio + Export", type="primary", disabled=not can_build)

if build:
    try:
        with tempfile.TemporaryDirectory() as td:
            # write uploaded music if provided
            if music_file:
                music_path = os.path.join(td, "music_upload.mp3")
                with open(music_path, "wb") as f:
                    f.write(music_file.read())

            if not music_path:
                st.error("Please upload a music file (or set DEFAULT_MUSIC_PATH to a valid file).")
                st.stop()

            # Render each section -> mixed mp3
            section_mp3s: Dict[int, bytes] = {}
            section_titles: Dict[int, str] = {}
            mixed_wavs: List[str] = []

            st.info("Generating TTS and mixing background music…")
            progress = st.progress(0)
            total = len(sections)

            for idx, sec in enumerate(sections, start=1):
                title = sec["title"]
                text = sec["text"]

                section_titles[idx] = title

                raw_wav = os.path.join(td, f"sec_{idx:02d}_voice.wav")
                mixed_wav = os.path.join(td, f"sec_{idx:02d}_mixed.wav")
                out_mp3 = os.path.join(td, f"sec_{idx:02d}.mp3")

                # TTS -> wav
                tts_to_wav(text, raw_wav)

                # Mix music under voice -> wav
                mix_music_under_voice(raw_wav, music_path, mixed_wav, music_db=music_db, fade_s=fade_s)

                # Convert to mp3
                wav_to_mp3(mixed_wav, out_mp3)

                with open(out_mp3, "rb") as f:
                    section_mp3s[idx] = f.read()

                mixed_wavs.append(mixed_wav)
                progress.progress(min(1.0, idx / total))

            # Full episode
            full_wav = os.path.join(td, "full_episode.wav")
            full_mp3 = os.path.join(td, "full_episode.mp3")
            concat_wavs(mixed_wavs, full_wav)
            wav_to_mp3(full_wav, full_mp3)

            with open(full_mp3, "rb") as f:
                full_mp3_bytes = f.read()

            # ZIP package
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("scripts/episode_topic.txt", topic)

                # scripts
                z.writestr("scripts/00_intro.txt", st.session_state.intro.strip() if st.session_state.intro else "")
                if st.session_state.outline:
                    for i, ch in enumerate(st.session_state.outline, 1):
                        txt = st.session_state.scripts.get(i, "").strip()
                        if txt:
                            z.writestr(f"scripts/{i:02d}_chapter_{clean_filename(ch['title'])}.txt", txt)
                z.writestr("scripts/99_outro.txt", st.session_state.outro.strip() if st.session_state.outro else "")

                # audio
                for idx in sorted(section_mp3s.keys()):
                    z.writestr(f"audio/{idx:02d}_{clean_filename(section_titles[idx])}.mp3", section_mp3s[idx])
                z.writestr("audio/full_episode.mp3", full_mp3_bytes)

            zip_buf.seek(0)

            st.session_state.built = {
                "zip": zip_buf.getvalue(),
                "full_mp3": full_mp3_bytes
            }

            st.success("Audio build complete! Downloads are ready below.")
    except Exception as e:
        st.error(f"Audio build failed: {e}")

st.divider()

# ----------------------------
# Downloads
# ----------------------------
st.header("4️⃣ Downloads")

if st.session_state.built:
    st.download_button(
        "⬇️ Download ZIP (scripts + section MP3s + full episode MP3)",
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
