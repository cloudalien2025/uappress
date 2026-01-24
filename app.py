import io
import os
import re
import json
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
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Generate Intro + chapters + Outro, then create Onyx narration with looped ambient music + fades + export pack.")

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

# Optional local default (ignored on Streamlit Cloud unless you ship the file in repo)
DEFAULT_MUSIC_PATH = "dark-ambient-soundscape-music-409350.mp3"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------
# Helpers
# ----------------------------
def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text.lower()[:80] or "episode"

def run_ffmpeg(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\n{p.stderr[-2000:]}")

def get_audio_duration_seconds(path: str) -> float:
    """
    Parse duration from ffmpeg stderr (works reliably for common formats).
    """
    p = subprocess.run([FFMPEG, "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", p.stderr)
    if not m:
        return 0.0
    hh, mm, ss = m.groups()
    return int(hh) * 3600 + int(mm) * 60 + float(ss)

def chunk_text(text: str, max_chars: int = 3600) -> List[str]:
    """
    Speech endpoint max input is 4096 chars. Keep margin.
    """
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

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    """
    Concatenate WAV files using ffmpeg concat demuxer.
    """
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

def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    cmd = [FFMPEG, "-y", "-i", wav_path, "-c:a", "libmp3lame", "-b:a", "192k", mp3_path]
    run_ffmpeg(cmd)

def mix_music_under_voice(
    voice_wav: str,
    music_path: str,
    out_wav: str,
    music_db: int = -24,
    fade_s: float = 7.0,
) -> None:
    """
    Loop music to match voice duration, apply volume + fade in/out, then mix under voice.
    """
    dur = max(0.0, get_audio_duration_seconds(voice_wav))
    if dur <= 0.1:
        # If we can't measure duration, just mix with -shortest
        dur = 600.0

    fade_s = float(max(0.0, min(fade_s, max(0.0, dur / 2.0 - 0.01))))
    fade_out_start = max(0.0, dur - fade_s)

    filter_complex = (
        f"[1:a]"
        f"volume={music_db}dB,"
        f"aloop=loop=-1:size=2e+09,"
        f"atrim=0:{dur:.3f},"
        f"afade=t=in:st=0:d={fade_s:.3f},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_s:.3f}"
        f"[m];"
        f"[0:a][m]"
        f"amix=inputs=2:duration=first:dropout_transition=2,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11"
        f"[aout]"
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
    Robustly extract a JSON object from a model response.
    Handles pure JSON, fenced JSON, and extra text.
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

# ----------------------------
# Script generators
# ----------------------------
def generate_outline(topic: str, total_minutes: int, chapter_minutes: int, global_style: str, episode_notes: str) -> List[dict]:
    chapters = max(6, int(total_minutes // chapter_minutes))
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
        temperature=0.3,
    )
    content = (r.choices[0].message.content or "").strip()
    data = extract_json_object(content)

    chapters_list = data.get("chapters")
    if not isinstance(chapters_list, list) or not chapters_list:
        raise ValueError("Outline JSON missing 'chapters' list.")

    normalized = []
    for ch in chapters_list:
        title = str(ch.get("title", "")).strip() or "Untitled Chapter"
        try:
            target = int(ch.get("target_minutes", chapter_minutes))
        except Exception:
            target = int(chapter_minutes)

        beats = ch.get("beats", [])
        if not isinstance(beats, list):
            beats = []
        beats = [str(b).strip() for b in beats if str(b).strip()]

        normalized.append({"title": title, "target_minutes": target, "beats": beats[:8]})
    return normalized

def generate_intro(topic: str, global_style: str, episode_notes: str) -> str:
    prompt = f"""
Write the INTRO for an audio-only investigative documentary episode.

Topic: {topic}

Style:
{global_style}

Episode notes:
{episode_notes}

Must include, naturally and once:
- This episode is sponsored by OPA Nutrition.
- Mention opanutrition.com.
- Briefly describe OPA Nutrition as premium daily wellness supplements (examples: focus/clarity support, daily vitality, fasting-friendly support), without medical claims.
- Engagement CTA: ask listeners to like, subscribe, and comment where they're listening from.
- Hook: tease a compelling question that will be examined, but do NOT spoil outcomes.

Length: 60–120 seconds spoken narration.
Return ONLY the intro narration text.
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip()

def generate_outro(topic: str, global_style: str, episode_notes: str) -> str:
    prompt = f"""
Write the OUTRO for an audio-only investigative documentary episode.

Topic: {topic}

Style:
{global_style}

Episode notes:
{episode_notes}

Must include, naturally:
- Thanks for listening
- Sponsored by OPA Nutrition (mention opanutrition.com)
- Briefly restate OPA Nutrition positioning as premium daily wellness supplements (examples: focus/clarity support, daily vitality, fasting-friendly support), without medical claims.
- Engagement CTA: ask for a comment with their theory and where they're listening from; invite them to like + subscribe.
- Close with a calm, haunting final line that fits the topic.

Length: 60–120 seconds spoken narration.
Return ONLY the outro narration text.
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip()

def generate_chapter_script(topic: str, chapter: dict, global_style: str, episode_notes: str) -> str:
    words = int(chapter["target_minutes"] * 140)  # ~140 wpm
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
- Vivid but grounded (specific dates/places/names when appropriate; no invented quotes)
- No hype, no sensationalism
- End with a subtle forward hook (one line)

Return ONLY narration text.
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.65,
    )
    return (r.choices[0].message.content or "").strip()

# ----------------------------
# TTS
# ----------------------------
def tts_to_wav(text: str, instructions: str, speed: float, out_wav: str) -> None:
    """
    Generates multiple WAV chunks via Speech API, then concatenates into one WAV.

    Speech API params:
    - response_format (NOT format) :contentReference[oaicite:1]{index=1}
    - instructions (works with gpt-4o-mini-tts) :contentReference[oaicite:2]{index=2}
    - speed :contentReference[oaicite:3]{index=3}
    """
    chunks = chunk_text(text, max_chars=3600)
    wav_parts = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, start=1):
            r = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                input=ch,
                instructions=instructions,
                speed=float(speed),
                response_format="wav",
            )
            part_path = os.path.join(td, f"part_{i}.wav")
            with open(part_path, "wb") as f:
                # openai python returns a binary response you can read
                f.write(r.read())
            wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)

# ----------------------------
# Session state
# ----------------------------
def ss_init(key: str, default):
    if key not in st.session_state:
        st.session_state[key] = default

ss_init("outline", None)
ss_init("intro", "")
ss_init("outro", "")
ss_init("scripts", {})  # int -> text
ss_init("built", None)

# ----------------------------
# UI — Episode setup
# ----------------------------
st.header("1️⃣ Episode Setup")

topic = st.text_input("Episode topic", "Roswell UFO Incident")
total_minutes = st.slider("Total length (minutes)", 30, 120, 90, 5)
chapter_minutes = st.slider("Minutes per chapter", 5, 15, 10)

default_global_style = (
    "Calm, authoritative male narrator.\n"
    "Serious investigative documentary tone.\n"
    "Measured pacing with subtle pauses.\n"
    "Grounded language. No hype, no jokes, no sensationalism.\n"
    "Use clear scene-setting, but keep claims careful and sourced in common public record.\n"
)
global_style = st.text_area("Global style instructions", value=default_global_style, height=140)

default_episode_notes = (
    "Focus on the historical timeline, key witnesses, military response, and official statements.\n"
    "Separate known facts from claims and later legends.\n"
    "When discussing controversies, present multiple possibilities without declaring certainty.\n"
)
episode_notes = st.text_area("Episode-specific notes", value=default_episode_notes, height=140)

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    gen_outline_btn = st.button("Generate Chapter Outline", type="primary")
with colB:
    gen_intro_btn = st.button("Generate Intro")
with colC:
    gen_outro_btn = st.button("Generate Outro")

if gen_outline_btn:
    try:
        st.session_state.outline = generate_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes)
        st.session_state.scripts = {}
        st.session_state.built = None
        st.success("Outline generated.")
    except Exception as e:
        st.error(f"Outline failed: {e}")

if gen_intro_btn:
    try:
        st.session_state.intro = generate_intro(topic, global_style, episode_notes)
        st.session_state.built = None
        st.success("Intro generated.")
    except Exception as e:
        st.error(f"Intro failed: {e}")

if gen_outro_btn:
    try:
        st.session_state.outro = generate_outro(topic, global_style, episode_notes)
        st.session_state.built = None
        st.success("Outro generated.")
    except Exception as e:
        st.error(f"Outro failed: {e}")

# Outline display
if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")

# ----------------------------
# Intro editor
# ----------------------------
st.header("2️⃣ Intro / Chapters / Outro")

st.subheader("Intro")
ss_init("intro_text", st.session_state.intro or "")
st.session_state.intro_text = st.text_area(
    "Intro text",
    value=st.session_state.intro_text,
    height=180,
    key="intro_text_widget",
)
# keep canonical copy
st.session_state.intro = st.session_state.intro_text

st.divider()

# ----------------------------
# Chapters
# ----------------------------
st.subheader("Chapters")

left, right = st.columns([1, 1])
with left:
    gen_all = st.button("Generate All Chapters")
with right:
    clear_all = st.button("Clear Chapters")

if clear_all:
    st.session_state.scripts = {}
    # also clear widget states for chapter text areas
    for k in list(st.session_state.keys()):
        if str(k).startswith("ch_text_"):
            del st.session_state[k]
    st.session_state.built = None
    st.success("Cleared.")

if st.session_state.outline and gen_all:
    with st.spinner("Writing chapters..."):
        try:
            for i, ch in enumerate(st.session_state.outline, 1):
                txt = generate_chapter_script(topic, ch, global_style, episode_notes)
                st.session_state.scripts[i] = txt
                # seed widget state so it SHOWS
                st.session_state[f"ch_text_{i}"] = txt
            st.session_state.built = None
            st.success("Chapters generated. Expand to edit.")
        except Exception as e:
            st.error(f"Chapter generation failed: {e}")

if st.session_state.outline:
    for i, ch in enumerate(st.session_state.outline, 1):
        with st.expander(f"Chapter {i}: {ch['title']}", expanded=(i == 1)):
            # per-chapter generate button
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Generate Chapter {i}", key=f"gen_ch_{i}"):
                    try:
                        txt = generate_chapter_script(topic, ch, global_style, episode_notes)
                        st.session_state.scripts[i] = txt
                        st.session_state[f"ch_text_{i}"] = txt  # force visible
                        st.session_state.built = None
                        st.success(f"Chapter {i} generated.")
                    except Exception as e:
                        st.error(f"Chapter {i} failed: {e}")
            with col2:
                st.caption(f"Target: ~{ch['target_minutes']} min")

            # ensure widget state exists (so it displays)
            ss_init(f"ch_text_{i}", st.session_state.scripts.get(i, ""))

            # editor (ALWAYS uses session_state so it doesn't blank out)
            st.session_state[f"ch_text_{i}"] = st.text_area(
                f"Chapter {i} text",
                value=st.session_state.get(f"ch_text_{i}", ""),
                height=260,
                key=f"ch_text_widget_{i}",
            )
            # sync back to canonical dict
            st.session_state.scripts[i] = st.session_state[f"ch_text_{i}"]

st.divider()

# ----------------------------
# Outro editor
# ----------------------------
st.subheader("Outro")
ss_init("outro_text", st.session_state.outro or "")
st.session_state.outro_text = st.text_area(
    "Outro text",
    value=st.session_state.outro_text,
    height=180,
    key="outro_text_widget",
)
st.session_state.outro = st.session_state.outro_text

# ----------------------------
# Audio
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

tts_instructions = st.text_area(
    "TTS delivery instructions (affects voice performance)",
    value=(
        "Calm, restrained investigative documentary delivery. "
        "Slightly slower than conversational pace. "
        "Use subtle pauses between paragraphs. "
        "Keep emotion controlled: serious, curious, and credible. "
        "Avoid exaggerated intensity."
    ),
    height=110,
)

speed = st.slider("Narration speed", 0.85, 1.10, 1.00, 0.01)
music_db = st.slider("Music volume (dB)", -35, -10, -24)
fade_s = st.slider("Music fade in/out (seconds)", 2, 12, 7)

music_file = st.file_uploader("Upload background music (MP3/WAV)", type=["mp3", "wav"])
music_path = None

# Offer repo local default if present
if not music_file and os.path.exists(DEFAULT_MUSIC_PATH):
    music_path = DEFAULT_MUSIC_PATH

def get_script_sequence() -> List[Tuple[str, str]]:
    """
    Returns ordered items: intro, chapters, outro (only if text exists).
    """
    items: List[Tuple[str, str]] = []
    intro_txt = (st.session_state.intro or "").strip()
    if intro_txt:
        items.append(("00_intro", intro_txt))

    # chapters in order
    for idx in sorted(st.session_state.scripts.keys()):
        txt = (st.session_state.scripts.get(idx) or "").strip()
        if txt:
            title = st.session_state.outline[idx - 1]["title"] if st.session_state.outline else f"Chapter {idx}"
            items.append((f"{idx:02d}_{clean_filename(title)}", txt))

    outro_txt = (st.session_state.outro or "").strip()
    if outro_txt:
        items.append(("99_outro", outro_txt))
    return items

sequence = get_script_sequence()
can_build = len(sequence) > 0

build = st.button("Generate Audio + Export", type="primary", disabled=not can_build)

if build:
    try:
        if not sequence:
            st.error("Add Intro/Outro and/or generate at least one chapter.")
            st.stop()

        with tempfile.TemporaryDirectory() as td:
            # Write music to temp if uploaded
            if music_file:
                music_path = os.path.join(td, "music_upload")
                # preserve extension
                ext = os.path.splitext(music_file.name)[1].lower() or ".mp3"
                music_path += ext
                with open(music_path, "wb") as f:
                    f.write(music_file.read())

            if not music_path or not os.path.exists(music_path):
                st.error("Please upload a music file (or add a default music file to the repo root).")
                st.stop()

            st.info("Generating TTS and mixing background music…")
            progress = st.progress(0.0)

            part_mp3s: Dict[str, bytes] = {}
            mixed_wavs: List[str] = []

            for i, (slug, txt) in enumerate(sequence, start=1):
                raw_wav = os.path.join(td, f"{slug}_voice.wav")
                mixed_wav = os.path.join(td, f"{slug}_mixed.wav")
                out_mp3 = os.path.join(td, f"{slug}.mp3")

                # TTS -> voice wav
                tts_to_wav(txt, tts_instructions, speed, raw_wav)

                # Mix background under voice
                mix_music_under_voice(raw_wav, music_path, mixed_wav, music_db=music_db, fade_s=float(fade_s))

                # Export MP3
                wav_to_mp3(mixed_wav, out_mp3)

                with open(out_mp3, "rb") as f:
                    part_mp3s[slug] = f.read()

                mixed_wavs.append(mixed_wav)
                progress.progress(min(1.0, i / float(len(sequence))))

            # Build full episode
            full_wav = os.path.join(td, "full_episode.wav")
            full_mp3 = os.path.join(td, "full_episode.mp3")
            concat_wavs(mixed_wavs, full_wav)
            wav_to_mp3(full_wav, full_mp3)

            with open(full_mp3, "rb") as f:
                full_mp3_bytes = f.read()

            # ZIP pack
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                # scripts
                z.writestr("scripts/episode_topic.txt", topic)
                if (st.session_state.intro or "").strip():
                    z.writestr("scripts/00_intro.txt", st.session_state.intro.strip())

                for idx in sorted(st.session_state.scripts.keys()):
                    title = st.session_state.outline[idx - 1]["title"] if st.session_state.outline else f"Chapter {idx}"
                    z.writestr(f"scripts/{idx:02d}_{clean_filename(title)}.txt", st.session_state.scripts[idx])

                if (st.session_state.outro or "").strip():
                    z.writestr("scripts/99_outro.txt", st.session_state.outro.strip())

                # audio parts
                for slug in sorted(part_mp3s.keys()):
                    z.writestr(f"audio/{slug}.mp3", part_mp3s[slug])
                z.writestr("audio/full_episode.mp3", full_mp3_bytes)

            zip_buf.seek(0)
            st.session_state.built = {
                "zip": zip_buf.getvalue(),
                "full_mp3": full_mp3_bytes
            }

            st.success("Done! Download your pack below.")

    except Exception as e:
        st.error(f"Audio build failed: {e}")

# ----------------------------
# Downloads
# ----------------------------
st.header("4️⃣ Downloads")

if st.session_state.built:
    st.download_button(
        "⬇️ Download ZIP (scripts + section MP3s + full episode MP3)",
        data=st.session_state.built["zip"],
        file_name=f"{clean_filename(topic)}_uappress_pack.zip",
        mime="application/zip",
    )

    st.audio(st.session_state.built["full_mp3"], format="audio/mp3")

    st.download_button(
        "⬇️ Download Full Episode MP3",
        data=st.session_state.built["full_mp3"],
        file_name=f"{clean_filename(topic)}_full_episode.mp3",
        mime="audio/mpeg",
    )
else:
    st.caption("After you generate audio, your download buttons will appear here.")
