import io
import os
import re
import json
import zipfile
import tempfile
import subprocess
from typing import List, Dict, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption(
    "Generate an Intro + chaptered documentary script + Outro, then create Onyx narration "
    "with looped ambient music + fades. Export section MP3s + full episode MP3."
)

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

# Default track path (useful locally/dev; on Streamlit Cloud you’ll likely upload)
DEFAULT_MUSIC_PATH = "/mnt/data/dark-ambient-soundscape-music-409350.mp3"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# ----------------------------
# Sponsor + Engagement (verbatim reads used in prompts)
# ----------------------------
SPONSOR_READ_INTRO = (
    "This documentary is sponsored by OPA Nutrition — a wellness company focused on clarity, "
    "resilience, and long-term health. Their lineup includes supplements designed to support "
    "cognitive focus, metabolic health, and daily vitality — built for people who value long-form "
    "thinking and sustained attention. Learn more at opanutrition.com."
)

SOFT_CTA_INTRO = (
    "If you appreciate calm, investigative documentaries like this one, you may want to subscribe — "
    "new long-form stories are released regularly. And if you enjoy the episode, tap like and tell me "
    "in the comments where you’re listening from."
)

SPONSOR_READ_OUTRO = (
    "Today’s documentary was sponsored by OPA Nutrition. If you want to explore products designed to support "
    "focus, metabolic balance, and daily vitality, visit opanutrition.com."
)

STRONG_CTA_OUTRO = (
    "If this investigation held your attention, please like the video, subscribe, and leave a comment with "
    "where you’re listening from — and which case you want covered next."
)

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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            safe_wp = wp.replace("'", "'\\''")
            f.write(f"file '{safe_wp}'\n")
        list_path = f.name

    cmd = [
        FFMPEG,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_path,
        "-c",
        "copy",
        out_wav,
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
    fade_s: int = 7,
) -> None:
    """
    Loops music to match voice duration, applies volume + fades, then mixes under voice.
    """
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
        FFMPEG,
        "-y",
        "-i",
        voice_wav,
        "-stream_loop",
        "-1",
        "-i",
        music_path,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(cmd)


def wav_to_mp3(wav_path: str, mp3_path: str) -> None:
    cmd = [
        FFMPEG,
        "-y",
        "-i",
        wav_path,
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        mp3_path,
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
# Script generation
# ----------------------------
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
        temperature=0.3,
    )

    data = extract_json_object(r.choices[0].message.content)
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

        normalized.append({"title": title, "target_minutes": target, "beats": beats[:8]})

    return normalized


def generate_chapter_script(topic, chapter, global_style, episode_notes):
    words = int(chapter["target_minutes"] * 140)
    beats = "\n".join([f"- {b}" for b in chapter["beats"]])

    prompt = f"""
Write a documentary narration chapter.

Topic: {topic}
Chapter: {chapter["title"]}
Target length: about {chapter["target_minutes"]} minutes (~{words} words)

Global style:
{global_style}

Episode notes:
{episode_notes}

Chapter beats (follow this order):
{beats}

Rules:
- Audio-only narration (no visual references).
- Calm, investigative documentary tone.
- Clearly separate documented facts from later interpretations or theories.
- Avoid hype, jokes, sensational language, or rhetorical clickbait.
- Use smooth transitions. Avoid repetition.
- End with a subtle forward hook (1–2 sentences), not a hard cliffhanger.

Return ONLY narration text.
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip()


def generate_intro(topic: str, global_style: str, episode_notes: str) -> str:
    prompt = f"""
Write a documentary INTRO for an audio-only YouTube episode.

Topic: {topic}

Global style:
{global_style}

Episode notes:
{episode_notes}

Requirements:
- 2–4 minutes of spoken narration
- Cold open feel: calm, atmospheric, investigative
- Establish timeframe and stakes without overexplaining
- Include this sponsor message verbatim, integrated naturally:
  "{SPONSOR_READ_INTRO}"
- After the sponsor message, include this engagement message verbatim:
  "{SOFT_CTA_INTRO}"
- No hype, no jokes, no sensationalism
- Return ONLY the intro narration text (no headings, no bullets).
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip()


def generate_outro(topic: str, global_style: str, episode_notes: str) -> str:
    prompt = f"""
Write a documentary OUTRO for an audio-only YouTube episode.

Topic: {topic}

Global style:
{global_style}

Episode notes:
{episode_notes}

Requirements:
- 2–3 minutes of spoken narration
- Reflective close: what seems established vs what remains uncertain
- Calm, grounded, credible
- Include this sponsor message verbatim, integrated naturally:
  "{SPONSOR_READ_OUTRO}"
- Then include this engagement message verbatim:
  "{STRONG_CTA_OUTRO}"
- End with a final lingering line that feels serious, not sensational
- Return ONLY the outro narration text (no headings, no bullets).
"""
    r = client.chat.completions.create(
        model=SCRIPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6,
    )
    return (r.choices[0].message.content or "").strip()


# ----------------------------
# TTS
# ----------------------------
def tts_to_wav(text: str, instructions: str, speed: float, out_wav: str) -> None:
    """
    Generates multiple short WAV chunks via OpenAI TTS, then concatenates into one WAV.
    """
    chunks = chunk_text(text, max_chars=3200)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, start=1):
            payload = f"[Delivery: {instructions}. Pace {speed:.2f}x]\n\n{ch}"
            r = client.audio.speech.create(
                model=TTS_MODEL,
                voice=TTS_VOICE,
                format="wav",
                input=payload,
            )
            part_path = os.path.join(td, f"part_{i}.wav")
            with open(part_path, "wb") as f:
                f.write(r.read())
            wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)


# ----------------------------
# Session state
# ----------------------------
st.session_state.setdefault("outline", None)
st.session_state.setdefault("built", None)

# Widget keys are the single source of truth (IMPORTANT)
st.session_state.setdefault("intro_text", "")
st.session_state.setdefault("outro_text", "")

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
        "No hype, no jokes, no sensationalism.\n"
        "Clean, structured storytelling with smooth transitions."
    ),
    height=140,
)

episode_notes = st.text_area(
    "Episode-specific notes (masterpiece brief)",
    value=(
        "Treat this as a serious historical investigation, not a hype piece.\n"
        "Anchor the narrative in a strict chronological timeline.\n"
        "Clearly separate documented facts and official statements from later interpretations and theories.\n"
        "Emphasize first-hand witness testimony, military communications, and shifting public explanations.\n"
        "Include Cold War context and how the story evolved culturally over time.\n"
        "Maintain an observational, restrained tone; allow uncertainty to remain where evidence is incomplete."
    ),
    height=180,
)

if st.button("Generate Chapter Outline", type="primary"):
    try:
        st.session_state.outline = generate_outline(
            topic, total_minutes, chapter_minutes, global_style, episode_notes
        )
        st.session_state.built = None

        # reset scripts when outline changes
        st.session_state["intro_text"] = ""
        st.session_state["outro_text"] = ""
        for i in range(1, 201):
            k = f"chapter_text_{i}"
            if k in st.session_state:
                st.session_state[k] = ""

        st.success("Outline generated.")
        st.rerun()
    except Exception as e:
        st.error(f"Outline generation failed: {e}")

if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")


# ----------------------------
# Scripts (Intro + Chapters + Outro)
# ----------------------------
st.header("2️⃣ Scripts (Intro → Chapters → Outro)")

if st.session_state.outline is None:
    st.info("Generate an outline first.")
else:
    # Intro
    st.subheader("Intro")

    c1, c2 = st.columns([1, 3], gap="large")
    with c1:
        if st.button("Generate Intro"):
            try:
                with st.spinner("Writing intro..."):
                    st.session_state["intro_text"] = generate_intro(topic, global_style, episode_notes)
                st.success("Intro generated.")
                st.session_state.built = None
                st.rerun()
            except Exception as e:
                st.error(f"Intro generation failed: {e}")

    with c2:
        st.text_area("Intro text", key="intro_text", height=220)

    # Chapters
    st.subheader("Chapters")

    if st.button("Generate All Chapters"):
        try:
            with st.spinner("Writing chapter scripts..."):
                for i, ch in enumerate(st.session_state.outline, 1):
                    txt = generate_chapter_script(topic, ch, global_style, episode_notes)
                    st.session_state[f"chapter_text_{i}"] = txt
            st.success("Chapters generated. Expand any chapter to edit.")
            st.session_state.built = None
            st.rerun()
        except Exception as e:
            st.error(f"Chapter generation failed: {e}")

    # Render chapters (keys hold the truth)
    for i, ch in enumerate(st.session_state.outline, 1):
        title = ch.get("title", f"Chapter {i}")
        st.session_state.setdefault(f"chapter_text_{i}", "")
        with st.expander(f"Chapter {i}: {title}", expanded=(i == 1)):
            st.text_area(f"Chapter {i} text", key=f"chapter_text_{i}", height=280)

    # Outro
    st.subheader("Outro")

    c3, c4 = st.columns([1, 3], gap="large")
    with c3:
        if st.button("Generate Outro"):
            try:
                with st.spinner("Writing outro..."):
                    st.session_state["outro_text"] = generate_outro(topic, global_style, episode_notes)
                st.success("Outro generated.")
                st.session_state.built = None
                st.rerun()
            except Exception as e:
                st.error(f"Outro generation failed: {e}")

    with c4:
        st.text_area("Outro text", key="outro_text", height=220)


# ----------------------------
# Audio
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

tts_instructions = st.text_area(
    "TTS delivery instructions",
    value=(
        "Calm, authoritative, restrained documentary delivery. "
        "Measured pace with subtle pauses. Neutral emotion. "
        "No exaggerated emphasis or sales tone."
    ),
    height=100,
)
speed = st.slider("Narration speed (guidance)", 0.85, 1.10, 1.00, 0.01)

music_db = st.slider("Music volume (dB)", -35, -10, -24)
fade_s = st.slider("Music fade in/out (seconds)", 2, 12, 7)

music_file = st.file_uploader("Upload background music (MP3/WAV)", type=["mp3", "wav"])
music_path = None
if not music_file and os.path.exists(DEFAULT_MUSIC_PATH):
    music_path = DEFAULT_MUSIC_PATH

# Determine if there's any text to render (from widget keys)
outline_len = len(st.session_state.outline) if st.session_state.outline else 0
has_any_text = bool(
    (st.session_state.get("intro_text") or "").strip()
    or (st.session_state.get("outro_text") or "").strip()
    or any((st.session_state.get(f"chapter_text_{i}") or "").strip() for i in range(1, outline_len + 1))
)

build = st.button("Generate Audio + Export", type="primary", disabled=not has_any_text)

if build:
    try:
        with tempfile.TemporaryDirectory() as td:
            # Save uploaded music into temp
            if music_file:
                ext = ".mp3"
                if music_file.name.lower().endswith(".wav"):
                    ext = ".wav"
                music_path = os.path.join(td, f"music_upload{ext}")
                with open(music_path, "wb") as f:
                    f.write(music_file.read())

            if not music_path:
                st.error("Please upload a music file (or set a default music path).")
                st.stop()

            # Build sections in order: Intro -> Chapters -> Outro
            sections: List[Tuple[str, str, str]] = []

            intro_txt = (st.session_state.get("intro_text") or "").strip()
            if intro_txt:
                sections.append(("intro", "Intro", intro_txt))

            for idx in range(1, outline_len + 1):
                txt = (st.session_state.get(f"chapter_text_{idx}") or "").strip()
                if not txt:
                    continue
                title = st.session_state.outline[idx - 1]["title"]
                sections.append((f"chapter_{idx:02d}", title, txt))

            outro_txt = (st.session_state.get("outro_text") or "").strip()
            if outro_txt:
                sections.append(("outro", "Outro", outro_txt))

            if not sections:
                st.error("No Intro/Chapters/Outro text found. Generate scripts first.")
                st.stop()

            mp3_blobs: Dict[str, bytes] = {}
            titles: Dict[str, str] = {}
            full_wav_paths: List[str] = []

            st.info("Generating TTS and mixing background music…")
            progress = st.progress(0)

            for n, (slug, title, text) in enumerate(sections, start=1):
                titles[slug] = title

                raw_wav = os.path.join(td, f"{slug}_voice.wav")
                mixed_wav = os.path.join(td, f"{slug}_mixed.wav")
                out_mp3 = os.path.join(td, f"{slug}.mp3")

                # TTS -> voice wav
                tts_to_wav(text, tts_instructions, speed, raw_wav)

                # Mix music
                mix_music_under_voice(raw_wav, music_path, mixed_wav, music_db=music_db, fade_s=fade_s)

                # Encode mp3
                wav_to_mp3(mixed_wav, out_mp3)

                with open(out_mp3, "rb") as f:
                    mp3_blobs[slug] = f.read()

                full_wav_paths.append(mixed_wav)
                progress.progress(min(1.0, n / len(sections)))

            # Full episode
            full_wav = os.path.join(td, "full_episode.wav")
            full_mp3 = os.path.join(td, "full_episode.mp3")
            concat_wavs(full_wav_paths, full_wav)
            wav_to_mp3(full_wav, full_mp3)

            with open(full_mp3, "rb") as f:
                full_mp3_bytes = f.read()

            # ZIP pack
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("scripts/episode_topic.txt", topic)

                # scripts
                if intro_txt:
                    z.writestr("scripts/intro.txt", intro_txt)

                for idx in range(1, outline_len + 1):
                    title = st.session_state.outline[idx - 1]["title"]
                    txt = st.session_state.get(f"chapter_text_{idx}") or ""
                    z.writestr(f"scripts/chapter_{idx:02d}_{clean_filename(title)}.txt", txt)

                if outro_txt:
                    z.writestr("scripts/outro.txt", outro_txt)

                # audio sections
                for slug, blob in mp3_blobs.items():
                    z.writestr(f"audio/{slug}_{clean_filename(titles.get(slug, slug))}.mp3", blob)

                # full audio
                z.writestr("audio/full_episode.mp3", full_mp3_bytes)

            zip_buf.seek(0)

            st.session_state.built = {
                "zip": zip_buf.getvalue(),
                "full_mp3": full_mp3_bytes,
            }

            st.success("Done! Download your audio below.")
            st.rerun()

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
