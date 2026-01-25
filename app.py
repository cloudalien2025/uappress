# app.py — UAPpress Documentary TTS Studio (MVP)
# ------------------------------------------------------------
# REQUIREMENTS (put these in requirements.txt):
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9
#
# Notes:
# - No pydub (avoids ffmpeg/avlib install issues on Streamlit Cloud)
# - Uses ffmpeg (bundled via imageio_ffmpeg) for concat + music mix
# - Fixes Streamlit state sync so generated text ALWAYS appears
# - FIX: avoids StreamlitAPIException by never assigning to widget-owned keys after widget creation

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


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio (MVP)")
st.caption("Generate Intro + chaptered documentary script, then create Onyx narration with background music mix.")


# ----------------------------
# OpenAI client
# ----------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found. Add it to Streamlit Secrets as OPENAI_API_KEY.")
    st.stop()

client = OpenAI(api_key=api_key)

SCRIPT_MODEL = "gpt-4o-mini"
TTS_MODEL = "gpt-4o-mini-tts"
TTS_VOICE = "onyx"

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

# Optional local default (won't exist on Streamlit Cloud unless you upload it)
DEFAULT_MUSIC_PATH = "/mnt/data/dark-ambient-soundscape-music-409350.mp3"


# ----------------------------
# Helpers
# ----------------------------
def run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def run_ffmpeg(cmd: List[str]) -> None:
    code, out, err = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"ffmpeg failed:\n{err[-4000:]}")

def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", text).strip()
    text = re.sub(r"\s+", "_", text)
    return (text.lower()[:80] or "episode")

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
        raise ValueError("Could not find a JSON object in the response.")
    return json.loads(m2.group(1))

def safe_chat(prompt: str, temperature: float = 0.4, tries: int = 2) -> str:
    last_err = None
    for _ in range(tries):
        try:
            r = client.chat.completions.create(
                model=SCRIPT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            last_err = e
            time.sleep(0.6)
    raise RuntimeError(f"Chat request failed: {last_err}")

def json_outline_from_model(prompt: str) -> dict:
    """
    Attempts to get valid JSON reliably. If parsing fails, retries with a stricter prompt.
    """
    content = safe_chat(prompt, temperature=0.25, tries=2)
    try:
        return extract_json_object(content)
    except Exception:
        stricter = prompt + "\n\nIMPORTANT: Return ONLY valid JSON. No prose, no markdown, no code fences."
        content2 = safe_chat(stricter, temperature=0.15, tries=2)
        return extract_json_object(content2)

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    # ffmpeg concat demuxer list file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for wp in wav_paths:
            f.write("file '{}'\n".format(wp.replace("'", "'\\''")))
        list_path = f.name

    try:
        cmd = [
            FFMPEG, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            out_wav,
        ]
        run_ffmpeg(cmd)
    finally:
        try:
            os.remove(list_path)
        except Exception:
            pass

def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    cmd = [
        FFMPEG, "-y",
        "-i", wav_path,
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_path,
    ]
    run_ffmpeg(cmd)

def get_audio_duration_seconds(path: str) -> float:
    """
    Uses 'ffmpeg -i' stderr parsing (works even without ffprobe).
    """
    cmd = [FFMPEG, "-i", path]
    code, out, err = run_cmd(cmd)
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
    if not m:
        return 0.0
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600 + mm * 60 + ss

def mix_music_under_voice(
    voice_wav: str,
    music_path: str,
    out_wav: str,
    music_db: int = -24,
    fade_s: int = 6,
) -> None:
    """
    Loops music to match voice duration, applies volume + fades, mixes under voice, normalizes loudness.
    """
    dur = max(0.0, get_audio_duration_seconds(voice_wav))
    fade_out_start = max(0.0, dur - fade_s) if dur > 0 else 0.0

    music_chain = (
        f"[1:a]volume={music_db}dB,"
        f"afade=t=in:st=0:d={fade_s},"
        f"afade=t=out:st={fade_out_start}:d={fade_s}[m];"
    )
    mix_chain = (
        f"[0:a][m]amix=inputs=2:duration=first:dropout_transition=2,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11[aout]"
    )
    filter_complex = music_chain + mix_chain

    cmd = [
        FFMPEG, "-y",
        "-i", voice_wav,
        "-stream_loop", "-1",
        "-i", music_path,
        "-filter_complex", filter_complex,
        "-map", "[aout]",
        "-c:a", "pcm_s16le",
        out_wav,
    ]
    run_ffmpeg(cmd)

def tts_to_wav(text: str, delivery_instructions: str, speed: float, out_wav: str) -> None:
    """
    Generates multiple short WAV chunks via OpenAI TTS, then concatenates into one WAV.
    """
    chunks = chunk_text(text, max_chars=3200)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, start=1):
            payload = (
                f"VOICE DIRECTION (follow closely): {delivery_instructions}\n"
                f"PACE: {speed:.2f}x\n"
                f"STYLE: investigative documentary; restrained.\n\n"
                f"{ch}"
            )

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


# ----------------------------
# Streamlit state (single source of truth)
# ----------------------------
def ensure_state():
    st.session_state.setdefault("outline", [])            # list[dict]
    st.session_state.setdefault("chapter_count", 0)
    st.session_state.setdefault("built", None)
    st.session_state.setdefault("topic", "Roswell UFO Incident")
    st.session_state.setdefault("last_outline_params", None)

ensure_state()

def text_key(kind: str, idx: int = 0) -> str:
    # kind in {"intro", "chapter", "outro"}
    return f"text::{kind}::{idx}"

def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> str:
    """
    IMPORTANT:
    Only initializes state if missing.
    Never overwrites widget-managed keys.
    """
    k = text_key(kind, idx)
    st.session_state.setdefault(k, default or "")
    return k

def get_text(kind: str, idx: int = 0) -> str:
    return st.session_state.get(text_key(kind, idx), "")


# ----------------------------
# Prompt builders
# ----------------------------
def prompt_outline(topic: str, total_minutes: int, chapter_minutes: int, global_style: str, episode_notes: str) -> str:
    chapters = max(6, total_minutes // chapter_minutes)
    return f"""
Create an outline for an audio-only investigative documentary.

TOPIC:
{topic}

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
  - beats (5–8 bullet beats; each beat is a single sentence)
- Beats should be chronological when possible
- Separate: established facts vs claims vs theories
- No sensationalism, no jokes
- Keep a strong narrative arc: hook → evidence → conflict → implications → unresolved questions

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

def prompt_intro(topic: str, global_style: str, episode_notes: str) -> str:
    return f"""
Write an INTRO for an audio-only investigative documentary.

TOPIC:
{topic}

STYLE (must follow):
{global_style}

NOTES:
{episode_notes}

INTRO REQUIREMENTS:
- 60–90 seconds spoken
- Strong hook and stakes within first 2–3 sentences
- Briefly preview what listeners will hear (without listing chapter numbers)
- Include a sponsor mention that feels natural and premium:
  - This episode is sponsored by OPA Nutrition
  - Mention they make premium wellness supplements (focus, clarity, energy, resilience, long-term health)
  - Mention the website: opanutrition.com
  - Keep it compliant and non-medical (no disease claims)
- Engagement CTA:
  - Ask listeners to subscribe
  - Ask them to comment where they’re listening from
  - Ask them to share what they believe happened (open-ended)

Return ONLY the intro narration text. No headings.
""".strip()

def prompt_outro(topic: str, global_style: str, episode_notes: str) -> str:
    return f"""
Write an OUTRO for an audio-only investigative documentary.

TOPIC:
{topic}

STYLE (must follow):
{global_style}

NOTES:
{episode_notes}

OUTRO REQUIREMENTS:
- 60–90 seconds spoken
- Summarize the most established facts (briefly)
- Clearly label what remains unknown
- End with a powerful, thoughtful final line (not cheesy)
- Include sponsor mention again (premium, natural):
  - Sponsored by OPA Nutrition
  - Mention premium wellness supplements that support focus, clarity, and daily performance
  - Mention: opanutrition.com
- Engagement CTA:
  - Ask for comments: where they’re listening from
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
) -> str:
    words = int(target_minutes * 145)
    beats_block = "\n".join([f"- {b}" for b in beats]) if beats else "- (No beats provided)"
    return f"""
Write a single chapter of an audio-only investigative documentary.

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

MUST COVER THESE BEATS:
{beats_block}

RULES:
- Calm, authoritative, investigative tone
- No hype, no jokes, no sensationalism
- No visual references (audio-only)
- Use concrete details where appropriate (dates, names, locations) but do not invent facts
- If something is uncertain or disputed, label it as such
- End with a subtle forward hook into the next chapter (1–2 sentences)

Return ONLY narration text. No headings, no bullet points.
""".strip()


# ----------------------------
# UI — Episode setup
# ----------------------------
st.header("1️⃣ Episode Setup")

colA, colB = st.columns([2, 1])

with colA:
    topic = st.text_input("Episode topic", value=st.session_state.topic)
    st.session_state.topic = topic

with colB:
    total_minutes = st.slider("Total length (minutes)", 30, 180, 90, 5)
    chapter_minutes = st.slider("Minutes per chapter", 5, 20, 10, 1)

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
        "Focus on the historical timeline, key witnesses, military response, official statements, "
        "and how the narrative evolved. Separate facts from theories. Avoid invented details."
    ),
    height=120,
)

outline_btn = st.button("Generate Chapter Outline", type="primary")
if outline_btn:
    with st.spinner("Creating outline…"):
        data = json_outline_from_model(
            prompt_outline(topic, total_minutes, chapter_minutes, global_style, episode_notes)
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
            normalized.append({"title": title, "target_minutes": tmin, "beats": beats[:10]})

        st.session_state.outline = normalized
        st.session_state.chapter_count = len(normalized)
        st.session_state.built = None

        # Seed widget keys safely (init-only; no overwrites)
        ensure_text_key("intro", 0, "")
        for i in range(1, st.session_state.chapter_count + 1):
            ensure_text_key("chapter", i, "")
        ensure_text_key("outro", 0, "")

        st.success("Outline generated.")

if st.session_state.outline:
    st.subheader("📑 Outline")
    for i, ch in enumerate(st.session_state.outline, 1):
        st.markdown(f"**{i}. {ch['title']}** ({ch['target_minutes']} min)")


# ----------------------------
# Scripts (Intro + Chapters + Outro)
# ----------------------------
st.header("2️⃣ Scripts")

# Ensure keys exist BEFORE widgets are created
ensure_text_key("intro", 0, "")
ensure_text_key("outro", 0, "")
for i in range(1, st.session_state.chapter_count + 1):
    ensure_text_key("chapter", i, "")

left, right = st.columns([1, 1])

with left:
    if st.button("Generate Intro", type="secondary", disabled=not st.session_state.outline):
        with st.spinner("Writing intro…"):
            intro_text = safe_chat(prompt_intro(topic, global_style, episode_notes), temperature=0.55, tries=2)
            # Safe: happens before the widget is created later in this run
            st.session_state[text_key("intro", 0)] = intro_text
            st.session_state.built = None
        st.success("Intro generated.")

with right:
    if st.button("Generate Outro", type="secondary", disabled=not st.session_state.outline):
        with st.spinner("Writing outro…"):
            outro_text = safe_chat(prompt_outro(topic, global_style, episode_notes), temperature=0.55, tries=2)
            st.session_state[text_key("outro", 0)] = outro_text
            st.session_state.built = None
        st.success("Outro generated.")

st.subheader("Intro")
st.text_area(
    "Intro text",
    key=text_key("intro", 0),
    height=220,
    placeholder="Click Generate Intro to populate…",
)

st.divider()

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    gen_all = st.button("Generate All Chapters", disabled=not st.session_state.outline)
with c2:
    clear_all = st.button("Clear Chapters")
with c3:
    st.caption("Tip: Expand a chapter to generate/edit. The text box will always show what’s stored.")

if clear_all:
    # Safe: occurs before chapter widgets are created? (Not necessarily)
    # Therefore: do NOT directly assign after widgets exist in the same run.
    # Use a rerun-safe pattern: set a flag, then clear keys before rendering editors.
    st.session_state["_clear_chapters_requested"] = True
    st.session_state.built = None
    st.rerun()

if st.session_state.get("_clear_chapters_requested"):
    for i in range(1, st.session_state.chapter_count + 1):
        k = text_key("chapter", i)
        # This happens at the top of the rerun BEFORE widgets render.
        st.session_state[k] = ""
    st.session_state["_clear_chapters_requested"] = False
    st.success("Chapters cleared.")

if gen_all and st.session_state.outline:
    with st.spinner("Writing all chapters…"):
        for i, ch in enumerate(st.session_state.outline, 1):
            txt = safe_chat(
                prompt_chapter(topic, ch["title"], ch["target_minutes"], ch["beats"], global_style, episode_notes),
                temperature=0.6,
                tries=2,
            )
            st.session_state[text_key("chapter", i)] = txt
        st.session_state.built = None
    st.success("All chapters generated.")

if st.session_state.outline:
    for i, ch in enumerate(st.session_state.outline, 1):
        with st.expander(f"Chapter {i}: {ch['title']}", expanded=(i == 1)):
            cc1, cc2 = st.columns([1, 3])
            with cc1:
                st.caption(f"Target: ~{ch['target_minutes']} min")
                if st.button(f"Generate Chapter {i}", key=f"btn_gen_ch_{i}"):
                    with st.spinner(f"Writing Chapter {i}…"):
                        txt = safe_chat(
                            prompt_chapter(topic, ch["title"], ch["target_minutes"], ch["beats"], global_style, episode_notes),
                            temperature=0.6,
                            tries=2,
                        )
                        st.session_state[text_key("chapter", i)] = txt
                        st.session_state.built = None
                    st.success(f"Chapter {i} generated.")
            with cc2:
                st.text_area(
                    f"Chapter {i} text",
                    key=text_key("chapter", i),
                    height=320,
                    placeholder="Click Generate Chapter to populate…",
                )

st.divider()

st.subheader("Outro")
st.text_area(
    "Outro text",
    key=text_key("outro", 0),
    height=220,
    placeholder="Click Generate Outro to populate…",
)


# ----------------------------
# Audio build
# ----------------------------
st.header("3️⃣ Create Audio (Onyx + Music)")

tts_instructions = st.text_area(
    "TTS delivery instructions",
    value=(
        "British-leaning neutral delivery if possible, but natural and not forced. "
        "Calm, authoritative, restrained documentary narration. "
        "Measured pace with subtle pauses. Crisp consonants. No exaggerated emotion."
    ),
    height=110,
)

speed = st.slider("Narration speed (guidance)", 0.85, 1.10, 1.00, 0.01)
music_db = st.slider("Music volume (dB)", -35, -10, -24, 1)
fade_s = st.slider("Music fade in/out (seconds)", 2, 12, 6, 1)

music_file = st.file_uploader("Upload background music (MP3/WAV)", type=["mp3", "wav"])

def has_any_script() -> bool:
    if get_text("intro", 0).strip():
        return True
    for i in range(1, st.session_state.chapter_count + 1):
        if get_text("chapter", i).strip():
            return True
    if get_text("outro", 0).strip():
        return True
    return False

build_disabled = (not st.session_state.outline) or (not has_any_script())
build = st.button("Generate Audio + Export", type="primary", disabled=build_disabled)

if build:
    try:
        with tempfile.TemporaryDirectory() as td:
            chosen_music_path = None
            if music_file:
                chosen_music_path = os.path.join(td, "music_upload")
                ext = ".mp3" if music_file.type == "audio/mpeg" else ".wav"
                chosen_music_path += ext
                with open(chosen_music_path, "wb") as f:
                    f.write(music_file.read())
            elif os.path.exists(DEFAULT_MUSIC_PATH):
                chosen_music_path = DEFAULT_MUSIC_PATH

            if not chosen_music_path:
                st.error("Please upload a music file (or ensure DEFAULT_MUSIC_PATH exists).")
                st.stop()

            segments: List[Tuple[str, str]] = []
            if get_text("intro", 0).strip():
                segments.append(("intro", get_text("intro", 0).strip()))

            for i in range(1, st.session_state.chapter_count + 1):
                txt = get_text("chapter", i).strip()
                if txt:
                    title = st.session_state.outline[i - 1]["title"]
                    segments.append((f"chapter_{i:02d}_{clean_filename(title)}", txt))

            if get_text("outro", 0).strip():
                segments.append(("outro", get_text("outro", 0).strip()))

            if not segments:
                st.error("No script text found to build audio.")
                st.stop()

            st.info("Generating TTS and mixing background music…")
            progress = st.progress(0)
            total = len(segments)

            per_segment_mp3: Dict[str, bytes] = {}
            mixed_wavs: List[str] = []

            for idx, (slug, txt) in enumerate(segments, start=1):
                voice_wav = os.path.join(td, f"{slug}_voice.wav")
                mixed_wav = os.path.join(td, f"{slug}_mixed.wav")
                out_mp3 = os.path.join(td, f"{slug}.mp3")

                clean_txt = sanitize_for_tts(txt)
                tts_to_wav(clean_txt, tts_instructions, speed, voice_wav)

                mix_music_under_voice(voice_wav, chosen_music_path, mixed_wav, music_db=music_db, fade_s=fade_s)
                wav_to_mp3(mixed_wav, out_mp3)

                with open(out_mp3, "rb") as f:
                    per_segment_mp3[slug] = f.read()

                mixed_wavs.append(mixed_wav)
                progress.progress(min(1.0, idx / total))

            full_wav = os.path.join(td, "full_episode.wav")
            full_mp3 = os.path.join(td, "full_episode.mp3")
            concat_wavs(mixed_wavs, full_wav)
            wav_to_mp3(full_wav, full_mp3)

            with open(full_mp3, "rb") as f:
                full_mp3_bytes = f.read()

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("scripts/topic.txt", topic)
                z.writestr("scripts/intro.txt", get_text("intro", 0))

                for i in range(1, st.session_state.chapter_count + 1):
                    title = st.session_state.outline[i - 1]["title"]
                    z.writestr(f"scripts/chapter_{i:02d}_{clean_filename(title)}.txt", get_text("chapter", i))

                z.writestr("scripts/outro.txt", get_text("outro", 0))

                for slug, b in per_segment_mp3.items():
                    z.writestr(f"audio/{slug}.mp3", b)
                z.writestr("audio/full_episode.mp3", full_mp3_bytes)

            zip_buf.seek(0)
            st.session_state.built = {"zip": zip_buf.getvalue(), "full_mp3": full_mp3_bytes}
            st.success("Done! Download below.")
    except Exception as e:
        st.error(f"Audio build failed: {e}")


# ----------------------------
# Downloads
# ----------------------------
st.header("4️⃣ Downloads")

if st.session_state.built:
    st.download_button(
        "⬇️ Download ZIP (scripts + segment MP3s + full episode MP3)",
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
