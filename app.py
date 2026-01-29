# ============================
# PART 1/5 — Core Setup, Sidebar, Clients, FFmpeg, Text + Script Parsing Utilities
# ============================
# app.py — UAPpress Documentary TTS Studio (Script → Audio) + OPTIONAL Master Prompt → Script
#
# REQUIREMENTS (requirements.txt):
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9
#
# DESIGN:
# - Primary mode: Script-to-audio production tool
# - Optional mode: Master Prompt → generates STRICT template script → auto-fills boxes
# - Streamlit Cloud safe: API key entered manually each run (session-only)

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
from typing import Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary TTS Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio")
st.caption("Script → audio production. (Optional) Master Prompt → Script → Audio.")


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
# Sidebar — OpenAI API Key (session-only) + Settings
# ----------------------------
with st.sidebar:
    st.header("🔐 Keys (not saved)")
    st.caption("Enter your OpenAI API key each run. Stored only in session memory.")

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
    st.session_state.setdefault("TTS_MODEL", "gpt-4o-mini-tts")
    st.session_state.setdefault("TTS_VOICE", "onyx")
    st.text_input("TTS model", key="TTS_MODEL")
    st.text_input("TTS voice", key="TTS_VOICE")

    st.session_state.setdefault("TTS_SPEED", 1.0)
    st.slider("Speed (reserved)", 0.75, 1.25, float(st.session_state["TTS_SPEED"]), 0.05, key="TTS_SPEED")

    st.divider()
    st.header("🧾 Script Generation (optional)")
    st.caption("Enable only if you want: Master Prompt → STRICT script → auto-fill boxes.")

    st.session_state.setdefault("ENABLE_SCRIPT_GEN", False)
    st.checkbox("Enable script generation", key="ENABLE_SCRIPT_GEN")

    st.session_state.setdefault("SCRIPT_MODEL", "gpt-4o-mini")
    st.text_input("Script model", key="SCRIPT_MODEL")

    st.session_state.setdefault("DEFAULT_CHAPTERS", 8)
    st.slider("Default chapters", 3, 20, int(st.session_state["DEFAULT_CHAPTERS"]), 1, key="DEFAULT_CHAPTERS")

    st.divider()
    st.header("🎧 Music Bed (optional)")
    st.session_state.setdefault("DEFAULT_MUSIC_PATH", "/mnt/data/dark-ambient-soundscape-music-409350.mp3")
    st.text_input(
        "Default music path",
        key="DEFAULT_MUSIC_PATH",
        help="Used if you don't upload a music file.",
    )

    st.session_state.setdefault("MUSIC_DB", -24)
    st.session_state.setdefault("MUSIC_FADE_S", 6)
    st.slider("Music level (dB)", -36, -10, int(st.session_state["MUSIC_DB"]), 1, key="MUSIC_DB")
    st.slider("Music fade (sec)", 2, 12, int(st.session_state["MUSIC_FADE_S"]), 1, key="MUSIC_FADE_S")

    st.divider()
    st.header("🧠 Cache (recommended)")
    st.session_state.setdefault("ENABLE_TTS_CACHE", True)
    st.session_state.setdefault("CACHE_DIR", ".uappress_tts_cache")
    st.checkbox("Enable TTS cache", key="ENABLE_TTS_CACHE")
    st.text_input("Cache directory", key="CACHE_DIR")
    st.caption("Cache stores WAV chunks by hash so reruns are much faster.")


# ----------------------------
# OpenAI client (requires sidebar key)
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


def chunk_text(text: str, max_chars: int = 2800) -> List[str]:
    """
    Conservative chunking for TTS stability.
    Splits on paragraph breaks; keeps chunks under max_chars.
    """
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


def sanitize_for_tts(text: str) -> str:
    """Remove invisible Unicode controls that can break TTS."""
    if not text:
        return ""
    t = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def strip_tts_directives(text: str) -> str:
    """Remove accidental voice/style directives from pasted/generated scripts."""
    if not text:
        return ""
    t = text
    t = re.sub(r"(?im)^\s*VOICE\s*DIRECTION.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*PACE\s*:.*$\n?", "", t)
    t = re.sub(r"(?im)^\s*STYLE\s*:.*$\n?", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ----------------------------
# Streamlit state (single source of truth)
# ----------------------------
def text_key(kind: str, idx: int = 0) -> str:
    return f"text::{kind}::{idx}"


def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> None:
    k = text_key(kind, idx)
    st.session_state.setdefault(k, default or "")


def reset_script_text_fields(chapter_count: int) -> None:
    ensure_text_key("intro", 0, "")
    ensure_text_key("outro", 0, "")
    for i in range(1, chapter_count + 1):
        ensure_text_key("chapter", i, "")


def ensure_state() -> None:
    st.session_state.setdefault("episode_title", "Untitled Episode")

    # Master inputs
    st.session_state.setdefault("MASTER_PROMPT", "")
    st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")

    # Script boxes
    st.session_state.setdefault("chapter_count", 0)
    reset_script_text_fields(int(st.session_state["chapter_count"]))

    # Build metadata
    st.session_state.setdefault("built_sections", {})
    st.session_state.setdefault("last_build", None)


ensure_state()


# ----------------------------
# Master Script parsing (STRICT headings)
# ----------------------------
def _normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


_CHAPTER_RE = re.compile(r"(?i)^\s*chapter\s+(\d+)\s*(?:[:\-–—]\s*(.*))?\s*$")


def detect_chapters_and_titles(master_text: str) -> Dict[int, str]:
    """
    Detect CHAPTER N headings and optional titles from the heading line.
    Examples:
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
        titles[n] = (m.group(2) or "").strip()
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

    positions: List[int] = []
    if intro_idx is not None:
        positions.append(intro_idx)
    if outro_idx is not None:
        positions.append(outro_idx)
    positions += list(chapter_idx.values())
    positions = sorted(set(positions))

    next_pos: Dict[int, int] = {}
    for k in range(len(positions)):
        cur = positions[k]
        nxt = positions[k + 1] if (k + 1) < len(positions) else len(lines)
        next_pos[cur] = nxt

    def slice_block(start_i: Optional[int]) -> str:
        if start_i is None:
            return ""
        end_i = next_pos.get(start_i, len(lines))
        body = "\n".join(lines[start_i + 1 : end_i]).strip()  # excludes heading line
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


# ============================
# PART 2/4 — TTS Engine, FFmpeg Audio Ops, Cache, File Packaging
# ============================

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
    """
    Cache key includes model + voice + text. (Speed reserved but not applied yet.)
    """
    key = _sha1(f"model={cfg.model}|voice={cfg.voice}|text={text}")
    _ensure_dir(cfg.cache_dir)
    return os.path.join(cfg.cache_dir, f"tts_{key}.wav")


def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    """
    Concatenate WAV files with ffmpeg concat demuxer (no re-encode).
    """
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
    Mix a looping music bed under voice, with fades + loudnorm.
    Output: PCM WAV (safe for later MP3 encoding).
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


def tts_to_wav(
    *,
    text: str,
    out_wav: str,
    cfg: TTSConfig,
) -> None:
    """
    Generate WAV narration via OpenAI TTS and concatenate chunks.
    Uses caching per-chunk (massive speedup on reruns).
    """
    text = strip_tts_directives(sanitize_for_tts(text))
    if not text.strip():
        # Create a tiny silent wav so pipeline doesn’t explode downstream
        run_ffmpeg([FFMPEG, "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.1", out_wav])
        return

    chunks = chunk_text(text, max_chars=2800)
    wav_parts: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        for i, ch in enumerate(chunks, 1):
            payload = strip_tts_directives(sanitize_for_tts(ch))

            cache_hit_path = _tts_cache_path(cfg, payload) if cfg.enable_cache else ""
            if cfg.enable_cache and os.path.exists(cache_hit_path) and os.path.getsize(cache_hit_path) > 2000:
                wav_parts.append(cache_hit_path)
                continue

            # OpenAI TTS: audio.speech.create(model, voice, input, response_format="wav") :contentReference[oaicite:1]{index=1}
            r = client.audio.speech.create(
                model=cfg.model,
                voice=cfg.voice,
                response_format="wav",
                input=payload,
            )

            part_path = os.path.join(td, f"part_{i:02d}.wav")
            with open(part_path, "wb") as f:
                f.write(r.read())

            # Store into cache
            if cfg.enable_cache:
                try:
                    _ensure_dir(cfg.cache_dir)
                    with open(cache_hit_path, "wb") as f2:
                        with open(part_path, "rb") as f1:
                            f2.write(f1.read())
                    wav_parts.append(cache_hit_path)
                except Exception:
                    wav_parts.append(part_path)
            else:
                wav_parts.append(part_path)

        concat_wavs(wav_parts, out_wav)


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
    Returns list of (section_id, text) in playback order.
    section_id is used for filenames.
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
# UI — Audio Assets
# ============================
st.header("3️⃣ Create Audio (Onyx + Music)")

music_file = st.file_uploader(
    "Optional music bed (mp3/wav). If omitted, Default music path is used.",
    type=["mp3", "wav", "m4a"],
)

music_path = ""
if music_file is not None:
    # Save uploaded file to temp for ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{music_file.name}") as tf:
        tf.write(music_file.read())
        music_path = tf.name
else:
    music_path = st.session_state.get("DEFAULT_MUSIC_PATH", "") or ""

use_music = st.checkbox("Mix music under voice", value=True, help="If enabled, a music bed is mixed under narration.")

cfg = TTSConfig(
    model=st.session_state.get("TTS_MODEL", "gpt-4o-mini-tts"),
    voice=st.session_state.get("TTS_VOICE", "onyx"),
    speed=float(st.session_state.get("TTS_SPEED", 1.0)),
    enable_cache=bool(st.session_state.get("ENABLE_TTS_CACHE", True)),
    cache_dir=str(st.session_state.get("CACHE_DIR", ".uappress_tts_cache")),
)

st.caption(f"TTS: model={cfg.model} | voice={cfg.voice} | cache={'on' if cfg.enable_cache else 'off'}")

# ============================
# PART 3/4 — Build Section Audio (Intro/Chapters/Outro) + Downloads
# ============================

# Where outputs live (Streamlit Cloud-safe temp folder)
def _workdir() -> str:
    st.session_state.setdefault("WORKDIR", tempfile.mkdtemp(prefix="uappress_tts_"))
    return st.session_state["WORKDIR"]


def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def section_output_paths(section_id: str, episode_slug: str) -> Dict[str, str]:
    wd = _workdir()
    base = f"{episode_slug}__{section_id}"
    return {
        "voice_wav": os.path.join(wd, f"{base}__voice.wav"),
        "mixed_wav": os.path.join(wd, f"{base}__mix.wav"),
        "mp3": os.path.join(wd, f"{base}.mp3"),
        "txt": os.path.join(wd, f"{base}.txt"),
    }


def episode_slug() -> str:
    title = st.session_state.get("episode_title", "Untitled Episode")
    return clean_filename(title)


def write_text_file(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")


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

    # 1) TTS → voice WAV (concat chunks)
    tts_to_wav(text=cleaned, out_wav=paths["voice_wav"], cfg=cfg)

    # 2) Optional music mix → WAV
    final_wav = paths["voice_wav"]
    if mix_music:
        if not music_path or not os.path.exists(music_path):
            raise FileNotFoundError("Music bed path not found. Upload music or set a valid Default music path in the sidebar.")
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


st.subheader("🎙️ Build audio for each section")

if not has_any_script():
    st.info("Paste your master script and fill the boxes before building audio.")
else:
    # Build controls
    b1, b2, b3 = st.columns([1, 1, 2])
    with b1:
        build_sections_btn = st.button("Build ALL Sections", type="primary")
    with b2:
        clear_build_btn = st.button("Clear Built Audio")
    with b3:
        st.caption("Build outputs go to a temp workdir. Use ZIP download when done.")

    if clear_build_btn:
        # Nuke workdir contents (best-effort)
        wd = _workdir()
        try:
            for name in os.listdir(wd):
                _safe_remove(os.path.join(wd, name))
        except Exception:
            pass
        st.session_state["last_build"] = None
        st.success("Cleared built audio from workdir.")

    # Per-section quick build buttons
    st.caption("Optional: build one section at a time")
    sec_list = section_texts()
    pick = st.selectbox(
        "Choose a section to build",
        options=[sid for sid, _ in sec_list],
        index=0,
        help="Build a single section MP3 so you can QC before generating the full episode.",
    )
    build_one_btn = st.button("Build Selected Section")

    # Shared parameters
    mix_music_flag = bool(use_music)
    music_db_i = int(st.session_state.get("MUSIC_DB", -24))
    fade_s_i = int(st.session_state.get("MUSIC_FADE_S", 6))

    # Storage for built file references (in session_state)
    st.session_state.setdefault("built_sections", {})  # {section_id: {paths...}}

    def _render_downloads_for_section(section_id: str, paths: Dict[str, str]) -> None:
        if not paths:
            return
        mp3_path = paths.get("mp3", "")
        txt_path = paths.get("txt", "")
        colx, coly, colz = st.columns([1, 1, 2])

        with colx:
            if mp3_path and os.path.exists(mp3_path):
                with open(mp3_path, "rb") as f:
                    st.download_button(
                        label=f"Download {section_id}.mp3",
                        data=f.read(),
                        file_name=os.path.basename(mp3_path),
                        mime="audio/mpeg",
                        key=f"dl_mp3_{section_id}",
                    )

        with coly:
            if txt_path and os.path.exists(txt_path):
                with open(txt_path, "rb") as f:
                    st.download_button(
                        label=f"Download {section_id}.txt",
                        data=f.read(),
                        file_name=os.path.basename(txt_path),
                        mime="text/plain",
                        key=f"dl_txt_{section_id}",
                    )

        with colz:
            # quick player
            if mp3_path and os.path.exists(mp3_path):
                st.audio(mp3_path)

    # Build selected section
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
                        mix_music=mix_music_flag,
                        music_path=music_path,
                        music_db=music_db_i,
                        fade_s=fade_s_i,
                    )
                    st.session_state["built_sections"][pick] = out_paths
                    st.success(f"Built: {pick}")
                except Exception as e:
                    st.error(str(e))

    # Build all sections
    if build_sections_btn:
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
                        mix_music=mix_music_flag,
                        music_path=music_path,
                        music_db=music_db_i,
                        fade_s=fade_s_i,
                    )
                    built[sid] = out_paths
                except Exception as e:
                    errors.append((sid, str(e)))

            # Merge into state
            st.session_state["built_sections"].update(built)
            st.session_state["last_build"] = {
                "ts": time.time(),
                "episode": st.session_state.get("episode_title", "Untitled Episode"),
                "sections_built": sorted(list(built.keys())),
                "errors": errors,
                "mix_music": mix_music_flag,
                "music_path": music_path,
                "tts_model": cfg.model,
                "tts_voice": cfg.voice,
                "cache": cfg.enable_cache,
            }

            if errors:
                st.warning("Some sections failed:\n" + "\n".join([f"- {sid}: {msg}" for sid, msg in errors]))
            else:
                st.success("All available sections built.")

    # Show built sections downloads
    built_sections: Dict[str, Dict[str, str]] = st.session_state.get("built_sections", {}) or {}
    if built_sections:
        st.subheader("✅ Built Sections")
        for sid in [x for x, _ in sec_list]:
            if sid not in built_sections:
                continue
            with st.expander(f"{sid} — downloads & preview", expanded=(sid == "00_intro")):
                _render_downloads_for_section(sid, built_sections[sid])

        # ZIP download of all section MP3s (+ txt)
        st.subheader("📦 Download Sections ZIP")
        zip_btn = st.button("Build ZIP of section MP3s + TXT")
        if zip_btn:
            slug = episode_slug()
            files: List[Tuple[str, str]] = []

            # Include per-section mp3 + txt if present
            for sid, paths in built_sections.items():
                mp3_path = paths.get("mp3", "")
                txt_path = paths.get("txt", "")
                if mp3_path and os.path.exists(mp3_path):
                    files.append((mp3_path, os.path.basename(mp3_path)))
                if txt_path and os.path.exists(txt_path):
                    files.append((txt_path, os.path.basename(txt_path)))

            # Include build metadata
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

# ============================
# PART 4/4 — Build Full Episode MP3 (concat sections) + Final ZIP
# ============================

def concat_mp3s(mp3_paths: List[str], out_mp3: str) -> None:
    """
    Concatenate MP3 files safely using ffmpeg concat demuxer.
    This re-muxes into MP3 stream copy when possible.
    """
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
    """
    Concats built section MP3s in the correct order.
    """
    mp3s = []
    for sid in ordered_section_ids:
        paths = built_sections.get(sid) or {}
        p = paths.get("mp3", "")
        if p and os.path.exists(p):
            mp3s.append(p)

    if not mp3s:
        raise ValueError("No built section MP3s found to concatenate.")

    concat_mp3s(mp3s, out_mp3)


st.subheader("🎬 Build FULL Episode MP3 (from built sections)")

built_sections: Dict[str, Dict[str, str]] = st.session_state.get("built_sections", {}) or {}
ordered_ids = [sid for sid, _ in section_texts()]

ep_slug = episode_slug()
full_mp3_path = os.path.join(_workdir(), f"{ep_slug}__FULL.mp3")

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

st.divider()

st.subheader("📦 Final Delivery ZIP (FULL MP3 + Sections + Text)")

final_zip_btn = st.button("Build FINAL ZIP (everything)")
if final_zip_btn:
    slug = episode_slug()
    files: List[Tuple[str, str]] = []

    # Full MP3 if exists
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

    # Also include the raw master script you pasted (for archive)
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

# Optional: show debug info
with st.expander("🔎 Build Metadata (debug)", expanded=False):
    st.json(st.session_state.get("last_build", {}))
    st.caption(f"Workdir: {_workdir()}")
