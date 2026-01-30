# ============================
# PART 1/4 — App Shell + Sidebar Settings
# File: app.py
# ============================
from __future__ import annotations

import io
import os
import re
import zipfile
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import streamlit as st
import imageio_ffmpeg
from openai import OpenAI


APP_TITLE = "UAPpress — Manual Script → MP3 Studio"
DEFAULT_MODEL = "gpt-4o-mini-tts"  # TTS model (keep configurable if you want)
DEFAULT_VOICE = "onyx"            # your requested voice

# ----------------------------
# Basic helpers
# ----------------------------
def _ensure_state() -> None:
    """
    Initialize Streamlit session_state keys safely.
    """
    if "project_title" not in st.session_state:
        st.session_state.project_title = "Untitled Documentary"

    # Each chapter is a dict: {"id": "<stable-id>", "text": ""}
    if "chapters" not in st.session_state:
        st.session_state.chapters = []

    if "intro_text" not in st.session_state:
        st.session_state.intro_text = ""

    if "outro_text" not in st.session_state:
        st.session_state.outro_text = ""

    if "generated_files" not in st.session_state:
        # maps filename -> bytes
        st.session_state.generated_files = {}

    if "last_build_log" not in st.session_state:
        st.session_state.last_build_log = ""


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "project"


def _stable_chapter_id() -> str:
    """
    Create a stable-ish ID without importing uuid (keep dependencies minimal).
    Uses a short random token from os.urandom.
    """
    return os.urandom(6).hex()


def _ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class AudioOptions:
    api_key: str
    model: str
    voice: str
    output_bitrate: str
    add_music: bool
    music_bytes: Optional[bytes]
    music_volume: float          # 0.0–1.0 (applied as multiplier)
    speech_gain_db: float        # e.g., 0, +2, -2
    music_loop: bool             # loop music to match speech if needed
    padding_ms: int              # silence at start/end


def _sidebar_controls() -> AudioOptions:
    st.sidebar.header("Settings")

    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        value=os.environ.get("OPENAI_API_KEY", ""),
        type="password",
        help="Required to generate Onyx TTS audio.",
        key="openai_api_key_input",
    )

    st.sidebar.subheader("Voice")
    model = st.sidebar.text_input("TTS model", value=DEFAULT_MODEL)
    voice = st.sidebar.text_input("Voice", value=DEFAULT_VOICE)

    st.sidebar.subheader("Audio Output")
    output_bitrate = st.sidebar.selectbox(
        "MP3 bitrate",
        options=["128k", "160k", "192k", "256k", "320k"],
        index=2,
        help="Higher bitrate = larger files.",
    )

    st.sidebar.subheader("Music (Optional)")
    add_music = st.sidebar.checkbox("Mix background music", value=False)
    music_bytes = None
    if add_music:
        music_file = st.sidebar.file_uploader(
            "Upload music file (mp3/wav)",
            type=["mp3", "wav", "m4a"],
            accept_multiple_files=False,
        )
        if music_file is not None:
            music_bytes = music_file.read()

        music_volume = st.sidebar.slider(
            "Music volume",
            min_value=0.0,
            max_value=1.0,
            value=0.18,
            step=0.01,
            help="How loud the music is relative to narration.",
        )
        music_loop = st.sidebar.checkbox(
            "Loop music to match narration length",
            value=True,
            help="If music is shorter than narration, loop it.",
        )
    else:
        music_volume = 0.0
        music_loop = True

    st.sidebar.subheader("Fine Tuning")
    speech_gain_db = st.sidebar.slider(
        "Narration gain (dB)",
        min_value=-6.0,
        max_value=6.0,
        value=0.0,
        step=0.5,
        help="Boost/cut narration volume before mixing.",
    )
    padding_ms = st.sidebar.selectbox(
        "Add silence padding (ms)",
        options=[0, 150, 250, 500, 750, 1000],
        index=2,
        help="Adds a little space before/after each section.",
    )

    return AudioOptions(
        api_key=api_key.strip(),
        model=model.strip() or DEFAULT_MODEL,
        voice=voice.strip() or DEFAULT_VOICE,
        output_bitrate=output_bitrate,
        add_music=add_music,
        music_bytes=music_bytes,
        music_volume=music_volume,
        speech_gain_db=float(speech_gain_db),
        music_loop=music_loop,
        padding_ms=int(padding_ms),
    )


# ----------------------------
# App page setup
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

_ensure_state()
opts = _sidebar_controls()

st.text_input("Project title", key="project_title")

st.caption(
    "Paste your Intro, Chapter 1..N, and Outro scripts below. "
    "Click **Create Audio** to generate separate MP3s for each non-empty section."
)

# ============================
# PART 2/4 — Script Builder UI (Intro + Dynamic Chapters + Outro)
# (append below Part 1 in app.py)
# ============================

def _add_chapter() -> None:
    st.session_state.chapters.append({"id": _stable_chapter_id(), "text": ""})


def _remove_chapter(idx: int) -> None:
    if 0 <= idx < len(st.session_state.chapters):
        st.session_state.chapters.pop(idx)


def _move_chapter(idx: int, direction: int) -> None:
    """
    direction: -1 up, +1 down
    """
    chapters = st.session_state.chapters
    new_idx = idx + direction
    if 0 <= idx < len(chapters) and 0 <= new_idx < len(chapters):
        chapters[idx], chapters[new_idx] = chapters[new_idx], chapters[idx]


def _section_word_count(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def _est_minutes(word_count: int) -> float:
    # rough narration pace ~150 wpm
    if word_count <= 0:
        return 0.0
    return word_count / 150.0


# --- Layout
left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Script Sections")

    st.markdown("### Intro")
    st.text_area(
        "Intro text (read exactly as pasted)",
        key="intro_text",
        height=220,
        placeholder="Paste your Intro here…",
        label_visibility="collapsed",
    )
    wc_intro = _section_word_count(st.session_state.intro_text)
    st.caption(f"Intro: {wc_intro} words • ~{_est_minutes(wc_intro):.1f} min")

    st.markdown("---")
    st.markdown("### Chapters")

    c1, c2 = st.columns([0.2, 0.8])
    with c1:
        if st.button("+ Add Chapter", use_container_width=True):
            _add_chapter()
    with c2:
        st.caption("Chapters are numbered automatically (Chapter 1, Chapter 2, …).")

    if len(st.session_state.chapters) == 0:
        st.info("No chapters yet. Click **+ Add Chapter** to add Chapter 1.")

    for idx, ch in enumerate(st.session_state.chapters):
        chap_num = idx + 1
        with st.container(border=True):
            header_cols = st.columns([0.55, 0.15, 0.15, 0.15])
            header_cols[0].markdown(f"**Chapter {chap_num}**")

            up_disabled = idx == 0
            down_disabled = idx == len(st.session_state.chapters) - 1

            if header_cols[1].button("↑", key=f"ch_up_{ch['id']}", disabled=up_disabled):
                _move_chapter(idx, -1)
                st.rerun()
            if header_cols[2].button("↓", key=f"ch_dn_{ch['id']}", disabled=down_disabled):
                _move_chapter(idx, +1)
                st.rerun()
            if header_cols[3].button("Remove", key=f"ch_rm_{ch['id']}"):
                _remove_chapter(idx)
                st.rerun()

            st.text_area(
                f"Chapter {chap_num} text",
                key=f"ch_text_{ch['id']}",
                height=220,
                placeholder=f"Paste Chapter {chap_num} here…",
                label_visibility="collapsed",
            )

            # Keep backing store synced (stable ID prevents scrambling)
            ch["text"] = st.session_state.get(f"ch_text_{ch['id']}", "")

            wc = _section_word_count(ch["text"])
            st.caption(f"Chapter {chap_num}: {wc} words • ~{_est_minutes(wc):.1f} min")

    st.markdown("---")
    st.markdown("### Outro")
    st.text_area(
        "Outro text (read exactly as pasted)",
        key="outro_text",
        height=220,
        placeholder="Paste your Outro here…",
        label_visibility="collapsed",
    )
    wc_outro = _section_word_count(st.session_state.outro_text)
    st.caption(f"Outro: {wc_outro} words • ~{_est_minutes(wc_outro):.1f} min")

with right:
    st.subheader("Build")

    # Quick status summary
    total_words = wc_intro + wc_outro + sum(_section_word_count(c["text"]) for c in st.session_state.chapters)
    total_min = _est_minutes(total_words)
    st.metric("Total words", f"{total_words}")
    st.metric("Estimated narration length", f"~{total_min:.1f} min")

    st.markdown("---")
    st.markdown("#### Actions")

    # Buttons (wired up in Part 3/4)
    st.button("Create Audio (MP3s)", type="primary", key="create_audio_btn")
    st.button("Download All MP3s (ZIP)", key="download_zip_btn")
    st.button("Export Script Backup (ZIP)", key="export_scripts_btn")

    st.markdown("---")
    st.markdown("#### Last build log")
    st.text_area(
        "Log",
        value=st.session_state.last_build_log or "",
        height=260,
        disabled=True,
        label_visibility="collapsed",
    )

# ============================
# PART 3/4 — Audio Generation Pipeline (Sequential, Crash-Safe) — FIXED (TTS)
# Replace your entire Part 3 block with this.
# ============================

def _require_api_key(api_key: str) -> bool:
    if not api_key:
        st.error("OpenAI API key is required (set it in the sidebar).")
        return False
    return True


def _collect_sections() -> List[Tuple[str, str]]:
    sections: List[Tuple[str, str]] = []

    intro = (st.session_state.intro_text or "").strip()
    if intro:
        sections.append(("Intro", intro))

    for idx, ch in enumerate(st.session_state.chapters):
        text = (ch.get("text") or "").strip()
        if text:
            sections.append((f"Chapter_{idx+1:02d}", text))

    outro = (st.session_state.outro_text or "").strip()
    if outro:
        sections.append(("Outro", outro))

    return sections


def _read_audio_response(resp) -> bytes:
    """
    SDK-safe: different OpenAI SDK versions return different response objects.
    """
    if resp is None:
        raise RuntimeError("Empty response from TTS API")

    # Newer SDK often provides .read()
    if hasattr(resp, "read") and callable(getattr(resp, "read")):
        b = resp.read()
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)

    # Some SDKs expose .content
    if hasattr(resp, "content"):
        b = getattr(resp, "content")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)

    # Some SDKs might already return bytes
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)

    # Fallback: try bytes() coercion
    try:
        return bytes(resp)
    except Exception:
        raise RuntimeError(f"Unrecognized TTS response type: {type(resp)}")


def _openai_tts_mp3_bytes(*, client: OpenAI, model: str, voice: str, text: str) -> bytes:
    """
    Generates MP3 bytes from OpenAI TTS.
    FIX: Use SDK-safe parameter handling (response_format vs format).
    Reads text exactly as provided.
    """
    # Try the most common/current parameter name first:
    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3",
        )
        return _read_audio_response(resp)
    except TypeError:
        # Older/newer SDK mismatch: try 'format'
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            format="mp3",
        )
        return _read_audio_response(resp)


def _run_ffmpeg(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout or ""
    except Exception as e:
        return 1, f"FFmpeg execution error: {e}"


def _apply_padding_mp3(mp3_in: str, mp3_out: str, padding_ms: int) -> Tuple[bool, str]:
    if padding_ms <= 0:
        try:
            with open(mp3_in, "rb") as f_in, open(mp3_out, "wb") as f_out:
                f_out.write(f_in.read())
            return True, ""
        except Exception as e:
            return False, str(e)

    ffmpeg = _ffmpeg_path()
    pad_s = padding_ms / 1000.0

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={pad_s}",
        "-i", mp3_in,
        "-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={pad_s}",
        "-filter_complex", "[0:a][1:a][2:a]concat=n=3:v=0:a=1[a]",
        "-map", "[a]",
        "-c:a", "libmp3lame",
        mp3_out
    ]
    code, out = _run_ffmpeg(cmd)
    return (code == 0), out


def _gain_and_mix_music(
    *,
    speech_mp3: str,
    music_path: str,
    out_mp3: str,
    speech_gain_db: float,
    music_volume: float,
    loop_music: bool,
    bitrate: str,
) -> Tuple[bool, str]:
    ffmpeg = _ffmpeg_path()

    music_input = ["-i", music_path]
    if loop_music:
        music_input = ["-stream_loop", "-1", "-i", music_path]

    cmd = [
        ffmpeg, "-y",
        "-i", speech_mp3,
        *music_input,
        "-filter_complex",
        (
            f"[0:a]volume={speech_gain_db}dB[s];"
            f"[1:a]volume={music_volume}[m];"
            f"[s][m]amix=inputs=2:duration=shortest:dropout_transition=2[a]"
        ),
        "-map", "[a]",
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        out_mp3,
    ]
    code, out = _run_ffmpeg(cmd)
    return (code == 0), out


def _maybe_gain_only(
    *,
    speech_mp3: str,
    out_mp3: str,
    speech_gain_db: float,
    bitrate: str,
) -> Tuple[bool, str]:
    if abs(speech_gain_db) < 0.01:
        try:
            with open(speech_mp3, "rb") as f_in, open(out_mp3, "wb") as f_out:
                f_out.write(f_in.read())
            return True, ""
        except Exception as e:
            return False, str(e)

    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", speech_mp3,
        "-filter:a", f"volume={speech_gain_db}dB",
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        out_mp3,
    ]
    code, out = _run_ffmpeg(cmd)
    return (code == 0), out


def _build_all_mp3s(opts: AudioOptions) -> None:
    if not _require_api_key(opts.api_key):
        return

    sections = _collect_sections()
    if not sections:
        st.warning("Nothing to generate — all sections are empty.")
        return

    st.session_state.generated_files = {}
    st.session_state.last_build_log = ""

    client = OpenAI(api_key=opts.api_key)

    proj_slug = _slugify(st.session_state.project_title)
    log_lines: List[str] = []

    with tempfile.TemporaryDirectory() as td:
        music_path = None
        if opts.add_music:
            if not opts.music_bytes:
                st.error("Music mixing is enabled, but no music file was uploaded in the sidebar.")
                return
            music_path = os.path.join(td, "bg_music")
            with open(music_path, "wb") as f:
                f.write(opts.music_bytes)

        progress = st.progress(0)
        status = st.empty()

        for i, (base_name, text) in enumerate(sections, start=1):
            status.write(f"Generating {base_name}… ({i}/{len(sections)})")

            # 1) TTS -> mp3
            try:
                raw_mp3 = _openai_tts_mp3_bytes(
                    client=client,
                    model=opts.model,
                    voice=opts.voice,
                    text=text,
                )
            except Exception as e:
                log_lines.append(f"[ERROR] {base_name}: OpenAI TTS failed: {repr(e)}")
                st.session_state.last_build_log = "\n".join(log_lines)
                progress.empty()
                status.empty()
                st.error(f"TTS failed on {base_name}. See log.")
                return

            raw_path = os.path.join(td, f"{base_name}_raw.mp3")
            with open(raw_path, "wb") as f:
                f.write(raw_mp3)

            # 2) Padding (optional)
            padded_path = os.path.join(td, f"{base_name}_padded.mp3")
            ok, out = _apply_padding_mp3(raw_path, padded_path, opts.padding_ms)
            if not ok:
                log_lines.append(f"[ERROR] {base_name}: padding failed\n{out}")
                st.session_state.last_build_log = "\n".join(log_lines)
                progress.empty()
                status.empty()
                st.error(f"Padding failed on {base_name}. See log.")
                return

            # 3) Gain and/or music mix
            final_path = os.path.join(td, f"{base_name}.mp3")
            if opts.add_music and music_path is not None:
                ok, out = _gain_and_mix_music(
                    speech_mp3=padded_path,
                    music_path=music_path,
                    out_mp3=final_path,
                    speech_gain_db=opts.speech_gain_db,
                    music_volume=opts.music_volume,
                    loop_music=opts.music_loop,
                    bitrate=opts.output_bitrate,
                )
                if not ok:
                    log_lines.append(f"[ERROR] {base_name}: music mix failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log_lines)
                    progress.empty()
                    status.empty()
                    st.error(f"Music mix failed on {base_name}. See log.")
                    return
            else:
                ok, out = _maybe_gain_only(
                    speech_mp3=padded_path,
                    out_mp3=final_path,
                    speech_gain_db=opts.speech_gain_db,
                    bitrate=opts.output_bitrate,
                )
                if not ok:
                    log_lines.append(f"[ERROR] {base_name}: gain/apply failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log_lines)
                    progress.empty()
                    status.empty()
                    st.error(f"Audio processing failed on {base_name}. See log.")
                    return

            # 4) Read final bytes into memory for downloads
            try:
                with open(final_path, "rb") as f:
                    final_bytes = f.read()
            except Exception as e:
                log_lines.append(f"[ERROR] {base_name}: could not read output mp3: {repr(e)}")
                st.session_state.last_build_log = "\n".join(log_lines)
                progress.empty()
                status.empty()
                st.error(f"Failed reading output for {base_name}. See log.")
                return

            filename = f"{proj_slug}__{base_name}.mp3"
            st.session_state.generated_files[filename] = final_bytes
            log_lines.append(f"[OK] {filename} ({len(final_bytes)/1024/1024:.2f} MB)")

            progress.progress(i / len(sections))

        status.write("Done.")
        progress.empty()

    st.session_state.last_build_log = "\n".join(log_lines)
    st.success(f"Generated {len(st.session_state.generated_files)} MP3 file(s).")


# Wire the Create Audio button from Part 2
# IMPORTANT: Do NOT assign to st.session_state.create_audio_btn (widget-owned key).
if st.session_state.get("create_audio_btn"):
    _build_all_mp3s(opts)

# ============================
# PART 4/4 — Downloads + Packaging (ZIP) + Script Backup Export — FIXED
# ============================

def _zip_bytes(file_map: Dict[str, bytes]) -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, b in file_map.items():
            z.writestr(name, b)
    return bio.getvalue()


def _export_scripts_zip() -> bytes:
    proj_slug = _slugify(st.session_state.project_title)

    file_map: Dict[str, bytes] = {}
    intro = (st.session_state.intro_text or "").strip()
    outro = (st.session_state.outro_text or "").strip()

    if intro:
        file_map[f"{proj_slug}__Intro.txt"] = intro.encode("utf-8")

    for idx, ch in enumerate(st.session_state.chapters):
        text = (ch.get("text") or "").strip()
        if text:
            file_map[f"{proj_slug}__Chapter_{idx+1:02d}.txt"] = text.encode("utf-8")

    if outro:
        file_map[f"{proj_slug}__Outro.txt"] = outro.encode("utf-8")

    return _zip_bytes(file_map)


st.markdown("---")
st.subheader("Outputs")

generated = st.session_state.generated_files or {}

if not generated:
    st.caption("No MP3s generated yet.")
else:
    st.caption("Download individual MP3s or download everything as a ZIP.")

    for fname, b in generated.items():
        st.download_button(
            label=f"Download {fname}",
            data=b,
            file_name=fname,
            mime="audio/mpeg",
            use_container_width=True,
            key=f"dl_{fname}",
        )


# ZIP download trigger
# IMPORTANT: Do NOT assign to st.session_state.download_zip_btn (widget-owned key).
if st.session_state.get("download_zip_btn"):
    if not generated:
        st.warning("No MP3s to zip yet. Click **Create Audio** first.")
    else:
        zip_b = _zip_bytes(generated)
        proj_slug = _slugify(st.session_state.project_title)
        st.download_button(
            label="Download ALL MP3s as ZIP",
            data=zip_b,
            file_name=f"{proj_slug}__mp3s.zip",
            mime="application/zip",
            use_container_width=True,
            key="dl_all_zip_now",
        )


# Script export trigger
# IMPORTANT: Do NOT assign to st.session_state.export_scripts_btn (widget-owned key).
if st.session_state.get("export_scripts_btn"):
    zip_b = _export_scripts_zip()
    proj_slug = _slugify(st.session_state.project_title)
    st.download_button(
        label="Download Script Backup (ZIP)",
        data=zip_b,
        file_name=f"{proj_slug}__scripts.zip",
        mime="application/zip",
        use_container_width=True,
        key="dl_scripts_zip_now",
    )

