# ============================
# PART 1/4 — App Shell + Sidebar (Onyx HQ Voice-Only)
# File: app.py
# ============================
from __future__ import annotations

import io
import os
import re
import zipfile
import tempfile
import subprocess
from typing import Dict, List, Optional, Tuple

import streamlit as st
import imageio_ffmpeg
from openai import OpenAI


APP_TITLE = "UAPpress — Manual Script → HQ Onyx MP3 Studio"

# Voice-only defaults (high quality)
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_VOICE = "onyx"
DEFAULT_MP3_BITRATE = "320k"   # highest practical MP3 setting
DEFAULT_SR = 48000             # 48kHz for clean masters


# ----------------------------
# Helpers
# ----------------------------
def _ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()


def _run(cmd: List[str]) -> Tuple[int, str]:
    try:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return p.returncode, p.stdout or ""
    except Exception as e:
        return 1, f"Subprocess error: {e}"


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "project"


def _ensure_state() -> None:
    if "project_title" not in st.session_state:
        st.session_state.project_title = "Untitled Documentary"
    if "intro_text" not in st.session_state:
        st.session_state.intro_text = ""
    if "outro_text" not in st.session_state:
        st.session_state.outro_text = ""
    if "chapters" not in st.session_state:
        # list of dicts: {"id": "<stable>", "text": ""}
        st.session_state.chapters = []
    if "built" not in st.session_state:
        st.session_state.built = None
    if "last_build_log" not in st.session_state:
        st.session_state.last_build_log = ""


def _stable_id() -> str:
    return os.urandom(6).hex()


def _word_count(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return len(re.findall(r"\b\w+\b", text))


def _est_minutes(words: int) -> float:
    # ~150 wpm
    if words <= 0:
        return 0.0
    return words / 150.0


# ----------------------------
# Page
# ----------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

_ensure_state()

# ----------------------------
# Sidebar (HQ settings)
# ----------------------------
st.sidebar.header("Settings (HQ Voice-Only)")

api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password",
    help="Required to generate Onyx TTS audio.",
)

st.sidebar.subheader("Voice")
tts_model = st.sidebar.text_input("TTS model", value=DEFAULT_TTS_MODEL)
voice = st.sidebar.text_input("Voice", value=DEFAULT_VOICE)

st.sidebar.subheader("Quality")
mp3_bitrate = st.sidebar.selectbox(
    "MP3 bitrate",
    options=["192k", "256k", "320k"],
    index=2,
)
sample_rate = st.sidebar.selectbox(
    "Master sample rate",
    options=[44100, 48000],
    index=1,
    help="48kHz recommended; keeps masters clean and consistent.",
)

st.sidebar.subheader("Loudness (broadcast-style)")
target_lufs = st.sidebar.selectbox(
    "Target loudness (LUFS-I)",
    options=[-18, -16, -14],
    index=1,
    help="For YouTube/podcast-style delivery, -16 LUFS is a solid target.",
)
true_peak = st.sidebar.selectbox(
    "True peak limit (dBTP)",
    options=[-2.0, -1.5, -1.0],
    index=2,
)

st.sidebar.subheader("Delivery Instructions")
tts_instructions = st.sidebar.text_area(
    "Instructions (used as guidance for TTS)",
    value=(
        "Calm, authoritative, restrained documentary narration. "
        "Measured pace with subtle pauses. Crisp consonants. "
        "No theatricality."
    ),
    height=120,
)

st.text_input("Project title", key="project_title")

st.caption(
    "Paste your Intro, Chapters, and Outro. The app reads exactly what you paste. "
    "Exports HQ voice-only MP3s and a full-episode MP3, normalized to your loudness target."
)

# ============================
# PART 2/4 — Manual Script Builder (Intro + Chapters 1..N + Outro)
# Append below Part 1
# ============================

def _add_chapter() -> None:
    st.session_state.chapters.append({"id": _stable_id(), "text": ""})


def _remove_chapter(idx: int) -> None:
    if 0 <= idx < len(st.session_state.chapters):
        st.session_state.chapters.pop(idx)


def _move_chapter(idx: int, direction: int) -> None:
    chapters = st.session_state.chapters
    j = idx + direction
    if 0 <= idx < len(chapters) and 0 <= j < len(chapters):
        chapters[idx], chapters[j] = chapters[j], chapters[idx]


left, right = st.columns([1.25, 0.75], gap="large")

with left:
    st.subheader("Script Sections")

    st.markdown("### Intro")
    st.text_area(
        "Intro",
        key="intro_text",
        height=220,
        placeholder="Paste your Intro here…",
        label_visibility="collapsed",
    )
    wc_intro = _word_count(st.session_state.intro_text)
    st.caption(f"Intro: {wc_intro} words • ~{_est_minutes(wc_intro):.1f} min")

    st.markdown("---")
    st.markdown("### Chapters")

    a, b = st.columns([0.25, 0.75])
    with a:
        if st.button("+ Add Chapter", use_container_width=True):
            _add_chapter()
            st.rerun()
    with b:
        st.caption("Chapters are numbered automatically: Chapter 1, Chapter 2, …")

    if len(st.session_state.chapters) == 0:
        st.info("No chapters yet. Click **+ Add Chapter** to add Chapter 1.")

    for idx, ch in enumerate(st.session_state.chapters):
        chap_num = idx + 1
        with st.container(border=True):
            cols = st.columns([0.55, 0.15, 0.15, 0.15])
            cols[0].markdown(f"**Chapter {chap_num}**")

            up_disabled = idx == 0
            dn_disabled = idx == len(st.session_state.chapters) - 1

            if cols[1].button("↑", key=f"ch_up_{ch['id']}", disabled=up_disabled):
                _move_chapter(idx, -1)
                st.rerun()
            if cols[2].button("↓", key=f"ch_dn_{ch['id']}", disabled=dn_disabled):
                _move_chapter(idx, +1)
                st.rerun()
            if cols[3].button("Remove", key=f"ch_rm_{ch['id']}"):
                _remove_chapter(idx)
                st.rerun()

            st.text_area(
                f"Chapter {chap_num}",
                key=f"ch_text_{ch['id']}",
                height=220,
                placeholder=f"Paste Chapter {chap_num} here…",
                label_visibility="collapsed",
            )
            ch["text"] = st.session_state.get(f"ch_text_{ch['id']}", "")

            wc = _word_count(ch["text"])
            st.caption(f"Chapter {chap_num}: {wc} words • ~{_est_minutes(wc):.1f} min")

    st.markdown("---")
    st.markdown("### Outro")
    st.text_area(
        "Outro",
        key="outro_text",
        height=220,
        placeholder="Paste your Outro here…",
        label_visibility="collapsed",
    )
    wc_outro = _word_count(st.session_state.outro_text)
    st.caption(f"Outro: {wc_outro} words • ~{_est_minutes(wc_outro):.1f} min")


with right:
    st.subheader("Build")

    total_words = wc_intro + wc_outro + sum(_word_count(c.get("text", "")) for c in st.session_state.chapters)
    st.metric("Total words", f"{total_words}")
    st.metric("Estimated duration", f"~{_est_minutes(total_words):.1f} min")

    st.markdown("---")
    st.markdown("#### Create Audio")
    build_clicked = st.button("Generate HQ Audio + Export", type="primary", use_container_width=True)

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
# PART 3/4 — HQ Audio Pipeline (Cloud-robust tail pad)
# COPY/PASTE THIS ENTIRE PART 3 BLOCK
#
# CHANGE SUMMARY:
# - Replaced the tpad-based tail padding (often missing on Streamlit Cloud FFmpeg)
#   with a concat-based method using anullsrc + concat (much more broadly supported).
# - Includes the missing build trigger at the bottom (button will work).
# ============================

def _collect_segments() -> List[Tuple[str, str]]:
    """
    Collect Intro/Chapters/Outro and ensure each segment ends cleanly
    so TTS doesn't stop abruptly on the last syllable.
    """
    def _ensure_terminal_punct(s: str) -> str:
        s = (s or "").strip()
        if not s:
            return s
        if re.search(r"[.!?…]\s*$", s):
            return s
        return s + "."

    segs: List[Tuple[str, str]] = []

    intro = _ensure_terminal_punct(st.session_state.intro_text)
    if intro:
        segs.append(("intro", intro))

    for i, ch in enumerate(st.session_state.chapters, start=1):
        txt = _ensure_terminal_punct(ch.get("text") or "")
        if txt:
            segs.append((f"chapter_{i:02d}", txt))

    outro = _ensure_terminal_punct(st.session_state.outro_text)
    if outro:
        segs.append(("outro", outro))

    return segs


def _read_audio_bytes(resp) -> bytes:
    if resp is None:
        raise RuntimeError("Empty TTS response")
    if hasattr(resp, "read") and callable(getattr(resp, "read")):
        b = resp.read()
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
    if hasattr(resp, "content"):
        b = getattr(resp, "content")
        if isinstance(b, (bytes, bytearray)):
            return bytes(b)
    if isinstance(resp, (bytes, bytearray)):
        return bytes(resp)
    raise RuntimeError(f"Unrecognized TTS response type: {type(resp)}")


def _tts_mp3_bytes(client: OpenAI, model: str, voice: str, text: str, instructions: str) -> bytes:
    """
    Voice-only TTS. We prepend instructions into the input in a lightweight way.
    """
    guided_input = f"[Delivery guidance: {instructions.strip()}]\n\n{text}"

    try:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=guided_input,
            response_format="mp3",
        )
        return _read_audio_bytes(resp)
    except TypeError:
        resp = client.audio.speech.create(
            model=model,
            voice=voice,
            input=guided_input,
            format="mp3",
        )
        return _read_audio_bytes(resp)


def _mp3_to_wav_master(mp3_in: str, wav_out: str, sr: int) -> Tuple[bool, str]:
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", mp3_in,
        "-ac", "2",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        wav_out,
    ]
    code, out = _run(cmd)
    return (code == 0), out


def _pad_wav_tail(wav_in: str, wav_out: str, pad_seconds: float, sr: int) -> Tuple[bool, str]:
    """
    Cloud-robust tail padding.

    Streamlit Cloud often ships an FFmpeg build that lacks the 'tpad' filter.
    This method instead appends silence by concatenating:
      [original audio] + [generated silence]
    using 'anullsrc' + 'concat', which is supported far more broadly.

    Output is PCM WAV, stereo, sr.
    """
    ffmpeg = _ffmpeg_path()

    # Ensure pad_seconds is a reasonable positive value
    try:
        pad_s = float(pad_seconds)
    except Exception:
        pad_s = 0.35
    if pad_s <= 0:
        pad_s = 0.20

    # filter_complex:
    #  - Take input audio as [a0]
    #  - Generate silence as [a1] with matching sr & stereo for pad duration
    #  - Concatenate them into [a]
    fc = (
        f"[0:a]aformat=sample_fmts=s16:channel_layouts=stereo,aresample={sr}[a0];"
        f"anullsrc=r={sr}:cl=stereo:d={pad_s}[a1];"
        f"[a0][a1]concat=n=2:v=0:a=1[a]"
    )

    cmd = [
        ffmpeg, "-y",
        "-i", wav_in,
        "-filter_complex", fc,
        "-map", "[a]",
        "-c:a", "pcm_s16le",
        "-ac", "2",
        "-ar", str(sr),
        wav_out,
    ]
    code, out = _run(cmd)
    return (code == 0), out


def _loudnorm_wav(wav_in: str, wav_out: str, target_lufs_i: int, true_peak_db: float, sr: int) -> Tuple[bool, str]:
    """
    Two-pass loudnorm for consistent, high-quality voice output.
    Pass 1 measures; pass 2 applies.
    """
    ffmpeg = _ffmpeg_path()

    cmd1 = [
        ffmpeg, "-y",
        "-i", wav_in,
        "-af", f"loudnorm=I={target_lufs_i}:TP={true_peak_db}:LRA=11:print_format=json",
        "-f", "null", "-"
    ]
    code1, out1 = _run(cmd1)
    if code1 != 0:
        return False, out1

    def _pick(key: str) -> Optional[str]:
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"([^"]+)"', out1)
        return m.group(1) if m else None

    measured_I = _pick("input_i")
    measured_TP = _pick("input_tp")
    measured_LRA = _pick("input_lra")
    measured_thresh = _pick("input_thresh")
    offset = _pick("target_offset")

    if not all([measured_I, measured_TP, measured_LRA, measured_thresh, offset]):
        cmd_fallback = [
            ffmpeg, "-y",
            "-i", wav_in,
            "-af", f"loudnorm=I={target_lufs_i}:TP={true_peak_db}:LRA=11",
            "-ac", "2",
            "-ar", str(sr),
            "-c:a", "pcm_s16le",
            wav_out,
        ]
        codef, outf = _run(cmd_fallback)
        return (codef == 0), outf

    af = (
        f"loudnorm=I={target_lufs_i}:TP={true_peak_db}:LRA=11:"
        f"measured_I={measured_I}:measured_TP={measured_TP}:measured_LRA={measured_LRA}:"
        f"measured_thresh={measured_thresh}:offset={offset}:linear=true:print_format=summary"
    )
    cmd2 = [
        ffmpeg, "-y",
        "-i", wav_in,
        "-af", af,
        "-ac", "2",
        "-ar", str(sr),
        "-c:a", "pcm_s16le",
        wav_out,
    ]
    code2, out2 = _run(cmd2)
    return (code2 == 0), out2


def _wav_to_mp3(wav_in: str, mp3_out: str, bitrate: str) -> Tuple[bool, str]:
    ffmpeg = _ffmpeg_path()
    cmd = [
        ffmpeg, "-y",
        "-i", wav_in,
        "-c:a", "libmp3lame",
        "-b:a", bitrate,
        mp3_out,
    ]
    code, out = _run(cmd)
    return (code == 0), out


def _concat_wavs(wavs: List[str], wav_out: str) -> Tuple[bool, str]:
    """
    Concatenate WAV files using ffmpeg concat demuxer.
    """
    ffmpeg = _ffmpeg_path()
    if not wavs:
        return False, "No wavs to concatenate."

    with tempfile.TemporaryDirectory() as td:
        list_path = os.path.join(td, "list.txt")
        with open(list_path, "w", encoding="utf-8") as f:
            for w in wavs:
                f.write(f"file '{w}'\n")

        cmd = [
            ffmpeg, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            wav_out,
        ]
        code, out = _run(cmd)
        return (code == 0), out


def _build_hq_voice_only(
    *,
    api_key: str,
    model: str,
    voice_name: str,
    instructions: str,
    mp3_bitrate: str,
    sr: int,
    target_lufs_i: int,
    true_peak_db: float,
) -> None:
    log: List[str] = []
    st.session_state.last_build_log = ""

    if not api_key.strip():
        st.error("OpenAI API key is required (set it in the sidebar).")
        return

    segments = _collect_segments()
    if not segments:
        st.warning("Nothing to generate — all sections are empty.")
        return

    client = OpenAI(api_key=api_key.strip())

    proj_slug = _slugify(st.session_state.project_title)
    progress = st.progress(0)
    status = st.empty()

    per_segment_mp3: Dict[str, bytes] = {}
    per_segment_wav_master: Dict[str, bytes] = {}
    norm_wavs_for_full: List[str] = []

    TAIL_PAD_SECONDS = 0.35

    try:
        with tempfile.TemporaryDirectory() as td:
            total = len(segments)

            for idx, (slug, txt) in enumerate(segments, start=1):
                status.write(f"Generating {slug}… ({idx}/{total})")

                # 1) TTS -> MP3 bytes
                try:
                    mp3_bytes = _tts_mp3_bytes(client, model, voice_name, txt, instructions)
                except Exception as e:
                    log.append(f"[ERROR] {slug}: TTS failed: {repr(e)}")
                    st.session_state.last_build_log = "\n".join(log)
                    progress.empty()
                    status.empty()
                    st.error(f"TTS failed on {slug}. See log.")
                    return

                raw_mp3_path = os.path.join(td, f"{slug}_tts.mp3")
                with open(raw_mp3_path, "wb") as f:
                    f.write(mp3_bytes)

                # 2) Decode to WAV master
                wav_master = os.path.join(td, f"{slug}_master.wav")
                ok, out = _mp3_to_wav_master(raw_mp3_path, wav_master, sr)
                if not ok:
                    log.append(f"[ERROR] {slug}: mp3->wav failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log)
                    progress.empty()
                    status.empty()
                    st.error(f"Audio decode failed on {slug}. See log.")
                    return

                # 2.5) Tail-pad BEFORE loudnorm + MP3 encode (cloud-robust)
                wav_padded = os.path.join(td, f"{slug}_padded.wav")
                ok, out = _pad_wav_tail(wav_master, wav_padded, TAIL_PAD_SECONDS, sr)
                if not ok:
                    log.append(f"[ERROR] {slug}: tail-pad failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log)
                    progress.empty()
                    status.empty()
                    st.error(f"Tail padding failed on {slug}. See log.")
                    return

                # 3) Loudness normalize to WAV
                wav_norm = os.path.join(td, f"{slug}_norm.wav")
                ok, out = _loudnorm_wav(wav_padded, wav_norm, target_lufs_i, true_peak_db, sr)
                if not ok:
                    log.append(f"[ERROR] {slug}: loudnorm failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log)
                    progress.empty()
                    status.empty()
                    st.error(f"Loudness normalization failed on {slug}. See log.")
                    return

                # 4) Encode final MP3
                out_mp3_path = os.path.join(td, f"{slug}.mp3")
                ok, out = _wav_to_mp3(wav_norm, out_mp3_path, mp3_bitrate)
                if not ok:
                    log.append(f"[ERROR] {slug}: wav->mp3 failed\n{out}")
                    st.session_state.last_build_log = "\n".join(log)
                    progress.empty()
                    status.empty()
                    st.error(f"MP3 encode failed on {slug}. See log.")
                    return

                with open(out_mp3_path, "rb") as f:
                    per_segment_mp3[slug] = f.read()

                with open(wav_norm, "rb") as f:
                    per_segment_wav_master[slug] = f.read()

                norm_wavs_for_full.append(wav_norm)

                log.append(f"[OK] {slug}.mp3")
                progress.progress(min(1.0, idx / total))

            # 5) Full episode from normalized WAVs -> normalize again -> MP3
            status.write("Building full episode…")
            full_wav = os.path.join(td, "full_episode.wav")
            ok, out = _concat_wavs(norm_wavs_for_full, full_wav)
            if not ok:
                log.append(f"[ERROR] full: concat failed\n{out}")
                st.session_state.last_build_log = "\n".join(log)
                progress.empty()
                status.empty()
                st.error("Full episode concat failed. See log.")
                return

            full_norm = os.path.join(td, "full_episode_norm.wav")
            ok, out = _loudnorm_wav(full_wav, full_norm, target_lufs_i, true_peak_db, sr)
            if not ok:
                log.append(f"[ERROR] full: loudnorm failed\n{out}")
                st.session_state.last_build_log = "\n".join(log)
                progress.empty()
                status.empty()
                st.error("Full episode loudness normalization failed. See log.")
                return

            full_mp3_path = os.path.join(td, "full_episode.mp3")
            ok, out = _wav_to_mp3(full_norm, full_mp3_path, mp3_bitrate)
            if not ok:
                log.append(f"[ERROR] full: wav->mp3 failed\n{out}")
                st.session_state.last_build_log = "\n".join(log)
                progress.empty()
                status.empty()
                st.error("Full episode MP3 encode failed. See log.")
                return

            with open(full_mp3_path, "rb") as f:
                full_mp3_bytes = f.read()

            # 6) ZIP bundle
            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
                z.writestr("scripts/intro.txt", st.session_state.intro_text or "")
                for i, ch in enumerate(st.session_state.chapters, start=1):
                    z.writestr(f"scripts/chapter_{i:02d}.txt", ch.get("text") or "")
                z.writestr("scripts/outro.txt", st.session_state.outro_text or "")

                for slug, b in per_segment_mp3.items():
                    z.writestr(f"audio/mp3/{slug}.mp3", b)

                for slug, b in per_segment_wav_master.items():
                    z.writestr(f"audio/wav_norm/{slug}.wav", b)

            zip_buf.seek(0)

            st.session_state.built = {
                "zip": zip_buf.getvalue(),
                "zip_name": f"{proj_slug}__uappress_pack.zip",
                "full_mp3": full_mp3_bytes,
                "full_name": f"{proj_slug}__full_episode.mp3",
            }

            progress.empty()
            status.empty()
            log.append("[OK] full_episode.mp3")
            st.success("Done! Download below.")

    finally:
        st.session_state.last_build_log = "\n".join(log)


# ----------------------------
# ✅ BUILD TRIGGER (must be after build_clicked exists in Part 2)
# ----------------------------
if build_clicked:
    _build_hq_voice_only(
        api_key=api_key,
        model=tts_model,
        voice_name=voice,
        instructions=tts_instructions,
        mp3_bitrate=mp3_bitrate,
        sr=int(sample_rate),
        target_lufs_i=int(target_lufs),
        true_peak_db=float(true_peak),
    )

# ============================
# PART 4/4 — Downloads (ZIP + Full Episode MP3)
# Append below Part 3
# ============================

st.header("4️⃣ Downloads")

built = st.session_state.get("built")

if built:
    st.download_button(
        "⬇️ Download ZIP (scripts + segment MP3s + WAV masters)",
        data=built["zip"],
        file_name=built["zip_name"],
        mime="application/zip",
    )

    st.audio(built["full_mp3"], format="audio/mp3")

    st.download_button(
        "⬇️ Download Full Episode MP3",
        data=built["full_mp3"],
        file_name=built["full_name"],
        mime="audio/mpeg",
    )
else:
    st.caption("After you generate audio, your download buttons will appear here.")
