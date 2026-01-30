# ============================
# PART 1/4 — Core Setup, Sidebar, Clients, FFmpeg, State, Utilities, Parsing (Hardened)
# ============================
# app.py — UAPpress Documentary TTS Studio
# MODE: Outline → Strict Script (Contract-Hardened) → Editable Sections → TTS Audio
#
# REQUIREMENTS:
# streamlit>=1.30
# openai>=1.0.0
# imageio-ffmpeg>=0.4.9

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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary TTS Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio")
st.caption("Credibility-first investigative documentaries. Outline → Script → Audio.")

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

def clean_filename(text: str) -> str:
    text = re.sub(r"[^\w\s\-]", "", (text or "")).strip()
    text = re.sub(r"\s+", "_", text)
    return (text.lower()[:80] or "episode")

def sanitize_for_tts(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def strip_tts_directives(text: str) -> str:
    """
    Removes any accidental stage directions or bracketed directives.
    Keeps evidence labels like [FACT] etc intact (Part 2 output).
    """
    if not text:
        return ""
    # Remove parenthetical stage directions e.g., (pause), (beat), (music)
    t = re.sub(r"\((?:pause|beat|music|silence|emphasis|dramatic)[^)]*\)", "", text, flags=re.IGNORECASE)
    # Remove lines that look like stage directions
    t = re.sub(r"(?im)^\s*(?:pause|beat|music|silence|emphasis|dramatic)\s*[:\-–—].*$", "", t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

# ----------------------------
# Workdir + Cache
# ----------------------------
def _workdir() -> str:
    st.session_state.setdefault("WORKDIR", tempfile.mkdtemp(prefix="uappress_tts_"))
    return st.session_state["WORKDIR"]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            if os.path.isdir(path):
                # Remove directory tree
                for root, dirs, files in os.walk(path, topdown=False):
                    for f in files:
                        _safe_remove(os.path.join(root, f))
                    for d in dirs:
                        try:
                            os.rmdir(os.path.join(root, d))
                        except Exception:
                            pass
                try:
                    os.rmdir(path)
                except Exception:
                    pass
            else:
                os.remove(path)
    except Exception:
        pass

def episode_slug() -> str:
    title = st.session_state.get("episode_title", "Untitled Episode")
    return clean_filename(title)

# ----------------------------
# Duration helper
# ----------------------------
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")

def get_media_duration_seconds(path: str) -> float:
    if not path or not os.path.exists(path):
        return 0.0
    code, _, err = run_cmd([FFMPEG, "-i", path])
    # ffmpeg prints duration to stderr even on code !=0
    m = _DURATION_RE.search(err or "")
    if not m:
        return 0.0
    hh = int(m.group(1))
    mm = int(m.group(2))
    ss = float(m.group(3))
    return hh * 3600.0 + mm * 60.0 + ss

# ----------------------------
# ZIP helper
# ----------------------------
def make_zip_bytes(files: List[Tuple[str, str]]) -> bytes:
    """
    files: list of (absolute_path, archive_name)
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for src, arc in files:
            if not src or not os.path.exists(src):
                continue
            z.write(src, arcname=arc)
    return buf.getvalue()

# ----------------------------
# Parsing (STRICT headings, with validation)
# ----------------------------
_CHAPTER_RE = re.compile(r"(?i)^\s*chapter\s+(\d+)\s*(?:[:\-–—]\s*(.*))?\s*$")

def detect_chapters_and_titles(master_text: str) -> Dict[int, str]:
    chapters: Dict[int, str] = {}
    if not master_text:
        return chapters
    for line in master_text.splitlines():
        m = _CHAPTER_RE.match(line)
        if m:
            n = int(m.group(1))
            title = (m.group(2) or "").strip()
            chapters[n] = title
    return chapters

def validate_master_script_structure(master_text: str) -> Tuple[bool, str, int]:
    """
    Validates:
      - INTRO present
      - OUTRO present
      - CHAPTER 1..N present and contiguous
      - OUTRO occurs after last chapter heading
    Returns (ok, error_message, chapter_count)
    """
    if not (master_text or "").strip():
        return False, "Script is empty.", 0

    txt = master_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")

    intro_idx = None
    outro_idx = None
    chapter_idx: Dict[int, int] = {}

    for i, line in enumerate(lines):
        u = line.strip().upper()
        if u == "INTRO":
            intro_idx = i
        elif u == "OUTRO":
            outro_idx = i
        else:
            m = _CHAPTER_RE.match(line)
            if m:
                chapter_idx[int(m.group(1))] = i

    if intro_idx is None:
        return False, "Missing INTRO heading.", 0
    if outro_idx is None:
        return False, "Missing OUTRO heading.", 0
    if not chapter_idx:
        return False, "No CHAPTER headings detected.", 0

    n = max(chapter_idx.keys())
    missing = [k for k in range(1, n + 1) if k not in chapter_idx]
    if missing:
        return False, f"Chapters missing or non-contiguous: {missing}", 0

    last_chapter_line = max(chapter_idx.values())
    if outro_idx <= last_chapter_line:
        return False, "OUTRO appears before the final chapter. Script headings are out of order.", 0

    if intro_idx >= min(chapter_idx.values()):
        return False, "INTRO appears after a chapter heading. Script headings are out of order.", 0

    return True, "", n

def parse_master_script(master_text: str, expected_chapters: int) -> Dict[str, Any]:
    """
    Parses INTRO / CHAPTER N / OUTRO.
    Heading lines are NOT included in narration output.
    Uses expected_chapters to enforce extraction of all chapters 1..N.
    """
    txt = (master_text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = txt.split("\n")

    intro_i, outro_i = None, None
    chapter_i: Dict[int, int] = {}

    for idx, line in enumerate(lines):
        u = line.strip().upper()
        if u == "INTRO":
            intro_i = idx
        elif u == "OUTRO":
            outro_i = idx
        else:
            m = _CHAPTER_RE.match(line)
            if m:
                chapter_i[int(m.group(1))] = idx

    # Marker order (validate already ensured order)
    markers = [i for i in [intro_i, *[chapter_i.get(k) for k in range(1, expected_chapters + 1)], outro_i] if i is not None]
    markers_sorted = sorted(markers)
    bounds = {markers_sorted[i]: markers_sorted[i + 1] if i + 1 < len(markers_sorted) else len(lines) for i in range(len(markers_sorted))}

    def block(start: Optional[int]) -> str:
        if start is None:
            return ""
        return "\n".join(lines[start + 1 : bounds[start]]).strip()

    return {
        "intro": block(intro_i),
        "outro": block(outro_i),
        "chapters": {k: block(chapter_i.get(k)) for k in range(1, expected_chapters + 1)},
    }

# ----------------------------
# Streamlit state helpers (single source of truth)
# ----------------------------
def text_key(kind: str, idx: int = 0) -> str:
    return f"text::{kind}::{idx}"

def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> None:
    st.session_state.setdefault(text_key(kind, idx), default or "")

def reset_script_text_fields(chapter_count: int) -> None:
    ensure_text_key("intro", 0, "")
    ensure_text_key("outro", 0, "")
    for i in range(1, int(chapter_count) + 1):
        ensure_text_key("chapter", i, "")

def ensure_state() -> None:
    st.session_state.setdefault("episode_title", "Untitled Episode")
    st.session_state.setdefault("DEFAULT_CHAPTERS", 8)

    # Outline state split (widget vs source)
    st.session_state.setdefault("OUTLINE_TEXT_UI", "")
    st.session_state.setdefault("OUTLINE_TEXT_SRC", "")
    st.session_state.setdefault("_PENDING_OUTLINE_SYNC", False)

    # Master script + parsed fields
    st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")
    st.session_state.setdefault("chapter_count", 0)

    # Part 2 outputs
    st.session_state.setdefault("generated_script_json", None)
    st.session_state.setdefault("generated_script_text", "")
    st.session_state.setdefault("script_generation_log", [])

    reset_script_text_fields(int(st.session_state.get("chapter_count", 0)))

ensure_state()

# ----------------------------
# Sidebar — API Key + Models
# ----------------------------
with st.sidebar:
    st.header("🔐 API Key (not saved)")
    st.session_state.setdefault("OPENAI_API_KEY_INPUT", "")
    api_key_input = st.text_input(
        "OpenAI API Key",
        type="password",
        key="OPENAI_API_KEY_INPUT",
        placeholder="sk-...",
        help="Stored only in session memory.",
    )

    st.divider()
    st.header("⚙️ Models")
    st.session_state.setdefault("SCRIPT_MODEL", "gpt-4o-mini")
    st.session_state.setdefault("TTS_MODEL", "gpt-4o-mini-tts")
    st.session_state.setdefault("TTS_VOICE", "onyx")

    st.text_input("Script model", key="SCRIPT_MODEL")
    st.text_input("TTS model", key="TTS_MODEL")
    st.text_input("TTS voice", key="TTS_VOICE")

    st.divider()
    st.header("🧠 Script Generation")
    st.session_state.setdefault("SCRIPT_MAX_PASSES", 3)
    st.number_input("Max repair passes", min_value=1, max_value=6, step=1, key="SCRIPT_MAX_PASSES")
    st.caption("Contract is enforced via JSON Schema + deterministic validators + repair loop.")

# ----------------------------
# OpenAI client
# ----------------------------
api_key = (api_key_input or "").strip()
if not api_key:
    st.warning("Enter your OpenAI API key in the sidebar to begin.")
    st.stop()

client = OpenAI(api_key=api_key)

# ============================
# PART 2/4 — STRICT Script Generation (FIX: no-arg wrapper + safe state writes)
# Paste this WHOLE block to replace your existing Part 2/4 top-to-bottom.
# ============================

from typing import Any, Dict, List, Tuple, Optional
import json
import re

# ----------------------------
# Robust JSON extraction
# ----------------------------

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()

    # fenced ```json
    m = _JSON_FENCE.search(s)
    if m:
        s = m.group(1).strip()

    # direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # find first {...}
    m2 = _JSON_OBJ.search(s)
    if not m2:
        return None
    try:
        obj = json.loads(m2.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _resp_text(resp: Any) -> str:
    v = getattr(resp, "output_text", None)
    if isinstance(v, str) and v.strip():
        return v.strip()
    try:
        out = getattr(resp, "output", None)
        if isinstance(out, list):
            parts: List[str] = []
            for item in out:
                content = getattr(item, "content", None)
                if isinstance(content, list):
                    for c in content:
                        t = getattr(c, "text", None)
                        if isinstance(t, str) and t.strip():
                            parts.append(t)
            if parts:
                return "\n".join(parts).strip()
    except Exception:
        pass
    return ""


# ----------------------------
# Contract checks (minimal but strict)
# ----------------------------

def _validate_script_json(script: Any, expected_chapters: int) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    if not isinstance(script, dict):
        return False, ["script_not_object"]

    intro = script.get("intro")
    outro = script.get("outro")
    chapters = script.get("chapters")

    if not isinstance(intro, str) or not intro.strip():
        problems.append("intro_missing_or_empty")
    if not isinstance(outro, str) or not outro.strip():
        problems.append("outro_missing_or_empty")

    if not isinstance(chapters, list):
        problems.append("chapters_missing_or_not_list")
        return False, problems

    if len(chapters) != expected_chapters:
        problems.append(f"chapter_count_mismatch_expected_{expected_chapters}_got_{len(chapters)}")

    nums: List[int] = []
    for i, ch in enumerate(chapters, start=1):
        if not isinstance(ch, dict):
            problems.append(f"chapter_{i}_not_object")
            continue
        n = ch.get("n")
        try:
            nums.append(int(n))
        except Exception:
            problems.append(f"chapter_{i}_missing_or_bad_n")
        if not isinstance(ch.get("title"), str) or not ch.get("title", "").strip():
            problems.append(f"chapter_{i}_missing_title")
        if not isinstance(ch.get("text"), str) or not ch.get("text", "").strip():
            problems.append(f"chapter_{i}_missing_text")

    if nums:
        expected_nums = list(range(1, len(chapters) + 1))
        if nums != expected_nums:
            problems.append("chapter_numbers_not_contiguous_from_1")

    ok = len(problems) == 0
    return ok, problems


# ----------------------------
# OpenAI call (Responses API)
# ----------------------------

def _call_openai_json_only(*, model: str, system: str, user: str, temperature: float = 0.2) -> Dict[str, Any]:
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=temperature,
    )

    # Prefer parsed if available
    parsed = getattr(resp, "output_parsed", None)
    if isinstance(parsed, dict):
        return parsed

    txt = _resp_text(resp)
    obj = _extract_json_object(txt)
    if not isinstance(obj, dict):
        raise ValueError("Model did not return valid JSON object.")
    return obj


# ----------------------------
# Internal strict generator (keyword-only)
# ----------------------------

def _generate_script_strict_impl(
    *,
    topic: str,
    outline: str,
    chapter_count: int,
    model: str,
    target_runtime_min: int,
    scope: str,
    constraints: str,
    max_passes: int,
) -> Tuple[bool, str, Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns: ok, msg, script_json, enforcement_log
    """
    log: List[Dict[str, Any]] = []
    chapter_count = int(chapter_count or 0)
    if chapter_count <= 0:
        return False, "chapter_count must be >= 1", None, log

    system = (
        "You MUST output ONLY valid JSON. No markdown. No commentary.\n"
        "Follow the INVESTIGATIVE CONTRACT: no novelization, no invented dialogue, no mind-reading.\n"
        "After Key Players section (in intro), use last names only.\n"
    )

    user_base = {
        "task": "generate_strict_documentary_script",
        "topic": topic,
        "target_runtime_min": int(target_runtime_min),
        "scope": scope,
        "constraints": constraints,
        "chapter_count": chapter_count,
        "outline": outline,
        "required_format": {
            "title": "string",
            "intro": "string",
            "chapters": [
                {"n": "int 1..N contiguous", "title": "string", "text": "string"}
            ],
            "outro": "string",
        },
        "hard_rules": [
            f"chapters MUST be EXACTLY {chapter_count} items",
            "chapter numbers MUST be contiguous starting at 1",
            "OUTPUT JSON ONLY",
        ],
    }

    problems: List[str] = []
    last: Optional[Dict[str, Any]] = None

    for p in range(1, max_passes + 1):
        payload = dict(user_base)
        payload["repair_pass"] = p
        payload["prior_problems"] = problems

        try:
            script_json = _call_openai_json_only(
                model=model,
                system=system,
                user=json.dumps(payload, ensure_ascii=False),
                temperature=0.2 if p == 1 else 0.1,
            )
            ok_render, problems = _validate_script_json(script_json, chapter_count)
            log.append({"pass": p, "ok_json": True, "ok_render": ok_render, "problems": problems[:]})
            last = script_json
            if ok_render:
                return True, "Script generated (strict).", script_json, log
        except Exception as e:
            problems = ["render_skipped_due_to_json_invalid", f"json_error_{type(e).__name__}"]
            log.append({"pass": p, "ok_json": False, "ok_render": False, "problems": problems[:]})
            last = None

    return False, "Script blocked by contract enforcement. Fix outline/title and retry.", last, log


# ----------------------------
# PUBLIC wrapper (NO-ARG) — this fixes your TypeError
# Your UI calls: ok, msg = generate_script_strict()
# ----------------------------

def generate_script_strict() -> Tuple[bool, str]:
    """
    UI-safe wrapper that reads session_state and writes results back.
    This function takes NO arguments (matches your current call site).
    """
    # Source of truth
    topic = (st.session_state.get("episode_title") or "").strip() or "Untitled Episode"
    outline = (st.session_state.get("OUTLINE_TEXT_SRC") or st.session_state.get("OUTLINE_TEXT_UI") or "").strip()

    # Chapter count: use DEFAULT_CHAPTERS (your UI) as canonical
    chapter_count = int(st.session_state.get("DEFAULT_CHAPTERS", 8) or 8)

    model = st.session_state.get("SCRIPT_MODEL", st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"))
    target_runtime_min = int(st.session_state.get("target_runtime_min", 45) or 45)
    scope = (st.session_state.get("scope") or "serious, historically grounded investigation").strip()
    constraints = (st.session_state.get("constraints") or "").strip()
    max_passes = int(st.session_state.get("script_max_passes", 3) or 3)

    st.session_state["script_generation_log"] = []

    if not outline:
        return False, "Outline is empty. Generate or paste an outline first."

    ok, msg, script_json, log = _generate_script_strict_impl(
        topic=topic,
        outline=outline,
        chapter_count=chapter_count,
        model=model,
        target_runtime_min=target_runtime_min,
        scope=scope,
        constraints=constraints,
        max_passes=max_passes,
    )

    st.session_state["script_generation_log"] = log

    if not ok or not isinstance(script_json, dict):
        st.session_state["MASTER_SCRIPT_TEXT"] = ""
        st.session_state["chapter_count"] = 0
        reset_script_text_fields(0)
        return False, msg

    # Render to your existing INTRO/CHAPTER/OUTRO headings so Part 1 parsers work
    title_line = (script_json.get("title") or topic).strip()
    intro = (script_json.get("intro") or "").strip()
    outro = (script_json.get("outro") or "").strip()
    chapters = script_json.get("chapters") or []

    lines: List[str] = []
    lines.append("INTRO")
    lines.append(intro)
    lines.append("")

    for ch in chapters:
        n = int(ch.get("n"))
        t = (ch.get("title") or f"Chapter {n}").strip()
        tx = (ch.get("text") or "").strip()
        lines.append(f"CHAPTER {n}: {t}")
        lines.append(tx)
        lines.append("")

    lines.append("OUTRO")
    lines.append(outro)
    master_text = "\n".join(lines).strip() + "\n"

    st.session_state["MASTER_SCRIPT_TEXT"] = master_text

    # Fill editable fields using your Part 1 parser utilities
    st.session_state["chapter_count"] = chapter_count
    reset_script_text_fields(chapter_count)

    parsed = parse_master_script(master_text, chapter_count)
    st.session_state[text_key("intro", 0)] = (parsed.get("intro", "") or "").strip()
    for i in range(1, chapter_count + 1):
        st.session_state[text_key("chapter", i)] = (parsed.get("chapters", {}).get(i, "") or "").strip()
    st.session_state[text_key("outro", 0)] = (parsed.get("outro", "") or "").strip()

    return True, msg

# ============================
# PART 3/4 — Outline UI → Generate Strict Script → Editable Review (Single Pipeline)
# ============================

st.header("1️⃣ Script Creation")

# Preload outline BEFORE widget (safe)
if st.session_state.get("_PENDING_OUTLINE_SYNC"):
    st.session_state["_PENDING_OUTLINE_SYNC"] = False
    st.session_state["OUTLINE_TEXT_UI"] = st.session_state.get("OUTLINE_TEXT_SRC", "")

def _queue_outline_to_ui(outline_text: str) -> None:
    st.session_state["OUTLINE_TEXT_SRC"] = (outline_text or "").strip()
    st.session_state["_PENDING_OUTLINE_SYNC"] = True
    st.rerun()

colA, colB = st.columns([3, 1])
with colA:
    st.text_input("Episode Title", key="episode_title")
with colB:
    st.number_input("Chapter Count", min_value=3, max_value=20, step=1, key="DEFAULT_CHAPTERS")

st.divider()
st.subheader("🧠 Outline / Treatment (Optional but recommended)")
st.text_area(
    "Outline (editable)",
    key="OUTLINE_TEXT_UI",
    height=300,
    help="Use this to lock chronology, witnesses, documents, and decision points before generating the script.",
)

# Mirror UI → SRC (safe)
st.session_state["OUTLINE_TEXT_SRC"] = (st.session_state.get("OUTLINE_TEXT_UI") or "").strip()

title_ok = bool((st.session_state.get("episode_title") or "").strip())

b1, b2, b3, b4 = st.columns([1, 1, 1, 1])

with b1:
    gen_outline_btn = st.button("Generate Outline", type="primary", disabled=not title_ok)
with b2:
    gen_script_btn = st.button("Generate STRICT Script", disabled=not title_ok)
with b3:
    clear_outline_btn = st.button("Clear Outline")
with b4:
    clear_script_btn = st.button("Clear Script")

if clear_outline_btn:
    st.session_state["OUTLINE_TEXT_SRC"] = ""
    st.session_state["_PENDING_OUTLINE_SYNC"] = True
    st.success("Outline cleared.")
    st.rerun()

if clear_script_btn:
    st.session_state["MASTER_SCRIPT_TEXT"] = ""
    st.session_state["chapter_count"] = 0
    st.session_state["generated_script_json"] = None
    st.session_state["generated_script_text"] = ""
    reset_script_text_fields(0)
    st.success("Script cleared.")
    st.rerun()

# Outline generator (contract-safe outline format; not narration)
if gen_outline_btn:
    title = (st.session_state.get("episode_title") or "").strip()
    chapters_n = int(st.session_state.get("DEFAULT_CHAPTERS") or 8)

    outline_prompt = f"""
DOCUMENTARY TITLE:
{title}

CHAPTER COUNT:
{chapters_n}

TASK:
Generate an OUTLINE ONLY (not narration). Procedural, chronology-first.

FORMAT:
INTRO
- Date/time anchor(s) (specific)
- Institutional context (who has jurisdiction)
- Key players (full name + role, 7–12 words why they matter)
- Central question (one sentence)

CHAPTER 1–{chapters_n}
For each chapter:
- Date/time window
- Central decision point (one)
- Key witness(es) (full name + role)
- Document / record / statement to anchor the chapter
- Institutional response (who did what)
- Consequence / unresolved tension (one)

OUTRO
- What is established vs disputed
- Human consequences
- Open questions for further investigation

RULES:
- No cinematic prose
- No quotes
- No invented dialogue
- No mind-reading
- No “we will” language
""".strip()

    with st.spinner("Generating outline…"):
        try:
            r = client.chat.completions.create(
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You write restrained investigative documentary outlines. No fiction, no quotes, no mind-reading."},
                    {"role": "user", "content": outline_prompt},
                ],
                temperature=0.35,
            )
            outline_text = (r.choices[0].message.content or "").strip()
            u = outline_text.upper()
            if "INTRO" not in u or "CHAPTER" not in u or "OUTRO" not in u:
                raise ValueError("Outline generation failed: missing INTRO/CHAPTER/OUTRO sections.")
            _queue_outline_to_ui(outline_text)
        except Exception as e:
            st.error(str(e))

# Strict script generation (single source of truth)
if gen_script_btn:
    with st.spinner("Generating contract-hardened script…"):
        ok, msg = generate_script_strict()
        if ok:
            st.success(msg)
        else:
            st.error(msg)

# Debug log
if st.session_state.get("script_generation_log"):
    with st.expander("Contract Enforcement Log", expanded=False):
        st.json(st.session_state["script_generation_log"])

# Editable script review
if int(st.session_state.get("chapter_count", 0)) > 0:
    st.header("2️⃣ Script Review (Editable)")

    st.subheader("INTRO")
    st.text_area("Intro narration", key=text_key("intro", 0), height=240)

    st.divider()
    chapter_count = int(st.session_state.get("chapter_count", 0))
    for i in range(1, chapter_count + 1):
        with st.expander(f"CHAPTER {i}", expanded=(i == 1)):
            st.text_area(f"Chapter {i} narration", key=text_key("chapter", i), height=340)

    st.divider()
    st.subheader("OUTRO")
    st.text_area("Outro narration", key=text_key("outro", 0), height=240)

    st.divider()
    st.subheader("Master Script Preview (Read-only)")
    # Rebuild a master script from edited sections using strict headings
    if st.button("Rebuild MASTER SCRIPT from edited sections"):
        # Rebuild script headings from edited text
        rebuilt: List[str] = ["INTRO", "", (st.session_state.get(text_key("intro", 0)) or "").strip(), ""]
        # Ensure sponsor lines remain present; if user deleted, reinsert safely at top/bottom
        if OPA_SPONSOR_INTRO_VERBATIM not in rebuilt[2]:
            rebuilt.insert(2, OPA_SPONSOR_INTRO_VERBATIM + "\n")
        for i in range(1, chapter_count + 1):
            rebuilt.extend([f"CHAPTER {i}:", "", (st.session_state.get(text_key("chapter", i)) or "").strip(), ""])
        rebuilt.extend(["OUTRO", "", (st.session_state.get(text_key("outro", 0)) or "").strip(), ""])
        if OPA_SPONSOR_OUTRO_VERBATIM not in rebuilt[-2]:
            rebuilt.insert(len(rebuilt) - 1, "\n" + OPA_SPONSOR_OUTRO_VERBATIM)
        master = "\n".join(rebuilt).strip() + "\n"

        ok_struct, err, n = validate_master_script_structure(master)
        if not ok_struct:
            st.error(f"MASTER SCRIPT rebuild failed structure validation: {err}")
        else:
            st.session_state["MASTER_SCRIPT_TEXT"] = master
            st.success("MASTER SCRIPT rebuilt.")
    st.text_area("MASTER_SCRIPT_TEXT", value=st.session_state.get("MASTER_SCRIPT_TEXT", ""), height=360)
else:
    st.info("Generate a script to enable the editable Intro / Chapters / Outro review panel.")

# ============================
# PART 4/4 — Audio Build (Per-Section + Full Episode) + QC + Downloads + Final ZIP (Hardened)
# ============================

st.header("3️⃣ Create Audio (Onyx + Music)")

@dataclass
class TTSConfig:
    model: str
    voice: str
    speed: float = 1.0
    enable_cache: bool = True
    cache_dir: str = ".uappress_tts_cache"

def _cache_key_tts(text: str, cfg: TTSConfig) -> str:
    h = hashlib.sha256()
    h.update((cfg.model or "").encode("utf-8"))
    h.update((cfg.voice or "").encode("utf-8"))
    h.update(str(cfg.speed).encode("utf-8"))
    h.update((text or "").encode("utf-8"))
    return h.hexdigest()

def _tts_cache_path(key: str, cfg: TTSConfig) -> str:
    _ensure_dir(cfg.cache_dir)
    return os.path.join(cfg.cache_dir, f"{key}.wav")

def tts_to_wav(*, text: str, out_wav: str, cfg: TTSConfig) -> None:
    """
    Generates WAV using OpenAI TTS and writes to out_wav.
    Uses a local cache keyed by (model, voice, speed, text).
    """
    _ensure_dir(os.path.dirname(out_wav))
    cleaned = sanitize_for_tts(text or "")
    if not cleaned:
        raise ValueError("TTS input text is empty after sanitization.")

    cache_hit = False
    if cfg.enable_cache:
        key = _cache_key_tts(cleaned, cfg)
        cpath = _tts_cache_path(key, cfg)
        if os.path.exists(cpath):
            # Copy cached wav
            with open(cpath, "rb") as src, open(out_wav, "wb") as dst:
                dst.write(src.read())
            cache_hit = True

    if cache_hit:
        return

    # Try Responses/Audio Speech API variants; fall back gracefully
    wav_bytes: Optional[bytes] = None
    last_err: Optional[Exception] = None

    # Variant A: audio.speech.create returns bytes in response.content (common SDK pattern)
    try:
        resp = client.audio.speech.create(
            model=cfg.model,
            voice=cfg.voice,
            input=cleaned,
            format="wav",
            speed=cfg.speed,
        )
        wav_bytes = getattr(resp, "content", None) or getattr(resp, "data", None)  # type: ignore
        if isinstance(wav_bytes, str):
            wav_bytes = wav_bytes.encode("utf-8")
    except Exception as e:
        last_err = e

    if wav_bytes is None:
        # Variant B: streaming response helper
        try:
            with client.audio.speech.with_streaming_response.create(
                model=cfg.model,
                voice=cfg.voice,
                input=cleaned,
                format="wav",
                speed=cfg.speed,
            ) as r:
                wav_bytes = r.read()
        except Exception as e:
            last_err = e

    if wav_bytes is None:
        raise RuntimeError(f"TTS failed: {last_err}")

    with open(out_wav, "wb") as f:
        f.write(wav_bytes)

    # Populate cache
    if cfg.enable_cache:
        key = _cache_key_tts(cleaned, cfg)
        cpath = _tts_cache_path(key, cfg)
        try:
            with open(out_wav, "rb") as src, open(cpath, "wb") as dst:
                dst.write(src.read())
        except Exception:
            pass

def wav_to_mp3(in_wav: str, out_mp3: str, bitrate: str = "192k") -> None:
    _ensure_dir(os.path.dirname(out_mp3))
    run_ffmpeg([FFMPEG, "-y", "-i", in_wav, "-b:a", bitrate, "-ar", "48000", "-ac", "2", out_mp3])

def mix_music_under_voice(*, voice_wav: str, music_path: str, out_wav: str, music_db: int, fade_s: int) -> None:
    """
    Mixes music under voice using ffmpeg:
      - loops/extends music to voice duration
      - reduces music volume (dB)
      - applies fades to music
      - loudnorm on final
    """
    if not os.path.exists(voice_wav):
        raise FileNotFoundError("Voice WAV not found.")
    if not os.path.exists(music_path):
        raise FileNotFoundError("Music bed not found.")

    dur = get_media_duration_seconds(voice_wav)
    if dur <= 0.0:
        dur = 60.0

    # Use -stream_loop -1 for music, trim to voice duration
    # Music volume adjustment: volume= -24dB means approx 0.063; ffmpeg supports dB via "volume=-24dB"
    # Fade in/out on music track
    fade_s = max(int(fade_s), 0)
    fade_out_start = max(dur - float(fade_s), 0.0)

    # Filter:
    #  music: looped, trimmed, volume, fades
    #  mix: amix, then loudnorm (gentle)
    filt = (
        f"[1:a]atrim=0:{dur},asetpts=PTS-STARTPTS,volume={music_db}dB,"
        f"afade=t=in:st=0:d={fade_s},afade=t=out:st={fade_out_start}:d={fade_s}[m];"
        f"[0:a]asetpts=PTS-STARTPTS[v];"
        f"[v][m]amix=inputs=2:duration=first:dropout_transition=2,volume=1.0,"
        f"loudnorm=I=-16:TP=-1.5:LRA=11"
    )

    _ensure_dir(os.path.dirname(out_wav))
    run_ffmpeg([
        FFMPEG, "-y",
        "-i", voice_wav,
        "-stream_loop", "-1", "-i", music_path,
        "-filter_complex", filt,
        "-t", f"{dur}",
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        out_wav
    ])

def write_text_file(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")

def section_output_paths(section_id: str, episode_slug_: str) -> Dict[str, str]:
    wd = _workdir()
    base = f"{episode_slug_}__{section_id}"
    return {
        "voice_wav": os.path.join(wd, f"{base}__voice.wav"),
        "mixed_wav": os.path.join(wd, f"{base}__mix.wav"),
        "final_wav": os.path.join(wd, f"{base}__final.wav"),
        "mp3": os.path.join(wd, f"{base}.mp3"),
        "txt": os.path.join(wd, f"{base}.txt"),
    }

def section_texts() -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    out.append(("00_intro", st.session_state.get(text_key("intro", 0), "")))
    n = int(st.session_state.get("chapter_count", 0))
    for i in range(1, n + 1):
        out.append((f"{i:02d}_chapter_{i}", st.session_state.get(text_key("chapter", i), "")))
    out.append(("99_outro", st.session_state.get(text_key("outro", 0), "")))
    return out

def has_any_script() -> bool:
    return any((t or "").strip() for _, t in section_texts())

# ----------------------------
# Music upload / selection (persist inside WORKDIR)
# ----------------------------
st.subheader("🎵 Music Bed (Optional)")
music_file = st.file_uploader("Optional music bed (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])

st.session_state.setdefault("MUSIC_PATH", "")
music_path = ""

if music_file is not None:
    wd = _workdir()
    _ensure_dir(wd)
    save_name = clean_filename(music_file.name)
    persisted = os.path.join(wd, f"__music__{save_name}")
    with open(persisted, "wb") as f:
        f.write(music_file.read())
    st.session_state["MUSIC_PATH"] = persisted
    music_path = persisted
else:
    music_path = st.session_state.get("MUSIC_PATH", "") or st.session_state.get("DEFAULT_MUSIC_PATH", "") or ""

use_music = st.checkbox("Mix music under voice", value=True)
music_db_i = int(st.session_state.get("MUSIC_DB", -24))
fade_s_i = int(st.session_state.get("MUSIC_FADE_S", 6))

if use_music:
    if music_path and os.path.exists(music_path):
        st.caption(f"Music bed loaded: {os.path.basename(music_path)}")
    else:
        st.warning("Music mixing enabled, but no music file is loaded. Upload one (or set DEFAULT_MUSIC_PATH).")

st.divider()

# ----------------------------
# TTS config
# ----------------------------
cfg = TTSConfig(
    model=st.session_state.get("TTS_MODEL", "gpt-4o-mini-tts"),
    voice=st.session_state.get("TTS_VOICE", "onyx"),
    speed=float(st.session_state.get("TTS_SPEED", 1.0)),
    enable_cache=bool(st.session_state.get("ENABLE_TTS_CACHE", True)),
    cache_dir=str(st.session_state.get("CACHE_DIR", ".uappress_tts_cache")),
)
st.caption(f"TTS: model={cfg.model} | voice={cfg.voice} | cache={'on' if cfg.enable_cache else 'off'}")

def build_one_section(
    *,
    section_id: str,
    text: str,
    cfg: TTSConfig,
    mix_music: bool,
    music_path: str,
    music_db: int,
    fade_s: int,
    force: bool = False,
) -> Dict[str, str]:
    slug = episode_slug()
    paths = section_output_paths(section_id, slug)

    cleaned = strip_tts_directives(sanitize_for_tts(text or ""))
    write_text_file(paths["txt"], cleaned)

    if force:
        for k in ["voice_wav", "mixed_wav", "final_wav", "mp3"]:
            _safe_remove(paths[k])

    # TTS → voice WAV
    tts_to_wav(text=cleaned, out_wav=paths["voice_wav"], cfg=cfg)

    # Optional music mix
    intermediate_wav = paths["voice_wav"]
    if mix_music:
        if not music_path or not os.path.exists(music_path):
            raise FileNotFoundError("Music bed not found (upload one or disable mixing).")
        mix_music_under_voice(
            voice_wav=paths["voice_wav"],
            music_path=music_path,
            out_wav=paths["mixed_wav"],
            music_db=music_db,
            fade_s=fade_s,
        )
        intermediate_wav = paths["mixed_wav"]

    # Re-encode to unified WAV for concat reliability
    run_ffmpeg([
        FFMPEG, "-y",
        "-i", intermediate_wav,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        paths["final_wav"]
    ])

    # Unified WAV → MP3
    wav_to_mp3(paths["final_wav"], paths["mp3"], bitrate="192k")
    return paths

# ----------------------------
# Build controls + QC
# ----------------------------
st.subheader("🎙️ Build Audio Sections")

if not has_any_script():
    st.info("Generate a script first (Intro / Chapters / Outro) before building audio.")
else:
    st.session_state.setdefault("built_sections", {})
    built_sections: Dict[str, Dict[str, str]] = st.session_state["built_sections"]

    sections = section_texts()
    ordered_ids = [sid for sid, _ in sections]
    ids_with_text = [sid for sid, txt in sections if (txt or "").strip()]
    default_select = ["00_intro"] if "00_intro" in ids_with_text else (ids_with_text[:1] if ids_with_text else [])
    selected_ids = st.multiselect(
        "Select sections to build",
        options=ids_with_text,
        default=default_select,
        help="Tip: build Intro first to QC voice + pacing before building everything.",
    )

    force_rebuild = st.checkbox("Force rebuild (overwrite outputs)", value=False)

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        build_selected_btn = st.button("Build SELECTED", type="primary", disabled=not bool(selected_ids))
    with c2:
        build_all_btn = st.button("Build ALL Sections")
    with c3:
        clear_built_btn = st.button("Clear Built Audio (keep music)")

    if clear_built_btn:
        wd = _workdir()
        # Only remove generated artifacts; keep persisted music and workdir itself
        for name in os.listdir(wd):
            if name.startswith("__music__"):
                continue
            _safe_remove(os.path.join(wd, name))
        st.session_state["built_sections"] = {}
        st.success("Cleared built audio (kept persisted music bed).")

    def _build_ids(to_build: List[str]) -> None:
        with st.spinner("Building audio…"):
            prog = st.progress(0)
            total = max(len(to_build), 1)
            built = 0
            for sid, txt in sections:
                if sid not in to_build:
                    continue
                if not (txt or "").strip():
                    continue
                out_paths = build_one_section(
                    section_id=sid,
                    text=txt,
                    cfg=cfg,
                    mix_music=use_music,
                    music_path=music_path,
                    music_db=music_db_i,
                    fade_s=fade_s_i,
                    force=force_rebuild,
                )
                built_sections[sid] = out_paths
                built += 1
                prog.progress(min(built / total, 1.0))
            st.success("Build complete.")

    if build_selected_btn:
        _build_ids(selected_ids)
    if build_all_btn:
        _build_ids(ids_with_text)

    st.divider()
    st.subheader("🔎 Section QC (Playback + Downloads)")

    any_built = False
    for sid in ordered_ids:
        paths = built_sections.get(sid) or {}
        mp3_path = paths.get("mp3", "")
        txt_path = paths.get("txt", "")
        if mp3_path and os.path.exists(mp3_path):
            any_built = True
            dur = get_media_duration_seconds(mp3_path)
            with st.expander(f"{sid}  —  {dur:.1f}s", expanded=(sid == "00_intro")):
                st.audio(mp3_path)

                colx, coly = st.columns([1, 1])
                with colx:
                    with open(mp3_path, "rb") as f:
                        st.download_button(
                            f"Download {sid} MP3",
                            data=f.read(),
                            file_name=os.path.basename(mp3_path),
                            mime="audio/mpeg",
                        )
                with coly:
                    if txt_path and os.path.exists(txt_path):
                        with open(txt_path, "rb") as f:
                            st.download_button(
                                f"Download {sid} TXT",
                                data=f.read(),
                                file_name=os.path.basename(txt_path),
                                mime="text/plain",
                            )

    if not any_built:
        st.info("No built sections yet. Build Intro first to verify voice + pacing.")

# ----------------------------
# Build FULL episode MP3 (robust concat via unified WAV)
# ----------------------------
st.divider()
st.subheader("🎬 Build FULL Episode MP3")

ep_slug = episode_slug()
full_mp3_path = os.path.join(_workdir(), f"{ep_slug}__FULL.mp3")
full_wav_path = os.path.join(_workdir(), f"{ep_slug}__FULL.wav")

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    # Write concat list using double quotes and escaping double quotes
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in wav_paths:
            safe_p = p.replace('"', r'\"')
            f.write(f'file "{safe_p}"\n')
        list_path = f.name
    try:
        run_ffmpeg([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_wav])
    finally:
        _safe_remove(list_path)

def build_full_episode(built_sections: Dict[str, Dict[str, str]], ordered_ids: List[str]) -> None:
    wavs: List[str] = []
    for sid in ordered_ids:
        p = (built_sections.get(sid) or {}).get("final_wav")
        if p and os.path.exists(p):
            wavs.append(p)
    if not wavs:
        raise ValueError("No built section WAVs found. Build sections first.")

    _safe_remove(full_wav_path)
    _safe_remove(full_mp3_path)

    concat_wavs(wavs, full_wav_path)
    wav_to_mp3(full_wav_path, full_mp3_path, bitrate="192k")

if st.button("Build FULL Episode MP3", type="primary"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    ordered_ids = [sid for sid, _ in section_texts()]
    with st.spinner("Building full episode…"):
        try:
            build_full_episode(built_sections, ordered_ids)
            st.success("Full episode MP3 built.")
        except Exception as e:
            st.error(str(e))

if os.path.exists(full_mp3_path):
    dur = get_media_duration_seconds(full_mp3_path)
    st.caption(f"FULL episode duration: {dur/60.0:.1f} min")
    st.audio(full_mp3_path)
    with open(full_mp3_path, "rb") as f:
        st.download_button(
            "Download FULL Episode MP3",
            data=f.read(),
            file_name=os.path.basename(full_mp3_path),
            mime="audio/mpeg",
        )

# ----------------------------
# Final ZIP (clean + deterministic)
# ----------------------------
st.divider()
st.subheader("📦 Final Delivery ZIP")

include_full_in_zip = st.checkbox("Include FULL episode MP3 in ZIP (if built)", value=True)

if st.button("Build FINAL ZIP"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    files: List[Tuple[str, str]] = []

    # Optional FULL
    if include_full_in_zip and os.path.exists(full_mp3_path):
        files.append((full_mp3_path, os.path.basename(full_mp3_path)))

    # Section MP3 + TXT
    for sid in sorted(built_sections.keys()):
        paths = built_sections.get(sid) or {}
        mp3p = paths.get("mp3")
        txtp = paths.get("txt")
        if mp3p and os.path.exists(mp3p):
            files.append((mp3p, os.path.basename(mp3p)))
        if txtp and os.path.exists(txtp):
            files.append((txtp, os.path.basename(txtp)))

    # Master script + outline (always include)
    wd = _workdir()
    raw_path = os.path.join(wd, f"{ep_slug}__MASTER_SCRIPT.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write((st.session_state.get("MASTER_SCRIPT_TEXT", "") or "").strip() + "\n")
    files.append((raw_path, os.path.basename(raw_path)))

    outline_path = os.path.join(wd, f"{ep_slug}__OUTLINE.txt")
    with open(outline_path, "w", encoding="utf-8") as f:
        f.write((st.session_state.get("OUTLINE_TEXT_SRC", "") or "").strip() + "\n")
    files.append((outline_path, os.path.basename(outline_path)))

    zip_bytes = make_zip_bytes(files)
    st.download_button(
        "Download FINAL ZIP",
        data=zip_bytes,
        file_name=f"{ep_slug}__FINAL.zip",
        mime="application/zip",
    )

