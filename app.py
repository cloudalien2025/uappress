# ============================
# PART 1/4 — Core Setup, Sidebar, Clients, FFmpeg, Text + Script Parsing Utilities
# ============================
# app.py — UAPpress Documentary TTS Studio
# MODE: Script → Audio
# OPTIONAL: Super Master Prompt → Full Script → Auto-fill → Audio
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
from typing import Dict, List, Optional, Tuple

import streamlit as st
from openai import OpenAI
import imageio_ffmpeg


# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="UAPpress Documentary TTS Studio", layout="wide")
st.title("🛸 UAPpress — Documentary TTS Studio")
st.caption("Credibility-first investigative documentaries. Script → Audio.")


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


# =============================================================================
# 🔒 SUPER MASTER PROMPT v3.1 — PRESTIGE UFO / UAP INVESTIGATIVE DOCUMENTARY SYSTEM
# (Universal | Fact-Anchored | Character-Driven | Witness-Centered | Audio-First | High-Retention)
#
# NOTE:
# - Keep ROLE separate from CONTRACT so you can pass ROLE as system message and
#   CONTRACT as user message (or concatenate for a single user message).
# - This prompt is optimized for OpenAI TTS output: cadence, paragraph rhythm, and modular chapters.
# =============================================================================

# ----------------------------
# 🔒 SUPER MASTER PROMPT — ROLE (STYLE ONLY, NEVER ACKNOWLEDGED IN SCRIPT)
# ----------------------------
SUPER_ROLE_PROMPT = """
ROLE & AUTHORITY
(STYLE-SETTING ONLY — NEVER ACKNOWLEDGED IN SCRIPT)

You are the most trusted long-form investigative documentary creator in the field of unidentified aerial phenomena.

Your work is consumed by:
• Scientists
• Military professionals
• Intelligence analysts
• Journalists
• Skeptical long-form audiences

Your documentaries are cited, archived, debated, replayed, and scrutinized.

You are not:
• A believer
• A debunker
• A theorist
• A mystery narrator

You are a forensic storyteller reconstructing human decisions under uncertainty.

IMPORTANT:
This role defines investigative standards and narrative discipline only.
Do NOT reference or acknowledge this role inside the script itself.
""".strip()


# ----------------------------
# 🔒 SUPER MASTER PROMPT — RULES + OUTPUT CONTRACT (v3.1)
# ----------------------------
SUPER_SCRIPT_CONTRACT = r"""
🎯 OBJECTIVE (NON-NEGOTIABLE)

Generate a complete, audio-only, long-form investigative documentary script.

This script will be:
• Narrated entirely using OpenAI Text-to-Speech
• Exported as MP3 chapters
• Converted into MP4 documentary video

Write for listening endurance, not silent reading.
Target runtime: 45–120 minutes narrated.

🧠 CORE STORY LAW (ABSOLUTE)

This documentary is not about UFOs.

It is about:
• People trained to follow procedure
• Moments when procedure proves insufficient
• Decisions made with careers, credibility, and responsibility at stake
• How individuals and institutions respond differently to uncertainty

The phenomenon matters only because it forces people to choose, act, and live with consequences.

🧭 FACTUAL & TEMPORAL ANCHORING (HARD REQUIREMENT)

At the earliest appropriate moment, the script MUST:
• State the specific date or date range
• Clarify time-of-day ambiguity, especially when events cross midnight
• Distinguish between:
  – Calendar date
  – Operational shift timing
  – Later recollection phrasing

If events span multiple nights:
• Treat each night as a separate operational window
• Identify who was present each time

Forbidden vagueness:
• “Around Christmas”
• “During the holidays”
• “Late December” without dates

Precision is mandatory, even when uncertainty exists.

🧍 CHARACTER CENTRALITY LAW (NON-NEGOTIABLE)

Named individuals are not background texture.

They are:
• Decision-makers
• Witnesses
• Record-keepers
• Consequence-bearers

Rules:
• Characters must be present in scenes
• Their choices, hesitations, and constraints must be explicit
• Their professional or social risk must be clear
• Their later consequences must be tracked over time

Forbidden:
• Mythologizing
• Hero or villain framing
• Psychological speculation not supported by record

Characters are defined by what they did, documented, or refused to do.

🔥 NARRATIVE PRESSURE LAW (MANDATORY)

Every paragraph must contain at least one human pressure point:
• A decision made without full information
• A reporting threshold crossed
• A procedural rule strained or bypassed
• A professional or social risk introduced
• A consequence that cannot be undone

If a paragraph contains no pressure: Cut it.

🎙️ WITNESS TESTIMONY & HUMAN REACTION ENGINE (MANDATORY)

Witness testimony is evidence of perception, reaction, and decision-making, not proof of explanation.

Each witness scene must include at least three:
• Initial perception (before interpretation)
• Immediate reaction (shown through action or hesitation)
• Contextual constraint (rank, crowd, duty, fear, credibility risk)
• Decision point (what they did instead of speculating)
• Aftereffect (doubt, silence, consequence, persistence)

Temporal discipline:
• During the event
• Immediately after
• Years later
Later recollections require greater restraint.

Quote discipline:
• Use short verbatim fragments only
• Never stack quotes
• Every quote must trigger a reaction, decision, or consequence

🎧 AUDIO-FIRST / OPENAI TTS ENGINE (MANDATORY)

Assume the script will be narrated entirely by AI TTS.
Write for spoken clarity, not visual formatting.

Breath & cadence:
• Average sentence length: 12–22 words
• Long sentences must include natural pause points
• Avoid stacked subordinate clauses
If it cannot be spoken comfortably in one breath, split it.

Paragraph rhythm:
• Paragraphs: 3–5 sentences max
• One audible beat or action per paragraph
• No paragraph may contain more than one decision or revelation
Paragraph breaks must sound intentional.

Transitions:
• Scene changes must be embedded in language and audible without visuals
Forbidden: “Cut to”, “Fade”, visual-only cues

No stage directions:
Never write: “Pause”, “Emphasis”, “Dramatic”
Use rhythm and sentence length for weight.

Chapter modularity:
Each chapter must function as a standalone MP3.
Begin with immediate orientation. End on a clean narrative stop.
Avoid mid-thought endings.

📢 MANDATORY PRESERVATIONS

OPA NUTRITION (HARD LOCK)
INTRO — verbatim:
“This episode is sponsored by OPA Nutrition, makers of premium wellness supplements designed to support focus, clarity, energy, resilience, and long-term health. Explore their offerings at opanutrition.com.”

OUTRO — verbatim:
“This episode was sponsored by OPA Nutrition. For premium wellness supplements designed to support focus, clarity, energy, resilience, and long-term health, visit opanutrition.com.”

Do not paraphrase. Do not shorten. No disease claims anywhere.

AUDIENCE ENGAGEMENT (INTRO + OUTRO)
Must explicitly ask listeners to:
• Subscribe
• Comment where they’re listening from
• Suggest future cases/topics/documents to investigate
Tone: measured, human, unforced.

🚨 STRUCTURAL LAWS (ABSOLUTE)

ZERO REPETITION
No recaps. No reintroductions. No recycled phrasing.
No “as mentioned earlier”, “once again”, or timeline resets.

STRICT CONTINUITY
Time moves forward only.
Each chapter begins exactly where the previous chapter ends.
Assume perfect listener memory.

KEY PLAYERS LOCK-IN (INTRO ONLY)
Early in INTRO, include a tight Key Players section:
• 2–4 sentences total
• Each: FULL NAME + ROLE + why they matter (7–12 words)
Afterward: LAST NAMES ONLY. No reintroductions.

🧾 EVIDENTIARY DISCIPLINE (CRITICAL)

Clearly distinguish between:
• Documented fact
• Firsthand testimony
• Official explanation
• Later reinterpretation
• Speculation

Never blur categories. Never imply certainty.

🎭 TONE & DELIVERY

Calm. Controlled. Human. Precise.
Authority comes from clarity about people under pressure, not detachment.

⏱️ LENGTH REQUIREMENTS

Total script: 7,000–13,000 words
Chapters: 900–1,300 words each
No filler or transition-only chapters.

🚫 FORBIDDEN META LANGUAGE

Never write:
• “In this chapter…”
• “In the next chapter…”
• “We will… / We’re going to…”
• “As discussed earlier…”
• “You won’t believe…”

📄 MODEL OUTPUT FORMAT (MANDATORY)

Return ONLY the script text in plain text.
No commentary. No analysis. No bullet points. No markdown.

Use exactly this structure and headings:

INTRO
<Intro narration>

CHAPTER 1: <Title>
<Chapter narration>

CHAPTER 2: <Title>
<Chapter narration>

CHAPTER 3: <Title>
<Chapter narration>

(continue sequentially as needed…)

OUTRO
<Outro narration>

✅ FINAL COMMAND

Generate the complete long-form investigative documentary script in one response.
""".strip()

# ----------------------------
# Convenience: one combined prompt (if you prefer single user message)
# ----------------------------
SUPER_PROMPT_FULL = f"{SUPER_ROLE_PROMPT}\n\n{SUPER_SCRIPT_CONTRACT}".strip()

# ----------------------------
# Sidebar — API Key + High-Level Settings
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
    st.session_state.setdefault("ENABLE_SCRIPT_GEN", True)
    st.checkbox("Enable Master Prompt → Script", key="ENABLE_SCRIPT_GEN")

    st.caption("Uses the locked investigative Master Prompt + strict template.")


# ----------------------------
# OpenAI client
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


def sanitize_for_tts(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ----------------------------
# Streamlit state (single source of truth)
# ----------------------------
def text_key(kind: str, idx: int = 0) -> str:
    return f"text::{kind}::{idx}"


def ensure_text_key(kind: str, idx: int = 0, default: str = "") -> None:
    st.session_state.setdefault(text_key(kind, idx), default or "")


def reset_script_text_fields(chapter_count: int) -> None:
    ensure_text_key("intro", 0, "")
    ensure_text_key("outro", 0, "")
    for i in range(1, chapter_count + 1):
        ensure_text_key("chapter", i, "")


def ensure_state() -> None:
    st.session_state.setdefault("episode_title", "Untitled Episode")
    st.session_state.setdefault("MASTER_PROMPT_INPUT", "")
    st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")
    st.session_state.setdefault("chapter_count", 0)
    reset_script_text_fields(st.session_state["chapter_count"])


ensure_state()


# ----------------------------
# Master Script parsing (STRICT headings)
# ----------------------------
_CHAPTER_RE = re.compile(r"(?i)^\s*chapter\s+(\d+)\s*(?:[:\-–—]\s*(.*))?\s*$")

def detect_chapters_and_titles(master_text: str) -> Dict[int, str]:
    """
    Returns {chapter_number: chapter_title_or_empty}
    Used ONLY to detect how many chapters exist.
    """
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

def parse_master_script(master_text: str, expected_chapters: int) -> Dict[str, object]:
    """
    Parses INTRO / CHAPTER N / OUTRO.
    Heading lines are NOT included in narration output.
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

    markers = sorted([i for i in [intro_i, outro_i, *chapter_i.values()] if i is not None])
    bounds = {markers[i]: markers[i + 1] if i + 1 < len(markers) else len(lines) for i in range(len(markers))}

    def block(start: Optional[int]) -> str:
        if start is None:
            return ""
        return "\n".join(lines[start + 1 : bounds[start]]).strip()

    return {
        "intro": block(intro_i),
        "outro": block(outro_i),
        "chapters": {n: block(i) for n, i in chapter_i.items()},
    }

# ============================
# PART 2/4 — Script Generation (Contract-Hardened)
# ============================
# Goal: Prevent fictional drift by forcing structured JSON output + deterministic validators + repair loop.
# - No novelization, no invented dialogue, no mind-reading
# - Last-names-only AFTER Key Players
# - Evidentiary labeling on every paragraph
# - Procedural pressure only (no cinematic prose)
#
# Assumes earlier parts provide:
# - st, OpenAI client as `client` (or you can instantiate here)
# - user inputs in st.session_state as needed (topic, scope, runtime, chapter count, etc.)
# - downstream parts expect `st.session_state["generated_script_text"]` and/or `st.session_state["generated_script_json"]`

import json
import re
from typing import Any, Dict, List, Tuple

# ----------------------------
# Contract + Schema
# ----------------------------

EVIDENCE_LABELS = ["FACT", "REPORT", "STATEMENT", "ANALYSIS", "DISPUTED"]
EVIDENCE_TAG_RE = re.compile(r"^\[(FACT|REPORT|STATEMENT|ANALYSIS|DISPUTED)\]\s+")

FORBIDDEN_DIALOGUE_RE = re.compile(
    r'(".*?"|“.*?”|‘.*?’|\b(he said|she said|they said|I said|we said|told me|told us|replied|shouted|whispered)\b)',
    re.IGNORECASE | re.DOTALL,
)

FORBIDDEN_MINDREADING_RE = re.compile(
    r"\b(thought|felt|feels|fear(ed)?|hoped|wanted|knew|realized|remembered|believed|decided|wondered|couldn't help but)\b",
    re.IGNORECASE,
)

FORBIDDEN_NOVELIZATION_RE = re.compile(
    r"\b(moonlit|moonlight|dusky|eerie|ominous|haunting|chilling|blood[- ]?red|piercing|deafening|silence fell|the air was thick|"
    r"shadows|glowed|shimmered|echoed|rushed|heartbeat|sweat|trembled|quivered|gasped)\b",
    re.IGNORECASE,
)

FORBIDDEN_CERTAINTY_RE = re.compile(
    r"\b(proves|proof that|definitively|without question|no doubt|certainly|undeniable|conclusively)\b",
    re.IGNORECASE,
)

FIRST_NAME_TOKEN_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")  # heuristic; filtered by allowlist later


SCRIPT_JSON_SCHEMA: Dict[str, Any] = {
    "name": "uappress_investigative_script_v1",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "central_question", "key_players", "chapters"],
        "properties": {
            "title": {"type": "string", "minLength": 8, "maxLength": 120},
            "central_question": {"type": "string", "minLength": 12, "maxLength": 220},
            "key_players": {
                "type": "array",
                "minItems": 2,
                "maxItems": 25,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["full_name", "last_name", "role", "why_relevant", "source_types"],
                    "properties": {
                        "full_name": {"type": "string", "minLength": 3, "maxLength": 80},
                        "last_name": {"type": "string", "minLength": 2, "maxLength": 40},
                        "role": {"type": "string", "minLength": 2, "maxLength": 120},
                        "why_relevant": {"type": "string", "minLength": 10, "maxLength": 260},
                        "source_types": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 8,
                            "items": {
                                "type": "string",
                                "enum": [
                                    "Official statement",
                                    "Contemporaneous press",
                                    "Court/Legal record",
                                    "Government document",
                                    "Academic/Scholarly work",
                                    "Book (secondary)",
                                    "Interview/Testimony",
                                    "FOIA release",
                                    "Media investigation",
                                    "Unknown/Unverified",
                                ],
                            },
                        },
                    },
                },
            },
            "chapters": {
                "type": "array",
                "minItems": 3,
                "maxItems": 20,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["chapter_number", "chapter_title", "start_where_previous_ended", "segments"],
                    "properties": {
                        "chapter_number": {"type": "integer", "minimum": 1, "maximum": 50},
                        "chapter_title": {"type": "string", "minLength": 6, "maxLength": 120},
                        "start_where_previous_ended": {"type": "string", "minLength": 10, "maxLength": 240},
                        "segments": {
                            "type": "array",
                            "minItems": 6,
                            "maxItems": 80,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["evidence_label", "text", "source_types"],
                                "properties": {
                                    "evidence_label": {"type": "string", "enum": EVIDENCE_LABELS},
                                    "text": {"type": "string", "minLength": 30, "maxLength": 1200},
                                    "source_types": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 5,
                                        "items": {
                                            "type": "string",
                                            "enum": [
                                                "Official statement",
                                                "Contemporaneous press",
                                                "Court/Legal record",
                                                "Government document",
                                                "Academic/Scholarly work",
                                                "Book (secondary)",
                                                "Interview/Testimony",
                                                "FOIA release",
                                                "Media investigation",
                                                "Unknown/Unverified",
                                            ],
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

CONTRACT_SYSTEM = """You are UAPpress Documentary TTS Studio — Script Generator.
You MUST obey the Investigative Documentary Contract below. If any instruction conflicts, the Contract wins.

INVESTIGATIVE DOCUMENTARY CONTRACT (NON-NEGOTIABLE)
1) NO NOVELIZATION: Do not write cinematic prose, scene-setting, emotions, sensory detail, or dramatization.
2) NO INVENTED DIALOGUE: No quotes, no reconstructed conversations, no “he said/she said” paraphrase.
3) NO MIND-READING: Do not claim what anyone thought, felt, intended, feared, or believed unless explicitly documented, and if documented label it as STATEMENT and attribute neutrally.
4) EVIDENTIARY LABELING: Every paragraph must be tagged as exactly one of: [FACT], [REPORT], [STATEMENT], [ANALYSIS], [DISPUTED].
   - FACT: widely corroborated / primary records; REPORT: contemporaneous journalism; STATEMENT: attributed claim/testimony; ANALYSIS: careful reasoning without new facts; DISPUTED: conflicts/uncertainty and why.
5) LAST-NAMES-ONLY RULE: You may introduce a person in Key Players with full name. After Key Players, refer to any person ONLY by last name.
6) PROCEDURAL PRESSURE ONLY: Tension comes from timelines, gaps, contradictions, incentives, and institutional process — not from horror/sci-fi tone.
7) NO OVERCLAIMS: Avoid certainty words like “proves/definitive/undeniable.” Use careful, qualified language.
8) OUTPUT FORMAT: You MUST output valid JSON that matches the provided JSON Schema. Do not include any non-JSON text.
"""

def _build_user_brief() -> str:
    topic = (st.session_state.get("topic") or st.session_state.get("case_topic") or "").strip()
    if not topic:
        topic = "User-provided case/topic (not specified)"
    runtime_min = st.session_state.get("target_runtime_min") or st.session_state.get("runtime_min") or 45
    chapter_count = st.session_state.get("chapter_count") or st.session_state.get("chapters") or 10
    region = (st.session_state.get("region") or "").strip()
    scope = (st.session_state.get("scope") or "serious, historically grounded investigation").strip()
    constraints = (st.session_state.get("constraints") or "").strip()

    brief = {
        "topic": topic,
        "scope": scope,
        "target_runtime_min": int(runtime_min),
        "chapter_count": int(chapter_count),
        "region": region,
        "extra_constraints": constraints,
        "hard_rules_reminder": {
            "no_novelization": True,
            "no_invented_dialogue": True,
            "no_mind_reading": True,
            "last_names_only_after_key_players": True,
            "every_paragraph_labeled": EVIDENCE_LABELS,
            "procedural_pressure_only": True,
        },
        "narration_style": "calm, precise, investigative; no hype; no recap; forward-moving chronology; each chapter starts where previous ends",
    }
    return json.dumps(brief, ensure_ascii=False, indent=2)

# ----------------------------
# Validators (deterministic)
# ----------------------------

def _collect_key_player_names(script_json: Dict[str, Any]) -> Tuple[set, set]:
    full_names = set()
    last_names = set()
    for kp in script_json.get("key_players", []) or []:
        fn = (kp.get("full_name") or "").strip()
        ln = (kp.get("last_name") or "").strip()
        if fn:
            full_names.add(fn)
        if ln:
            last_names.add(ln)
    return full_names, last_names

def _text_has_forbidden_patterns(text: str) -> List[str]:
    hits = []
    if FORBIDDEN_DIALOGUE_RE.search(text or ""):
        hits.append("invented_dialogue_or_quotes")
    if FORBIDDEN_MINDREADING_RE.search(text or ""):
        hits.append("mind_reading_language")
    if FORBIDDEN_NOVELIZATION_RE.search(text or ""):
        hits.append("novelization_cinematic_prose")
    if FORBIDDEN_CERTAINTY_RE.search(text or ""):
        hits.append("overclaiming_certainty")
    return hits

def _find_first_name_mentions_outside_key_players(script_json: Dict[str, Any]) -> List[str]:
    """
    After Key Players, only last names allowed. This flags likely first-name mentions
    by scanning capitalized tokens and filtering out:
    - evidence labels, common sentence starters, months, acronyms
    - last names from key players (allowed)
    """
    _, allowed_last = _collect_key_player_names(script_json)

    # common allowlist tokens (expand if needed)
    allow_tokens = {
        "The", "A", "An", "In", "On", "At", "By", "To", "From", "And", "But", "Or", "For", "With", "Without",
        "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December",
        "UAP", "UFO", "US", "USA", "Navy", "Air", "Force", "DoD", "CIA", "NSA", "FBI", "DIA", "AARO", "NASA", "NORAD",
        "Project", "Operation", "Department", "Committee", "Congress", "Senate", "House",
    }
    suspicious: List[str] = []

    for ch in script_json.get("chapters", []) or []:
        for seg in ch.get("segments", []) or []:
            txt = seg.get("text") or ""
            # Collect capitalized tokens
            tokens = FIRST_NAME_TOKEN_RE.findall(txt)
            for t in tokens:
                if t in allow_tokens:
                    continue
                if t in allowed_last:
                    continue
                # If token appears as evidence label prefix, skip
                if t in EVIDENCE_LABELS:
                    continue
                # Heuristic: If token ends sentence start, we still consider suspicious (can't be sure)
                # but reduce spam by ignoring tokens following punctuation + space pattern? (Not reliable)
                suspicious.append(t)

    # De-duplicate, keep stable order
    seen = set()
    out = []
    for t in suspicious:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out

def _validate_script_json(script_json: Dict[str, Any]) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    # Basic structure checks
    if not isinstance(script_json, dict):
        return False, ["output_not_json_object"]

    if "key_players" not in script_json or not script_json.get("key_players"):
        problems.append("missing_key_players")

    if "chapters" not in script_json or not script_json.get("chapters"):
        problems.append("missing_chapters")

    # Segment-level checks
    for ch in script_json.get("chapters", []) or []:
        segments = ch.get("segments", []) or []
        if not segments:
            problems.append(f"chapter_{ch.get('chapter_number','?')}_has_no_segments")
            continue

        for seg in segments:
            label = (seg.get("evidence_label") or "").strip()
            txt = (seg.get("text") or "").strip()

            if label not in EVIDENCE_LABELS:
                problems.append("missing_or_invalid_evidence_label")
                break

            if not txt:
                problems.append("empty_segment_text")
                continue

            # Must not include a leading bracket label in text (label stored separately)
            if EVIDENCE_TAG_RE.match(txt):
                problems.append("embedded_bracket_label_in_text_instead_of_field")
                break

            hits = _text_has_forbidden_patterns(txt)
            for h in hits:
                problems.append(h)

            # procedural pressure only: flag sci-fi/horror keywords
            if re.search(r"\b(alien(s)?|extraterrestrial|spaceship|craft from another world|interdimensional)\b", txt, re.IGNORECASE):
                problems.append("sensational_terms_present")
            # discourage "we will show" style promises
            if re.search(r"\b(you will see|we will prove|this will reveal)\b", txt, re.IGNORECASE):
                problems.append("future_promise_hype_language")

    # Last-names-only check
    suspicious_firsts = _find_first_name_mentions_outside_key_players(script_json)
    if suspicious_firsts:
        # Don’t fail on one token; fail if there are multiple or if a known first-name appears
        if len(suspicious_firsts) >= 2:
            problems.append(f"possible_first_name_mentions_after_key_players: {', '.join(suspicious_firsts[:12])}")

    ok = len(problems) == 0
    # de-dup problems
    problems_unique = []
    seen = set()
    for p in problems:
        if p not in seen:
            seen.add(p)
            problems_unique.append(p)
    return ok, problems_unique

# ----------------------------
# OpenAI call (JSON Schema enforced) + Repair Loop
# ----------------------------

def _call_openai_json(schema: Dict[str, Any], user_brief_json: str, model: str) -> Dict[str, Any]:
    """
    Attempts to force valid JSON via responses API with json_schema.
    Falls back to chat.completions if responses API not available, still enforcing JSON-only text.
    """
    # Primary: Responses API (preferred for strict response_format)
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CONTRACT_SYSTEM},
                {"role": "user", "content": f"Generate the script JSON now. Use only the provided brief.\n\nBRIEF (JSON):\n{user_brief_json}"},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": schema,
            },
        )
        text = getattr(resp, "output_text", None)
        if not text:
            # Some SDK variants store content in output[0].content[0].text
            try:
                text = resp.output[0].content[0].text  # type: ignore
            except Exception:
                text = ""
        return json.loads(text)
    except Exception:
        pass

    # Fallback: Chat Completions (best-effort JSON-only)
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CONTRACT_SYSTEM},
            {"role": "user", "content": f"OUTPUT MUST BE VALID JSON ONLY (no markdown). Generate the script JSON now.\n\nBRIEF (JSON):\n{user_brief_json}"},
        ],
        temperature=0.2,
    )
    content = (resp2.choices[0].message.content or "").strip()
    # Strip accidental code fences
    content = re.sub(r"^```(json)?\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()
    return json.loads(content)

def _repair_with_openai(
    *,
    prior_json: Dict[str, Any],
    violations: List[str],
    user_brief_json: str,
    model: str,
) -> Dict[str, Any]:
    repair_instructions = {
        "task": "Repair the JSON to fully comply with the Investigative Documentary Contract.",
        "violations_detected": violations,
        "repair_rules": [
            "Return JSON ONLY. No commentary.",
            "Do NOT add invented dialogue or quotes.",
            "Remove mind-reading and cinematic prose.",
            "Ensure last-names-only after Key Players.",
            "Keep every segment labeled via evidence_label field.",
            "Avoid certainty/overclaim language.",
        ],
    }
    # Primary: Responses API with schema again
    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": CONTRACT_SYSTEM},
                {"role": "user", "content": f"BRIEF (JSON):\n{user_brief_json}"},
                {"role": "user", "content": f"VIOLATIONS + REPAIR INSTRUCTIONS (JSON):\n{json.dumps(repair_instructions, ensure_ascii=False, indent=2)}"},
                {"role": "user", "content": f"PRIOR OUTPUT JSON:\n{json.dumps(prior_json, ensure_ascii=False)}"},
                {"role": "user", "content": "Return corrected JSON that matches the JSON Schema now."},
            ],
            response_format={"type": "json_schema", "json_schema": SCRIPT_JSON_SCHEMA},
        )
        text = getattr(resp, "output_text", None)
        if not text:
            try:
                text = resp.output[0].content[0].text  # type: ignore
            except Exception:
                text = ""
        return json.loads(text)
    except Exception:
        pass

    # Fallback: Chat Completions
    resp2 = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CONTRACT_SYSTEM},
            {"role": "user", "content": f"BRIEF (JSON):\n{user_brief_json}"},
            {"role": "user", "content": f"VIOLATIONS + REPAIR INSTRUCTIONS (JSON):\n{json.dumps(repair_instructions, ensure_ascii=False, indent=2)}"},
            {"role": "user", "content": f"PRIOR OUTPUT JSON:\n{json.dumps(prior_json, ensure_ascii=False)}"},
            {"role": "user", "content": "OUTPUT MUST BE VALID JSON ONLY. Return corrected JSON that matches the JSON Schema now."},
        ],
        temperature=0.1,
    )
    content = (resp2.choices[0].message.content or "").strip()
    content = re.sub(r"^```(json)?\s*", "", content, flags=re.IGNORECASE).strip()
    content = re.sub(r"\s*```$", "", content).strip()
    return json.loads(content)

def _render_script_text(script_json: Dict[str, Any]) -> str:
    """
    Produces narration-ready plain text with explicit evidence tags on each paragraph.
    Ensures output is predictable for downstream TTS.
    """
    lines: List[str] = []
    lines.append(script_json.get("title", "").strip())
    lines.append("")
    lines.append(f"CENTRAL QUESTION: {script_json.get('central_question','').strip()}")
    lines.append("")
    lines.append("KEY PLAYERS")
    for kp in script_json.get("key_players", []) or []:
        full_name = (kp.get("full_name") or "").strip()
        role = (kp.get("role") or "").strip()
        why = (kp.get("why_relevant") or "").strip()
        srcs = kp.get("source_types") or []
        srcs_s = ", ".join(srcs) if isinstance(srcs, list) else str(srcs)
        lines.append(f"- {full_name} — {role}. {why} (Source types: {srcs_s})")
    lines.append("")
    lines.append("SCRIPT")
    lines.append("")

    chapters = script_json.get("chapters", []) or []
    for ch in chapters:
        n = ch.get("chapter_number")
        title = (ch.get("chapter_title") or "").strip()
        start = (ch.get("start_where_previous_ended") or "").strip()
        lines.append(f"CHAPTER {n}: {title}")
        lines.append(f"START: {start}")
        lines.append("")
        for seg in ch.get("segments", []) or []:
            label = (seg.get("evidence_label") or "").strip()
            txt = (seg.get("text") or "").strip()
            srcs = seg.get("source_types") or []
            srcs_s = ", ".join(srcs) if isinstance(srcs, list) else str(srcs)
            # Each paragraph must be tagged
            lines.append(f"[{label}] {txt} (Source types: {srcs_s})")
            lines.append("")
        lines.append("")  # chapter gap

    return "\n".join(lines).strip() + "\n"

# ----------------------------
# Streamlit UI + Generation
# ----------------------------

st.subheader("Part 2/4 — Script Generation (Contract-Hardened)")

if "generated_script_json" not in st.session_state:
    st.session_state["generated_script_json"] = None
if "generated_script_text" not in st.session_state:
    st.session_state["generated_script_text"] = ""
if "script_generation_log" not in st.session_state:
    st.session_state["script_generation_log"] = []

model_name = st.session_state.get("script_model") or st.session_state.get("model") or "gpt-4.1-mini"
max_passes = int(st.session_state.get("script_max_passes") or 3)

with st.expander("Generator Settings", expanded=False):
    st.write("Model:", model_name)
    st.write("Max repair passes:", max_passes)
    st.caption("Contract is enforced via JSON Schema + deterministic validators + repair loop.")

user_brief_json = _build_user_brief()

colA, colB = st.columns([1, 1])
with colA:
    if st.button("Generate Script (Strict)", type="primary", use_container_width=True):
        st.session_state["script_generation_log"] = []
        last_json: Dict[str, Any] = {}
        try:
            # Pass 1
            out_json = _call_openai_json(SCRIPT_JSON_SCHEMA, user_brief_json, model_name)
            ok, problems = _validate_script_json(out_json)
            st.session_state["script_generation_log"].append({"pass": 1, "ok": ok, "problems": problems})
            last_json = out_json

            # Repairs
            pass_num = 1
            while (not ok) and pass_num < max_passes:
                pass_num += 1
                out_json = _repair_with_openai(
                    prior_json=last_json,
                    violations=problems,
                    user_brief_json=user_brief_json,
                    model=model_name,
                )
                ok, problems = _validate_script_json(out_json)
                st.session_state["script_generation_log"].append({"pass": pass_num, "ok": ok, "problems": problems})
                last_json = out_json

            # Final gate: if still not ok, hard fail (do not emit unsafe prose)
            if not ok:
                st.error("Script blocked by contract enforcement. Fix your brief (scope/players) and retry.")
                st.session_state["generated_script_json"] = None
                st.session_state["generated_script_text"] = ""
            else:
                st.session_state["generated_script_json"] = last_json
                st.session_state["generated_script_text"] = _render_script_text(last_json)
                st.success("Script generated and validated against the contract.")
        except Exception as e:
            st.session_state["generated_script_json"] = None
            st.session_state["generated_script_text"] = ""
            st.error(f"Generation failed: {e}")

with colB:
    if st.button("Clear Generated Script", use_container_width=True):
        st.session_state["generated_script_json"] = None
        st.session_state["generated_script_text"] = ""
        st.session_state["script_generation_log"] = []
        st.toast("Cleared.", icon="🧹")

# Output previews
if st.session_state.get("script_generation_log"):
    with st.expander("Contract Enforcement Log", expanded=False):
        st.json(st.session_state["script_generation_log"])

script_text = st.session_state.get("generated_script_text") or ""
if script_text:
    st.text_area("Generated Script (Narration-Ready)", value=script_text, height=520)

    # Optional: expose JSON for downstream parts (TTS segmentation, etc.)
    with st.expander("Generated Script JSON", expanded=False):
        st.json(st.session_state.get("generated_script_json"))


# ============================
# PART 3/4 — Outline → Full Script Pipeline (STREAMLIT-SAFE) — COPY/PASTE
# ============================
# Fixes included:
# ✅ No widget-owned key writes after widget is created (OUTLINE_TEXT_UI vs OUTLINE_TEXT_SRC)
# ✅ "Generate Full Script" calls generate_master_script_one_shot(master_prompt=OUTLINE_TEXT_SRC)
# ✅ Script Review (Editable) UI is OUTSIDE the try/except (prevents SyntaxError)
# ✅ Clear Outline + Clear Script included
# ✅ Works even on reruns (renders Script Review if chapter_count > 0)

st.header("1️⃣ Script Creation")

# ----------------------------
# Session defaults
# ----------------------------
st.session_state.setdefault("episode_title", "Untitled Episode")
st.session_state.setdefault("DEFAULT_CHAPTERS", 8)

# Outline state split (widget vs source)
st.session_state.setdefault("OUTLINE_TEXT_UI", "")
st.session_state.setdefault("OUTLINE_TEXT_SRC", "")
st.session_state.setdefault("_PENDING_OUTLINE_SYNC", False)

# Script state
st.session_state.setdefault("MASTER_SCRIPT_TEXT", "")
st.session_state.setdefault("chapter_count", 0)

# ----------------------------
# Helpers
# ----------------------------
def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return int(default)

def _queue_outline_to_ui(outline_text: str) -> None:
    st.session_state["OUTLINE_TEXT_SRC"] = (outline_text or "").strip()
    st.session_state["_PENDING_OUTLINE_SYNC"] = True
    st.rerun()

# Preload outline BEFORE widget is instantiated (safe)
if st.session_state.get("_PENDING_OUTLINE_SYNC"):
    st.session_state["_PENDING_OUTLINE_SYNC"] = False
    st.session_state["OUTLINE_TEXT_UI"] = st.session_state.get("OUTLINE_TEXT_SRC", "")

# ----------------------------
# Episode metadata
# ----------------------------
colA, colB = st.columns([3, 1])
with colA:
    st.text_input("Episode Title", key="episode_title")
with colB:
    st.number_input(
        "Chapter Count",
        min_value=3,
        max_value=20,
        step=1,
        key="DEFAULT_CHAPTERS",
    )

st.divider()

# ----------------------------
# Outline UI
# ----------------------------
st.subheader("🧠 Outline / Treatment")

st.text_area(
    "Outline (editable)",
    key="OUTLINE_TEXT_UI",
    height=320,
    help="Edit freely. This outline guides the full documentary. It is NOT narrated.",
)

# Mirror UI → SRC (safe)
st.session_state["OUTLINE_TEXT_SRC"] = (st.session_state.get("OUTLINE_TEXT_UI") or "").strip()

title_ok = bool((st.session_state.get("episode_title") or "").strip())
outline_ok = bool((st.session_state.get("OUTLINE_TEXT_SRC") or "").strip())

b1, b2, b3 = st.columns([1, 1, 1])

with b1:
    gen_outline = st.button("Generate Outline", type="primary", disabled=not title_ok)
with b2:
    gen_script = st.button("Generate Full Script", disabled=not outline_ok)
with b3:
    clear_outline = st.button("Clear Outline")

if clear_outline:
    st.session_state["OUTLINE_TEXT_SRC"] = ""
    st.session_state["OUTLINE_TEXT_UI"] = ""  # safe because widget reads it at next rerun
    st.session_state["_PENDING_OUTLINE_SYNC"] = False
    st.success("Outline cleared.")
    st.rerun()

st.caption(
    "Recommended: Review the outline carefully. This locks structure, witnesses, and chronology "
    "before generating a long-form script."
)

st.divider()

# ----------------------------
# Generate OUTLINE (one-shot)
# ----------------------------
if gen_outline:
    title = (st.session_state.get("episode_title") or "").strip()
    chapters_n = _safe_int(st.session_state.get("DEFAULT_CHAPTERS", 8), 8)

    with st.spinner("Generating outline…"):
        try:
            outline_prompt = f"""
DOCUMENTARY TITLE:
{title}

CHAPTER COUNT:
{chapters_n}

TASK:
Generate a documentary OUTLINE ONLY.

FORMAT:
INTRO
- Time, place, institutional context
- Key players (full names + roles)
- Stakes and triggering decision

CHAPTER 1–N:
Each chapter must specify:
- Central decision
- Key witness(es)
- Document or evidence
- Institutional response
- Consequence or unresolved tension

OUTRO:
- Human consequences
- Institutional outcome
- Open questions

RULES:
- No narration prose
- No summaries
- No speculation
- No meta commentary
- Clear chronology
""".strip()

            r = client.chat.completions.create(
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": SUPER_ROLE_PROMPT},
                    {"role": "system", "content": SUPER_SCRIPT_CONTRACT},
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

# ----------------------------
# Generate FULL SCRIPT (uses OUTLINE_TEXT_SRC)
# ----------------------------
if gen_script:
    title = (st.session_state.get("episode_title") or "").strip()
    outline = (st.session_state.get("OUTLINE_TEXT_SRC") or "").strip()
    chapters_n = _safe_int(st.session_state.get("DEFAULT_CHAPTERS", 8), 8)

    with st.spinner("Generating full script…"):
        try:
            raw = generate_master_script_one_shot(
                topic=title,
                master_prompt=outline,  # ✅ correct argument name
                chapters=chapters_n,
                model=st.session_state.get("SCRIPT_MODEL", "gpt-4o-mini"),
            )

            raw = (raw or "").strip()
            if not raw:
                raise ValueError("Script generation returned empty output.")

            st.session_state["MASTER_SCRIPT_TEXT"] = raw

            chapter_map = detect_chapters_and_titles(raw)
            n = max(chapter_map.keys()) if chapter_map else 0
            if n <= 0:
                raise ValueError("No chapters detected in generated script.")

            st.session_state["chapter_count"] = n
            reset_script_text_fields(n)

            parsed = parse_master_script(raw, n)
            st.session_state[text_key("intro", 0)] = (parsed.get("intro", "") or "").strip()
            for i in range(1, n + 1):
                st.session_state[text_key("chapter", i)] = (parsed.get("chapters", {}).get(i, "") or "").strip()
            st.session_state[text_key("outro", 0)] = (parsed.get("outro", "") or "").strip()

            st.success(f"Script generated: Intro + {n} chapters + Outro")

        except Exception as e:
            st.error(str(e))

# ----------------------------
# Script Review (Editable) — IMPORTANT: OUTSIDE try/except
# ----------------------------
if int(st.session_state.get("chapter_count", 0)) > 0:
    st.header("2️⃣ Script Review (Editable)")

    # Intro
    st.subheader("INTRO")
    st.text_area(
        "Intro narration",
        key=text_key("intro", 0),
        height=240,
    )

    st.divider()

    # Chapters
    chapter_count = int(st.session_state.get("chapter_count", 0))
    for i in range(1, chapter_count + 1):
        with st.expander(f"CHAPTER {i}", expanded=(i == 1)):
            st.text_area(
                f"Chapter {i} narration",
                key=text_key("chapter", i),
                height=340,
            )

    st.divider()

    # Outro
    st.subheader("OUTRO")
    st.text_area(
        "Outro narration",
        key=text_key("outro", 0),
        height=240,
    )

    # ----------------------------
    # Reset controls
    # ----------------------------
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        if st.button("Clear Script"):
            reset_script_text_fields(st.session_state.get("chapter_count", 0))
            st.session_state["chapter_count"] = 0
            st.session_state["MASTER_SCRIPT_TEXT"] = ""
            st.success("Script cleared.")
            st.rerun()

    with c2:
        if st.button("Clear Outline (keep script)"):
            st.session_state["OUTLINE_TEXT_SRC"] = ""
            st.session_state["_PENDING_OUTLINE_SYNC"] = True
            st.rerun()

    with c3:
        st.caption("Clearing text does not affect any already-generated audio.")
else:
    st.info("Generate a script to enable the editable Intro / Chapters / Outro review panel.")

            
# ============================
# PART 4/4 — Audio Build (Per-Section + Full Episode) + QC + Downloads + Final ZIP — PATCHED
# ============================
# Goals of this patch:
# ✅ Build ALL or build SELECTED sections (Intro-only QC is fast)
# ✅ Per-section QC: audio player + duration + download
# ✅ Robust FULL episode build (re-encode WAVs → concat WAV → encode MP3) to avoid mp3 concat/copy glitches
# ✅ Optional "Force rebuild" per run (ignores cache + overwrites outputs)
# ✅ Music upload stored into WORKDIR (persists across reruns; avoids temp file disappearing)
# ✅ Final ZIP is clean: section mp3 + section txt + optional FULL mp3 + master script txt

st.header("3️⃣ Create Audio (Onyx + Music)")

import tempfile
import os
from typing import Dict, List, Tuple

# ----------------------------
# Workdir (Streamlit-safe)
# ----------------------------
def _workdir() -> str:
    st.session_state.setdefault("WORKDIR", tempfile.mkdtemp(prefix="uappress_tts_"))
    return st.session_state["WORKDIR"]

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _safe_remove(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

def write_text_file(path: str, text: str) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write((text or "").strip() + "\n")

def episode_slug() -> str:
    title = st.session_state.get("episode_title", "Untitled Episode")
    return clean_filename(title)

def section_output_paths(section_id: str, episode_slug_: str) -> Dict[str, str]:
    wd = _workdir()
    base = f"{episode_slug_}__{section_id}"
    return {
        "voice_wav": os.path.join(wd, f"{base}__voice.wav"),
        "mixed_wav": os.path.join(wd, f"{base}__mix.wav"),
        "final_wav": os.path.join(wd, f"{base}__final.wav"),  # unified 48k stereo WAV for concat safety
        "mp3": os.path.join(wd, f"{base}.mp3"),
        "txt": os.path.join(wd, f"{base}.txt"),
    }

# ----------------------------
# Duration helper (ffprobe via imageio_ffmpeg bundled ffmpeg)
# ----------------------------
def _duration_seconds(media_path: str) -> float:
    try:
        return float(get_media_duration_seconds(media_path))
    except Exception:
        return 0.0

# ----------------------------
# Music upload / selection (persist inside WORKDIR)
# ----------------------------
st.subheader("🎵 Music Bed (Optional)")

music_file = st.file_uploader(
    "Optional music bed (mp3/wav/m4a)",
    type=["mp3", "wav", "m4a"],
)

# Persist chosen/uploaded music path in session_state
st.session_state.setdefault("MUSIC_PATH", "")
music_path = ""

if music_file is not None:
    # Save upload into WORKDIR to survive reruns
    wd = _workdir()
    _ensure_dir(wd)
    save_name = clean_filename(music_file.name)
    persisted = os.path.join(wd, f"__music__{save_name}")
    with open(persisted, "wb") as f:
        f.write(music_file.read())
    st.session_state["MUSIC_PATH"] = persisted
    music_path = persisted
else:
    # Fall back to last uploaded or a configured default
    music_path = st.session_state.get("MUSIC_PATH", "") or st.session_state.get("DEFAULT_MUSIC_PATH", "") or ""

use_music = st.checkbox(
    "Mix music under voice",
    value=True,
    help="If enabled, a music bed is mixed under narration with fades + loudnorm.",
)

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

# ----------------------------
# Build one section
# ----------------------------
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
    """
    Produces:
      - voice WAV
      - optional mixed WAV
      - final_wav (unified WAV for safe concat)
      - MP3
      - TXT (clean narration)
    """
    slug = episode_slug()
    paths = section_output_paths(section_id, slug)

    cleaned = strip_tts_directives(sanitize_for_tts(text or ""))
    write_text_file(paths["txt"], cleaned)

    # Force rebuild: remove outputs (cache behavior still depends on your tts_to_wav implementation;
    # this guarantees files are overwritten even if cache is on)
    if force:
        for k in ["voice_wav", "mixed_wav", "final_wav", "mp3"]:
            _safe_remove(paths[k])

    # 1) TTS → voice WAV
    tts_to_wav(text=cleaned, out_wav=paths["voice_wav"], cfg=cfg)

    # 2) Optional music mix
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

    # 3) Re-encode to a UNIFIED WAV for concat reliability (48kHz, 2ch, pcm_s16le)
    run_ffmpeg([
        FFMPEG, "-y",
        "-i", intermediate_wav,
        "-ar", "48000",
        "-ac", "2",
        "-c:a", "pcm_s16le",
        paths["final_wav"]
    ])

    # 4) Unified WAV → MP3
    wav_to_mp3(paths["final_wav"], paths["mp3"], bitrate="192k")

    return paths

# ----------------------------
# Section list helper
# ----------------------------
def section_texts() -> List[Tuple[str, str]]:
    """
    Playback order: intro → chapters → outro
    """
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
# Build controls
# ----------------------------
st.subheader("🎙️ Build Audio Sections")

if not has_any_script():
    st.info("Generate a script first (Intro / Chapters / Outro) before building audio.")
else:
    st.session_state.setdefault("built_sections", {})
    built_sections: Dict[str, Dict[str, str]] = st.session_state["built_sections"]

    sections = section_texts()
    ordered_ids = [sid for sid, _ in sections]

    # Build mode controls
    ids_with_text = [sid for sid, txt in sections if (txt or "").strip()]
    default_select = ["00_intro"] if "00_intro" in ids_with_text else (ids_with_text[:1] if ids_with_text else [])
    selected_ids = st.multiselect(
        "Select sections to build",
        options=ids_with_text,
        default=default_select,
        help="Tip: build Intro first to QC voice + pacing before building everything.",
    )

    force_rebuild = st.checkbox(
        "Force rebuild (overwrite outputs)",
        value=False,
        help="If checked, deletes section outputs first and rebuilds them.",
    )

    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        build_selected_btn = st.button("Build SELECTED", type="primary", disabled=not bool(selected_ids))
    with c2:
        build_all_btn = st.button("Build ALL Sections")
    with c3:
        clear_built_btn = st.button("Clear Built Audio")

    if clear_built_btn:
        wd = _workdir()
        for name in os.listdir(wd):
            # Don't delete persisted music unless it's one of our generated artifacts
            if name.startswith("__music__"):
                continue
            _safe_remove(os.path.join(wd, name))
        st.session_state["built_sections"] = {}
        st.success("Cleared built audio (kept persisted music bed).")

    def _build_ids(to_build: List[str]):
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

    # ----------------------------
    # Per-section QC + downloads
    # ----------------------------
    st.divider()
    st.subheader("🔎 Section QC (Playback + Downloads)")

    any_built = False
    for sid in ordered_ids:
        paths = built_sections.get(sid) or {}
        mp3_path = paths.get("mp3", "")
        txt_path = paths.get("txt", "")
        if mp3_path and os.path.exists(mp3_path):
            any_built = True
            dur = _duration_seconds(mp3_path)
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
# Build FULL episode MP3 (robust concat via WAV)
# ----------------------------
st.divider()
st.subheader("🎬 Build FULL Episode MP3")

ep_slug = episode_slug()
full_mp3_path = os.path.join(_workdir(), f"{ep_slug}__FULL.mp3")
full_wav_path = os.path.join(_workdir(), f"{ep_slug}__FULL.wav")

def concat_wavs(wav_paths: List[str], out_wav: str) -> None:
    # concat demuxer (WAV) is reliable if formats match (we enforce unified WAV per section)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for p in wav_paths:
            # ffmpeg concat list expects escaping; safest is to wrap and replace single quotes
            safe_p = p.replace("'", r"'\''")
            f.write(f"file '{safe_p}'\n")
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

    # Build full wav then encode mp3 (uniform, predictable)
    _safe_remove(full_wav_path)
    _safe_remove(full_mp3_path)

    concat_wavs(wavs, full_wav_path)
    wav_to_mp3(full_wav_path, full_mp3_path, bitrate="192k")

if st.button("Build FULL Episode MP3", type="primary"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    sections = section_texts()
    ordered_ids = [sid for sid, _ in sections]
    with st.spinner("Building full episode…"):
        try:
            build_full_episode(built_sections, ordered_ids)
            st.success("Full episode MP3 built.")
        except Exception as e:
            st.error(str(e))

if os.path.exists(full_mp3_path):
    dur = _duration_seconds(full_mp3_path)
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

include_full_in_zip = st.checkbox(
    "Include FULL episode MP3 in ZIP (if built)",
    value=True,
)

if st.button("Build FINAL ZIP"):
    built_sections = st.session_state.get("built_sections", {}) or {}
    files: List[Tuple[str, str]] = []

    # Optional FULL
    if include_full_in_zip and os.path.exists(full_mp3_path):
        files.append((full_mp3_path, os.path.basename(full_mp3_path)))

    # Section MP3 + TXT (only if exists)
    for sid, paths in built_sections.items():
        mp3p = paths.get("mp3")
        txtp = paths.get("txt")
        if mp3p and os.path.exists(mp3p):
            files.append((mp3p, os.path.basename(mp3p)))
        if txtp and os.path.exists(txtp):
            files.append((txtp, os.path.basename(txtp)))

    # Master script
    raw_path = os.path.join(_workdir(), f"{ep_slug}__MASTER_SCRIPT.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write((st.session_state.get("MASTER_SCRIPT_TEXT", "") or "").strip() + "\n")
    files.append((raw_path, os.path.basename(raw_path)))

    # Outline (nice to ship)
    outline_path = os.path.join(_workdir(), f"{ep_slug}__OUTLINE.txt")
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

