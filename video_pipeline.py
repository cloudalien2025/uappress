# video_pipeline.py
# ------------------------------------------------------------
# OpenAI video-per-scene pipeline + ffmpeg stitching for Streamlit
#
# Requires:
#   openai>=1.0.0
#   imageio-ffmpeg>=0.4.9
#
# Notes:
# - Video API is async: create -> poll -> download_content (MP4 bytes)
# - Short clips (4–8s) stitch better than long renders
# ------------------------------------------------------------

from __future__ import annotations

import json
import os
import re
import time
import tempfile
import subprocess
from dataclasses import dataclass
from typing import List, Dict, Optional

import imageio_ffmpeg
from openai import OpenAI, AsyncOpenAI


# ----------------------------
# Helpers
# ----------------------------

def _safe_slug(s: str, max_len: int = 60) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s[:max_len] if s else "chapter"

def _ffmpeg_path() -> str:
    return imageio_ffmpeg.get_ffmpeg_exe()

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed:\n{' '.join(cmd)}\n\nSTDERR:\n{p.stderr}\n\nSTDOUT:\n{p.stdout}")


# ----------------------------
# Scene planning (text -> JSON)
# ----------------------------

SCENE_PLANNER_SYSTEM = (
    "You convert a documentary narration chapter into a list of short visual scenes for AI video generation. "
    "Return STRICT JSON only: a list of objects with keys: scene, seconds, prompt, on_screen_text(optional). "
    "Keep prompts cinematic, documentary-realistic, no brand names, no copyrighted characters, no celebrity likeness. "
    "Avoid gore or graphic violence. Prefer atmospheric b-roll and reenactment-style shots."
)

def plan_scenes_for_chapter(
    client: OpenAI,
    chapter_title: str,
    chapter_text: str,
    *,
    max_scenes: int = 10,
    seconds_per_scene: int = 8,
    model: str = "gpt-5-mini",
) -> List[Dict]:
    """
    Returns a list like:
      [{"scene":1,"seconds":8,"prompt":"...","on_screen_text":"..."}, ...]
    """
    user = {
        "chapter_title": chapter_title,
        "max_scenes": max_scenes,
        "seconds_per_scene": seconds_per_scene,
        "chapter_text": chapter_text,
        "rules": [
            "Shots should be visually distinct and easy to generate.",
            "Keep each prompt 1–3 sentences max.",
            "Include camera/motion/lighting details.",
            "No text inside the image unless specified via on_screen_text."
        ],
        "output_format": "STRICT_JSON_LIST_ONLY"
    }

    # Using Responses API style via OpenAI SDK "responses.create" is common,
    # but to keep this drop-in compatible with many apps, we use a simple chat-like call.
    # If your app already uses responses, swap this with your existing helper.
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SCENE_PLANNER_SYSTEM},
            {"role": "user", "content": json.dumps(user)}
        ],
    )

    text = resp.output_text.strip()

    # Try to extract JSON list even if wrapped
    m = re.search(r"(\[\s*\{.*\}\s*\])", text, flags=re.S)
    if m:
        text = m.group(1)

    scenes = json.loads(text)
    # Basic validation + normalization
    norm: List[Dict] = []
    for i, sc in enumerate(scenes, start=1):
        norm.append({
            "scene": int(sc.get("scene", i)),
            "seconds": int(sc.get("seconds", seconds_per_scene)),
            "prompt": str(sc["prompt"]).strip(),
            "on_screen_text": (str(sc["on_screen_text"]).strip() if sc.get("on_screen_text") else None),
        })
    return norm


# ----------------------------
# Video generation (prompt -> mp4 bytes)
# ----------------------------

async def generate_video_clip_bytes(
    client: AsyncOpenAI,
    *,
    prompt: str,
    seconds: int = 8,
    size: str = "1280x720",
    model: str = "sora-2",
) -> bytes:
    """
    Creates a video job, polls until completed, then downloads MP4 bytes.
    The Videos API supports sora-2 / sora-2-pro, and seconds 4/8/12. :contentReference[oaicite:1]{index=1}
    """
    # Create & poll
    video = await client.videos.create_and_poll(
        model=model,
        prompt=prompt,
        seconds=str(seconds),
        size=size,
    )
    if video.status != "completed":
        raise RuntimeError(f"Video job did not complete. status={video.status} id={video.id}")

    # Download MP4 bytes
    resp = client.videos.download_content(video_id=video.id)
    content = resp.read()
    return content


def write_bytes(path: str, b: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b)


# ----------------------------
# Stitching
# ----------------------------

def concat_mp4s(mp4_paths: List[str], out_path: str) -> str:
    """
    Concatenates MP4s using concat demuxer (fast).
    Assumes clips share codec params. If concat fails, you can re-encode.
    """
    ffmpeg = _ffmpeg_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        for p in mp4_paths:
            tf.write(f"file '{p}'\n")
        list_path = tf.name

    try:
        _run([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", out_path])
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass

    return out_path


def mux_audio(video_path: str, audio_path: str, out_path: str) -> str:
    """
    Adds narration audio, trims to shortest stream.
    """
    ffmpeg = _ffmpeg_path()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    _run([
        ffmpeg, "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        out_path
    ])
    return out_path


def build_chapter_video(
    scene_mp4s: List[str],
    chapter_audio_mp3: str,
    out_dir: str,
    chapter_slug: str,
) -> str:
    stitched = os.path.join(out_dir, f"{chapter_slug}_stitched.mp4")
    final = os.path.join(out_dir, f"{chapter_slug}_final.mp4")
    concat_mp4s(scene_mp4s, stitched)
    mux_audio(stitched, chapter_audio_mp3, final)
    return final


def build_full_documentary(chapter_final_mp4s: List[str], out_path: str) -> str:
    return concat_mp4s(chapter_final_mp4s, out_path)
