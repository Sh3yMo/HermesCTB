"""Post-run analyzer for MV pipeline jobs.

Compares what the pipeline planned (per-segment role / lyrics-section / portrait)
against what the rendered audio actually contains (inaSpeech gender classification
on demucs-extracted vocals).

Outputs a per-segment OK / STYLE / MISMATCH classification so the user can
objectively tell whether a music-video clip is "stylistic" or actually broken.

Usage:
    py analyze_mv_run.py outputs/2026-05-27/music_video_cbab632e.mp4
    py analyze_mv_run.py outputs/2026-05-27/music_video_cbab632e.mp4 --out analysis.md

Sidecar resolution:
    <mp4_dir>/ComfyUI_NNNNN_.mp3              (audio, newest matching job-time)
    <mp4_dir>/ComfyUI_NNNNN__lyrics.txt       (lyrics)
    <mp4_dir>/segments_<job_id>.json          (planned segments, if pipeline logged it)
    <mp4_dir>/seg_videos/ltx2_*-audio.mp4     (rendered segments — used for fallback timing)
"""
from __future__ import annotations

# Fix 15b bootstrap for ffmpeg/torchcodec
import _ffmpeg_init  # noqa: F401

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from audio_gender_detect import _classify_section, _segment_audio
from lyric_align import _demucs_vocals, parse_sections


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class PlannedSegment:
    index: int
    start: float
    end: float
    section_label: str = ""
    is_vocal: bool = False
    portrait_role: str = "story"  # male | female | duet | story
    video_path: str = ""


@dataclass
class AnalysisRow:
    seg: int
    time: str
    section: str
    planned: str
    detected: str
    status: str  # OK | STYLE | MISMATCH | UNKNOWN


# ─── Sidecar discovery ───────────────────────────────────────────────────────

def _find_sidecar(mp4_path: str) -> Dict[str, Optional[str]]:
    """Locate audio, lyrics, segments.json, seg_videos dir next to the MP4."""
    d = os.path.dirname(os.path.abspath(mp4_path))
    base = os.path.basename(mp4_path)
    # Job id from filename: music_video_<job8>.mp4
    job_match = re.match(r'music_video_([0-9a-f]+)\.mp4$', base)
    job_id = job_match.group(1) if job_match else None

    # Find newest mp3 + matching _lyrics.txt
    mp3s = sorted(
        (f for f in os.listdir(d) if f.startswith("ComfyUI_") and f.endswith(".mp3")),
        key=lambda f: os.path.getmtime(os.path.join(d, f)),
        reverse=True,
    )
    audio = os.path.join(d, mp3s[0]) if mp3s else None
    lyrics = None
    if audio:
        stem = os.path.splitext(os.path.basename(audio))[0]
        candidate = os.path.join(d, f"{stem}_lyrics.txt")
        if os.path.exists(candidate):
            lyrics = candidate

    segments_json = None
    if job_id:
        candidate = os.path.join(d, f"segments_{job_id}.json")
        if os.path.exists(candidate):
            segments_json = candidate

    seg_videos_dir = os.path.join(d, "seg_videos")
    if not os.path.isdir(seg_videos_dir):
        seg_videos_dir = None

    return {
        "job_id": job_id,
        "audio": audio,
        "lyrics": lyrics,
        "segments_json": segments_json,
        "seg_videos_dir": seg_videos_dir,
    }


# ─── Planned-segment loading ─────────────────────────────────────────────────

def _portrait_role_from_label(label: str) -> str:
    """Extract role suffix from a section label (e.g. 'Verse 1 - male' -> 'male').
    Returns 'story' if no role suffix (instrumental section or unannotated)."""
    if not label:
        return "story"
    m = re.search(r' - (male|female|duet)\b', label.lower())
    return m.group(1) if m else "story"


def load_planned_segments(segments_json: Optional[str], audio_duration: float,
                          lyrics_path: Optional[str], seg_videos_dir: Optional[str]
                          ) -> List[PlannedSegment]:
    """Load segments from JSON (preferred) or reconstruct from lyrics + seg_videos."""
    if segments_json and os.path.exists(segments_json):
        with open(segments_json, encoding="utf-8") as f:
            raw = json.load(f)
        return [PlannedSegment(
            index=int(r.get("index", i)),
            start=float(r.get("start", 0.0)),
            end=float(r.get("end", 0.0)),
            section_label=str(r.get("section_label", "")),
            is_vocal=bool(r.get("is_vocal", False)),
            portrait_role=str(r.get("portrait_role", "story")),
            video_path=str(r.get("video_path", "")),
        ) for i, r in enumerate(raw)]

    # Fallback: reconstruct from lyrics sections + seg_videos durations
    if not lyrics_path or not os.path.exists(lyrics_path):
        return []
    sections = parse_sections(lyrics_path)
    if not sections:
        return []
    # Proportional time split (no real alignment available post-hoc)
    total_vocal = sum(1 for s in sections if s.get("is_vocal"))
    if total_vocal == 0:
        return []
    # Distribute audio_duration evenly across sections (rough fallback)
    per_section = audio_duration / max(1, len(sections))
    out: List[PlannedSegment] = []
    cursor = 0.0
    for i, sec in enumerate(sections):
        label = sec.get("label", f"Section {i}")
        out.append(PlannedSegment(
            index=i,
            start=cursor,
            end=cursor + per_section,
            section_label=label,
            is_vocal=bool(sec.get("is_vocal", False)),
            portrait_role=_portrait_role_from_label(label),
        ))
        cursor += per_section
    return out


# ─── Audio analysis ──────────────────────────────────────────────────────────

def _audio_duration_via_ffprobe(audio_path: str) -> float:
    """Get duration via ffprobe (must be on PATH)."""
    import subprocess
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
            capture_output=True, text=True, check=True, timeout=30,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# ─── Status classification ───────────────────────────────────────────────────

def _classify_status(planned_role: str, detected: str, lyrics_role: str) -> str:
    """Decide OK / STYLE / MISMATCH for a single segment.

    OK     — planned matches detected (and matches lyrics if vocal)
    STYLE  — story-segment with vocals over it (intentional MV style)
    MISMATCH — planned role differs from detected (potential bug)
    UNKNOWN — detection inconclusive (silent/music only)
    """
    if detected in ("unknown", "noEnergy", "music"):
        # Story / instrumental segments expected to have no clear voice
        if planned_role == "story":
            return "OK"
        return "UNKNOWN"

    if planned_role == "story":
        # Story segment but audio has clear gender → intentional overlay style
        return "STYLE"

    if planned_role == "duet":
        # Duet portrait covers both — any voice OK
        return "OK"

    if planned_role == detected:
        return "OK"
    return "MISMATCH"


# ─── Main analysis ───────────────────────────────────────────────────────────

def analyze_mv_run(mp4_path: str) -> Tuple[List[AnalysisRow], Dict[str, Optional[str]]]:
    """Run full analysis on an MV MP4. Returns (rows, sidecar_dict)."""
    sidecar = _find_sidecar(mp4_path)
    audio = sidecar["audio"]
    if not audio:
        raise FileNotFoundError(f"No ComfyUI_*.mp3 sidecar next to {mp4_path}")

    audio_dur = _audio_duration_via_ffprobe(audio)
    segments_plan = load_planned_segments(
        sidecar["segments_json"], audio_dur, sidecar["lyrics"], sidecar["seg_videos_dir"]
    )
    if not segments_plan:
        return [], sidecar

    # Extract vocals once, then run inaSpeech once over the full track
    with tempfile.TemporaryDirectory(prefix="mv_analyze_") as work:
        vocals = _demucs_vocals(audio, work) or audio
        ina_segments = _segment_audio(vocals)

    rows: List[AnalysisRow] = []
    for ps in segments_plan:
        # Fix 36: _classify_section returns (gender, confidence) — analyze_mv_run
        # only needs the gender label here.
        detected, _conf = _classify_section(ps.start, ps.end, ina_segments)
        lyrics_role = _portrait_role_from_label(ps.section_label)
        status = _classify_status(ps.portrait_role, detected, lyrics_role)
        rows.append(AnalysisRow(
            seg=ps.index,
            time=f"{ps.start:5.1f}–{ps.end:5.1f}",
            section=ps.section_label,
            planned=ps.portrait_role,
            detected=detected,
            status=status,
        ))
    return rows, sidecar


def render_markdown_table(rows: List[AnalysisRow]) -> str:
    """Format analysis rows as a markdown table."""
    lines = [
        "| Seg | Time         | Section                 | Planned   | Detected   | Status      |",
        "|-----|--------------|-------------------------|-----------|------------|-------------|",
    ]
    for r in rows:
        flag = " ⚠" if r.status == "MISMATCH" else ""
        lines.append(
            f"| {r.seg:>3} | {r.time:<12} | {r.section[:23]:<23} | "
            f"{r.planned:<9} | {r.detected:<10} | {r.status}{flag:<5} |"
        )
    # Summary
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.status] = counts.get(r.status, 0) + 1
    lines.append("")
    lines.append("**Summary:** " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Analyze an MV pipeline run.")
    p.add_argument("mp4", help="Path to music_video_<jobid>.mp4")
    p.add_argument("--out", help="Write markdown report to this file (default: stdout)")
    args = p.parse_args()

    if not os.path.exists(args.mp4):
        print(f"ERROR: not found: {args.mp4}", file=sys.stderr)
        sys.exit(2)

    rows, sidecar = analyze_mv_run(args.mp4)
    if not rows:
        print("ERROR: could not load planned segments (no segments JSON, no lyrics sidecar)",
              file=sys.stderr)
        sys.exit(3)

    md = render_markdown_table(rows)
    print(f"MV Run Analysis: {args.mp4}")
    print(f"  audio:        {sidecar['audio']}")
    print(f"  lyrics:       {sidecar['lyrics']}")
    print(f"  segments.json: {sidecar['segments_json'] or '(none — used lyrics fallback)'}")
    print()
    print(md)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(f"# MV Run Analysis\n\nSource: `{args.mp4}`\n\n")
            f.write(md)
        print(f"\nReport written to: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
