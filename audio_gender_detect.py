"""Audio gender detection for music-video frame routing.

After ACE-Step generates the audio and lyric_align produces section timestamps,
this module detects which gender actually sings each section by running
inaSpeechSegmenter over the demucs-extracted vocals track.

The detected gender is the ground truth for frame selection in
music_video_pipeline.py — overrides the LLM-emitted lyrics tag when they
disagree, because the audio is what the viewer actually hears.
"""
from __future__ import annotations

# Fix 15 windows bootstrap: register FFmpeg shared-DLL dir before any
# torchcodec/torchaudio/TF imports (inaSpeechSegmenter chains those).
import _ffmpeg_init  # noqa: F401

from typing import Dict, List, Optional, Tuple

# Module-level cache: Segmenter init is expensive (TF model load), do it once.
_SEGMENTER = None


def _get_segmenter():
    global _SEGMENTER
    if _SEGMENTER is None:
        from inaSpeechSegmenter import Segmenter  # heavy import, defer
        _SEGMENTER = Segmenter(vad_engine="smn", detect_gender=True)
    return _SEGMENTER


def _segment_audio(audio_path: str) -> List[Tuple[str, float, float]]:
    """Run inaSpeechSegmenter; returns list of (label, start_s, end_s).

    Labels: 'male', 'female', 'noEnergy', 'music', 'noise'.
    """
    seg = _get_segmenter()
    return seg(audio_path)


def _classify_section(
    section_start: float,
    section_end: float,
    segments: List[Tuple[str, float, float]],
    dominant_threshold: float = 0.7,
    duet_threshold: float = 0.2,
) -> str:
    """Determine the dominant gender for a single section based on overlapping segments.

    Returns 'male' | 'female' | 'duet' | 'unknown'.
    """
    male_s = 0.0
    female_s = 0.0
    for label, s_start, s_end in segments:
        # Compute overlap with section window
        overlap_start = max(section_start, s_start)
        overlap_end = min(section_end, s_end)
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap <= 0:
            continue
        if label == "male":
            male_s += overlap
        elif label == "female":
            female_s += overlap

    speech_total = male_s + female_s
    if speech_total <= 0:
        return "unknown"

    male_ratio = male_s / speech_total
    female_ratio = female_s / speech_total

    # Both significantly present → duet
    if male_ratio >= duet_threshold and female_ratio >= duet_threshold:
        return "duet"
    # Dominant gender
    if male_ratio >= dominant_threshold:
        return "male"
    if female_ratio >= dominant_threshold:
        return "female"
    # Mixed but not duet-balanced → pick winner
    return "male" if male_s > female_s else "female"


def detect_section_genders(
    vocals_path: str,
    sections: List[Dict],
) -> Dict[str, str]:
    """Detect gender per section by overlap with inaSpeechSegmenter segments.

    Args:
        vocals_path: path to vocals-only WAV (from demucs).
        sections: list of dicts with keys 'label', 'start' (or 'start_time'),
                  'end' (or 'end_time'). Time values in seconds.

    Returns:
        Mapping section_label → detected gender ('male'|'female'|'duet'|'unknown').
    """
    if not sections:
        return {}
    segments = _segment_audio(vocals_path)
    result: Dict[str, str] = {}
    for sec in sections:
        label = sec.get("label") or sec.get("name") or ""
        start = sec.get("start", sec.get("start_time", 0.0))
        end = sec.get("end", sec.get("end_time", 0.0))
        if not label or end <= start:
            continue
        result[label] = _classify_section(float(start), float(end), segments)
    return result


def compare_with_lyrics_tags(
    detected: Dict[str, str],
    lyrics_role_extractor,
) -> List[Tuple[str, Optional[str], str]]:
    """Compare detected genders against role extracted from lyrics labels.

    Args:
        detected: output of detect_section_genders().
        lyrics_role_extractor: callable label -> Optional[str]. Pass
            music_video_pipeline.extract_section_role here.

    Returns:
        List of (label, expected_role, detected_role) for ALL sections — even
        matches. Callers can filter by `expected != detected` for mismatches.
    """
    rows: List[Tuple[str, Optional[str], str]] = []
    for label, det in detected.items():
        expected = lyrics_role_extractor(label)
        rows.append((label, expected, det))
    return rows
