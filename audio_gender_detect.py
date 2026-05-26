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


def _find_voice_activity_transitions(
    segments: List[Tuple[str, float, float]],
) -> List[Tuple[float, str]]:
    """Return list of (time, kind) for every voice-state transition in segments.

    Transition kinds:
      'voice_start' : noEnergy/music/noise -> male/female
      'voice_end'   : male/female -> noEnergy/music/noise
      'gender_swap' : male <-> female (vocal -> vocal but different gender)
    """
    transitions: List[Tuple[float, str]] = []
    voice_set = {"male", "female"}
    for i in range(len(segments) - 1):
        cur_label, _, cur_end = segments[i]
        nxt_label, _, _ = segments[i + 1]
        cur_voice = cur_label in voice_set
        nxt_voice = nxt_label in voice_set
        if not cur_voice and nxt_voice:
            transitions.append((cur_end, "voice_start"))
        elif cur_voice and not nxt_voice:
            transitions.append((cur_end, "voice_end"))
        elif cur_voice and nxt_voice and cur_label != nxt_label:
            transitions.append((cur_end, "gender_swap"))
    return transitions


def refine_section_boundaries(
    sections: List[Dict],
    segments: List[Tuple[str, float, float]],
    max_shift_s: float = 2.0,
    close_threshold_s: float = 1.0,
) -> List[Dict]:
    """Refine section start/end times using inaSpeech voice-activity transitions.

    For each pair of adjacent sections, looks for a transition (voice_start /
    voice_end / gender_swap) in `segments` within ±max_shift_s of the
    whisperx-derived boundary:
      - shift <= close_threshold_s : adopt the inaSpeech boundary (more precise)
      - close < shift <= max_shift_s : average the two
      - no transition in window : keep whisperx boundary

    Returns a NEW list with adjusted start/end times. Original list untouched.
    Each section's start aligns with the previous section's end (no gap, no
    overlap) so downstream Segment-builders remain consistent.
    """
    if not sections or not segments:
        return [dict(s) for s in sections]

    transitions = _find_voice_activity_transitions(segments)
    if not transitions:
        return [dict(s) for s in sections]

    refined = [dict(s) for s in sections]

    for i in range(len(refined) - 1):
        cur, nxt = refined[i], refined[i + 1]
        # whisperx-boundary = cur.end == nxt.start (in aligned output)
        wx_boundary = float(cur.get("end", cur.get("end_time", 0.0)))

        # Pick nearest transition within window
        best_shift = None
        best_time = None
        for t_time, _kind in transitions:
            shift = abs(t_time - wx_boundary)
            if shift > max_shift_s:
                continue
            if best_shift is None or shift < best_shift:
                best_shift = shift
                best_time = t_time

        if best_time is None:
            continue  # keep whisperx

        if best_shift <= close_threshold_s:
            new_boundary = best_time
        else:
            new_boundary = (wx_boundary + best_time) / 2.0

        # Apply — overwrite cur.end and nxt.start so they remain contiguous
        if "end" in cur:
            cur["end"] = new_boundary
        if "end_time" in cur:
            cur["end_time"] = new_boundary
        if "start" in nxt:
            nxt["start"] = new_boundary
        if "start_time" in nxt:
            nxt["start_time"] = new_boundary

    return refined


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
