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
) -> Tuple[str, float]:
    """Determine the dominant gender for a single section based on overlapping segments.

    Fix 36: returns (label, confidence). Confidence semantics:
      - dominant solo (male/female above dominant_threshold): confidence
        equals the winning ratio (≥0.70 by default).
      - duet: confidence = 2 * min(male_ratio, female_ratio); 50/50 → 1.0,
        20/80 → 0.4.
      - majority-vote fallback (mixed but not duet-balanced): confidence
        equals the winning ratio (< 0.70, signals uncertainty so the caller
        can withhold the override).
      - no speech / unknown: confidence = 0.0.

    Returns ('male' | 'female' | 'duet' | 'unknown', confidence_in_[0,1]).
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
        return ("unknown", 0.0)

    male_ratio = male_s / speech_total
    female_ratio = female_s / speech_total

    # Both significantly present → duet
    if male_ratio >= duet_threshold and female_ratio >= duet_threshold:
        balance = 2.0 * min(male_ratio, female_ratio)
        return ("duet", balance)
    # Dominant gender (high confidence)
    if male_ratio >= dominant_threshold:
        return ("male", male_ratio)
    if female_ratio >= dominant_threshold:
        return ("female", female_ratio)
    # Mixed but not duet-balanced → pick winner with the actual winning
    # ratio as confidence (caller can use this to withhold weak overrides).
    if male_s > female_s:
        return ("male", male_ratio)
    return ("female", female_ratio)


def detect_section_genders(
    vocals_path: str,
    sections: List[Dict],
) -> Dict[str, Tuple[str, float]]:
    """Detect gender per section by overlap with inaSpeechSegmenter segments.

    Args:
        vocals_path: path to vocals-only WAV (from demucs).
        sections: list of dicts with keys 'label', 'start' (or 'start_time'),
                  'end' (or 'end_time'). Time values in seconds.

    Returns:
        Mapping section_label → (detected gender, confidence). Gender is one
        of 'male'|'female'|'duet'|'unknown'; confidence is in [0, 1].
    """
    if not sections:
        return {}
    segments = _segment_audio(vocals_path)
    result: Dict[str, Tuple[str, float]] = {}
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
    voice_end_tail_offset_s: float = 0.5,
) -> List[Dict]:
    """Refine section start/end times using inaSpeech voice-activity transitions.

    For each pair of adjacent sections, looks for a transition (voice_start /
    voice_end / gender_swap) in `segments` within ±max_shift_s of the
    whisperx-derived boundary:
      - shift <= close_threshold_s : adopt the inaSpeech boundary (more precise)
      - close < shift <= max_shift_s : average the two
      - no transition in window : keep whisperx boundary

    Fix 16: when the chosen transition is a 'voice_end' (vocal -> silence/music),
    add `voice_end_tail_offset_s` to compensate for sung-vowel sustain/reverb
    that the inaSpeech VAD threshold cuts off prematurely. voice_start and
    gender_swap onsets are sharp and need no offset.

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
        best_kind = None
        for t_time, t_kind in transitions:
            shift = abs(t_time - wx_boundary)
            if shift > max_shift_s:
                continue
            if best_shift is None or shift < best_shift:
                best_shift = shift
                best_time = t_time
                best_kind = t_kind

        if best_time is None:
            continue  # keep whisperx

        # Fix 16: tail bias only for voice_end transitions
        effective_time = best_time
        if best_kind == "voice_end":
            effective_time = best_time + voice_end_tail_offset_s

        if best_shift <= close_threshold_s:
            new_boundary = effective_time
        else:
            new_boundary = (wx_boundary + effective_time) / 2.0

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


def split_sections_at_mid_swaps(
    sections: List[Dict],
    segments: List[Tuple[str, float, float]],
    min_subsection_s: float = 4.0,
) -> List[Dict]:
    """Fix 17: Split a section in two when the audio swaps gender mid-section.

    For each section, looks at the inaSpeech segments overlapping its
    [start, end] window. If a gender_swap transition occurs strictly INSIDE
    the section (not at either boundary) AND both resulting halves have at
    least `min_subsection_s` duration, the section is split into two
    sub-sections at that swap point. The label gets a gender suffix per half
    (overwriting any pre-existing "- male/female/duet" suffix).

    Lyrics stay on the first sub-section. is_vocal and reuse_of are preserved
    on the first sub-section; reuse_of is cleared on the second (it is a
    NEW section that did not exist in the LLM's original plan).

    Returns a NEW list (input untouched).
    """
    if not sections or not segments:
        return [dict(s) for s in sections]

    voice_set = {"male", "female"}

    def _section_start_end(s: Dict) -> Tuple[float, float]:
        st = float(s.get("start", s.get("start_time", 0.0)))
        en = float(s.get("end", s.get("end_time", 0.0)))
        return st, en

    def _strip_gender_suffix(label: str) -> str:
        import re
        return re.sub(r' - (male|female|duet)\b', '', label, count=1)

    out: List[Dict] = []
    for sec in sections:
        st, en = _section_start_end(sec)
        if en - st < 2 * min_subsection_s:
            out.append(dict(sec))
            continue

        # Find first gender_swap inside the section that yields both halves >= min
        swap_time = None
        prev_voice_label = None
        first_voice_label = None
        for label, s_start, s_end in segments:
            # Only voice segments inside the window
            ov_start = max(st, s_start)
            ov_end = min(en, s_end)
            if ov_end <= ov_start:
                continue
            if label not in voice_set:
                # Reset prev_voice on silence/music — swaps across a gap don't count
                prev_voice_label = None
                continue
            if first_voice_label is None:
                first_voice_label = label
            if prev_voice_label is not None and prev_voice_label != label:
                # Swap point = start of this segment (clamped to section window)
                candidate = max(s_start, st)
                if (candidate - st) >= min_subsection_s and (en - candidate) >= min_subsection_s:
                    swap_time = candidate
                    second_voice_label = label
                    break
            prev_voice_label = label

        if swap_time is None or first_voice_label is None:
            out.append(dict(sec))
            continue

        # Build two sub-sections
        base_label = _strip_gender_suffix(sec.get("label", ""))
        first = dict(sec)
        second = dict(sec)
        # Times
        for k_end in ("end", "end_time"):
            if k_end in first:
                first[k_end] = swap_time
            if k_end in second:
                # leave second's end at original end
                pass
        for k_start in ("start", "start_time"):
            if k_start in second:
                second[k_start] = swap_time
        # Labels
        first["label"] = f"{base_label} - {first_voice_label}"
        second["label"] = f"{base_label} - {second_voice_label}"
        # Lyrics on first only; reuse_of cleared on second
        if "lyrics" in second:
            second["lyrics"] = ""
        if "reuse_of" in second:
            second["reuse_of"] = None
        out.append(first)
        out.append(second)

    return out


def compare_with_lyrics_tags(
    detected: Dict[str, Tuple[str, float]],
    lyrics_role_extractor,
) -> List[Tuple[str, Optional[str], str]]:
    """Compare detected genders against role extracted from lyrics labels.

    Args:
        detected: output of detect_section_genders() — Dict[label, (gender, confidence)].
        lyrics_role_extractor: callable label -> Optional[str]. Pass
            music_video_pipeline.extract_section_role here.

    Returns:
        List of (label, expected_role, detected_role) for ALL sections — even
        matches. Callers can filter by `expected != detected` for mismatches.
    """
    rows: List[Tuple[str, Optional[str], str]] = []
    for label, det in detected.items():
        # Fix 36: detected entries are (gender, confidence) tuples; back-compat
        # accept legacy plain strings too.
        if isinstance(det, tuple):
            det_gender = det[0]
        else:
            det_gender = det
        expected = lyrics_role_extractor(label)
        rows.append((label, expected, det_gender))
    return rows
