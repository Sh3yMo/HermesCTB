"""Tests for audio_gender_detect.py — Fix 15 detection helpers.

Use mocks for inaSpeechSegmenter; we don't load the real TF model in unit tests.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_gender_detect import (  # noqa: E402
    _classify_section,
    _find_voice_activity_transitions,
    compare_with_lyrics_tags,
    detect_section_genders,
    refine_section_boundaries,
)


# ─── _classify_section (pure logic, no TF) ───────────────────────────────────

def test_classify_male_dominant():
    segs = [("male", 0.0, 10.0), ("noEnergy", 10.0, 12.0)]
    assert _classify_section(0.0, 12.0, segs) == "male"


def test_classify_female_dominant():
    segs = [("female", 0.0, 10.0)]
    assert _classify_section(0.0, 10.0, segs) == "female"


def test_classify_duet_balanced():
    segs = [("male", 0.0, 5.0), ("female", 5.0, 10.0)]
    assert _classify_section(0.0, 10.0, segs) == "duet"


def test_classify_unknown_silent():
    segs = [("noEnergy", 0.0, 10.0), ("music", 0.0, 10.0)]
    assert _classify_section(0.0, 10.0, segs) == "unknown"


def test_classify_partial_overlap_weighting():
    # Section 0-10, male only overlaps 5s, female only 1s → male wins
    segs = [("male", 0.0, 5.0), ("female", 5.0, 6.0)]
    assert _classify_section(0.0, 10.0, segs) == "male"


def test_classify_outside_section_ignored():
    segs = [("male", 100.0, 110.0)]  # well outside section 0-10
    assert _classify_section(0.0, 10.0, segs) == "unknown"


def test_classify_mixed_not_duet():
    # Section 0-10, male 9s, female 1s → not balanced for duet → male
    segs = [("male", 0.0, 9.0), ("female", 9.0, 10.0)]
    assert _classify_section(0.0, 10.0, segs) == "male"


def test_classify_duet_threshold_edge():
    # 80/20 split → exactly at duet threshold
    segs = [("male", 0.0, 8.0), ("female", 8.0, 10.0)]
    assert _classify_section(0.0, 10.0, segs) == "duet"


# ─── detect_section_genders (with mocked segmenter) ──────────────────────────

def test_detect_section_genders_full_pipeline():
    segs = [("male", 0.0, 12.0), ("female", 12.0, 24.0), ("noEnergy", 24.0, 26.0)]
    sections = [
        {"label": "Verse 1 - male", "start": 0.0, "end": 12.0},
        {"label": "Chorus - female", "start": 12.0, "end": 24.0},
    ]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert result == {
        "Verse 1 - male": "male",
        "Chorus - female": "female",
    }


def test_detect_section_genders_mismatch_detected():
    """LLM tagged Verse as male but audio is actually female — detection catches it."""
    segs = [("female", 0.0, 12.0)]
    sections = [{"label": "Verse 1 - male", "start": 0.0, "end": 12.0}]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert result["Verse 1 - male"] == "female"


def test_detect_section_genders_empty_sections():
    assert detect_section_genders("/fake/vocals.wav", []) == {}


def test_detect_section_genders_alt_keys():
    """Accepts 'start_time'/'end_time' as alternate keys."""
    segs = [("male", 0.0, 5.0)]
    sections = [{"label": "Verse", "start_time": 0.0, "end_time": 5.0}]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert result == {"Verse": "male"}


def test_detect_section_genders_skips_invalid():
    segs = [("male", 0.0, 5.0)]
    sections = [
        {"label": "", "start": 0.0, "end": 5.0},          # missing label
        {"label": "X", "start": 5.0, "end": 5.0},          # zero duration
        {"label": "Y", "start": 0.0, "end": 5.0},          # valid
    ]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert result == {"Y": "male"}


# ─── refine_section_boundaries (Fix 15c) ─────────────────────────────────────

def test_transitions_detected_voice_start_end_swap():
    segs = [
        ("noEnergy", 0.0, 2.0),
        ("male", 2.0, 10.0),
        ("female", 10.0, 18.0),
        ("noEnergy", 18.0, 20.0),
    ]
    trans = _find_voice_activity_transitions(segs)
    # Expected: voice_start @2.0, gender_swap @10.0, voice_end @18.0
    kinds = [t[1] for t in trans]
    times = [t[0] for t in trans]
    assert "voice_start" in kinds
    assert "gender_swap" in kinds
    assert "voice_end" in kinds
    assert 2.0 in times and 10.0 in times and 18.0 in times


def test_refine_snaps_to_close_transition():
    """whisperx boundary at 10.3s; inaSpeech transition at 10.0s (shift 0.3s < 1.0) → snap to 10.0"""
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.3},
        {"label": "C", "start": 10.3, "end": 20.0},
    ]
    segments = [("male", 0.0, 10.0), ("female", 10.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    assert refined[0]["end"] == 10.0
    assert refined[1]["start"] == 10.0


def test_refine_averages_when_moderate_shift():
    """whisperx 10.0s, inaSpeech 11.5s (shift 1.5s, between close_threshold and max_shift) → avg 10.75"""
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.0},
        {"label": "C", "start": 10.0, "end": 20.0},
    ]
    segments = [("male", 0.0, 11.5), ("female", 11.5, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    assert abs(refined[0]["end"] - 10.75) < 0.01
    assert abs(refined[1]["start"] - 10.75) < 0.01


def test_refine_keeps_whisperx_when_no_transition_in_window():
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.0},
        {"label": "C", "start": 10.0, "end": 20.0},
    ]
    # transition far away (at 5s and 15s, but whisperx boundary at 10s, max_shift=2)
    segments = [("male", 0.0, 5.0), ("noEnergy", 5.0, 15.0), ("female", 15.0, 20.0)]
    refined = refine_section_boundaries(sections, segments, max_shift_s=2.0)
    assert refined[0]["end"] == 10.0  # unchanged
    assert refined[1]["start"] == 10.0


def test_refine_keeps_contiguous_no_gap():
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.5},
        {"label": "C", "start": 10.5, "end": 20.0},
    ]
    segments = [("male", 0.0, 10.0), ("female", 10.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    assert refined[0]["end"] == refined[1]["start"]


def test_refine_empty_inputs_safe():
    assert refine_section_boundaries([], [("male", 0, 10)]) == []
    assert refine_section_boundaries([{"label": "A", "start": 0, "end": 5}], []) == \
           [{"label": "A", "start": 0, "end": 5}]


def test_refine_uses_alt_time_keys():
    sections = [
        {"label": "V1", "start_time": 0.0, "end_time": 10.3},
        {"label": "C", "start_time": 10.3, "end_time": 20.0},
    ]
    segments = [("male", 0.0, 10.0), ("female", 10.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    assert refined[0]["end_time"] == 10.0
    assert refined[1]["start_time"] == 10.0


# ─── compare_with_lyrics_tags ────────────────────────────────────────────────

def test_compare_returns_all_sections_with_expected_and_detected():
    detected = {"Verse 1 - male": "female", "Chorus - female": "female"}

    def extractor(label):
        if "female" in label.lower():
            return "female"
        if "male" in label.lower():
            return "male"
        return None

    rows = compare_with_lyrics_tags(detected, extractor)
    # Convert to dict by label for stable assert
    by_label = {r[0]: r for r in rows}
    assert by_label["Verse 1 - male"] == ("Verse 1 - male", "male", "female")  # MISMATCH
    assert by_label["Chorus - female"] == ("Chorus - female", "female", "female")  # match
