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
    split_sections_at_mid_swaps,
)


# ─── _classify_section (pure logic, no TF) ───────────────────────────────────

def test_classify_male_dominant():
    segs = [("male", 0.0, 10.0), ("noEnergy", 10.0, 12.0)]
    gender, conf = _classify_section(0.0, 12.0, segs)
    assert gender == "male"
    assert conf >= 0.99  # 100% male


def test_classify_female_dominant():
    segs = [("female", 0.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "female"
    assert conf >= 0.99


def test_classify_duet_balanced():
    segs = [("male", 0.0, 5.0), ("female", 5.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    # 50/50 → balance = 2 * 0.5 = 1.0
    assert abs(conf - 1.0) < 0.01


def test_classify_unknown_silent():
    segs = [("noEnergy", 0.0, 10.0), ("music", 0.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "unknown"
    assert conf == 0.0


def test_classify_partial_overlap_weighting():
    # Section 0-10, male only overlaps 5s, female only 1s → male wins
    segs = [("male", 0.0, 5.0), ("female", 5.0, 6.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "male"
    # male_ratio = 5/6 ≈ 0.83 → dominant, conf = 0.83
    assert conf > 0.7


def test_classify_outside_section_ignored():
    segs = [("male", 100.0, 110.0)]  # well outside section 0-10
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "unknown"
    assert conf == 0.0


def test_classify_mixed_not_duet():
    # Section 0-10, male 9s, female 1s → not balanced for duet → male
    segs = [("male", 0.0, 9.0), ("female", 9.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "male"
    assert conf > 0.7  # 90% male


def test_classify_duet_threshold_edge():
    # 80/20 split → exactly at duet threshold
    segs = [("male", 0.0, 8.0), ("female", 8.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    # balance = 2 * 0.2 = 0.4
    assert abs(conf - 0.4) < 0.01


def test_classify_dominant_with_silence_padding():
    """Silence/noEnergy doesn't count toward speech_total — only male/female
    overlap matters for the ratio."""
    segs = [("male", 0.0, 6.0), ("female", 6.0, 7.0), ("noEnergy", 7.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    # speech_total = 7s; male_ratio = 6/7 ≈ 0.857 → dominant male.
    assert gender == "male"
    assert conf > 0.8


def test_classify_duet_unbalanced_returns_balance_confidence():
    """Duet but one side dominates — confidence is the balance score."""
    # 75/25 split: male 75% female 25% → both > 0.2 → duet branch
    segs = [("male", 0.0, 7.5), ("female", 7.5, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    # balance = 2 * 0.25 = 0.5
    assert abs(conf - 0.5) < 0.01


# ─── detect_section_genders (with mocked segmenter) ──────────────────────────

def test_detect_section_genders_full_pipeline():
    segs = [("male", 0.0, 12.0), ("female", 12.0, 24.0), ("noEnergy", 24.0, 26.0)]
    sections = [
        {"label": "Verse 1 - male", "start": 0.0, "end": 12.0},
        {"label": "Chorus - female", "start": 12.0, "end": 24.0},
    ]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    # Fix 36: dict values are (gender, confidence) tuples.
    assert result["Verse 1 - male"][0] == "male"
    assert result["Verse 1 - male"][1] >= 0.99
    assert result["Chorus - female"][0] == "female"
    assert result["Chorus - female"][1] >= 0.99


def test_detect_section_genders_mismatch_detected():
    """LLM tagged Verse as male but audio is actually female — detection catches it."""
    segs = [("female", 0.0, 12.0)]
    sections = [{"label": "Verse 1 - male", "start": 0.0, "end": 12.0}]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    gender, conf = result["Verse 1 - male"]
    assert gender == "female"
    assert conf >= 0.99


def test_detect_section_genders_empty_sections():
    assert detect_section_genders("/fake/vocals.wav", []) == {}


def test_detect_section_genders_alt_keys():
    """Accepts 'start_time'/'end_time' as alternate keys."""
    segs = [("male", 0.0, 5.0)]
    sections = [{"label": "Verse", "start_time": 0.0, "end_time": 5.0}]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert result["Verse"][0] == "male"


def test_detect_section_genders_skips_invalid():
    segs = [("male", 0.0, 5.0)]
    sections = [
        {"label": "", "start": 0.0, "end": 5.0},          # missing label
        {"label": "X", "start": 5.0, "end": 5.0},          # zero duration
        {"label": "Y", "start": 0.0, "end": 5.0},          # valid
    ]
    with patch("audio_gender_detect._segment_audio", return_value=segs):
        result = detect_section_genders("/fake/vocals.wav", sections)
    assert list(result.keys()) == ["Y"]
    assert result["Y"][0] == "male"


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


# ─── Fix 16: voice_end tail-bias ─────────────────────────────────────────────

def test_refine_voice_end_adds_tail_bias():
    """voice_end transition gets +0.5s tail offset (sustain/reverb compensation)."""
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.3},
        {"label": "Interlude", "start": 10.3, "end": 20.0},
    ]
    # male ends at 10.0, then silence — voice_end transition @ 10.0
    segments = [("male", 0.0, 10.0), ("noEnergy", 10.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    # whisperx 10.3, transition 10.0, shift 0.3 (close) → adopt + 0.5 tail = 10.5
    assert abs(refined[0]["end"] - 10.5) < 0.01
    assert abs(refined[1]["start"] - 10.5) < 0.01


def test_refine_voice_start_no_bias():
    """voice_start transition does NOT add tail offset (sharp onset)."""
    sections = [
        {"label": "Intro", "start": 0.0, "end": 5.3},
        {"label": "V1", "start": 5.3, "end": 20.0},
    ]
    segments = [("noEnergy", 0.0, 5.0), ("male", 5.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    # voice_start @ 5.0, no offset → 5.0 exact
    assert abs(refined[0]["end"] - 5.0) < 0.01


def test_refine_gender_swap_no_bias():
    """gender_swap transition is a sharp onset — no tail offset."""
    sections = [
        {"label": "V1", "start": 0.0, "end": 10.3},
        {"label": "C", "start": 10.3, "end": 20.0},
    ]
    segments = [("male", 0.0, 10.0), ("female", 10.0, 20.0)]
    refined = refine_section_boundaries(sections, segments)
    assert abs(refined[0]["end"] - 10.0) < 0.01  # no +0.5


# ─── Fix 17: split_sections_at_mid_swaps ─────────────────────────────────────

def test_split_at_mid_swap_creates_two_subsections():
    sections = [{"label": "Verse 2", "start": 0.0, "end": 16.0, "is_vocal": True,
                 "lyrics": "line1\nline2"}]
    segments = [("male", 0.0, 8.0), ("female", 8.0, 16.0)]
    out = split_sections_at_mid_swaps(sections, segments, min_subsection_s=4.0)
    assert len(out) == 2
    assert out[0]["label"] == "Verse 2 - male"
    assert out[1]["label"] == "Verse 2 - female"
    assert out[0]["end"] == 8.0
    assert out[1]["start"] == 8.0
    assert out[1]["end"] == 16.0


def test_split_skips_swap_if_subsection_too_short():
    """Swap @ 2s but section starts at 0 → first half only 2s < min 4s → no split."""
    sections = [{"label": "V1", "start": 0.0, "end": 16.0}]
    segments = [("male", 0.0, 2.0), ("female", 2.0, 16.0)]
    out = split_sections_at_mid_swaps(sections, segments, min_subsection_s=4.0)
    assert len(out) == 1
    assert out[0]["label"] == "V1"


def test_split_preserves_lyrics_on_first_subsection():
    sections = [{"label": "V2", "start": 0.0, "end": 16.0,
                 "lyrics": "verse2 text", "reuse_of": "V1"}]
    segments = [("male", 0.0, 8.0), ("female", 8.0, 16.0)]
    out = split_sections_at_mid_swaps(sections, segments)
    assert out[0]["lyrics"] == "verse2 text"
    assert out[1]["lyrics"] == ""
    assert out[0]["reuse_of"] == "V1"
    assert out[1]["reuse_of"] is None


def test_split_no_swap_returns_unchanged():
    sections = [{"label": "V1", "start": 0.0, "end": 16.0}]
    segments = [("male", 0.0, 16.0)]
    out = split_sections_at_mid_swaps(sections, segments)
    assert len(out) == 1
    assert out[0]["label"] == "V1"


def test_split_strips_existing_gender_suffix():
    """Original label had '- male' but audio swapped — re-tag based on detected."""
    sections = [{"label": "Verse 2 - male", "start": 0.0, "end": 16.0}]
    segments = [("male", 0.0, 8.0), ("female", 8.0, 16.0)]
    out = split_sections_at_mid_swaps(sections, segments)
    assert out[0]["label"] == "Verse 2 - male"
    assert out[1]["label"] == "Verse 2 - female"  # not "Verse 2 - male - female"


def test_split_empty_inputs_safe():
    assert split_sections_at_mid_swaps([], [("male", 0, 10)]) == []
    assert split_sections_at_mid_swaps([{"label": "A", "start": 0, "end": 5}], []) == \
           [{"label": "A", "start": 0, "end": 5}]


def test_split_section_below_min_double_not_processed():
    """Section length 7s < 2 * min(4) = 8s → not even considered for splitting."""
    sections = [{"label": "V", "start": 0.0, "end": 7.0}]
    segments = [("male", 0.0, 3.5), ("female", 3.5, 7.0)]
    out = split_sections_at_mid_swaps(sections, segments, min_subsection_s=4.0)
    assert len(out) == 1


# ─── compare_with_lyrics_tags ────────────────────────────────────────────────

def test_compare_returns_all_sections_with_expected_and_detected():
    # Fix 36: detected is now Dict[label, (gender, confidence)].
    detected = {
        "Verse 1 - male": ("female", 0.92),
        "Chorus - female": ("female", 0.88),
    }

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


def test_compare_back_compat_accepts_plain_string_values():
    """compare_with_lyrics_tags accepts legacy plain strings too."""
    detected_legacy = {"Verse 1 - male": "female", "Chorus - female": "female"}

    def extractor(label):
        if "female" in label.lower():
            return "female"
        if "male" in label.lower():
            return "male"
        return None

    rows = compare_with_lyrics_tags(detected_legacy, extractor)
    by_label = {r[0]: r for r in rows}
    assert by_label["Verse 1 - male"] == ("Verse 1 - male", "male", "female")
