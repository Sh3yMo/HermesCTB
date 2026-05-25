"""Tests for audio_gender_detect.py — Fix 15 detection helpers.

Use mocks for inaSpeechSegmenter; we don't load the real TF model in unit tests.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_gender_detect import (  # noqa: E402
    _classify_section,
    compare_with_lyrics_tags,
    detect_section_genders,
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
