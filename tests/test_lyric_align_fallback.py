"""Tests for Fix 21: proportional gender-fallback in lyric_align.align_sections.

When whisperx is unavailable (no word alignment), align_sections must still
return sections WITH gender-suffix labels + coarse proportional timestamps,
so portrait routing works (instead of returning None -> legacy path -> no
gender routing).
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lyric_align  # noqa: E402
from lyric_align import _proportional_fallback, align_sections  # noqa: E402


def _sections():
    return [
        {"label": "Intro", "lyrics": "", "is_vocal": False},
        {"label": "Verse 1 - female", "lyrics": "a\nb\nc\nd", "is_vocal": True},
        {"label": "Chorus - male", "lyrics": "E\nF\nG\nH", "is_vocal": True},
        {"label": "Bridge - duet", "lyrics": "x\ny", "is_vocal": True},
        {"label": "Outro", "lyrics": "", "is_vocal": False},
    ]


# ─── _proportional_fallback ──────────────────────────────────────────────────

def test_fallback_preserves_gender_labels():
    out = _proportional_fallback(_sections(), 120.0)
    labels = [s["label"] for s in out]
    assert "Verse 1 - female" in labels
    assert "Chorus - male" in labels
    assert "Bridge - duet" in labels


def test_fallback_timestamps_monotonic_contiguous():
    out = _proportional_fallback(_sections(), 120.0)
    for i in range(1, len(out)):
        assert out[i]["start"] == out[i - 1]["end"], "no gaps/overlaps between sections"
        assert out[i]["end"] >= out[i]["start"]


def test_fallback_sum_equals_audio_dur():
    out = _proportional_fallback(_sections(), 120.0)
    assert out[0]["start"] == 0.0
    assert abs(out[-1]["end"] - 120.0) < 0.01


def test_fallback_vocal_weighted_longer_than_instrumental():
    out = _proportional_fallback(_sections(), 120.0)
    by_label = {s["label"]: (s["end"] - s["start"]) for s in out}
    # Verse 1 (4 lyric chars-ish) should get more time than empty Intro
    assert by_label["Verse 1 - female"] > by_label["Intro"]
    assert by_label["Chorus - male"] > by_label["Outro"]


def test_fallback_empty_sections_returns_none():
    assert _proportional_fallback([], 120.0) is None


def test_fallback_zero_duration_returns_none():
    assert _proportional_fallback(_sections(), 0.0) is None


def test_fallback_preserves_is_vocal():
    out = _proportional_fallback(_sections(), 120.0)
    by_label = {s["label"]: s["is_vocal"] for s in out}
    assert by_label["Verse 1 - female"] is True
    assert by_label["Intro"] is False


# ─── align_sections falls back when whisperx missing ─────────────────────────

def test_align_sections_uses_fallback_when_no_words(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text(
        "[Intro]\n\n[Verse 1 - female]\na\nb\n\n[Chorus - male]\nC\nD\n\n[Outro]"
    )
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")

    # whisperx returns nothing → must hit proportional fallback, NOT None
    with patch.object(lyric_align, "_demucs_vocals", return_value=None), \
         patch.object(lyric_align, "_whisperx_words", return_value=None):
        out = align_sections(str(audio), str(lyrics), 60.0, "en")

    assert out is not None, "must return proportional fallback, not None"
    labels = [s["label"] for s in out]
    assert "Verse 1 - female" in labels
    assert "Chorus - male" in labels
    # contiguous + covers full duration
    assert out[0]["start"] == 0.0
    assert abs(out[-1]["end"] - 60.0) < 0.01


def test_align_sections_fallback_when_no_vtimes(tmp_path):
    """whisperx returns words but none match sections → still fallback, not None."""
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse 1 - female]\nhello world")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")

    with patch.object(lyric_align, "_demucs_vocals", return_value=None), \
         patch.object(lyric_align, "_whisperx_words", return_value=[{"word": "zzz", "start": 0, "end": 1}]), \
         patch.object(lyric_align, "_section_times_from_words", return_value={}):
        out = align_sections(str(audio), str(lyrics), 30.0, "en")

    assert out is not None
    assert out[0]["label"] == "Verse 1 - female"


def test_align_sections_still_none_when_no_sections(tmp_path):
    """Empty lyrics → genuinely nothing to align → None (legacy fallback)."""
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("")
    audio = tmp_path / "song.mp3"
    audio.write_bytes(b"fake")
    out = align_sections(str(audio), str(lyrics), 30.0, "en")
    assert out is None
