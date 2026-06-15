"""Tests for analyze_mv_run.py — Fix 18 post-run analyzer."""
import json
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_mv_run import (  # noqa: E402
    AnalysisRow,
    PlannedSegment,
    _classify_status,
    _find_sidecar,
    _portrait_role_from_label,
    analyze_mv_run,
    load_planned_segments,
    render_markdown_table,
)


# ─── _portrait_role_from_label ───────────────────────────────────────────────

def test_role_from_label_male():
    assert _portrait_role_from_label("Verse 1 - male") == "male"


def test_role_from_label_female():
    assert _portrait_role_from_label("Chorus - female") == "female"


def test_role_from_label_duet():
    assert _portrait_role_from_label("Bridge - duet") == "duet"


def test_role_from_label_no_role_is_story():
    assert _portrait_role_from_label("Intro") == "story"
    assert _portrait_role_from_label("Guitar Solo") == "story"
    assert _portrait_role_from_label("") == "story"


def test_role_from_label_case_insensitive():
    assert _portrait_role_from_label("Verse 1 - MALE") == "male"


# ─── _classify_status ────────────────────────────────────────────────────────

def test_status_ok_when_planned_matches_detected():
    assert _classify_status("male", "male", "male") == "OK"


def test_status_mismatch_when_planned_differs_from_detected():
    assert _classify_status("male", "female", "male") == "MISMATCH"


def test_status_style_for_story_with_audible_voice():
    assert _classify_status("story", "male", "story") == "STYLE"
    assert _classify_status("story", "female", "story") == "STYLE"


def test_status_ok_for_story_with_silence_or_music():
    assert _classify_status("story", "unknown", "story") == "OK"
    assert _classify_status("story", "noEnergy", "story") == "OK"
    assert _classify_status("story", "music", "story") == "OK"


def test_status_unknown_for_vocal_segment_with_silence():
    """Planned vocal but no clear voice detected — inconclusive."""
    assert _classify_status("male", "unknown", "male") == "UNKNOWN"


def test_status_ok_for_duet_with_any_voice():
    """Duet portrait covers both — any voice is OK."""
    assert _classify_status("duet", "male", "duet") == "OK"
    assert _classify_status("duet", "female", "duet") == "OK"


# ─── load_planned_segments ───────────────────────────────────────────────────

def test_load_planned_segments_from_json(tmp_path):
    seg_json = tmp_path / "segs.json"
    seg_json.write_text(json.dumps([
        {"index": 0, "start": 0.0, "end": 6.5, "section_label": "Intro",
         "is_vocal": False, "portrait_role": "story", "video_path": "x.mp4"},
        {"index": 1, "start": 6.5, "end": 18.0, "section_label": "Verse 1 - male",
         "is_vocal": True, "portrait_role": "male"},
    ]))
    out = load_planned_segments(str(seg_json), 120.0, None, None)
    assert len(out) == 2
    assert out[0].section_label == "Intro"
    assert out[0].portrait_role == "story"
    assert out[1].portrait_role == "male"
    assert out[1].is_vocal is True


def test_load_planned_segments_fallback_from_lyrics(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Intro]\n\n[Verse 1 - male]\nline\n\n[Chorus - female]\nLINE")
    out = load_planned_segments(None, 30.0, str(lyrics), None)
    assert len(out) == 3
    assert out[0].section_label == "Intro"
    assert out[0].portrait_role == "story"
    assert out[1].portrait_role == "male"
    assert out[2].portrait_role == "female"
    # Proportional time split
    assert out[0].start == 0.0
    assert abs(out[2].end - 30.0) < 0.1


def test_load_planned_segments_empty_when_no_sources(tmp_path):
    assert load_planned_segments(None, 30.0, None, None) == []


def test_find_sidecar_supports_artist_title_filenames(tmp_path):
    mp4 = tmp_path / "Zion Breeze - Island Romance.mp4"
    mp4.write_bytes(b"mp4")
    audio = tmp_path / "Zion Breeze - Island Romance.mp3"
    audio.write_bytes(b"mp3")
    lyrics = tmp_path / "Zion Breeze - Island Romance_lyrics.txt"
    lyrics.write_text("[Verse - male]\nline")
    seg = tmp_path / "segments_dec78c72.json"
    seg.write_text("[]")
    os.utime(seg, (mp4.stat().st_mtime, mp4.stat().st_mtime))

    sidecar = _find_sidecar(str(mp4))

    assert sidecar["audio"] == str(audio)
    assert sidecar["lyrics"] == str(lyrics)
    assert sidecar["segments_json"] == str(seg)


def test_find_sidecar_chooses_segments_json_nearest_to_title_mp4(tmp_path):
    mp4 = tmp_path / "Artist - Title.mp4"
    mp4.write_bytes(b"mp4")
    audio = tmp_path / "Artist - Title.mp3"
    audio.write_bytes(b"mp3")
    old = tmp_path / "segments_aaaaaaaa.json"
    new = tmp_path / "segments_bbbbbbbb.json"
    old.write_text("[]")
    new.write_text("[]")
    os.utime(old, (mp4.stat().st_mtime - 5000, mp4.stat().st_mtime - 5000))
    os.utime(new, (mp4.stat().st_mtime + 5, mp4.stat().st_mtime + 5))

    sidecar = _find_sidecar(str(mp4))

    assert sidecar["segments_json"] == str(new)


# ─── analyze_mv_run integration ──────────────────────────────────────────────

def test_analyze_mv_run_full_pipeline(tmp_path):
    """End-to-end with mocked demucs + inaSpeech."""
    # Build fake sidecar layout
    audio = tmp_path / "ComfyUI_99999_.mp3"
    audio.write_bytes(b"fake mp3")
    lyrics = tmp_path / "ComfyUI_99999__lyrics.txt"
    lyrics.write_text("[Intro]\n\n[Verse 1 - male]\nline\n\n[Chorus - female]\nLINE")
    mp4 = tmp_path / "music_video_abcd1234.mp4"
    mp4.write_bytes(b"fake mp4")

    # Mock ffprobe duration + demucs (returns None) + inaSpeech segments
    with patch("analyze_mv_run._audio_duration_via_ffprobe", return_value=30.0), \
         patch("analyze_mv_run._demucs_vocals", return_value=None), \
         patch("analyze_mv_run._segment_audio", return_value=[
             ("noEnergy", 0.0, 10.0),
             ("male", 10.0, 20.0),
             ("female", 20.0, 30.0),
         ]):
        rows, sidecar = analyze_mv_run(str(mp4))

    assert sidecar["audio"] == str(audio)
    assert sidecar["lyrics"] == str(lyrics)
    assert len(rows) == 3
    # Intro (story) + noEnergy → OK
    assert rows[0].section == "Intro"
    assert rows[0].planned == "story"
    assert rows[0].status == "OK"
    # Verse 1 - male + male → OK
    assert rows[1].planned == "male"
    assert rows[1].detected == "male"
    assert rows[1].status == "OK"
    # Chorus - female + female → OK
    assert rows[2].planned == "female"
    assert rows[2].status == "OK"


def test_analyze_mv_run_detects_mismatch(tmp_path):
    """Planned male section but audio has female → MISMATCH flag."""
    audio = tmp_path / "ComfyUI_99999_.mp3"
    audio.write_bytes(b"x")
    lyrics = tmp_path / "ComfyUI_99999__lyrics.txt"
    lyrics.write_text("[Verse 1 - male]\nline")
    mp4 = tmp_path / "music_video_abcd1234.mp4"
    mp4.write_bytes(b"x")

    with patch("analyze_mv_run._audio_duration_via_ffprobe", return_value=10.0), \
         patch("analyze_mv_run._demucs_vocals", return_value=None), \
         patch("analyze_mv_run._segment_audio", return_value=[("female", 0.0, 10.0)]):
        rows, _ = analyze_mv_run(str(mp4))

    assert len(rows) == 1
    assert rows[0].planned == "male"
    assert rows[0].detected == "female"
    assert rows[0].status == "MISMATCH"


def test_analyze_missing_audio_raises(tmp_path):
    mp4 = tmp_path / "music_video_x.mp4"
    mp4.write_bytes(b"x")
    import pytest
    with pytest.raises(FileNotFoundError):
        analyze_mv_run(str(mp4))


# ─── render_markdown_table ───────────────────────────────────────────────────

def test_render_markdown_table_includes_mismatch_flag():
    rows = [
        AnalysisRow(0, "0.0–6.5", "Intro", "story", "noEnergy", "OK"),
        AnalysisRow(1, "6.5–18.0", "Verse 1 - male", "male", "female", "MISMATCH"),
    ]
    md = render_markdown_table(rows)
    assert "MISMATCH ⚠" in md
    assert "Summary:" in md
    assert "MISMATCH=1" in md
    assert "OK=1" in md


def test_render_markdown_table_empty_rows():
    md = render_markdown_table([])
    assert "| Seg |" in md  # header still there
    assert "Summary:" in md
