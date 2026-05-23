"""Unit tests for pure music-video planning helpers (no ComfyUI / network)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    ACE_STEP_LANGS,
    build_aligned_timeline,
    build_segment_timeline,
    chunk_list,
    clamp_song_duration,
    parse_segment_plan,
    to_ace_language,
)


def _al(label, lyrics, start, end, is_vocal=True, reuse_of=None):
    return {"label": label, "lyrics": lyrics, "start": start, "end": end,
            "is_vocal": is_vocal, "reuse_of": reuse_of}


def test_aligned_basic_keeps_real_boundaries():
    a = [_al("Verse 1", "x", 0.0, 12.0), _al("Chorus", "y", 12.0, 20.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert [(r["start_time"], r["end_time"]) for r in rows] == [(0.0, 12.0), (12.0, 20.0)]
    assert [r["label"] for r in rows] == ["Verse 1", "Chorus"]
    assert all(r["is_vocal"] for r in rows)


def test_aligned_splits_over_cap_within_span():
    a = [_al("Long", "L", 0.0, 70.0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 3  # ceil(70/30)
    assert rows[0]["start_time"] == 0.0 and rows[-1]["end_time"] == 70.0
    assert rows[0]["lyrics"] == "L" and rows[1]["lyrics"] == "" and rows[2]["lyrics"] == ""
    for x, y in zip(rows, rows[1:]):
        assert y["start_time"] == x["end_time"]


def test_aligned_merges_tiny_section_into_previous():
    a = [_al("Verse", "v", 0.0, 12.0), _al("Stab", "", 12.0, 13.0, is_vocal=False)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 13.0


def test_aligned_carries_reuse_of_first_row_only():
    a = [_al("Chorus", "c", 0.0, 12.0),
         _al("Chorus2", "c", 40.0, 80.0, reuse_of=0)]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    reuse_rows = [r for r in rows if r.get("reuse_of") == 0]
    assert len(reuse_rows) == 1  # only first sub-row of the repeated section
    assert reuse_rows[0]["start_time"] == 40.0


def test_aligned_empty_returns_empty():
    assert build_aligned_timeline([], 8.0, 30.0) == []


def test_aligned_zero_duration_first_section_absorbed():
    """RC: an instrumental [Intro] with 0 duration (WhisperX gave it no time)
    has no predecessor to merge into. It must be absorbed INTO the next section
    so no 0-duration row reaches _extract_audio_clip (→ empty WAV → LTX crash)."""
    a = [
        _al("Intro", "", 0.0, 0.0, is_vocal=False),
        _al("Verse 1", "v", 0.0, 12.0),
        _al("Chorus", "c", 12.0, 24.0),
    ]
    rows = build_aligned_timeline(a, min_seg=4.0, max_seg=20.0)
    assert rows, "expected non-empty timeline"
    for r in rows:
        assert (r["end_time"] - r["start_time"]) >= 0.05, f"degenerate row: {r}"
    # the intro's time span is folded into the first kept row
    assert rows[0]["start_time"] == 0.0


def test_aligned_tiny_first_section_absorbed_forward():
    """A short-but-nonzero leading section also merges forward (no predecessor)."""
    a = [
        _al("Intro", "", 0.0, 0.4, is_vocal=False),
        _al("Verse 1", "v", 0.4, 12.0),
    ]
    rows = build_aligned_timeline(a, min_seg=8.0, max_seg=30.0)
    assert len(rows) == 1
    assert rows[0]["start_time"] == 0.0 and rows[0]["end_time"] == 12.0

CAP = 30.0
MIN = 8.0


# --- clamp_song_duration ---------------------------------------------------

def test_clamp_none_returns_default():
    assert clamp_song_duration(None) == 150


def test_clamp_below_floor():
    assert clamp_song_duration(5) == 20


def test_clamp_explicit_short_honored():
    assert clamp_song_duration(30) == 30


def test_clamp_above_ceiling():
    assert clamp_song_duration(9999) == 300


def test_clamp_valid_passthrough():
    assert clamp_song_duration(200) == 200


def test_clamp_float_and_string_coerced():
    assert clamp_song_duration("210") == 210
    assert clamp_song_duration(180.7) == 180


def test_clamp_garbage_returns_default():
    assert clamp_song_duration("not-a-number") == 150


# --- to_ace_language -------------------------------------------------------

def test_lang_name_to_iso():
    assert to_ace_language("german", "") == "de"
    assert to_ace_language("German", "") == "de"
    assert to_ace_language("französisch", "") == "fr"


def test_lang_iso_passthrough():
    assert to_ace_language("de", "") == "de"
    assert to_ace_language("EN", "") == "en"


def test_lang_infer_from_brief():
    assert to_ace_language("", "deutsches HipHop-Video über X") == "de"
    assert to_ace_language(None, "a french chanson") == "fr"


def test_lang_unknown_returns_none():
    assert to_ace_language("klingon", "") is None
    assert to_ace_language("", "a song about cats") is None
    assert to_ace_language(None, None) is None


def test_lang_result_always_valid_enum():
    for v in ("german", "de", "english", "französisch", "spanish"):
        r = to_ace_language(v, "")
        assert r in ACE_STEP_LANGS


# --- chunk_list ------------------------------------------------------------

def test_chunk_even():
    assert chunk_list([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_remainder():
    assert chunk_list([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_empty():
    assert chunk_list([], 4) == []


def test_chunk_batch_larger_than_list():
    assert chunk_list([1, 2], 10) == [[1, 2]]


def test_chunk_nonpositive_batch_is_single_chunk():
    assert chunk_list([1, 2, 3], 0) == [[1, 2, 3]]
    assert chunk_list([1, 2, 3], -1) == [[1, 2, 3]]


# --- parse_segment_plan ----------------------------------------------------

def test_parse_plain_json_array():
    txt = '[{"video_prompt":"a","frame_variant_prompt":"b","duration":12,"label":"Intro"}]'
    out = parse_segment_plan(txt)
    assert len(out) == 1
    assert out[0]["video_prompt"] == "a"
    assert out[0]["frame_variant_prompt"] == "b"
    assert out[0]["duration"] == 12
    assert out[0]["label"] == "Intro"


def test_parse_with_code_fence_and_prose():
    txt = 'Sure!\n```json\n[{"video_prompt":"x","frame_variant_prompt":"y","duration":20}]\n```\nDone'
    out = parse_segment_plan(txt)
    assert len(out) == 1 and out[0]["video_prompt"] == "x"


def test_parse_missing_fields_defaulted():
    out = parse_segment_plan('[{"video_prompt":"only"}]')
    assert out[0]["video_prompt"] == "only"
    assert out[0]["frame_variant_prompt"] == "only"  # falls back to video_prompt
    assert out[0]["duration"] is None
    assert out[0]["label"] == ""


def test_parse_malformed_returns_empty():
    assert parse_segment_plan("not json at all") == []
    assert parse_segment_plan("") == []
    assert parse_segment_plan('{"not":"a list"}') == []


# --- build_segment_timeline ------------------------------------------------

def _check_contiguous(segs, total):
    assert segs, "must produce at least one segment"
    assert abs(segs[0]["start_time"]) < 1e-6
    assert abs(segs[-1]["end_time"] - total) < 1e-3, (segs[-1]["end_time"], total)
    for a, b in zip(segs, segs[1:]):
        assert abs(a["end_time"] - b["start_time"]) < 1e-6
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert d > 0


def test_timeline_empty_specs_fallback_tiles_total():
    segs = build_segment_timeline([], 180.0, MIN, CAP)
    _check_contiguous(segs, 180.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert MIN - 1e-6 <= d <= CAP + 1e-6


def test_timeline_clamps_oversized_duration_to_cap():
    specs = [{"video_prompt": "p", "frame_variant_prompt": "f", "duration": 35},
             {"video_prompt": "p2", "frame_variant_prompt": "f2", "duration": 35}]
    segs = build_segment_timeline(specs, 60.0, MIN, CAP)
    _check_contiguous(segs, 60.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert d <= CAP + 1e-6  # never exceeds A/V-safe cap


def test_timeline_normalizes_sum_to_total():
    specs = [{"video_prompt": "a", "frame_variant_prompt": "fa", "duration": 10},
             {"video_prompt": "b", "frame_variant_prompt": "fb", "duration": 10},
             {"video_prompt": "c", "frame_variant_prompt": "fc", "duration": 10}]
    segs = build_segment_timeline(specs, 150.0, MIN, CAP)
    _check_contiguous(segs, 150.0)
    for s in segs:
        d = s["end_time"] - s["start_time"]
        assert MIN - 1e-6 <= d <= CAP + 1e-6


def test_timeline_preserves_prompts_in_order():
    specs = [{"video_prompt": "v1", "frame_variant_prompt": "f1", "duration": 12, "label": "Intro"},
             {"video_prompt": "v2", "frame_variant_prompt": "f2", "duration": 12, "label": "Verse"}]
    segs = build_segment_timeline(specs, 24.0, MIN, CAP)
    assert [s["video_prompt"] for s in segs[:2]] == ["v1", "v2"]
    assert [s["frame_variant_prompt"] for s in segs[:2]] == ["f1", "f2"]


def test_timeline_short_total_single_segment():
    segs = build_segment_timeline([], 12.0, MIN, CAP)
    _check_contiguous(segs, 12.0)
    assert len(segs) == 1
