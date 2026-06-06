"""Tests for PromptRelay schema additions to music_video_pipeline.

Covers:
- Segment.video_prompt_relay roundtrip via to_dict/from_dict
- pick_relay_beat_count adaptive logic (duration -> beat count)
- extract_relay_spec strict parser (LLM spec dict -> relay block | None)
- RELAY_MIN_BEAT_SECONDS / RELAY_MAX_BEATS constants
"""

from music_video_pipeline import (
    RELAY_MAX_BEATS,
    RELAY_MIN_BEAT_SECONDS,
    Segment,
    extract_relay_spec,
    pick_relay_beat_count,
)


def test_relay_constants():
    assert RELAY_MIN_BEAT_SECONDS == 5.0
    assert RELAY_MAX_BEATS == 4


def test_pick_relay_beat_count_below_minimum():
    assert pick_relay_beat_count(0.0) == 0
    assert pick_relay_beat_count(4.9) == 0


def test_pick_relay_beat_count_at_minimum():
    assert pick_relay_beat_count(5.0) == 1


def test_pick_relay_beat_count_adaptive():
    assert pick_relay_beat_count(10.0) == 2
    assert pick_relay_beat_count(15.0) == 3
    assert pick_relay_beat_count(20.0) == 4


def test_pick_relay_beat_count_capped_at_max():
    assert pick_relay_beat_count(30.0) == 4
    assert pick_relay_beat_count(60.0) == 4


def test_extract_relay_spec_missing_returns_none():
    assert extract_relay_spec({}) is None
    assert extract_relay_spec({"video_prompt": "x"}) is None


def test_extract_relay_spec_wrong_shape_returns_none():
    assert extract_relay_spec({"video_prompt_relay": "string instead of dict"}) is None
    assert extract_relay_spec({"video_prompt_relay": {"global": 1, "beats": []}}) is None
    assert extract_relay_spec({"video_prompt_relay": {"global": "x", "beats": "not a list"}}) is None


def test_extract_relay_spec_empty_beats_returns_none():
    assert extract_relay_spec({"video_prompt_relay": {"global": "anchor", "beats": []}}) is None
    assert extract_relay_spec({"video_prompt_relay": {"global": "anchor", "beats": ["", "  "]}}) is None


def test_extract_relay_spec_valid():
    spec = {
        "video_prompt_relay": {
            "global": "  golden hour, cinematic medium shot  ",
            "beats": ["beat 1", "  beat 2  ", "beat 3"],
        }
    }
    out = extract_relay_spec(spec)
    assert out == {
        "global": "golden hour, cinematic medium shot",
        "beats": ["beat 1", "beat 2", "beat 3"],
    }


def test_extract_relay_spec_caps_at_max_beats():
    spec = {
        "video_prompt_relay": {
            "global": "g",
            "beats": ["b1", "b2", "b3", "b4", "b5", "b6"],
        }
    }
    out = extract_relay_spec(spec)
    assert out is not None
    assert len(out["beats"]) == RELAY_MAX_BEATS
    assert out["beats"] == ["b1", "b2", "b3", "b4"]


def test_segment_relay_default_none():
    s = Segment(index=0, start_time=0.0, end_time=10.0)
    assert s.video_prompt_relay is None


def test_segment_to_dict_includes_relay_field():
    s = Segment(
        index=0,
        start_time=0.0,
        end_time=10.0,
        video_prompt_relay={"global": "g", "beats": ["b1", "b2"]},
    )
    d = s.to_dict()
    assert d["video_prompt_relay"] == {"global": "g", "beats": ["b1", "b2"]}
    assert d["wardrobe_slot"] == ""


def test_segment_from_dict_roundtrip_relay():
    payload = {
        "index": 3,
        "start_time": 15.0,
        "end_time": 25.0,
        "video_prompt_relay": {"global": "anchor", "beats": ["a", "b"]},
        "wardrobe_slot": "slot_a",
    }
    s = Segment.from_dict(payload)
    assert s.video_prompt_relay == {"global": "anchor", "beats": ["a", "b"]}
    assert s.wardrobe_slot == "slot_a"
    # to_dict round-trip preserves
    assert s.to_dict()["video_prompt_relay"] == {"global": "anchor", "beats": ["a", "b"]}
