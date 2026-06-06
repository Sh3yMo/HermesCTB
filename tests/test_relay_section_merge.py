"""Tests for Stage 7 / Option A: section-merge continuity.

Covers merge_continuous_segments() + _build_merged_segment() helpers in
music_video_pipeline. Adjacent VOCAL segments sharing wardrobe_slot and
tod_state collapse into one Smart-Node multi-beat clip so consecutive
same-context lyric sections render as a continuous take instead of
hard-cut clips.
"""

from music_video_pipeline import (
    RELAY_MAX_BEATS,
    Segment,
    _build_merged_segment,
    merge_continuous_segments,
)


def _mk(idx, start, end, lyrics="la la", slot="slot_a", relay=None, reuse_of=None, prompt="p"):
    return Segment(
        index=idx,
        start_time=start,
        end_time=end,
        lyrics=lyrics,
        prompt=prompt,
        wardrobe_slot=slot,
        reuse_of=reuse_of,
        video_prompt_relay=relay,
    )


def test_merge_empty_returns_empty():
    assert merge_continuous_segments([], []) == []


def test_merge_single_unchanged():
    s = _mk(0, 0.0, 10.0)
    out = merge_continuous_segments([s], ["dawn"])
    assert out == [s]


def test_merge_same_context_two_vocal_merged():
    s1 = _mk(0, 0.0, 10.0, slot="slot_a", relay={"global": "g", "beats": ["sit"]})
    s2 = _mk(1, 10.0, 18.0, slot="slot_a", relay={"global": "g", "beats": ["stand"]})
    out = merge_continuous_segments([s1, s2], ["dawn", "dawn"])
    assert len(out) == 1
    m = out[0]
    assert m.start_time == 0.0
    assert m.end_time == 18.0
    assert m.video_prompt_relay == {"global": "g", "beats": ["sit", "stand"]}


def test_merge_breaks_on_different_wardrobe_slot():
    s1 = _mk(0, 0.0, 10.0, slot="slot_a")
    s2 = _mk(1, 10.0, 18.0, slot="slot_b")
    out = merge_continuous_segments([s1, s2], ["dawn", "dawn"])
    assert len(out) == 2


def test_merge_breaks_on_different_tod_state():
    s1 = _mk(0, 0.0, 10.0)
    s2 = _mk(1, 10.0, 18.0)
    out = merge_continuous_segments([s1, s2], ["dawn", "noon"])
    assert len(out) == 2


def test_merge_breaks_on_empty_lyrics_story_segment():
    s1 = _mk(0, 0.0, 10.0, lyrics="la la")
    s2 = _mk(1, 10.0, 18.0, lyrics="")  # story
    s3 = _mk(2, 18.0, 26.0, lyrics="la la")
    out = merge_continuous_segments([s1, s2, s3], ["dawn"] * 3)
    # story between blocks merge across it
    assert len(out) == 3


def test_merge_breaks_on_reuse_of_set():
    s1 = _mk(0, 0.0, 10.0)
    s2 = _mk(1, 10.0, 18.0, reuse_of=0)
    out = merge_continuous_segments([s1, s2], ["dawn", "dawn"])
    assert len(out) == 2


def test_merge_respects_max_clip_duration():
    # combined 0-25s would fit cap=30, but next jump to 32s exceeds
    s1 = _mk(0, 0.0, 15.0)
    s2 = _mk(1, 15.0, 25.0)
    s3 = _mk(2, 25.0, 32.0)
    out = merge_continuous_segments([s1, s2, s3], ["dawn"] * 3, max_clip_duration=30.0)
    # first two merge (0-25), third stays separate (would be 0-32 > 30)
    assert len(out) == 2
    assert out[0].start_time == 0.0
    assert out[0].end_time == 25.0
    assert out[1].start_time == 25.0


def test_merge_respects_relay_max_beats_cap():
    # 5 segments * 1 beat each would be 5 beats > RELAY_MAX_BEATS=4
    segs = [
        _mk(i, i * 5.0, (i + 1) * 5.0, relay={"global": "g", "beats": [f"b{i}"]})
        for i in range(5)
    ]
    out = merge_continuous_segments(segs, ["dawn"] * 5)
    # first 4 merge (beat budget exhausted), 5th stays separate
    assert len(out) == 2
    assert len(out[0].video_prompt_relay["beats"]) == RELAY_MAX_BEATS


def test_merge_three_adjacent_same_context():
    s1 = _mk(0, 0.0, 8.0, relay={"global": "g1", "beats": ["a"]})
    s2 = _mk(1, 8.0, 16.0, relay={"global": "g2", "beats": ["b"]})
    s3 = _mk(2, 16.0, 24.0, relay={"global": "g3", "beats": ["c"]})
    out = merge_continuous_segments([s1, s2, s3], ["dawn"] * 3)
    assert len(out) == 1
    m = out[0]
    assert m.start_time == 0.0
    assert m.end_time == 24.0
    assert m.video_prompt_relay["beats"] == ["a", "b", "c"]
    # first global wins
    assert m.video_prompt_relay["global"] == "g1"


def test_merge_mixes_relay_with_single_prompt_members():
    s1 = _mk(0, 0.0, 8.0, prompt="walks across rooftop", relay=None)
    s2 = _mk(1, 8.0, 16.0, relay={"global": "anchor", "beats": ["turns to camera"]})
    out = merge_continuous_segments([s1, s2], ["dawn"] * 2)
    assert len(out) == 1
    m = out[0]
    # s1's video_prompt becomes a beat, s2's relay beat appended
    assert m.video_prompt_relay["beats"] == ["walks across rooftop", "turns to camera"]
    # s2's global picked up since s1 had no relay
    assert m.video_prompt_relay["global"] == "anchor"


def test_build_merged_segment_concatenates_labels_and_lyrics():
    g = [
        _mk(0, 0.0, 8.0, lyrics="line one"),
        _mk(1, 8.0, 16.0, lyrics="line two"),
    ]
    g[0].label = "Verse 1"
    g[1].label = "Chorus"
    out = _build_merged_segment(g)
    assert out.label == "Verse 1 + Chorus"
    assert out.lyrics == "line one\nline two"
    assert out.start_time == 0.0
    assert out.end_time == 16.0
    assert out.reuse_of is None


def test_build_merged_segment_no_beats_returns_none_relay():
    # both members single-prompt with empty prompts → no beats
    s1 = _mk(0, 0.0, 8.0, prompt="", relay=None)
    s2 = _mk(1, 8.0, 16.0, prompt="", relay=None)
    out = _build_merged_segment([s1, s2])
    assert out.video_prompt_relay is None
