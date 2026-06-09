"""Stage L8: Intro / Outro are always STORY regardless of lyrics or gender tag."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pytest

from music_video_pipeline import (
    build_aligned_timeline,
    extract_section_role,
    partition_anchors_by_role,
)


@dataclass
class _StubSegment:
    label: str
    reuse_of: Optional[int] = None


def test_partition_intro_outro_force_none_role_even_with_gender_tag():
    segments = [
        _StubSegment("Intro - male"),
        _StubSegment("Verse 1 - male"),
        _StubSegment("Chorus - female"),
        _StubSegment("Outro - female"),
    ]
    out = partition_anchors_by_role(segments)
    assert None in out
    intro_idx = 0
    outro_idx = 3
    assert intro_idx in out[None]
    assert outro_idx in out[None]
    # Verse + Chorus still go to their tagged role buckets
    assert 1 in out.get("male", [])
    assert 2 in out.get("female", [])


def test_partition_plain_intro_outro_stay_none():
    segments = [
        _StubSegment("Intro"),
        _StubSegment("Verse 1 - male"),
        _StubSegment("Outro"),
    ]
    out = partition_anchors_by_role(segments)
    assert None in out
    assert 0 in out[None] and 2 in out[None]


def test_partition_extract_section_role_still_returns_gender():
    # Pure parser still works the same way for downstream tooling
    assert extract_section_role("Intro - male") == "male"
    assert extract_section_role("Outro - female") == "female"
    assert extract_section_role("Verse 1 - duet") == "duet"


def test_build_aligned_timeline_marks_intro_outro_not_vocal():
    sections = [
        {"label": "Intro", "is_vocal": True, "lyrics": "spoken intro line", "start": 0.0, "end": 6.0},
        {"label": "Verse 1 - male", "is_vocal": True, "lyrics": "verse line", "start": 6.0, "end": 24.0},
        {"label": "Chorus - male", "is_vocal": True, "lyrics": "chorus line", "start": 24.0, "end": 42.0},
        {"label": "Outro", "is_vocal": True, "lyrics": "outro line", "start": 42.0, "end": 60.0},
    ]
    rows = build_aligned_timeline(sections, min_seg=4.0, max_seg=30.0, total_duration=60.0)
    intro = next(r for r in rows if r["label"].lower().startswith("intro"))
    verse = next(r for r in rows if "verse" in r["label"].lower())
    chorus = next(r for r in rows if "chorus" in r["label"].lower())
    outro = next(r for r in rows if r["label"].lower().startswith("outro"))
    assert intro["is_vocal"] is False
    assert outro["is_vocal"] is False
    assert verse["is_vocal"] is True
    assert chorus["is_vocal"] is True


def test_build_aligned_timeline_intro_outro_with_explicit_gender_still_story():
    sections = [
        {"label": "Intro - male", "is_vocal": True, "lyrics": "x", "start": 0.0, "end": 5.0},
        {"label": "Verse 1 - male", "is_vocal": True, "lyrics": "y", "start": 5.0, "end": 25.0},
        {"label": "Outro - male", "is_vocal": True, "lyrics": "z", "start": 25.0, "end": 60.0},
    ]
    rows = build_aligned_timeline(sections, min_seg=4.0, max_seg=30.0, total_duration=60.0)
    intro = next(r for r in rows if r["label"].lower().startswith("intro"))
    outro = next(r for r in rows if r["label"].lower().startswith("outro"))
    assert intro["is_vocal"] is False
    assert outro["is_vocal"] is False
