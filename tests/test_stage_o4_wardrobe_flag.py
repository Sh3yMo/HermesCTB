"""Stage O4 — wardrobe arcs disabled by default.

plan_segments(wardrobe_enabled=False) must produce constant empty wardrobe
slots, never append ", wearing ..." tags, skip the wardrobe-arc LLM pick,
and instruct the director that the outfit is fixed by the reference images.
Network-free (LLM calls mocked).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import MusicVideoPrompter  # noqa: E402


def _make_prompter() -> MusicVideoPrompter:
    return MusicVideoPrompter({
        "openrouter_api_key": "test-key",
        "openrouter_model": "test/model",
        "openrouter_base_url": "http://localhost/api",
        "max_tokens": 4000,
        "max_retries": 1,
        "retry_backoff": [0.0],
        "fallback_models": [],
        "vision_model": "test/vision",
        "vision_fallback_models": [],
        "disable_reasoning": True,
    })


def _aligned_sections() -> list[dict]:
    return [
        {"section_label": "Intro", "is_vocal": False, "start": 0.0, "end": 6.0, "lyrics": ""},
        {"section_label": "Verse 1 - male", "is_vocal": True, "start": 6.0, "end": 18.0, "lyrics": "lonely streets at night"},
        {"section_label": "Chorus - female", "is_vocal": True, "start": 18.0, "end": 30.0, "lyrics": "i miss you"},
        {"section_label": "Verse 2 - male", "is_vocal": True, "start": 30.0, "end": 42.0, "lyrics": "second verse"},
        {"section_label": "Chorus - female", "is_vocal": True, "start": 42.0, "end": 54.0, "lyrics": "i miss you"},
        {"section_label": "Outro", "is_vocal": False, "start": 54.0, "end": 60.0, "lyrics": ""},
    ]


def test_plan_segments_wardrobe_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MV_DIRECTOR_ENABLED", "0")
    prompter = _make_prompter()
    captured: dict[str, str] = {"system": "", "user": ""}

    async def fake_openrouter(messages, model=None, max_tokens=None):
        for m in messages:
            captured[m["role"]] = m["content"]
        return json.dumps([
            {"video_prompt": f"singer performs in the city, segment {i}",
             "frame_variant_prompt": f"singer still frame, segment {i}",
             "background_prompt": "empty neon-lit alley at night"}
            for i in range(6)
        ])

    async def _no_wardrobe_pick(**kwargs):
        raise AssertionError(
            "_pick_wardrobe_arc must not be called when wardrobe is disabled"
        )

    async def _tod(**kwargs):
        return "day_to_night"

    with patch.object(prompter, "_call_openrouter", side_effect=fake_openrouter), \
         patch.object(prompter, "_pick_time_of_day_arc", side_effect=_tod), \
         patch.object(prompter, "_pick_wardrobe_arc", side_effect=_no_wardrobe_pick):
        segments = asyncio.run(prompter.plan_segments(
            lyrics_text=(
                "[Verse 1 - male]\nlonely streets\n[Chorus - female]\ni miss you"
            ),
            theme="city nights",
            total_duration=60.0,
            genre="pop",
            aligned_sections=_aligned_sections(),
            wardrobe_enabled=False,
        ))

    assert segments, "plan_segments returned no segments"
    for s in segments:
        assert s.wardrobe_slot == "", f"segment {s.index} got slot {s.wardrobe_slot!r}"
        assert ", wearing" not in (s.prompt or "").lower()
        assert ", wearing" not in (s.frame_variant_prompt or "").lower()
    assert "wardrobe arcs DISABLED" in captured["user"], (
        "director user prompt must state that wardrobe is disabled"
    )
