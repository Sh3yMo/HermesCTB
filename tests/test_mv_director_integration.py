"""Integration: verify MVDirector brief is injected into plan_segments system prompt."""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import patch

import pytest

from music_video_pipeline import MusicVideoPrompter


def _make_prompter() -> MusicVideoPrompter:
    config = {
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
    }
    return MusicVideoPrompter(config)


def _aligned_sections() -> list[dict]:
    return [
        {"section_label": "Intro", "is_vocal": False, "start": 0.0, "end": 6.0, "lyrics": ""},
        {"section_label": "Verse 1 - male", "is_vocal": True, "start": 6.0, "end": 18.0, "lyrics": "lonely streets at night"},
        {"section_label": "Chorus - male", "is_vocal": True, "start": 18.0, "end": 30.0, "lyrics": "i miss you"},
        {"section_label": "Verse 2 - male", "is_vocal": True, "start": 30.0, "end": 42.0, "lyrics": "second verse"},
        {"section_label": "Chorus - male", "is_vocal": True, "start": 42.0, "end": 54.0, "lyrics": "i miss you"},
        {"section_label": "Outro", "is_vocal": False, "start": 54.0, "end": 60.0, "lyrics": ""},
    ]


def test_director_brief_injected_for_rock_song(monkeypatch: pytest.MonkeyPatch) -> None:
    """Confirms a rock song's system prompt contains DIRECTOR BRIEF + dj_pult forbidden."""
    monkeypatch.delenv("MV_DIRECTOR_ENABLED", raising=False)
    prompter = _make_prompter()
    captured: dict[str, str] = {"system": "", "user": ""}

    async def fake_openrouter(messages, model=None, max_tokens=None):
        for m in messages:
            if m["role"] == "system":
                captured["system"] = m["content"]
            elif m["role"] == "user":
                captured["user"] = m["content"]
        if "DIRECTOR BRIEF" in captured["system"]:
            n_sections = 6
            return json.dumps([
                {"video_prompt": f"male singer performing on stage segment {i}",
                 "frame_variant_prompt": f"male singer segment {i}"}
                for i in range(n_sections)
            ])
        return json.dumps([{"sub_genre": "grunge",
                            "per_section_sentiment": [{"section_idx": i, "label": "anthemic"} for i in range(6)]}])

    async def smart_openrouter(messages, model=None, max_tokens=None):
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        if "Classify" in (next((m["content"] for m in messages if m["role"] == "system"), "")):
            return json.dumps({
                "sub_genre": "grunge",
                "per_section_sentiment": [{"section_idx": i, "label": "anthemic"} for i in range(6)],
            })
        for m in messages:
            if m["role"] == "system":
                captured["system"] = m["content"]
            if m["role"] == "user":
                captured["user"] = m["content"]
        return json.dumps([
            {"video_prompt": f"male singer segment {i}", "frame_variant_prompt": f"male singer {i}"}
            for i in range(6)
        ])

    with patch.object(prompter, "_call_openrouter", side_effect=smart_openrouter):
        with patch.object(prompter, "_pick_time_of_day_arc", return_value=asyncio_return("day_to_night")):
            with patch.object(prompter, "_pick_wardrobe_arc", return_value=asyncio_return("street_casual")):
                asyncio.run(prompter.plan_segments(
                    lyrics_text="[Intro]\n[Verse 1 - male]\nlonely streets\n[Chorus - male]\ni miss you",
                    theme="grunge rock garage band",
                    total_duration=60.0,
                    genre="rock",
                    aligned_sections=_aligned_sections(),
                ))

    assert "DIRECTOR BRIEF" in captured["system"], "director brief not injected"
    assert "Producer style reference" in captured["system"]
    assert "PER-SEGMENT SHOT DIRECTIVES" in captured["system"]
    assert "dj_pult" in captured["system"], "rock song must forbid dj_pult"
    assert "HARD RULES" in captured["system"]


def test_director_brief_disabled_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MV_DIRECTOR_ENABLED", "0")
    prompter = _make_prompter()
    captured: dict[str, str] = {"system": ""}

    async def fake_openrouter(messages, model=None, max_tokens=None):
        for m in messages:
            if m["role"] == "system":
                captured["system"] = m["content"]
        return json.dumps([
            {"video_prompt": f"seg {i}", "frame_variant_prompt": f"seg {i}"}
            for i in range(6)
        ])

    with patch.object(prompter, "_call_openrouter", side_effect=fake_openrouter):
        with patch.object(prompter, "_pick_time_of_day_arc", return_value=asyncio_return("day_to_night")):
            with patch.object(prompter, "_pick_wardrobe_arc", return_value=asyncio_return("street_casual")):
                asyncio.run(prompter.plan_segments(
                    lyrics_text="[Intro]\n[Verse 1]\nx",
                    theme="generic",
                    total_duration=60.0,
                    genre="pop",
                    aligned_sections=_aligned_sections(),
                ))

    assert "DIRECTOR BRIEF" not in captured["system"], "flag=0 should disable brief"


def asyncio_return(value):
    """Helper: returns an awaitable coroutine yielding `value` for AsyncMock substitution."""
    async def _coro(*_args, **_kwargs):
        return value
    return _coro()
