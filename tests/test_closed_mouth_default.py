"""Stage M: closed-mouth default in MCA + frame prompts."""

from __future__ import annotations

import asyncio

import pytest

from mv_director import MVDirector
import music_video_pipeline as mvp


# ── M1: duet-composite portrait ───────────────────────────────────


def test_duet_composite_portrait_demands_closed_mouths():
    prompt = mvp.build_duet_portrait_prompt(theme="rock anthem", wardrobe_slot="", duet_kind="mixed")
    assert "mouths CLOSED" in prompt or "mouth CLOSED" in prompt
    # old open-style "mouths clearly visible" sentence must be gone
    assert "both faces and mouths clearly visible and in sharp focus" not in prompt


# ── M1: aligned VOCAL system prompt ──────────────────────────────


def test_aligned_vocal_rules_demand_closed_mouth_in_still():
    text = mvp._SEG_DIRECTOR_RULES
    assert "Stage M" in text
    assert "CLOSED" in text
    assert "open mouth" in text.lower()
    assert "stroke-like" in text


# ── M1: legacy VOCAL system prompt (constructed inside plan_segments) ─


def test_legacy_vocal_block_demands_closed_mouth():
    # The legacy block is built inline inside plan_segments; we check the
    # module source directly for the Stage M markers.
    import inspect

    src = inspect.getsource(mvp)
    # both blocks must carry the Stage M closed-mouth instruction
    assert src.count("Stage M") >= 4
    assert "closed-mouth resting state" in src
    assert "lips lightly together, no teeth showing" in src or \
        "lips lightly together, no teeth" in src


# ── M1: MCA singer portrait prompt ───────────────────────────────


def test_singer_portrait_system_prompt_demands_closed_mouth():
    import inspect

    src = inspect.getsource(mvp)
    # generate_character_portrait_prompt now contains closed-mouth hard req
    assert "mouth CLOSED in a relaxed neutral expression" in src
    assert "closed-mouth resting state" in src


# ── M2: director brief universal forbidden ───────────────────────


def _make_director() -> MVDirector:
    return MVDirector()


def test_director_brief_lists_universal_mouth_forbidden_for_rock():
    d = _make_director()
    profile = d.select_producer_profile(genre="rock", sub_genre=None, mood="anthemic", song_seed=1)
    profile = d.apply_sub_genre_modifiers(profile, sub_genre=None)
    fake_sections = [
        {"label": "Intro", "is_vocal": False, "lyrics": ""},
        {"label": "Verse 1 - male", "is_vocal": True, "lyrics": "x"},
        {"label": "Chorus - male", "is_vocal": True, "lyrics": "y"},
        {"label": "Outro", "is_vocal": False, "lyrics": ""},
    ]
    shot_plan = d.build_shot_plan(fake_sections, profile, sentiment={"per_section_sentiment": []})
    brief = d.render_director_brief(profile, shot_plan, song_genre="rock")
    for needle in (
        "open mouth at frame start",
        "mid-singing pose in still",
        "teeth bared",
        "microphone touching lips",
    ):
        assert needle in brief, f"missing universal mouth forbidden: {needle}"


def test_director_brief_lists_universal_mouth_forbidden_for_pop():
    d = _make_director()
    profile = d.select_producer_profile(genre="pop", sub_genre=None, mood="happy", song_seed=2)
    profile = d.apply_sub_genre_modifiers(profile, sub_genre=None)
    fake_sections = [
        {"label": "Intro", "is_vocal": False, "lyrics": ""},
        {"label": "Verse 1 - female", "is_vocal": True, "lyrics": "x"},
        {"label": "Chorus - female", "is_vocal": True, "lyrics": "y"},
        {"label": "Outro", "is_vocal": False, "lyrics": ""},
    ]
    shot_plan = d.build_shot_plan(fake_sections, profile, sentiment={"per_section_sentiment": []})
    brief = d.render_director_brief(profile, shot_plan, song_genre="pop")
    assert "open mouth at frame start" in brief
