"""Unit tests for MVDirector (Producer Style Director)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from mv_director import (
    CLOSEUP_SHOTS,
    DEFAULT_PROFILES_PATH,
    ESTABLISHING_SHOTS,
    MVDirector,
    SHOT_CODES,
)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def director() -> MVDirector:
    return MVDirector()


@pytest.fixture()
def aligned_rock_sections() -> list[dict]:
    labels = ["Intro", "Verse 1", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus", "Outro"]
    return [
        {"section_label": lbl, "role": "male" if "Chorus" in lbl or "Verse" in lbl else "story"}
        for lbl in labels
    ]


@pytest.fixture()
def aligned_pop_sections() -> list[dict]:
    labels = ["Intro", "Verse 1", "Pre-Chorus", "Chorus", "Verse 2", "Chorus", "Bridge", "Chorus", "Outro"]
    return [{"section_label": lbl, "role": "female"} for lbl in labels]


# ── Profile loading ─────────────────────────────────────────────────


def test_profile_loads_all_nine(director: MVDirector) -> None:
    assert len(director._producers) == 9
    ids = {p["id"] for p in director._producers}
    expected = {
        "dave_meyers", "melina_matsoukas", "floria_sigismondi",
        "anton_corbijn", "samuel_bayer", "mark_romanek",
        "hype_williams", "hiro_murai", "cole_bennett",
    }
    assert ids == expected


def test_every_profile_has_required_fields(director: MVDirector) -> None:
    required = {
        "id", "name", "genres", "sub_genres", "ethos_oneliner",
        "videos_analyzed", "signature_shots", "shot_distribution",
        "color_palette", "lighting_recipes", "edit_tempo",
        "motion_preference", "story_archetypes", "closeup_frequency",
        "mood_mapping", "forbidden_for_genres",
    }
    for p in director._producers:
        missing = required - p.keys()
        assert not missing, f"profile {p.get('id')} missing fields: {missing}"


def test_shot_distributions_sum_to_one(director: MVDirector) -> None:
    for p in director._producers:
        total = sum(p["shot_distribution"].values())
        assert abs(total - 1.0) < 0.02, f"{p['id']} shot_distribution sums to {total}"


def test_every_video_has_three_entries(director: MVDirector) -> None:
    for p in director._producers:
        assert len(p["videos_analyzed"]) == 3, f"{p['id']} has !=3 videos"


# ── Producer selection ─────────────────────────────────────────────


def test_select_rock_never_returns_pop_or_hiphop(director: MVDirector) -> None:
    for sub in [None, "grunge", "alt_rock", "post_punk", "industrial_rock"]:
        for seed in range(20):
            picked = director.select_producer_profile("rock", sub, None, seed)
            assert "rock" in picked["genres"], f"rock query returned {picked['id']} for sub={sub}"


def test_select_hiphop_never_returns_rock(director: MVDirector) -> None:
    for sub in [None, "gangster_rap", "trap", "conscious"]:
        for seed in range(20):
            picked = director.select_producer_profile("hip_hop", sub, None, seed)
            assert "hip_hop" in picked["genres"]


def test_select_gangster_rap_prefers_hype_williams(director: MVDirector) -> None:
    counts: dict[str, int] = {}
    for seed in range(30):
        picked = director.select_producer_profile("hip_hop", "gangster_rap", None, seed)
        counts[picked["id"]] = counts.get(picked["id"], 0) + 1
    assert counts.get("hype_williams", 0) > 0, "gangster_rap should bias toward Hype Williams"


def test_select_conscious_prefers_hiro_murai(director: MVDirector) -> None:
    counts: dict[str, int] = {}
    for seed in range(30):
        picked = director.select_producer_profile("hip_hop", "conscious", None, seed)
        counts[picked["id"]] = counts.get(picked["id"], 0) + 1
    assert counts.get("hiro_murai", 0) > 0


def test_select_trap_prefers_cole_bennett(director: MVDirector) -> None:
    counts: dict[str, int] = {}
    for seed in range(30):
        picked = director.select_producer_profile("hip_hop", "trap", None, seed)
        counts[picked["id"]] = counts.get(picked["id"], 0) + 1
    assert counts.get("cole_bennett", 0) > 0


def test_seeded_choice_is_deterministic(director: MVDirector) -> None:
    a = director.select_producer_profile("pop", None, None, "song_123")
    b = director.select_producer_profile("pop", None, None, "song_123")
    assert a["id"] == b["id"]


# ── Forbidden lists ────────────────────────────────────────────────


def test_every_rock_profile_forbids_dj_pult_for_rock_genre(director: MVDirector) -> None:
    """The user-reported bug: DJ pult shot appeared in a rock song.
    Every rock profile must explicitly include dj_pult in its hip_hop or edm
    forbidden lists AND the universal_forbidden_by_genre['rock'] must include it.
    """
    universal_rock = director._universal_forbidden.get("rock", [])
    assert "dj_pult" in universal_rock
    assert "dj_booth" in universal_rock
    assert "turntables" in universal_rock

    for p in director._producers:
        if "rock" not in p["genres"]:
            continue
        fb_all = []
        for vals in p["forbidden_for_genres"].values():
            fb_all.extend(vals)
        assert "dj_pult" in fb_all or "dj_booth" in fb_all, \
            f"rock profile {p['id']} doesn't forbid dj_pult/dj_booth"


def test_collect_forbidden_for_rock_includes_dj_pult(
    director: MVDirector, aligned_rock_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("rock", "grunge", None, 1)
    forbidden = director._collect_forbidden(profile, "rock")
    assert "dj_pult" in forbidden
    assert "nightclub_dancefloor" in forbidden


# ── Sub-genre modifiers ────────────────────────────────────────────


def test_rnb_modifier_raises_closeup_frequency(director: MVDirector) -> None:
    base = director.select_producer_profile("hip_hop", None, None, 1)
    base_freq = base["closeup_frequency"]
    modified = director.apply_sub_genre_modifiers(base, "rnb")
    assert modified["closeup_frequency"] > base_freq
    assert "fisheye_distortion" in modified.get("_extra_forbidden", [])


def test_power_ballad_overrides_edit_tempo(director: MVDirector) -> None:
    base = director.select_producer_profile("rock", None, None, 1)
    modified = director.apply_sub_genre_modifiers(base, "power_ballad")
    assert modified["edit_tempo"] == "slow_lingering"


def test_trap_enforces_closeup_floor(director: MVDirector) -> None:
    base = director.select_producer_profile("hip_hop", None, None, 1)
    modified = director.apply_sub_genre_modifiers(base, "trap")
    assert modified["closeup_frequency"] >= 0.45


def test_unknown_subgenre_returns_profile_unchanged(director: MVDirector) -> None:
    base = director.select_producer_profile("pop", None, None, 1)
    same = director.apply_sub_genre_modifiers(base, "nonexistent_subgenre_xyz")
    assert same["closeup_frequency"] == base["closeup_frequency"]


# ── Shot plan ───────────────────────────────────────────────────────


def test_shot_plan_no_consecutive_duplicate_shot_types(
    director: MVDirector, aligned_rock_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("rock", "grunge", None, 42)
    sentiment = {"sub_genre": "grunge", "per_section_sentiment": [
        {"section_idx": i, "label": "anthemic"} for i in range(len(aligned_rock_sections))
    ]}
    plan = director.build_shot_plan(aligned_rock_sections, profile, sentiment, song_seed=42)
    assert len(plan) == len(aligned_rock_sections)
    for i in range(1, len(plan)):
        assert plan[i]["shot_type"] != plan[i - 1]["shot_type"], \
            f"consecutive duplicate at idx {i}: {plan[i-1]['shot_type']}"


def test_shot_plan_meets_closeup_quota(
    director: MVDirector, aligned_pop_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("pop", "dance_pop", None, 7)
    profile["closeup_frequency"] = 0.40
    sentiment = {"sub_genre": "dance_pop", "per_section_sentiment": [
        {"section_idx": i, "label": "anthemic"} for i in range(len(aligned_pop_sections))
    ]}
    plan = director.build_shot_plan(aligned_pop_sections, profile, sentiment, song_seed=7)
    cu_count = sum(1 for e in plan if e["shot_type"] in CLOSEUP_SHOTS)
    required = max(1, int(round(0.40 * len(plan))))
    assert cu_count >= required, f"only {cu_count}/{len(plan)} closeups, needed {required}"


def test_shot_plan_establishing_in_first_three(
    director: MVDirector, aligned_rock_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("rock", "grunge", None, 1)
    sentiment = {"sub_genre": "grunge", "per_section_sentiment": []}
    plan = director.build_shot_plan(aligned_rock_sections, profile, sentiment, song_seed=1)
    establishing = [e for e in plan[:3] if e["shot_type"] in ESTABLISHING_SHOTS]
    assert establishing, "no establishing shot in first 3 segments"


def test_shot_plan_sad_chorus_forced_to_closeup(director: MVDirector) -> None:
    aligned = [
        {"section_label": "Intro", "role": "story"},
        {"section_label": "Verse 1", "role": "female"},
        {"section_label": "Chorus", "role": "female"},
        {"section_label": "Verse 2", "role": "female"},
        {"section_label": "Chorus", "role": "female"},
    ]
    profile = director.select_producer_profile("pop", "power_ballad", None, 99)
    profile = director.apply_sub_genre_modifiers(profile, "power_ballad")
    sentiment = {"sub_genre": "power_ballad", "per_section_sentiment": [
        {"section_idx": 0, "label": "intimate"},
        {"section_idx": 1, "label": "melancholic"},
        {"section_idx": 2, "label": "sad"},
        {"section_idx": 3, "label": "melancholic"},
        {"section_idx": 4, "label": "sad"},
    ]}
    plan = director.build_shot_plan(aligned, profile, sentiment, song_seed=99)
    sad_choruses = [p for p in plan if "chorus" in p["section_label"].lower()]
    assert sad_choruses
    for entry in sad_choruses:
        assert entry["shot_type"] in CLOSEUP_SHOTS, \
            f"sad chorus not forced to CU/ECU: {entry}"


def test_shot_plan_empty_sections_returns_empty(director: MVDirector) -> None:
    profile = director.select_producer_profile("pop", None, None, 1)
    plan = director.build_shot_plan([], profile, {"per_section_sentiment": []})
    assert plan == []


# ── Sentiment classifier ───────────────────────────────────────────


def test_fallback_classification_no_llm(director: MVDirector) -> None:
    result = asyncio.run(director.classify_sub_genre_and_sentiment(
        lyrics={"Intro": "", "Verse 1": "lonely night", "Chorus": "i miss you"},
        genre="acoustic ballad",
        key="A minor",
        tempo=72.0,
        aligned_sections=None,
        text_caller=None,
    ))
    assert result["sub_genre"] == "power_ballad"
    assert len(result["per_section_sentiment"]) == 3
    assert all(e["label"] in {"melancholic", "sad", "intimate", "neutral", "calm"}
               for e in result["per_section_sentiment"])


def test_llm_classification_with_mock(director: MVDirector) -> None:
    async def mock_caller(system: str, user: str) -> str:
        return json.dumps({
            "sub_genre": "gangster_rap",
            "per_section_sentiment": [
                {"section_idx": 0, "label": "tense"},
                {"section_idx": 1, "label": "aggressive"},
                {"section_idx": 2, "label": "celebratory"},
            ],
        })

    aligned = [
        {"section_label": "Intro", "role": "story"},
        {"section_label": "Verse 1", "role": "male"},
        {"section_label": "Chorus", "role": "male"},
    ]
    result = asyncio.run(director.classify_sub_genre_and_sentiment(
        lyrics={"Intro": "", "Verse 1": "rolling deep", "Chorus": "we run this"},
        genre="hip hop", key=None, tempo=95.0,
        aligned_sections=aligned, text_caller=mock_caller,
    ))
    assert result["sub_genre"] == "gangster_rap"
    assert result["per_section_sentiment"][1]["label"] == "aggressive"


def test_llm_classification_invalid_json_falls_back(director: MVDirector) -> None:
    async def bad_caller(system: str, user: str) -> str:
        return "totally not json"

    aligned = [{"section_label": "Intro", "role": "story"}, {"section_label": "Verse 1", "role": "male"}]
    result = asyncio.run(director.classify_sub_genre_and_sentiment(
        lyrics={"Intro": "", "Verse 1": "x"}, genre="rock", key="E minor", tempo=100.0,
        aligned_sections=aligned, text_caller=bad_caller,
    ))
    assert "per_section_sentiment" in result
    assert len(result["per_section_sentiment"]) == 2


def test_llm_classification_invalid_label_normalized(director: MVDirector) -> None:
    async def odd_caller(system: str, user: str) -> str:
        return json.dumps({
            "sub_genre": "trap",
            "per_section_sentiment": [
                {"section_idx": 0, "label": "explosive_party_vibes"},
            ],
        })

    aligned = [{"section_label": "Intro", "role": "story"}, {"section_label": "Verse 1", "role": "male"}]
    result = asyncio.run(director.classify_sub_genre_and_sentiment(
        lyrics="[Intro]\n[Verse 1]", genre="hip hop", key=None, tempo=140.0,
        aligned_sections=aligned, text_caller=odd_caller,
    ))
    assert result["per_section_sentiment"][0]["label"] == "neutral"
    assert len(result["per_section_sentiment"]) == 2


# ── Director brief renderer ────────────────────────────────────────


def test_render_director_brief_contains_required_sections(
    director: MVDirector, aligned_rock_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("rock", "grunge", None, 1)
    sentiment = {"sub_genre": "grunge", "per_section_sentiment": []}
    plan = director.build_shot_plan(aligned_rock_sections, profile, sentiment, song_seed=1)
    brief = director.render_director_brief(profile, plan, "rock")
    assert "Producer style reference" in brief
    assert "PER-SEGMENT SHOT DIRECTIVES" in brief
    assert "HARD RULES" in brief
    assert "FORBIDDEN" in brief
    assert "dj_pult" in brief, "rock brief must list dj_pult as forbidden"


def test_render_director_brief_includes_each_section(
    director: MVDirector, aligned_pop_sections: list[dict]
) -> None:
    profile = director.select_producer_profile("pop", None, None, 1)
    sentiment = {"sub_genre": None, "per_section_sentiment": []}
    plan = director.build_shot_plan(aligned_pop_sections, profile, sentiment, song_seed=1)
    brief = director.render_director_brief(profile, plan, "pop")
    for idx in range(len(aligned_pop_sections)):
        assert f"| {idx} |" in brief, f"segment {idx} missing from brief"


# ── Genre canonicalisation ─────────────────────────────────────────


def test_canonicalise_genre_variants(director: MVDirector) -> None:
    assert director._canonicalise_genre("hip_hop") == "hip_hop"
    assert director._canonicalise_genre("hiphop_trap") == "hip_hop"
    assert director._canonicalise_genre("alternative_rock") == "rock"
    assert director._canonicalise_genre("grunge_rock") == "rock"
    assert director._canonicalise_genre("dance_pop") == "pop"
    assert director._canonicalise_genre("electronic_dance") == "pop"
    assert director._canonicalise_genre("unknown_genre") == "pop"
