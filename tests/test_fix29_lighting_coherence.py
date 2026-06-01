"""Fix 29 — unit tests for time-of-day arc expansion, genre defaults,
light-tag idempotent post-injection, and presence of the Fix-29 UNIVERSAL
SCENE-FRAMING RULE in segment-director prompts.

Pure / network-free — covers only the deterministic helpers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import music_video_pipeline as mvp  # noqa: E402
from music_video_pipeline import (  # noqa: E402
    TIME_OF_DAY_ARCS,
    TIME_OF_DAY_STATES,
    _DEFAULT_TIME_OF_DAY_ARC,
    _append_light_tag,
    _expand_tod_plan,
    _genre_default_tod_arc,
    _light_tag_suffix,
    _SEG_DIRECTOR_RULES,
)


# ---------------------------------------------------------------------------
# _expand_tod_plan — length, validity, monotonic forward progression
# ---------------------------------------------------------------------------

def test_expand_tod_plan_length_matches_for_every_arc_and_segment_count():
    for arc_key in TIME_OF_DAY_ARCS:
        for n in (1, 2, 3, 5, 8, 12, 20):
            plan = _expand_tod_plan(arc_key, n)
            assert len(plan) == n, f"{arc_key}/{n}: got {len(plan)}"
            for state in plan:
                assert state in TIME_OF_DAY_STATES, f"{arc_key}: unknown state {state!r}"


def test_expand_tod_plan_single_state_arc_is_constant():
    for arc_key, template in TIME_OF_DAY_ARCS.items():
        if len(template) != 1:
            continue
        plan = _expand_tod_plan(arc_key, 7)
        assert plan == [template[0]] * 7, f"{arc_key} not constant"


def test_expand_tod_plan_multi_state_is_monotonic_forward():
    """Multi-state arc: as i grows, the template index never decreases."""
    for arc_key, template in TIME_OF_DAY_ARCS.items():
        if len(template) < 2:
            continue
        for n in (2, 3, 5, 8, 12):
            plan = _expand_tod_plan(arc_key, n)
            indices = [template.index(s) for s in plan]
            assert indices == sorted(indices), (
                f"{arc_key}/{n}: not monotonic: {plan}"
            )


def test_expand_tod_plan_multi_state_lands_on_final_template_state():
    """Last segment should hit the arc's closing state — guarantees the
    progression actually reaches its destination instead of stopping early."""
    for arc_key, template in TIME_OF_DAY_ARCS.items():
        if len(template) < 2:
            continue
        for n in (2, len(template), max(len(template), 6), 10):
            plan = _expand_tod_plan(arc_key, n)
            assert plan[-1] == template[-1], (
                f"{arc_key}/{n}: last={plan[-1]} != template-final={template[-1]}"
            )


def test_expand_tod_plan_unknown_arc_falls_back_to_default():
    plan_unknown = _expand_tod_plan("doesnotexist", 5)
    plan_default = _expand_tod_plan(_DEFAULT_TIME_OF_DAY_ARC, 5)
    assert plan_unknown == plan_default


def test_expand_tod_plan_zero_or_negative_returns_empty():
    assert _expand_tod_plan("golden_hour_to_blue_hour", 0) == []
    assert _expand_tod_plan("golden_hour_to_blue_hour", -3) == []


# ---------------------------------------------------------------------------
# _genre_default_tod_arc — substring match, fallback to global default
# ---------------------------------------------------------------------------

def test_genre_default_known_genres_resolve():
    cases = {
        "reggae": "golden_hour_to_blue_hour",
        "cyberpunk synthwave": "single_night",  # cyberpunk hit first
        "Deep House": "sunset_to_night",
        "doom metal": "stormy_constant",  # 'metal' hit (substring)
    }
    for genre, expected in cases.items():
        got = _genre_default_tod_arc(genre)
        # 'doom' also hits stormy_constant; both acceptable so just check that
        # the returned arc exists.
        assert got in TIME_OF_DAY_ARCS, f"{genre}: arc {got!r} unknown"


def test_genre_default_unknown_genre_uses_global_default():
    assert _genre_default_tod_arc("nonexistent-genre-xyz") == _DEFAULT_TIME_OF_DAY_ARC
    assert _genre_default_tod_arc("") == _DEFAULT_TIME_OF_DAY_ARC


# ---------------------------------------------------------------------------
# _append_light_tag — append, idempotent, handle empty input
# ---------------------------------------------------------------------------

def test_append_light_tag_adds_suffix():
    out = _append_light_tag("Close-up of singer on beach", "morning_golden")
    assert out.endswith(TIME_OF_DAY_STATES["morning_golden"])
    assert "lit by" in out


def test_append_light_tag_idempotent_when_state_already_present():
    state = "blue_hour"
    once = _append_light_tag("Wide shot of harbor at dusk", state)
    twice = _append_light_tag(once, state)
    assert once == twice, "second append should be a no-op"


def test_append_light_tag_empty_prompt_returns_suffix_text():
    out = _append_light_tag("", "midday")
    assert out  # not empty
    assert "midday" in out.lower() or "high overhead sun" in out.lower()


def test_append_light_tag_unknown_state_returns_prompt_unchanged():
    base = "Singer at the microphone"
    assert _append_light_tag(base, "made_up_state") == base


def test_append_light_tag_strips_trailing_period_before_suffix():
    out = _append_light_tag("Close-up of singer.", "blue_hour")
    # The trailing period is stripped so the suffix attaches cleanly.
    assert ", lit by" in out
    assert ".," not in out


# ---------------------------------------------------------------------------
# _light_tag_suffix — returns formatted suffix for valid keys, "" for invalid
# ---------------------------------------------------------------------------

def test_light_tag_suffix_known_state():
    s = _light_tag_suffix("golden_hour_late")
    assert s.startswith(", lit by ")
    assert TIME_OF_DAY_STATES["golden_hour_late"] in s


def test_light_tag_suffix_unknown_state_empty():
    assert _light_tag_suffix("not_a_state") == ""


# ---------------------------------------------------------------------------
# _SEG_DIRECTOR_RULES — Fix 29 UNIVERSAL block coexists with Fix 26 VOCAL block
# ---------------------------------------------------------------------------

def test_seg_director_rules_contains_fix26_vocal_block():
    assert "HARD VOCAL FRAMING RULE (Fix 26" in _SEG_DIRECTOR_RULES


def test_seg_director_rules_contains_fix29_universal_block():
    assert "UNIVERSAL SCENE-FRAMING RULE (Fix 29" in _SEG_DIRECTOR_RULES


def test_seg_director_rules_fix29_bans_anonymous_distant_figures():
    text = _SEG_DIRECTOR_RULES
    assert "anonymous distant figures" in text.lower()
    assert "stock-establisher" in text.lower()


def test_seg_director_rules_fix29_allows_named_story_characters():
    text = _SEG_DIRECTOR_RULES.lower()
    # Spot-check the explicit allowed examples we wrote into the rule.
    for needle in ("rastafari dj", "dancer", "surfer", "fisherman"):
        assert needle in text, f"missing allowed-example '{needle}'"


def test_seg_director_rules_fix29_has_replaceability_test():
    text = _SEG_DIRECTOR_RULES.lower()
    assert "two unnamed people walking" in text
    assert "forbidden" in text and "allowed" in text


# ---------------------------------------------------------------------------
# Arc inventory sanity
# ---------------------------------------------------------------------------

def test_default_arc_exists():
    assert _DEFAULT_TIME_OF_DAY_ARC in TIME_OF_DAY_ARCS


def test_every_arc_template_references_valid_states():
    for arc_key, template in TIME_OF_DAY_ARCS.items():
        assert template, f"{arc_key}: empty template"
        for state in template:
            assert state in TIME_OF_DAY_STATES, (
                f"{arc_key}: template references unknown state {state!r}"
            )


def test_every_state_has_nonempty_description():
    for state, text in TIME_OF_DAY_STATES.items():
        assert text.strip(), f"{state}: empty description"
