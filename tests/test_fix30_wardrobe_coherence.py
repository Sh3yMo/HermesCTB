"""Fix 30 + Fix 31 — unit tests for wardrobe arc expansion, genre defaults,
role-aware wardrobe-tag post-injection, duet-portrait prompt identity +
outfit anchors, and presence of the wardrobe rule blocks in segment-director
prompts.

Pure / network-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import music_video_pipeline as mvp  # noqa: E402
from music_video_pipeline import (  # noqa: E402
    WARDROBE_ARCS,
    WARDROBE_STATES,
    _DEFAULT_WARDROBE_ARC,
    _append_wardrobe_tag,
    _expand_wardrobe_plan,
    _genre_default_wardrobe_arc,
    _get_wardrobe_outfit,
    _wardrobe_tag_suffix,
    _SEG_DIRECTOR_RULES,
    build_duet_portrait_prompt,
)


# ---------------------------------------------------------------------------
# _expand_wardrobe_plan — length, validity, monotonic forward progression
# ---------------------------------------------------------------------------

def test_expand_wardrobe_plan_length_matches_for_every_arc_and_segment_count():
    for arc_key in WARDROBE_ARCS:
        for n in (1, 2, 3, 5, 8, 12, 20):
            plan = _expand_wardrobe_plan(arc_key, n)
            assert len(plan) == n, f"{arc_key}/{n}: got {len(plan)}"
            for slot in plan:
                assert slot in WARDROBE_STATES, f"{arc_key}: unknown slot {slot!r}"


def test_expand_wardrobe_plan_single_slot_arc_is_constant():
    for arc_key, template in WARDROBE_ARCS.items():
        if len(template) != 1:
            continue
        plan = _expand_wardrobe_plan(arc_key, 7)
        assert plan == [template[0]] * 7, f"{arc_key} not constant"


def test_expand_wardrobe_plan_multi_slot_is_monotonic_forward():
    for arc_key, template in WARDROBE_ARCS.items():
        if len(template) < 2:
            continue
        for n in (2, 3, 5, 8, 12):
            plan = _expand_wardrobe_plan(arc_key, n)
            indices = [template.index(s) for s in plan]
            assert indices == sorted(indices), (
                f"{arc_key}/{n}: not monotonic: {plan}"
            )


def test_expand_wardrobe_plan_multi_slot_lands_on_final_template_slot():
    for arc_key, template in WARDROBE_ARCS.items():
        if len(template) < 2:
            continue
        for n in (2, len(template), max(len(template), 6), 10):
            plan = _expand_wardrobe_plan(arc_key, n)
            assert plan[-1] == template[-1], (
                f"{arc_key}/{n}: last={plan[-1]} != template-final={template[-1]}"
            )


def test_expand_wardrobe_plan_unknown_arc_falls_back_to_default():
    plan_unknown = _expand_wardrobe_plan("nonexistent_arc", 5)
    plan_default = _expand_wardrobe_plan(_DEFAULT_WARDROBE_ARC, 5)
    assert plan_unknown == plan_default


def test_expand_wardrobe_plan_zero_or_negative_returns_empty():
    assert _expand_wardrobe_plan("daywear_to_evening", 0) == []
    assert _expand_wardrobe_plan("daywear_to_evening", -3) == []


# ---------------------------------------------------------------------------
# _genre_default_wardrobe_arc
# ---------------------------------------------------------------------------

def test_genre_default_known_genres_resolve_to_valid_arcs():
    for genre in ("reggae", "synthwave summer", "Deep House", "indie rock", "country", "lofi"):
        got = _genre_default_wardrobe_arc(genre)
        assert got in WARDROBE_ARCS, f"{genre}: arc {got!r} unknown"


def test_genre_default_unknown_genre_uses_global_default():
    assert _genre_default_wardrobe_arc("nonexistent-genre-xyz") == _DEFAULT_WARDROBE_ARC
    assert _genre_default_wardrobe_arc("") == _DEFAULT_WARDROBE_ARC


def test_default_arc_is_soft_arc_not_single_slot():
    template = WARDROBE_ARCS[_DEFAULT_WARDROBE_ARC]
    assert len(template) >= 2, (
        f"default arc '{_DEFAULT_WARDROBE_ARC}' should have ≥2 slots, "
        f"got {len(template)}"
    )


# ---------------------------------------------------------------------------
# Fix 31 — WARDROBE_STATES schema: each slot has female + male keys
# ---------------------------------------------------------------------------

def test_every_wardrobe_state_has_female_and_male_outfits():
    for slot, entry in WARDROBE_STATES.items():
        assert isinstance(entry, dict), f"{slot}: not a dict, got {type(entry).__name__}"
        assert "female" in entry, f"{slot}: missing 'female' key"
        assert "male" in entry, f"{slot}: missing 'male' key"
        assert entry["female"].strip(), f"{slot}: empty female outfit"
        assert entry["male"].strip(), f"{slot}: empty male outfit"


def test_get_wardrobe_outfit_resolves_per_sex():
    f = _get_wardrobe_outfit("casual_beachwear", "female")
    m = _get_wardrobe_outfit("casual_beachwear", "male")
    assert "sundress" in f.lower()
    assert "linen shirt" in m.lower()
    assert f != m


def test_get_wardrobe_outfit_unknown_slot_returns_empty():
    assert _get_wardrobe_outfit("nope", "female") == ""
    assert _get_wardrobe_outfit("nope", "male") == ""


def test_get_wardrobe_outfit_unknown_sex_returns_empty():
    assert _get_wardrobe_outfit("casual_beachwear", "duet") == ""
    assert _get_wardrobe_outfit("casual_beachwear", "child") == ""
    assert _get_wardrobe_outfit("casual_beachwear", "") == ""


# ---------------------------------------------------------------------------
# Fix 31 — role-aware _wardrobe_tag_suffix
# ---------------------------------------------------------------------------

def test_wardrobe_tag_suffix_female_role_returns_female_outfit():
    s = _wardrobe_tag_suffix("casual_beachwear", role="female")
    assert s.startswith(", wearing ")
    assert "sundress" in s
    assert "linen shirt" not in s


def test_wardrobe_tag_suffix_male_role_returns_male_outfit():
    s = _wardrobe_tag_suffix("casual_beachwear", role="male")
    assert s.startswith(", wearing ")
    assert "linen shirt" in s
    assert "sundress" not in s


def test_wardrobe_tag_suffix_duet_role_contains_both_outfits():
    s = _wardrobe_tag_suffix("casual_beachwear", role="duet")
    assert "female performer" in s.lower()
    assert "male performer" in s.lower()
    assert "sundress" in s
    assert "linen shirt" in s


# ---------------------------------------------------------------------------
# Fix 32 — duet_kind awareness (ff / mm / mixed)
# ---------------------------------------------------------------------------

def test_wardrobe_tag_suffix_duet_ff_uses_only_female_outfit():
    s = _wardrobe_tag_suffix("casual_beachwear", role="duet", duet_kind="ff")
    assert "both female performers" in s.lower()
    assert "sundress" in s
    assert "linen shirt" not in s
    # The mixed-duet phrase "the male performer wearing" must not appear.
    assert "male performer wearing" not in s.lower()


def test_wardrobe_tag_suffix_duet_mm_uses_only_male_outfit():
    s = _wardrobe_tag_suffix("casual_beachwear", role="duet", duet_kind="mm")
    assert "both male performers" in s.lower()
    assert "linen shirt" in s
    assert "sundress" not in s
    assert "female performer wearing" not in s.lower()


def test_wardrobe_tag_suffix_duet_mixed_keeps_both_outfits():
    s = _wardrobe_tag_suffix("casual_beachwear", role="duet", duet_kind="mixed")
    assert "female performer" in s.lower()
    assert "male performer" in s.lower()
    assert "sundress" in s
    assert "linen shirt" in s


def test_wardrobe_tag_suffix_solo_roles_ignore_duet_kind():
    """Solo female/male should be unaffected by duet_kind."""
    fem_mixed = _wardrobe_tag_suffix("casual_beachwear", role="female", duet_kind="mixed")
    fem_ff = _wardrobe_tag_suffix("casual_beachwear", role="female", duet_kind="ff")
    assert fem_mixed == fem_ff
    male_mixed = _wardrobe_tag_suffix("casual_beachwear", role="male", duet_kind="mixed")
    male_mm = _wardrobe_tag_suffix("casual_beachwear", role="male", duet_kind="mm")
    assert male_mixed == male_mm


def test_append_wardrobe_tag_duet_ff_idempotent():
    once = _append_wardrobe_tag("Wide shot", "casual_beachwear", role="duet", duet_kind="ff")
    twice = _append_wardrobe_tag(once, "casual_beachwear", role="duet", duet_kind="ff")
    assert once == twice
    assert "both female performers" in once.lower()


def test_wardrobe_tag_suffix_none_role_returns_empty():
    """STORY-style segments without a named recurring performer get NO tag."""
    assert _wardrobe_tag_suffix("casual_beachwear", role=None) == ""
    assert _wardrobe_tag_suffix("casual_beachwear", role="") == ""
    assert _wardrobe_tag_suffix("casual_beachwear", role="story") == ""


def test_wardrobe_tag_suffix_unknown_role_returns_empty():
    assert _wardrobe_tag_suffix("casual_beachwear", role="child") == ""
    assert _wardrobe_tag_suffix("casual_beachwear", role="dj") == ""


def test_wardrobe_tag_suffix_unknown_slot_returns_empty():
    assert _wardrobe_tag_suffix("nope", role="female") == ""


# ---------------------------------------------------------------------------
# Fix 31 — role-aware _append_wardrobe_tag
# ---------------------------------------------------------------------------

def test_append_wardrobe_tag_female_appends_female_outfit():
    out = _append_wardrobe_tag("Close-up of singer on beach", "casual_beachwear", role="female")
    assert "sundress" in out
    assert "linen shirt" not in out


def test_append_wardrobe_tag_male_appends_male_outfit():
    out = _append_wardrobe_tag("Close-up of singer on beach", "casual_beachwear", role="male")
    assert "linen shirt" in out
    assert "sundress" not in out


def test_append_wardrobe_tag_duet_appends_both_outfits():
    out = _append_wardrobe_tag("Wide shot of duet", "casual_beachwear", role="duet")
    assert "sundress" in out
    assert "linen shirt" in out


def test_append_wardrobe_tag_no_role_leaves_prompt_unchanged():
    """The headline Fix 31 guarantee: no role = no outfit forced on the prompt."""
    base = "Wide shot of a child building a sandcastle on the beach"
    assert _append_wardrobe_tag(base, "casual_beachwear", role=None) == base
    assert _append_wardrobe_tag(base, "casual_beachwear", role="") == base
    assert _append_wardrobe_tag(base, "casual_beachwear", role="story") == base


def test_append_wardrobe_tag_idempotent_per_role():
    once = _append_wardrobe_tag("Medium shot of singer", "performance_stage", role="female")
    twice = _append_wardrobe_tag(once, "performance_stage", role="female")
    assert once == twice


def test_append_wardrobe_tag_unknown_slot_returns_prompt_unchanged():
    base = "Singer on a rooftop"
    assert _append_wardrobe_tag(base, "made_up_slot", role="female") == base


def test_append_wardrobe_tag_strips_trailing_period():
    out = _append_wardrobe_tag("Close-up of singer.", "casual_beachwear", role="female")
    assert ", wearing" in out
    assert ".," not in out


# ---------------------------------------------------------------------------
# _SEG_DIRECTOR_RULES — outfit-per-section clause softened + role-aware
# ---------------------------------------------------------------------------

def test_seg_director_rules_no_longer_demands_outfit_variation_per_section():
    assert "vary location, outfit detail, pose and background per section" \
        not in _SEG_DIRECTOR_RULES


def test_seg_director_rules_references_wardrobe_plan_and_role_awareness():
    text = _SEG_DIRECTOR_RULES.lower()
    assert "wardrobe" in text
    assert "fix 30" in text
    # Fix 31 role-aware addendum.
    assert "fix 31" in text
    assert "role-aware" in text or "applies only to the named" in text


def test_seg_director_rules_warns_against_dressing_other_characters():
    text = _SEG_DIRECTOR_RULES.lower()
    # The "do not dress other characters in the performer's outfit" rule.
    assert "never put the recurring performer's outfit on a different character" in text


# ---------------------------------------------------------------------------
# build_duet_portrait_prompt — identity + role-aware outfit anchors
# ---------------------------------------------------------------------------

def test_duet_prompt_contains_extended_identity_anchor():
    p = build_duet_portrait_prompt("reggae beach")
    text = p.lower()
    for needle in (
        "eye colour", "eyebrow shape", "nose shape", "lip shape",
        "jawline", "body proportions", "height",
    ):
        assert needle in text, f"missing identity anchor: {needle!r}"
    assert "same two people" in text or "same two performers" in text


def test_duet_prompt_includes_both_female_and_male_outfits_when_slot_provided():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear")
    assert "sundress" in p
    assert "linen shirt" in p
    assert "female performer wears" in p
    assert "male performer wears" in p


def test_duet_prompt_ff_uses_only_female_outfit_and_locks_gender():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="ff")
    assert "sundress" in p
    assert "linen shirt" not in p
    assert "Both performers are female" in p
    assert "never depict a male performer" in p.lower()


def test_duet_prompt_mm_uses_only_male_outfit_and_locks_gender():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mm")
    assert "linen shirt" in p
    assert "sundress" not in p
    assert "Both performers are male" in p
    assert "never depict a female performer" in p.lower()


def test_duet_prompt_mixed_default_keeps_legacy_wording():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    assert "female performer wears" in p
    assert "male performer wears" in p
    # No gender-lock clause for mixed duets.
    assert "Both performers are female" not in p
    assert "Both performers are male" not in p


def test_duet_prompt_unknown_slot_falls_back_to_generic_costume_clause():
    p = build_duet_portrait_prompt("any", wardrobe_slot="made_up_slot")
    assert "established costumes from the reference images" in p
    assert "sundress" not in p


def test_duet_prompt_default_no_slot_keeps_generic_costume_clause():
    p = build_duet_portrait_prompt("any")
    assert "established costumes from the reference images" in p


def test_duet_prompt_preserves_legacy_identity_neutral_core():
    p = build_duet_portrait_prompt("any")
    assert "Preserve each performer" in p
    assert "do not restyle or recolour them" in p
    assert "reference images" in p


# ---------------------------------------------------------------------------
# Arc inventory sanity
# ---------------------------------------------------------------------------

def test_default_wardrobe_arc_exists():
    assert _DEFAULT_WARDROBE_ARC in WARDROBE_ARCS


def test_every_wardrobe_arc_references_valid_slots():
    for arc_key, template in WARDROBE_ARCS.items():
        assert template, f"{arc_key}: empty template"
        for slot in template:
            assert slot in WARDROBE_STATES, (
                f"{arc_key}: template references unknown slot {slot!r}"
            )


def test_wardrobe_states_contain_clothing_keywords_in_both_sexes():
    clothing_keywords = (
        "dress", "shirt", "pants", "jeans", "shorts", "top", "skirt",
        "jacket", "blouse", "tee", "hoodie", "bodysuit", "leggings",
        "sweater", "pullover", "gown", "boots", "sandals", "sneakers",
        "heels", "trousers", "suit", "tuxedo", "blazer",
    )
    for slot, entry in WARDROBE_STATES.items():
        for sex in ("female", "male"):
            text = entry[sex].lower()
            assert any(k in text for k in clothing_keywords), (
                f"{slot}/{sex}: description has no recognizable clothing word: {entry[sex]!r}"
            )
