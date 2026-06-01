"""Fix 30 — unit tests for wardrobe arc expansion, genre defaults,
wardrobe-tag idempotent post-injection, duet-portrait prompt identity +
outfit anchors, and presence of the Fix-30 WARDROBE COHERENCE rule in
segment-director prompts.

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
    """As segment index grows, the template index never decreases —
    the singer never goes BACKWARD in the wardrobe story."""
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
# _genre_default_wardrobe_arc — soft-arc defaults, fallback to global default
# ---------------------------------------------------------------------------

def test_genre_default_known_genres_resolve_to_valid_arcs():
    for genre in ("reggae", "synthwave summer", "Deep House", "indie rock", "country", "lofi"):
        got = _genre_default_wardrobe_arc(genre)
        assert got in WARDROBE_ARCS, f"{genre}: arc {got!r} unknown"


def test_genre_default_unknown_genre_uses_global_default():
    assert _genre_default_wardrobe_arc("nonexistent-genre-xyz") == _DEFAULT_WARDROBE_ARC
    assert _genre_default_wardrobe_arc("") == _DEFAULT_WARDROBE_ARC


def test_default_arc_is_soft_arc_not_single_slot():
    """The default arc should be a 2- or 3-slot arc (soft-arc strategy),
    NOT a single-outfit lockdown."""
    template = WARDROBE_ARCS[_DEFAULT_WARDROBE_ARC]
    assert len(template) >= 2, (
        f"default arc '{_DEFAULT_WARDROBE_ARC}' should have ≥2 slots, "
        f"got {len(template)}"
    )


# ---------------------------------------------------------------------------
# _append_wardrobe_tag — append, idempotent, handle empty input
# ---------------------------------------------------------------------------

def test_append_wardrobe_tag_adds_suffix():
    out = _append_wardrobe_tag("Close-up of singer on beach", "casual_beachwear")
    assert "wearing" in out
    assert WARDROBE_STATES["casual_beachwear"] in out


def test_append_wardrobe_tag_idempotent():
    slot = "performance_stage"
    once = _append_wardrobe_tag("Medium shot of singer on stage", slot)
    twice = _append_wardrobe_tag(once, slot)
    assert once == twice, "second append should be a no-op"


def test_append_wardrobe_tag_empty_prompt_returns_suffix_text():
    out = _append_wardrobe_tag("", "intimate_indoor")
    assert out
    assert "knit sweater" in out.lower() or "intimate" in out.lower() or "sweater" in out.lower()


def test_append_wardrobe_tag_unknown_slot_returns_prompt_unchanged():
    base = "Singer on a rooftop"
    assert _append_wardrobe_tag(base, "made_up_slot") == base


def test_append_wardrobe_tag_strips_trailing_period():
    out = _append_wardrobe_tag("Close-up of singer.", "casual_beachwear")
    assert ", wearing" in out
    assert ".," not in out


# ---------------------------------------------------------------------------
# _wardrobe_tag_suffix — returns formatted suffix for valid keys, "" for invalid
# ---------------------------------------------------------------------------

def test_wardrobe_tag_suffix_known_slot():
    s = _wardrobe_tag_suffix("formal_evening")
    assert s.startswith(", wearing ")
    assert WARDROBE_STATES["formal_evening"] in s


def test_wardrobe_tag_suffix_unknown_slot_empty():
    assert _wardrobe_tag_suffix("not_a_slot") == ""


# ---------------------------------------------------------------------------
# _SEG_DIRECTOR_RULES — outfit-per-section clause softened
# ---------------------------------------------------------------------------

def test_seg_director_rules_no_longer_demands_outfit_variation_per_section():
    """The original wording 'vary location, outfit detail, pose and background
    per section' caused outfit-chaos. Fix 30 changes this — the literal phrase
    must no longer appear verbatim."""
    assert "vary location, outfit detail, pose and background per section" \
        not in _SEG_DIRECTOR_RULES


def test_seg_director_rules_references_wardrobe_plan():
    """The new wording should reference the wardrobe plan / Fix 30 explicitly."""
    text = _SEG_DIRECTOR_RULES.lower()
    assert "wardrobe" in text
    assert "fix 30" in text


def test_seg_director_rules_requires_identity_continuity_via_outfit():
    text = _SEG_DIRECTOR_RULES.lower()
    assert "re-using the same outfit" in text or "identity continuity" in text


# ---------------------------------------------------------------------------
# build_duet_portrait_prompt — identity + outfit anchors
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


def test_duet_prompt_includes_wardrobe_slot_when_provided():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear")
    assert WARDROBE_STATES["casual_beachwear"] in p
    assert "wardrobe slot" in p.lower()


def test_duet_prompt_unknown_slot_falls_back_to_generic_costume_clause():
    p = build_duet_portrait_prompt("any", wardrobe_slot="made_up_slot")
    # No specific outfit description, but the generic "established costumes
    # from the reference images" clause is still present.
    assert "established costumes from the reference images" in p


def test_duet_prompt_default_no_slot_keeps_generic_costume_clause():
    p = build_duet_portrait_prompt("any")
    assert "established costumes from the reference images" in p


def test_duet_prompt_preserves_legacy_identity_neutral_core():
    """Fix 24A core must still be intact: no LLM-invented identity attributes;
    explicit identity-lock clause."""
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


def test_every_wardrobe_slot_has_nonempty_description():
    for slot, text in WARDROBE_STATES.items():
        assert text.strip(), f"{slot}: empty description"


def test_wardrobe_states_contain_clothing_keywords():
    """Sanity-check that descriptions actually describe clothing, not just labels."""
    clothing_keywords = (
        "dress", "shirt", "pants", "jeans", "shorts", "top", "skirt",
        "jacket", "blouse", "tee", "hoodie", "bodysuit", "leggings",
        "sweater", "gown", "boots", "sandals", "sneakers", "heels",
    )
    for slot, text in WARDROBE_STATES.items():
        lower = text.lower()
        assert any(k in lower for k in clothing_keywords), (
            f"{slot}: description has no recognizable clothing word: {text!r}"
        )
