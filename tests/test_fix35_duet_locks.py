"""Fix 35 — verify duet portrait prompt has garment-lock + anatomy-lock +
composition-hint, and that mixed-duet names male outfit before female to
match portrait_a reference order.

Pure / network-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    build_duet_portrait_prompt,
)


# ---------------------------------------------------------------------------
# Fix 35 A — Garment-Lock + Cross-Outfit Negation (mixed duet)
# ---------------------------------------------------------------------------

def test_mixed_duet_has_garment_lock_negation():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    text = p.lower()
    # Lock per performer
    assert "the man wears:" in text
    assert "the woman wears:" in text
    # Cross-swap negation
    assert "never swap" in text or "no cross-dressing" in text
    assert "no shared pieces" in text
    assert "no half-and-half" in text or "no garment split" in text


def test_mixed_duet_male_outfit_named_before_female():
    """Fix 35 D: portrait_a (lead, duplicated weight) is the male performer.
    Naming his outfit first reduces model-bias of associating the FIRST
    outfit with the FIRST reference image."""
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    assert "The man wears:" in p
    assert "The woman wears:" in p
    assert p.index("The man wears:") < p.index("The woman wears:")
    # Also verify the actual outfit text order
    male_outfit = "linen shirt"
    female_outfit = "sundress"
    assert p.index(male_outfit) < p.index(female_outfit)


def test_mixed_duet_garment_lock_works_for_other_slots():
    """Slot-agnostic — works for any slot, not just beachwear."""
    for slot in ("performance_stage", "intimate_indoor", "formal_evening", "streetwear_neon"):
        p = build_duet_portrait_prompt("any", wardrobe_slot=slot, duet_kind="mixed")
        assert "The man wears:" in p
        assert "The woman wears:" in p
        assert p.index("The man wears:") < p.index("The woman wears:")


# ---------------------------------------------------------------------------
# Fix 35 B — Anatomy-Lock (all duet kinds)
# ---------------------------------------------------------------------------

def test_anatomy_lock_present_in_mixed_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    text = p.lower()
    assert "exactly two arms and two legs per person" in text
    assert "exactly one head per person" in text
    assert "no merged limbs" in text
    assert "no shared body parts" in text
    assert "no extra arms or legs" in text


def test_anatomy_lock_present_in_ff_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="ff")
    text = p.lower()
    assert "exactly two arms and two legs per person" in text
    assert "no merged limbs" in text


def test_anatomy_lock_present_in_mm_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mm")
    text = p.lower()
    assert "exactly two arms and two legs per person" in text
    assert "no merged limbs" in text


# ---------------------------------------------------------------------------
# Fix 35 C — Composition-Hint (all duet kinds)
# ---------------------------------------------------------------------------

def test_composition_hint_present_in_mixed_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    text = p.lower()
    assert "small visible gap" in text
    assert "never overlapping at the torso" in text
    assert "never merged or fused" in text


def test_composition_hint_present_in_ff_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="ff")
    text = p.lower()
    assert "small visible gap" in text
    assert "never overlapping at the torso" in text


def test_composition_hint_present_in_mm_duet():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mm")
    text = p.lower()
    assert "small visible gap" in text
    assert "never overlapping at the torso" in text


# ---------------------------------------------------------------------------
# Fix 35 A — Same-gender garment-lock (no cross-swap needed, but split-prevention)
# ---------------------------------------------------------------------------

def test_ff_duet_includes_split_prevention():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="ff")
    text = p.lower()
    assert "no half-and-half outfits" in text
    assert "no shared pieces" in text or "no garments split between bodies" in text


def test_mm_duet_includes_split_prevention():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mm")
    text = p.lower()
    assert "no half-and-half outfits" in text
    assert "no shared pieces" in text or "no garments split between bodies" in text


# ---------------------------------------------------------------------------
# Back-compat — Fix-24A / Fix-30 / Fix-32 core still intact
# ---------------------------------------------------------------------------

def test_duet_prompt_still_has_identity_anchor():
    p = build_duet_portrait_prompt("any", wardrobe_slot="casual_beachwear", duet_kind="mixed")
    assert "Preserve each performer" in p
    assert "eye colour" in p
    assert "jawline" in p
    assert "reference images" in p


def test_duet_prompt_no_slot_still_returns_valid_prompt():
    """Without a wardrobe_slot, the prompt still has anatomy + composition
    locks and the generic costume clause."""
    p = build_duet_portrait_prompt("any")
    assert "Two clearly separate full-body figures" in p
    assert "small visible gap" in p
    assert "established costumes from the reference images" in p
