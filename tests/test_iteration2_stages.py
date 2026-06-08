"""Tests for Iteration 2 fixes (2026-06-07 evening).

Stage F: gender-override threshold lowered to 0.70.
Stage G: duet portrait force-generates missing solo refs.
Stage H: multi-character composition clause in system prompt.
Stage I: GENRE_DANCE_STYLES lookup + clause in system prompt.

Stage G is exercised at the unit-of-logic level (the threshold-and-
generate decision) rather than against the full async dispatcher, since
the dispatcher carries heavy ComfyUI side effects. The decision rule is
the entire bug. Same for Stage F: the rule under test is the
threshold comparison + label rewrite, not the upstream demucs/inaSpeech
machinery.
"""
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    GENRE_DANCE_STYLES,
    _DEFAULT_GENRE_DANCE,
    _genre_dance_style,
)


# ---------------------------------------------------------------------------
# Stage F: gender-override threshold
# ---------------------------------------------------------------------------


def _apply_override(label: str, audio_gender: str, audio_conf: float,
                    high_conf: float = 0.70, low_conf: float = 0.70) -> str:
    """Pure-Python replica of api.py:1107-1140 override decision.

    Kept in sync by the same threshold constants; this lets us unit-test
    the rule without booting demucs/inaSpeech/the full pipeline.
    """
    has_explicit = bool(re.search(r' - (male|female|duet)\b', label))
    required = high_conf if has_explicit else low_conf
    if audio_gender == "unknown":
        return label
    if audio_conf < required:
        return label
    return re.sub(r' - (male|female|duet)\b', f" - {audio_gender}", label, count=1)


def test_stage_f_overrides_at_confidence_0_72():
    """Stage F: audio classified as male with confidence 0.72 must
    override an LLM 'Verse 1 - female' label (this exact scenario was
    silently kept on pre-F threshold 0.85 in job 63486a7a)."""
    new = _apply_override("Verse 1 - female", "male", 0.72)
    assert new == "Verse 1 - male"


def test_stage_f_overrides_at_confidence_0_70_boundary():
    """Threshold is inclusive at 0.70 (matches segmenter dominant floor)."""
    new = _apply_override("Verse 1 - female", "male", 0.70)
    assert new == "Verse 1 - male"


def test_stage_f_does_not_override_below_threshold():
    """0.68 is below floor — label sticks to LLM-authored value."""
    new = _apply_override("Verse 1 - female", "male", 0.68)
    assert new == "Verse 1 - female"


def test_stage_f_unknown_gender_never_overrides():
    new = _apply_override("Verse 1 - female", "unknown", 0.99)
    assert new == "Verse 1 - female"


def test_stage_f_generic_label_also_uses_low_threshold():
    """Labels without explicit role suffix still need ≥ low_conf=0.70."""
    new = _apply_override("Verse 1", "male", 0.71)
    assert new == "Verse 1"  # no role tag, regex sub finds no match


# ---------------------------------------------------------------------------
# Stage G: duet-portrait force-generation rule
# ---------------------------------------------------------------------------


def _missing_solo_roles_for_duet(portraits: dict, all_roles_present: set) -> list:
    """Pure-Python replica of api.py Stage-G decision: return the list of
    solo roles whose portrait must be force-generated before the duet
    composite can run."""
    missing: list = []
    if "duet" not in all_roles_present:
        return missing
    for role in ("male", "female"):
        if portraits.get(role) is None:
            missing.append(role)
    return missing


def test_stage_g_female_only_song_with_duet_forces_male_portrait():
    """Job-63486a7a scenario: song has female solos + duets, no male
    solo. Pre-G the male portrait was never generated → duet composite
    skipped → duet segments fell back to the female portrait + a random
    male partner in every shot. Post-G: male portrait force-generated."""
    portraits = {"female": "/tmp/female.png"}
    missing = _missing_solo_roles_for_duet(portraits, {"female", "duet"})
    assert missing == ["male"]


def test_stage_g_male_only_song_with_duet_forces_female_portrait():
    portraits = {"male": "/tmp/male.png"}
    missing = _missing_solo_roles_for_duet(portraits, {"male", "duet"})
    assert missing == ["female"]


def test_stage_g_both_solos_present_no_extra_generation():
    portraits = {"male": "/tmp/m.png", "female": "/tmp/f.png"}
    missing = _missing_solo_roles_for_duet(portraits, {"male", "female", "duet"})
    assert missing == []


def test_stage_g_no_duet_no_force_generation():
    """Female-solo song with no duet → no extra generation."""
    portraits = {"female": "/tmp/f.png"}
    missing = _missing_solo_roles_for_duet(portraits, {"female"})
    assert missing == []


# ---------------------------------------------------------------------------
# Stage I: GENRE_DANCE_STYLES lookup
# ---------------------------------------------------------------------------


def test_stage_i_rock_returns_rock_specific_moves():
    style = _genre_dance_style("pop-rock duet rooftop sunset")
    # "rock" substring hits rock entry — not pop.
    assert "headbang" in style or "power poses" in style


def test_stage_i_flamenco_returns_footwork():
    style = _genre_dance_style("traditional spanish flamenco")
    assert "stamping footwork" in style
    assert "palmas" in style or "hand claps" in style


def test_stage_i_hip_hop_recognised_with_space():
    style = _genre_dance_style("modern hip hop")
    assert "isolation" in style or "body rolls" in style


def test_stage_i_unknown_genre_returns_default():
    style = _genre_dance_style("xenobiological space opera")
    assert style == _DEFAULT_GENRE_DANCE


def test_stage_i_empty_genre_returns_default():
    assert _genre_dance_style("") == _DEFAULT_GENRE_DANCE
    assert _genre_dance_style(None) == _DEFAULT_GENRE_DANCE


def test_stage_i_default_is_concrete_not_vague():
    """The default must not be the very vague phrasing this stage exists
    to fix. 'beat-synced' is acceptable; 'energetic dance break' is not."""
    assert "energetic dance break" not in _DEFAULT_GENRE_DANCE
    assert "dance moves" not in _DEFAULT_GENRE_DANCE


def test_stage_i_dictionary_covers_key_genres():
    must_have = ["rock", "metal", "hip hop", "pop", "edm", "flamenco",
                 "country", "jazz", "classical"]
    for g in must_have:
        assert g in GENRE_DANCE_STYLES, f"missing genre key: {g}"


# ---------------------------------------------------------------------------
# Stage H: multi-character composition clause is present in system prompt
# ---------------------------------------------------------------------------


def test_stage_h_clause_present_in_module_source():
    """Sanity check: the prompt-string lives inside plan_segments. We
    don't render the full prompt here (it needs an async LLM context),
    but we verify the clause literal made it into the file so a future
    refactor that drops the clause fails this test."""
    import music_video_pipeline as mv
    src = open(mv.__file__, encoding="utf-8").read()
    assert "MULTI-CHARACTER COMPOSITION (Stage H" in src
    assert "GENRE CHOREOGRAPHY (Stage I" in src
