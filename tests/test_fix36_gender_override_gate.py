"""Fix 36 — verify confidence-gated, asymmetric-trust audio override logic.

Tests the override decision in isolation (the api.py code path is small
and inlined; we replicate its pure-function semantics here so we can
unit-test it without spinning up the full FastAPI pipeline).

Pure / network-free.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HIGH_CONF = 0.85
LOW_CONF = 0.70


def _override_decision(label, detected_gender, confidence):
    """Pure replica of the Fix 36 override-decision branch in api.py.

    Returns (new_label, corrected_bool, kept_bool).
    """
    if detected_gender == "unknown":
        return (label, False, False)
    has_explicit = bool(re.search(r' - (male|female|duet)\b', label))
    required = HIGH_CONF if has_explicit else LOW_CONF
    if confidence < required:
        return (label, False, True)  # kept
    new_label = re.sub(r' - (male|female|duet)\b',
                       f' - {detected_gender}', label, count=1)
    if new_label != label:
        return (new_label, True, False)
    return (label, False, False)


# ---------------------------------------------------------------------------
# Asymmetric trust — explicit gender suffix requires HIGH confidence
# ---------------------------------------------------------------------------

def test_explicit_label_kept_when_audio_confidence_low():
    """The 5f6565b3 Flamenco regression: LLM said 'Verse - male', audio
    said 'female' but with confidence 0.55 (close-call). Override must NOT
    fire."""
    new, corrected, kept = _override_decision("Verse - male", "female", 0.55)
    assert new == "Verse - male"
    assert corrected is False
    assert kept is True


def test_explicit_label_overridden_when_audio_confidence_high():
    """If audio is 92% sure the LLM was wrong, do flip."""
    new, corrected, kept = _override_decision("Verse - male", "female", 0.92)
    assert new == "Verse - female"
    assert corrected is True
    assert kept is False


def test_explicit_label_kept_at_exact_threshold_minus_epsilon():
    new, corrected, kept = _override_decision("Verse - male", "female", 0.849)
    assert new == "Verse - male"
    assert kept is True


def test_explicit_label_overridden_at_exact_threshold():
    new, corrected, kept = _override_decision("Verse - male", "female", 0.85)
    assert new == "Verse - female"
    assert corrected is True


# ---------------------------------------------------------------------------
# Generic LLM label — no explicit suffix, lower gate
# ---------------------------------------------------------------------------

def test_generic_label_not_modified_by_override_no_suffix_to_replace():
    """Generic labels like 'Intro' or 'Chorus' have no '- gender' suffix —
    the regex sub leaves them unchanged regardless of confidence."""
    new, corrected, kept = _override_decision("Chorus", "female", 0.80)
    assert new == "Chorus"
    assert corrected is False  # nothing to replace


def test_generic_label_passes_gate_but_no_suffix_exists():
    """Confidence 0.71 > LOW_CONF, but label has no suffix to rewrite."""
    new, corrected, kept = _override_decision("Intro", "female", 0.71)
    assert new == "Intro"
    assert corrected is False


# ---------------------------------------------------------------------------
# Unknown gender — never overrides
# ---------------------------------------------------------------------------

def test_unknown_audio_never_overrides():
    new, corrected, kept = _override_decision("Verse - male", "unknown", 0.99)
    assert new == "Verse - male"
    assert corrected is False
    assert kept is False  # not "kept" semantically — just skipped


# ---------------------------------------------------------------------------
# Duet labels — same gate as male/female
# ---------------------------------------------------------------------------

def test_explicit_duet_label_kept_when_audio_confidence_low():
    new, corrected, kept = _override_decision("Outro - duet", "female", 0.60)
    assert new == "Outro - duet"
    assert kept is True


def test_explicit_duet_label_overridden_at_high_confidence():
    new, corrected, kept = _override_decision("Outro - duet", "female", 0.90)
    assert new == "Outro - female"
    assert corrected is True


# ---------------------------------------------------------------------------
# detect_section_genders integration — verify the gate sees real confidence
# ---------------------------------------------------------------------------

def test_close_call_50_50_returns_duet_with_balance_one():
    """50/50 split — confidence is the balance score (= 1.0)."""
    from audio_gender_detect import _classify_section
    segs = [("male", 0.0, 5.0), ("female", 5.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    assert abs(conf - 1.0) < 0.01


def test_skewed_duet_returns_lower_balance_confidence():
    """80/20 → balance = 0.4, the gate will block a 'duet' override of an
    explicit gender label."""
    from audio_gender_detect import _classify_section
    segs = [("male", 0.0, 8.0), ("female", 8.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    assert abs(conf - 0.4) < 0.01
    # Apply gate: explicit gender label needs HIGH_CONF=0.85
    new, corrected, _ = _override_decision("Verse - male", gender, conf)
    assert new == "Verse - male"
    assert corrected is False
