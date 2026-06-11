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


from audio_gender_detect import _DUET_OVERRIDE_CONF  # noqa: E402

# Stage F lowered HIGH_CONF 0.85 -> 0.70 (matches the segmenter's
# dominant_threshold floor); replica mirrors api.py.
HIGH_CONF = 0.70
LOW_CONF = 0.70


def _override_decision(label, detected_gender, confidence):
    """Pure replica of the Fix 36 override-decision branch in api.py.

    Returns (new_label, corrected_bool, kept_bool).
    """
    if detected_gender == "unknown":
        return (label, False, False)
    has_explicit = bool(re.search(r' - (male|female|duet)\b', label))
    required = HIGH_CONF if has_explicit else LOW_CONF
    # Stage O5 duet bias: duet confidence (2×min ratio) caps at 1.0 only for
    # a perfect 50/50 split — it gets its own, lower gate.
    if detected_gender == "duet":
        required = _DUET_OVERRIDE_CONF
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
    new, corrected, kept = _override_decision("Verse - male", "female", 0.699)
    assert new == "Verse - male"
    assert kept is True


def test_explicit_label_overridden_at_exact_threshold():
    new, corrected, kept = _override_decision("Verse - male", "female", 0.70)
    assert new == "Verse - female"
    assert corrected is True


# ---------------------------------------------------------------------------
# Stage O5 — duet bias: balanced alternating vocals must win over solo labels
# ---------------------------------------------------------------------------

def test_duet_bias_overrides_solo_label_at_balanced_split():
    """70/30 alternating duet (job ffd3b9a6 pattern): classification must say
    'duet' and its 0.6 balance confidence must pass the duet gate and rewrite
    an explicit solo lyrics label."""
    from audio_gender_detect import _classify_section
    segs = [("male", 0.0, 7.0), ("female", 7.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "duet"
    assert abs(conf - 0.6) < 0.01
    new, corrected, kept = _override_decision("Verse 1 - male", gender, conf)
    assert new == "Verse 1 - duet"
    assert corrected is True
    assert kept is False


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
    """Stage O5: 80/20 is no duet anymore (below _DUET_MIN_RATIO=0.25) —
    it classifies as solo male whose 0.8 ratio passes the solo gate."""
    from audio_gender_detect import _classify_section
    segs = [("male", 0.0, 8.0), ("female", 8.0, 10.0)]
    gender, conf = _classify_section(0.0, 10.0, segs)
    assert gender == "male"
    assert abs(conf - 0.8) < 0.01
