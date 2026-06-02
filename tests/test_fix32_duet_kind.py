"""Fix 32 — verify same-gender duet recovery: roles_present must include
'duet' even when the first duet section is a reuse row, and
plan_same_gender_portraits must trigger on ff/mm inputs.

Pure / network-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    Segment,
    extract_section_role,
    plan_same_gender_portraits,
)


def _seg(idx, label, reuse_of=None):
    return Segment(
        index=idx,
        start_time=float(idx) * 10.0,
        end_time=float(idx + 1) * 10.0,
        label=label,
        reuse_of=reuse_of,
    )


# ---------------------------------------------------------------------------
# all_roles_present (computed in api.py) — verify the set-comprehension
# pattern works when the first duet appearance is a reuse row.
# ---------------------------------------------------------------------------

def _all_roles_present(segments):
    """Mirror the api.py Fix 32 logic so we can unit-test it without
    importing api.py (which would pull FastAPI / ComfyUI deps)."""
    out = {extract_section_role(s.label) for s in segments}
    out.discard(None)
    return out


def test_all_roles_present_picks_up_duet_role_from_reuse_row():
    """Repro of the 60895ff6 regression: first duet section is a reuse of
    an earlier chorus. roles_present (anchor-only) misses 'duet';
    all_roles_present must NOT."""
    segments = [
        _seg(0, "Intro"),
        _seg(1, "Verse 1 - female"),
        _seg(2, "Chorus - female"),
        _seg(3, "Verse 2 - female"),
        _seg(4, "Chorus - duet", reuse_of=2),  # first duet IS a reuse row
        _seg(5, "Bridge - whispered"),
        _seg(6, "Final Chorus - duet"),
        _seg(7, "Outro"),
    ]
    roles = _all_roles_present(segments)
    assert "duet" in roles
    assert "female" in roles


def test_all_roles_present_only_yields_valid_roles():
    segments = [
        _seg(0, "Intro"),
        _seg(1, "Verse 1 - female"),
        _seg(2, "Chorus - duet"),
    ]
    roles = _all_roles_present(segments)
    assert roles == {"female", "duet"}


# ---------------------------------------------------------------------------
# plan_same_gender_portraits with the all_roles_present set must trigger
# the ff/mm path correctly.
# ---------------------------------------------------------------------------

def test_plan_same_gender_portraits_ff_with_duet_role_returns_make_duet_true():
    roles = {"female", "duet"}
    sg = plan_same_gender_portraits("ff", roles, True, "auto")
    assert sg is not None
    base, partner, make_duet = sg
    assert base == "female"
    assert partner == "female2"
    assert make_duet is True


def test_plan_same_gender_portraits_mm_with_duet_role_returns_make_duet_true():
    roles = {"male", "duet"}
    sg = plan_same_gender_portraits("mm", roles, True, "auto")
    assert sg is not None
    base, partner, make_duet = sg
    assert base == "male"
    assert partner == "male2"
    assert make_duet is True


def test_plan_same_gender_portraits_ff_without_duet_role_skips_partner():
    """Pure solo sections, no duet — partner+duet shouldn't render."""
    roles = {"female"}
    sg = plan_same_gender_portraits("ff", roles, True, "auto")
    # Implementation may still return a plan but with make_duet=False; the
    # contract is that no partner portrait+duet portrait gets rendered.
    if sg is not None:
        _, _, make_duet = sg
        assert make_duet is False
