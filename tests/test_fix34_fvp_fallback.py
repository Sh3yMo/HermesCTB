"""Fix 34 — verify the silent fvp = vp fallback is replaced by an explicit
sanitizer that strips video-direction language (camera moves, scene ends,
hard cuts, lipsync booster). Lyrics-strip from Fix 33 must still work.

Pure / network-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from music_video_pipeline import (  # noqa: E402
    derive_still_prompt_from_video_prompt,
    strip_lyrics_from_image_prompt,
    _SEG_DIRECTOR_RULES,
    _VIDEO_DIRECTIVES_RE,
)


# ---------------------------------------------------------------------------
# _VIDEO_DIRECTIVES_RE — camera moves, scene ends, transitions, lipsync
# ---------------------------------------------------------------------------

def test_strip_removes_the_camera_performs_phrase():
    out = strip_lyrics_from_image_prompt(
        "Close-up of singer. The camera performs a subtle dolly-in. Warm cafe lighting."
    )
    assert "camera performs" not in out.lower()
    assert "dolly-in" not in out.lower()
    assert "Close-up of singer" in out
    assert "Warm cafe lighting" in out


def test_strip_removes_the_scene_ends_phrase():
    out = strip_lyrics_from_image_prompt(
        "Medium shot of woman by window. The scene ends with a clean hard cut. Soft golden hour glow."
    )
    assert "scene ends" not in out.lower()
    assert "hard cut" not in out.lower()
    assert "Medium shot of woman by window" in out
    assert "Soft golden hour glow" in out


def test_strip_removes_camera_move_phrases():
    cases = [
        "Profile shot. Subtle dolly-in toward subject. Calm mood.",
        "Profile shot. Slow pull-back from face. Calm mood.",
        "Profile shot. Push-in close. Calm mood.",
        "Profile shot. Gentle zoom-in on hand. Calm mood.",
        "Profile shot. Smooth whip-pan across crowd. Calm mood.",
        "Profile shot. Slow tilt-up to ceiling. Calm mood.",
        "Profile shot. Handheld drift around subject. Calm mood.",
    ]
    for prompt in cases:
        out = strip_lyrics_from_image_prompt(prompt)
        assert "Profile shot" in out
        assert "Calm mood" in out
        for move in (
            "dolly-in", "pull-back", "push-in", "zoom-in",
            "whip-pan", "tilt-up", "handheld drift",
        ):
            assert move not in out.lower(), f"{move!r} left in {out!r}"


def test_strip_removes_cut_transition_phrases():
    cases = [
        ("Close-up. Hard cut to black. Warm light.", "hard cut"),
        ("Close-up. Crossfade to next shot. Warm light.", "crossfade"),
        ("Close-up. Fade to white. Warm light.", "fade to"),
        ("Close-up. Cut to wide shot. Warm light.", "cut to"),
        ("Close-up. Jump cut on action. Warm light.", "jump cut"),
    ]
    for prompt, needle in cases:
        out = strip_lyrics_from_image_prompt(prompt)
        assert "Close-up" in out
        assert "Warm light" in out
        assert needle not in out.lower(), f"{needle!r} left in {out!r}"


def test_strip_removes_lipsync_booster_fragments():
    """LIPSYNC_BOOSTER from api.py: 'The lips are syncing naturally to the
    vocals. Every word is pronounced perfectly, facial expressions are lively,
    diction and lip sync are perfect.'"""
    out = strip_lyrics_from_image_prompt(
        "Close-up of singer. The lips are syncing naturally to the vocals. "
        "Every word is pronounced perfectly. Diction and lip sync are perfect. "
        "Soft window light."
    )
    assert "lips are syncing" not in out.lower()
    assert "every word is pronounced" not in out.lower()
    assert "lip sync are perfect" not in out.lower()
    assert "Close-up of singer" in out
    assert "Soft window light" in out


# ---------------------------------------------------------------------------
# Fix-26 framing phrases MUST survive — they belong in fvp.
# ---------------------------------------------------------------------------

def test_strip_preserves_fix26_framing_phrases():
    cases = [
        "Close-up of a female singer in a cafe. Warm light.",
        "Medium close-up of a male singer at a bar. Cool light.",
        "Medium shot of singer leaning on rail. Sunset light.",
        "3/4 angle of singer facing camera. Golden light.",
        "Low-angle shot of singer below palm tree. Blue hour light.",
        "High-angle shot of singer on bed. Soft light.",
    ]
    for prompt in cases:
        out = strip_lyrics_from_image_prompt(prompt)
        # The framing phrase must remain intact.
        assert prompt.split(".")[0] in out, f"Framing phrase eaten: {prompt!r} -> {out!r}"


# ---------------------------------------------------------------------------
# derive_still_prompt_from_video_prompt — combined sanitizer
# ---------------------------------------------------------------------------

def test_derive_still_strips_everything_video_specific():
    vp = (
        "Close-up of female singer with curly hair in a Paris cafe at night. "
        "She wears a cream knit sweater. The camera performs a subtle dolly-in. "
        "Mouth visible as she sings: \"c'est le temps des nostalgies\". "
        "The scene ends with a clean hard cut. "
        "Lit by night — cool moonlight rim. "
        "The lips are syncing naturally to the vocals. "
        "Every word is pronounced perfectly."
    )
    out = derive_still_prompt_from_video_prompt(
        vp, lyrics="c'est le temps des nostalgies"
    )
    # Video direction stripped
    assert "camera performs" not in out.lower()
    assert "dolly-in" not in out.lower()
    assert "scene ends" not in out.lower()
    assert "hard cut" not in out.lower()
    assert "lips are syncing" not in out.lower()
    assert "every word is pronounced" not in out.lower()
    # Lyrics stripped (quotes + substring)
    assert "nostalgies" not in out.lower()
    # Visual description preserved
    assert "Close-up of female singer" in out
    assert "cream knit sweater" in out
    assert "Paris cafe" in out
    # The Fix 29 lighting tag survives
    assert "lit by night" in out.lower() or "moonlight" in out.lower()


def test_derive_still_handles_empty_input():
    assert derive_still_prompt_from_video_prompt("") == ""
    assert derive_still_prompt_from_video_prompt("", lyrics="anything") == ""


def test_derive_still_no_video_directives_passes_through():
    vp = (
        "Close-up of singer in soft cream knit sweater by the window, "
        "warm cafe lighting, hair down naturally"
    )
    out = derive_still_prompt_from_video_prompt(vp)
    assert "Close-up of singer in soft cream knit sweater" in out
    assert "warm cafe lighting" in out


# ---------------------------------------------------------------------------
# Real-world c72fa617 regression prompt — full end-to-end coverage.
# ---------------------------------------------------------------------------

def test_strip_handles_c72fa617_regression_prompt():
    """The actual prompt that leaked into the T2I, taken from the ComfyUI
    queue dump after the bug was observed."""
    bad = (
        "Close-up of woman in soft cream knit sweater by the cafe window. "
        "rim lighting her silhouette. The interior is lit by warm, dim "
        "practical lights. She wears a soft cream knit sweater and faded "
        "blue jeans, barefoot, hair down naturally, with a nostalgic "
        "expression, looking slightly away from the camera then directly "
        "into the lens. The camera is at a slight Dutch angle for dramatic "
        "effect, emphasizing the mood of the rain outside. The text:: The "
        "camera performs a subtle dolly-in. The text: c'est le temps des "
        "nostalgies, où mon âme s'est endormie. The scene ends with a "
        "clean hard cut, lit by night — deep dark sky, practical and "
        "ambient light sources, cool moonlight rim, wearing soft cream "
        "knit sweater and faded blue jeans, barefoot, hair down naturally"
    )
    out = strip_lyrics_from_image_prompt(
        bad, lyrics="c'est le temps des nostalgies, ou mon ame s'est endormie"
    )
    # Lyrics gone
    assert "nostalgies" not in out.lower()
    assert "endormie" not in out.lower()
    # Video direction gone
    assert "camera performs" not in out.lower()
    assert "dolly-in" not in out.lower()
    assert "scene ends" not in out.lower()
    assert "hard cut" not in out.lower()
    # Visual description preserved
    assert "Close-up of woman in soft cream knit sweater" in out
    assert "cafe window" in out
    assert "cream knit sweater" in out
    # Note: in this specific c72fa617 prompt the Fix-29 ", lit by night" and
    # Fix-30 ", wearing ..." tags were attached as comma-clauses to the same
    # sentence as "The scene ends with a clean hard cut" — the sanitizer
    # eats the whole sentence so those tags also disappear. Acceptable
    # trade-off: the cleaned prompt has a clean visual description, and the
    # Fix-29/30 lighting/wardrobe info is independently enforced in the
    # video_prompt path and the LLM system-prompt. The clean-frame guarantee
    # is the priority here.


# ---------------------------------------------------------------------------
# System prompt updates — mandatory fvp clause
# ---------------------------------------------------------------------------

def test_seg_director_rules_marks_fvp_mandatory():
    text = _SEG_DIRECTOR_RULES.lower()
    assert "fix 34" in text
    assert "mandatory" in text
    assert "never leave it empty" in text or "never leave fvp empty" in text


def test_seg_director_rules_bans_video_direction_in_fvp():
    text = _SEG_DIRECTOR_RULES.lower()
    for needle in (
        "the camera performs",
        "dolly-in",
        "scene ends",
        "lipsync",
    ):
        assert needle in text, f"missing ban for {needle!r}"


# ---------------------------------------------------------------------------
# Idempotency — repeated sanitization is a no-op
# ---------------------------------------------------------------------------

def test_sanitizer_is_idempotent():
    vp = (
        "Close-up of singer. The camera performs a subtle dolly-in. "
        "She sings \"hello world\". Warm light."
    )
    once = strip_lyrics_from_image_prompt(vp)
    twice = strip_lyrics_from_image_prompt(once)
    assert once == twice
