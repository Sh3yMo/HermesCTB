from api import _build_flux2_segment_frame_prompt, _build_flux2_last_frame_prompt


def test_flux2_segment_frame_prompt_is_flux_clean():
    out = _build_flux2_segment_frame_prompt(
        "Singer stands in a neon alley.",
        "photorealistic cinematic still, teal and magenta palette",
    ).lower()

    # scene + style preserved
    assert "neon alley" in out
    assert "photorealistic cinematic still" in out
    # identity / wardrobe lock + scene integration kept (positive phrasing)
    assert "exact garments" in out
    assert "contact shadows" in out
    assert "inside the scene" in out

    # LTX video-grading language must NOT leak into a Flux2 still
    assert "render this video frame" not in out
    assert "color grade" not in out
    assert "film grain" not in out
    # Flux2 has no negative prompts — no negative phrasing
    assert "no green screen" not in out
    assert "not pasted" not in out
    assert "do not change" not in out


def test_flux2_last_frame_prompt_no_colon_no_motion_language():
    out = _build_flux2_last_frame_prompt(
        "Singer stands in a neon alley.",
        "photorealistic cinematic still",
    ).lower()
    assert ":" not in out
    assert "end frame" not in out
    assert "render this video frame" not in out
    assert "same location" in out
