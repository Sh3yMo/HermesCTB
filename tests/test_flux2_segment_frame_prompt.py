from api import _build_flux2_segment_frame_prompt


def test_flux2_segment_frame_prompt_locks_clothing_and_scene_integration():
    out = _build_flux2_segment_frame_prompt(
        "Singer stands in a neon alley.",
        "photorealistic cinematic still, teal and magenta palette",
    ).lower()

    assert "photorealistic cinematic still" in out
    assert "same exact garments" in out
    assert "do not change clothing color, cut, or style" in out
    assert "natural contact shadows" in out
    assert "physically inside the background" in out
    assert "not pasted" in out
    assert "no green screen" in out
