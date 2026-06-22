"""Unit tests for mv_prompt_hygiene helpers."""

from __future__ import annotations

import pytest

from mv_prompt_hygiene import (
    NEUTRAL_BG_HEX,
    NEUTRAL_BG_RGB,
    assert_no_duplicate_framing_phrases,
    beats_similarity_ratio,
    clean_beat_text,
    collapse_duplicate_beats,
    has_duplicate_framing_phrases,
    is_instructional_language,
    normalize_ltx_text,
    validate_wardrobe_contract,
    wardrobe_contract_to_compact_string,
)


# ---------- normalize_ltx_text (R2-4/RC4 deterministic safety net) ----------

def test_normalize_ltx_text_removes_colons():
    out = normalize_ltx_text("Male performer wearing onyx trench coat: belted, no sunglasses")
    assert ":" not in out
    # content words preserved
    assert "onyx trench coat" in out
    assert "belted" in out


def test_normalize_ltx_text_collapses_newlines_to_single_line():
    out = normalize_ltx_text("rain falls on the street.\nthe singer looks up.\n\nneon glows")
    assert "\n" not in out
    assert "rain falls on the street" in out
    assert "neon glows" in out


def test_normalize_ltx_text_tidies_punctuation_without_dropping_words():
    out = normalize_ltx_text("a man stands ,  rain drips : cold wind , , blows")
    assert ":" not in out
    assert ", ," not in out
    assert "  " not in out
    for w in ("man", "stands", "rain", "drips", "cold", "wind", "blows"):
        assert w in out


def test_normalize_ltx_text_empty():
    assert normalize_ltx_text("") == ""
    assert normalize_ltx_text(None) == ""


# ---------- clean_beat_text ----------

def test_clean_beat_strips_dangling_connector_as():
    assert clean_beat_text("She raises her hand as") == "She raises her hand"


def test_clean_beat_strips_dangling_connector_while():
    assert clean_beat_text("He sings the line while") == "He sings the line"


def test_clean_beat_drops_empty_result():
    assert clean_beat_text("") is None
    assert clean_beat_text("   ") is None
    assert clean_beat_text(None) is None


def test_clean_beat_drops_short_result():
    assert clean_beat_text("Hi") is None


def test_clean_beat_keeps_complete_sentence():
    assert (
        clean_beat_text("She stands at the platform, eyes closed.")
        == "She stands at the platform, eyes closed."
    )


def test_clean_beat_handles_multiple_sentences_with_dangling_last():
    result = clean_beat_text("She stands still. Her lips tremble as")
    assert result == "She stands still. Her lips tremble"


def test_clean_beat_drops_pure_connector():
    assert clean_beat_text("as") is None


def test_clean_beat_add_trailing_period_opt_in():
    assert clean_beat_text("She stands", add_trailing_period=True) == "She stands."


# ---------- beats_similarity_ratio ----------

def test_similarity_identical_is_one():
    assert beats_similarity_ratio("hello world", "hello world") == 1.0


def test_similarity_disjoint_is_low():
    assert beats_similarity_ratio("hello world", "completely different text") < 0.5


def test_similarity_near_duplicate_intro_beats():
    a = (
        "Wide shot of an old abandoned train station waiting hall in morning "
        "golden hour, warm low sun, long soft shadows."
    )
    b = (
        "Wide establishing shot of a surreal ritual tableau inside an "
        "abandoned train station, morning golden hour lighting, long soft shadows."
    )
    assert beats_similarity_ratio(a, b) > 0.5


def test_similarity_handles_empty():
    assert beats_similarity_ratio("", "anything") == 0.0
    assert beats_similarity_ratio("anything", "") == 0.0


# ---------- collapse_duplicate_beats ----------

def test_collapse_drops_duplicates():
    a = "She stands on the platform, eyes closed."
    b = "She stands on the platform, eyes closed."
    assert collapse_duplicate_beats([a, b]) == [a]


def test_collapse_keeps_distinct_beats():
    a = "She stands on the platform."
    b = "Sun sets behind the tracks."
    assert collapse_duplicate_beats([a, b]) == [a, b]


def test_collapse_skips_empty():
    assert collapse_duplicate_beats(["", "real beat here"]) == ["real beat here"]


# ---------- is_instructional_language ----------

def test_instructional_language_detects_use_the():
    assert is_instructional_language("Use the scene reference as the real location")


def test_instructional_language_detects_should():
    assert is_instructional_language("The performer should look physically present")


def test_instructional_language_detects_must():
    assert is_instructional_language("Scene location must be exactly: X")


def test_visual_prose_is_not_instructional():
    prose = (
        "Old abandoned train station at golden hour, warm low sun, long soft "
        "shadows, dust motes in beams of light, photoreal cinematic still."
    )
    assert not is_instructional_language(prose)


# ---------- assert_no_duplicate_framing_phrases ----------

def test_assert_passes_when_no_dupes():
    assert_no_duplicate_framing_phrases(
        "Female vocalist, fair skin, long dark hair, standing, head to toe visible."
    )


def test_assert_raises_when_standing_duplicated():
    with pytest.raises(ValueError):
        assert_no_duplicate_framing_phrases(
            "Female vocalist standing, head to toe visible, standing, facing the camera."
        )


def test_has_duplicate_framing_phrases_helper():
    assert has_duplicate_framing_phrases("full body shot, full body shot")
    assert not has_duplicate_framing_phrases("full body shot, fair skin")


# ---------- wardrobe ----------

def test_compact_contract_has_no_or_disjunction():
    contract = {
        "top": {"garment": "blouse", "color": "cream", "fit": "fitted"},
        "bottom": {"garment": "denim shorts", "color": "indigo"},
        "footwear": {"garment": "sneakers", "color": "white"},
        "accessories": ["silver hoop earrings"],
    }
    out = wardrobe_contract_to_compact_string(contract)
    assert " or " not in out.lower()
    assert "cream" in out
    assert "indigo" in out
    assert "white" in out
    assert "silver hoop earrings" in out


def test_compact_contract_skips_missing_fields():
    contract = {"top": {"garment": "tee", "color": "black"}}
    out = wardrobe_contract_to_compact_string(contract)
    assert "black" in out
    assert "tee" in out


def test_validate_wardrobe_requires_colors():
    ok, errs = validate_wardrobe_contract(
        {
            "top": {"garment": "blouse", "color": "cream"},
            "bottom": {"garment": "denim shorts", "color": "indigo"},
        }
    )
    assert ok, errs


def test_validate_wardrobe_fails_on_missing_color():
    ok, errs = validate_wardrobe_contract(
        {
            "top": {"garment": "blouse"},
            "bottom": {"garment": "denim shorts", "color": "indigo"},
        }
    )
    assert not ok
    assert any("top.color" in e for e in errs)


# ---------- shared constants ----------

def test_neutral_bg_hex_and_rgb_match():
    assert NEUTRAL_BG_HEX == "#808080"
    assert NEUTRAL_BG_RGB == (128, 128, 128)


# ---------- api._dedupe_portrait_framing ----------

def test_dedupe_portrait_framing_removes_doubles():
    from api import _dedupe_portrait_framing
    src = (
        "Female vocalist, hourglass build, standing, head to toe visible, "
        "facing the camera directly, full body, "
        "full body shot, standing, facing the camera directly, head to toe "
        "visible, entire figure including footwear in frame"
    )
    out = _dedupe_portrait_framing(src)
    # Each framing token appears exactly once (the suffix preserved at the end).
    for tok in ("standing", "head to toe", "facing the camera", "full body"):
        assert out.lower().count(tok) == 1, (tok, out)


def test_dedupe_portrait_framing_passes_through_no_dupes():
    from api import _dedupe_portrait_framing
    src = "Female vocalist, fair skin, full body shot, standing"
    assert _dedupe_portrait_framing(src) == src


# ---------- character sheet 16:9 padding ----------

def test_compose_sheet_three_cells_native_16_9_no_padding(tmp_path):
    # Stage R (RC5/RC7): 3 cells of 16:27 tile horizontally to exactly 16:9
    # with NO grey padding (the grey field used to bleed into the video bg).
    from PIL import Image
    from msr_refs import compose_character_sheet, NEUTRAL_BG_RGB

    src = tmp_path / "view.png"
    Image.new("RGB", (1024, 1728), (50, 50, 50)).save(src)
    out = tmp_path / "sheet.png"
    compose_character_sheet([str(src)] * 3, str(out))  # default cells 1024x1728
    img = Image.open(out).convert("RGB")
    w, h = img.size
    assert (w, h) == (3072, 1728)
    assert w % 32 == 0 and h % 32 == 0
    assert abs(w / h - 16 / 9) < 1e-6
    # No padding: corners are view content, NOT the neutral pad grey.
    assert img.getpixel((0, 0)) != NEUTRAL_BG_RGB
    assert img.getpixel((w - 1, h - 1)) != NEUTRAL_BG_RGB


def test_compose_sheet_horizontal_row_no_padding(tmp_path):
    # Stage R: target_aspect=None lays cells out in a single horizontal row.
    from PIL import Image
    from msr_refs import compose_character_sheet

    src = tmp_path / "view.png"
    Image.new("RGB", (512, 864), (50, 50, 50)).save(src)
    out = tmp_path / "sheet.png"
    compose_character_sheet([str(src)] * 3, str(out),
                            cell_w=512, cell_h=864, target_aspect=None)
    img = Image.open(out).convert("RGB")
    assert img.size == (3 * 512, 864)


# ---------- inject_msr_resolution ----------

def test_inject_msr_resolution_sets_scalar_width_height():
    from comfyui import inject_msr_resolution
    wf = {
        "2006": {
            "class_type": "LiconMSR",
            "inputs": {
                "1": ["2001", 0],
                "width": ["759:1080", 1],
                "height": ["759:1081", 1],
                "frame_count": 41,
            },
        }
    }
    inject_msr_resolution(wf, 1024, 576)
    assert wf["2006"]["inputs"]["width"] == 1024
    assert wf["2006"]["inputs"]["height"] == 576


def test_inject_msr_resolution_noop_on_standard_workflow():
    from comfyui import inject_msr_resolution
    wf = {
        "1": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": 768, "height": 768},
        }
    }
    inject_msr_resolution(wf, 1024, 576)
    assert wf["1"]["inputs"]["width"] == 768


# ---------- wardrobe contract async helper ----------

def test_build_role_clothing_async_uses_llm_when_present():
    import asyncio
    from music_video_pipeline import build_role_clothing_contracts_async

    class FakePrompter:
        async def generate_wardrobe_contract(self, theme, genre, role):
            if role == "female":
                return {
                    "top": {"garment": "blouse", "color": "cream", "fit": "fitted"},
                    "bottom": {"garment": "denim shorts", "color": "indigo"},
                    "footwear": {"garment": "sneakers", "color": "white"},
                    "accessories": [],
                }
            return {
                "top": {"garment": "shirt", "color": "burgundy", "fit": "loose"},
                "bottom": {"garment": "trousers", "color": "tan"},
                "footwear": {"garment": "sandals", "color": "brown"},
                "accessories": [],
            }

    out = asyncio.run(
        build_role_clothing_contracts_async(FakePrompter(), "love song", "pop")
    )
    assert "cream" in out["female"]
    assert "indigo" in out["female"]
    assert " or " not in out["female"]
    assert "burgundy" in out["male"]


def test_build_role_clothing_async_falls_back_on_invalid_llm_result():
    import asyncio
    from music_video_pipeline import build_role_clothing_contracts_async

    class BadPrompter:
        async def generate_wardrobe_contract(self, theme, genre, role):
            return {"top": {"garment": "shirt"}}  # missing color

    out = asyncio.run(
        build_role_clothing_contracts_async(BadPrompter(), "love song", "pop")
    )
    # Falls back to the regex extractor → at minimum a non-empty string per role.
    assert out["female"]
    assert out["male"]


def test_build_role_clothing_async_no_prompter_uses_regex():
    import asyncio
    from music_video_pipeline import build_role_clothing_contracts_async

    out = asyncio.run(
        build_role_clothing_contracts_async(None, "love song", "pop")
    )
    assert out["female"]
    assert out["male"]
