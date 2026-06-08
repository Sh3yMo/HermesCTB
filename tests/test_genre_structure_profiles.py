"""Genre → structure profile mapping + prompt injection + parser round-trip."""
import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_enhancer import (
    AudioEnhancer, AudioSettings,
    _default_structure_profile, VALID_STRUCTURE_PROFILES,
)


def _make_enhancer():
    return AudioEnhancer({
        "openrouter_api_key": "test",
        "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
    })


@pytest.mark.parametrize("genre,expected", [
    ("synthwave", "edm"),
    ("Big Room EDM", "edm"),
    ("future bass", "edm"),
    ("techno", "edm"),
    ("trance", "edm"),
    ("trap", "hiphop"),
    ("drill", "hiphop"),
    ("Hip-Hop", "hiphop"),
    ("boom bap", "hiphop"),
    ("acoustic", "ballad"),
    ("piano ballad", "ballad"),
    ("singer-songwriter", "ballad"),
    ("folk", "ballad"),
    # Default fallback.
    ("pop", "pop"),
    ("indie rock", "pop"),
    ("contemporary r&b", "pop"),
    ("", "pop"),
    (None, "pop"),
    ("klezmer", "pop"),  # unknown → safe pop default
])
def test_default_structure_profile_mapping(genre, expected):
    assert _default_structure_profile(genre) == expected


def test_valid_structure_profiles_set():
    assert VALID_STRUCTURE_PROFILES == {"edm", "pop", "hiphop", "ballad"}


@pytest.mark.parametrize("genre,token", [
    ("synthwave", "edm"),
    ("trap", "hiphop"),
    ("piano ballad", "ballad"),
    ("indie pop", "pop"),
])
def test_prompt_injects_recommended_profile(genre, token):
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=90,
        genre=genre, language="en",
    )
    prompt = enh._build_vocal_prompt(
        "an idea", f"Genre: {genre}\nDuration: 90 seconds\nLanguage: en\nType: vocal", settings,
    )
    assert "STRUCTURE PROFILES BY GENRE" in prompt
    assert f"RECOMMENDED PROFILE for this song (chosen from genre): {token}" in prompt
    # All four profile labels must be listed so the LLM can override.
    for profile in ("edm", "pop", "hiphop", "ballad"):
        assert f"• {profile}" in prompt


def test_parser_normalizes_structure_profile_to_known_value():
    enh = _make_enhancer()
    settings = AudioSettings(type="vocal", voice="any", duration=120, genre="trap")
    response = (
        '{"caption": "trap heat", '
        '"lyrics": "[Intro]\\n\\n[Verse 1]\\nbars\\n\\n[Chorus]\\nhook", '
        '"bpm": 140, "key": "A minor", "time_signature": "4", '
        '"duration": 120, "genre": "trap", '
        '"artist": "Skyline Kid", "title": "Late Run", '
        '"structure_profile": "HIPHOP"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.structure_profile == "hiphop"


def test_parser_falls_back_to_default_profile_when_llm_omits():
    enh = _make_enhancer()
    settings = AudioSettings(type="vocal", voice="any", duration=90, genre="house")
    response = (
        '{"caption": "house groove", '
        '"lyrics": "[Intro]\\n\\n[Verse 1]\\nline", '
        '"bpm": 124, "key": "C major", "time_signature": "4", '
        '"duration": 90, "genre": "house", '
        '"artist": "Beat Atlas", "title": "Move"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.structure_profile == "edm"


def test_parser_rejects_unknown_profile_and_uses_genre_default():
    enh = _make_enhancer()
    settings = AudioSettings(type="vocal", voice="any", duration=90, genre="folk")
    response = (
        '{"caption": "folk", "lyrics": "[Intro]\\n\\n[Verse 1]\\nl", '
        '"bpm": 90, "key": "G major", "time_signature": "4", '
        '"duration": 90, "genre": "folk", '
        '"artist": "Wren Hollow", "title": "Riverbed", '
        '"structure_profile": "polka"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.structure_profile == "ballad"
