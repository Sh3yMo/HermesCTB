"""Artist/title threading: AudioSettings → _generate_song → file rename.

Verifies:
1. AudioSettings carries artist/title/structure_profile fields with sane
   defaults and round-trips through to_dict/from_dict.
2. _build_vocal_prompt embeds the explicit-artist clause when the user
   supplied one (override mode) vs the invent-a-name clause when not.
3. _parse_generation_response honours the user-supplied override (user
   value wins over LLM value).
"""
import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_enhancer import AudioEnhancer, AudioSettings


def _make_enhancer():
    return AudioEnhancer({
        "openrouter_api_key": "test",
        "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
    })


def test_audio_settings_defaults():
    s = AudioSettings()
    assert s.artist == ""
    assert s.title == ""
    assert s.structure_profile == ""


def test_audio_settings_roundtrip_preserves_new_fields():
    s = AudioSettings(artist="Neon Vipers", title="Midnight Run",
                      structure_profile="edm", genre="synthwave")
    d = s.to_dict()
    assert d["artist"] == "Neon Vipers"
    assert d["title"] == "Midnight Run"
    assert d["structure_profile"] == "edm"
    restored = AudioSettings.from_dict(d)
    assert restored.artist == "Neon Vipers"
    assert restored.title == "Midnight Run"
    assert restored.structure_profile == "edm"


def test_prompt_includes_artist_override_when_user_supplied():
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=120,
        genre="pop", language="en",
        artist="My Custom Band",
    )
    prompt = enh._build_vocal_prompt(
        "a song", "Genre: pop\nDuration: 120 seconds\nLanguage: en\nType: vocal", settings,
    )
    assert "ARTIST NAME OVERRIDE" in prompt
    assert 'You MUST emit "artist": "My Custom Band"' in prompt
    assert "do not invent a new one" in prompt


def test_prompt_invents_artist_when_no_user_value():
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=120,
        genre="pop", language="en",
    )
    prompt = enh._build_vocal_prompt(
        "a song", "Genre: pop\nDuration: 120 seconds\nLanguage: en\nType: vocal", settings,
    )
    assert "ARTIST NAME OVERRIDE" not in prompt
    assert "invent a creative, genre-fitting stage / band name" in prompt


def test_prompt_always_includes_title_rule():
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=60,
        genre="trap", language="en",
        artist="Fixed Name",
    )
    prompt = enh._build_vocal_prompt(
        "x", "Duration: 60 seconds\nLanguage: en\nType: vocal", settings,
    )
    assert "title: 1-5 words" in prompt
    assert "ASCII letters/spaces/apostrophe/hyphen only" in prompt


def test_parser_takes_llm_artist_when_user_value_empty():
    enh = _make_enhancer()
    settings = AudioSettings(type="vocal", voice="any", duration=120, genre="pop")
    response = (
        '{"caption": "x", "lyrics": "[Intro]\\n\\n[Verse 1]\\nline", '
        '"bpm": 120, "key": "C major", "time_signature": "4", '
        '"duration": 120, "genre": "pop", '
        '"artist": "Skyline Echo", "title": "Stay Alive"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.artist == "Skyline Echo"
    assert settings.title == "Stay Alive"


def test_parser_preserves_user_artist_override():
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=120, genre="pop",
        artist="User Picked Name",
    )
    # LLM tries to suggest a different artist — must be ignored.
    response = (
        '{"caption": "x", "lyrics": "[Intro]\\n\\n[Verse 1]\\nline", '
        '"bpm": 120, "key": "C major", "time_signature": "4", '
        '"duration": 120, "genre": "pop", '
        '"artist": "LLM Invented", "title": "Their Title"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.artist == "User Picked Name"
    # Title from LLM still wins (always derived from hook).
    assert settings.title == "Their Title"


def test_parser_title_falls_back_to_empty_when_missing():
    enh = _make_enhancer()
    settings = AudioSettings(type="vocal", voice="any", duration=60, genre="pop")
    response = (
        '{"caption": "x", "lyrics": "[Intro]\\n\\n[Verse 1]\\nl", '
        '"bpm": 120, "key": "C major", "time_signature": "4", '
        '"duration": 60, "genre": "pop"}'
    )
    enh._parse_generation_response(response, settings)
    assert settings.title == ""
    assert settings.artist == ""
