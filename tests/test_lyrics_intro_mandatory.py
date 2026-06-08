"""Mandatory [Intro] rule: every duration must yield [Intro] as the first tag.

Covers the prompt-side contract added to AudioEnhancer._build_vocal_prompt: the
system prompt forces `[Intro]` as the opening tag at every supported duration
(20s, 30s, 45s, 60s, 90s, 120s+). The parser-side assertion runs against a
mocked LLM response — we check that the parser preserves the intro position.
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


@pytest.mark.parametrize("duration", [20, 30, 45, 60, 90, 120, 180])
def test_prompt_contains_mandatory_intro_rule(duration):
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=duration,
        genre="synthwave", language="en",
    )
    prompt = enh._build_vocal_prompt("a song about midnight", "Duration: %ds\nLanguage: en\nType: vocal" % duration, settings)
    assert "MANDATORY INTRO RULE" in prompt
    assert "[Intro]" in prompt
    assert "Never start with [Verse 1]" in prompt


@pytest.mark.parametrize("duration,lyrics", [
    (20, "[Intro]\n\n[Chorus]\nshort hook line"),
    (30, "[Intro]\n\n[Verse 1]\nverse line"),
    (45, "[Intro]\n\n[Verse 1]\nv\n\n[Chorus]\nc"),
    (60, "[Intro]\n\n[Verse 1]\nv\n\n[Chorus]\nc\n\n[Outro]\n"),
    (90, "[Intro]\n\n[Verse 1]\nv\n\n[Chorus]\nc\n\n[Verse 2]\nv2\n\n[Chorus]\nc\n\n[Outro]\n"),
])
def test_parsed_lyrics_keep_intro_first(duration, lyrics):
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=duration,
        genre="pop", language="en",
    )
    enh._parse_lyrics_into_settings(lyrics, settings)
    assert settings.structure, "structure should not be empty"
    first_tag = settings.structure[0]
    assert first_tag.startswith("Intro"), (
        f"first tag at {duration}s must be [Intro], got {first_tag!r}"
    )


def test_system_prompt_intro_clause_present():
    """The system prompt added to generate_song must contain the rule (6) clause."""
    enh = _make_enhancer()
    settings = AudioSettings(
        type="vocal", voice="any", duration=60,
        genre="trap", language="en",
    )
    enh._call_openrouter = AsyncMock(return_value=(
        '{"caption": "x", "lyrics": "[Intro]\\n\\n[Verse 1]\\nl", '
        '"bpm": 120, "key": "C major", "time_signature": "4", '
        '"duration": 60, "genre": "trap", "artist": "Test", "title": "Run", '
        '"structure_profile": "hiphop"}'
    ))
    import asyncio
    asyncio.run(enh._generate_song_impl(settings, "test idea"))
    # System prompt was forwarded as kwarg → grab the captured args.
    call_kwargs = enh._call_openrouter.await_args.kwargs
    system_prompt = call_kwargs["system_prompt"]
    assert "EVERY song MUST begin with [Intro]" in system_prompt
    assert "(6)" in system_prompt
