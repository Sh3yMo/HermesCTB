"""Tests for Fix 13: vocal gender caption enhancement in AudioEnhancer."""
import os
import sys
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio_enhancer import AudioEnhancer, AudioSettings


def _make_enhancer():
    config = {
        "openrouter_api_key": "test",
        "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
    }
    return AudioEnhancer(config)


# ─── _build_vocal_role_tags ──────────────────────────────────────────────────

def test_vocal_role_tags_mixed():
    enh = _make_enhancer()
    lyrics = "[Verse - male]\nline\n\n[Chorus - female]\nline\n\n[Bridge - duet]\nline"
    result = enh._build_vocal_role_tags(lyrics)
    assert "male vocals in verse" in result
    assert "female vocals in chorus" in result
    assert "harmonizing simultaneously in bridge" in result
    assert "both voices layered together in bridge" in result
    assert "male and female vocal harmony" in result


def test_vocal_role_tags_no_gender():
    enh = _make_enhancer()
    lyrics = "[Verse 1]\nline\n\n[Chorus]\nline\n\n[Bridge]\nline"
    result = enh._build_vocal_role_tags(lyrics)
    assert result == ""


def test_vocal_role_tags_texture_only():
    """Non-gender modifiers (raspy, anthemic) must not produce a role description."""
    enh = _make_enhancer()
    lyrics = "[Verse - raspy]\nline\n\n[Chorus - anthemic]\nline"
    result = enh._build_vocal_role_tags(lyrics)
    assert result == ""


def test_vocal_role_tags_deduplicates_section():
    """Same section type appearing multiple times (Verse 1, Verse 2) → one entry."""
    enh = _make_enhancer()
    lyrics = "[Verse 1 - male]\nline\n\n[Verse 2 - male]\nline\n\n[Chorus - female]\nline"
    result = enh._build_vocal_role_tags(lyrics)
    assert result.count("male vocals in verse") == 1


# ─── inject_audio_settings ───────────────────────────────────────────────────

def _workflow_with_node_94():
    return {
        "94": {"inputs": {"tags": "", "lyrics": "", "bpm": 120, "keyscale": "C major",
                          "timesignature": "4", "language": "en", "duration": 60}},
        "98": {"inputs": {"seconds": 60}},
    }


def test_inject_prepends_vocal_roles():
    enh = _make_enhancer()
    settings = AudioSettings()
    settings.genre = "pop"
    settings.caption = "pop, modern pop, bright guitars"
    settings.voice = "any"
    settings.bpm = 120
    settings.key = "C major"
    settings.language = "en"
    settings.duration = 60
    settings.structure = ["Verse 1 - male", "Chorus - female", "Bridge - duet"]
    settings.lyrics = {
        "Verse 1 - male": "line",
        "Chorus - female": "LINE",
        "Bridge - duet": "both",
    }
    wf = _workflow_with_node_94()
    result = enh.inject_audio_settings(wf, settings)
    tags = result["94"]["inputs"]["tags"]
    assert tags.startswith("male vocals in verse")
    assert "female vocals in chorus" in tags
    assert "harmonizing simultaneously in bridge" in tags
    assert "male and female vocal harmony" in tags
    assert "pop" in tags


def test_inject_single_voice_no_prepend():
    """voice='male' with no gender tags in lyrics → no vocal role prefix."""
    enh = _make_enhancer()
    settings = AudioSettings()
    settings.genre = "rock"
    settings.caption = "rock, powerful male vocal"
    settings.voice = "male"
    settings.bpm = 130
    settings.key = "E minor"
    settings.language = "en"
    settings.duration = 60
    settings.structure = ["Verse 1", "Chorus"]
    settings.lyrics = {"Verse 1": "line", "Chorus": "LINE"}
    wf = _workflow_with_node_94()
    result = enh.inject_audio_settings(wf, settings)
    tags = result["94"]["inputs"]["tags"]
    assert not tags.startswith("male vocals in")
    assert "rock" in tags


# ─── LLM prompt wording ──────────────────────────────────────────────────────

def test_build_vocal_prompt_any_section_specific():
    """voice='any' voice_rule must demand section-specific description in caption."""
    enh = _make_enhancer()
    settings = AudioSettings()
    settings.voice = "any"
    prompt = enh._build_vocal_prompt("summer pop song", "genre: pop", settings)
    # Must instruct the LLM to use section-specific vocal role descriptions
    assert "section" in prompt.lower() or ("in verse" in prompt.lower() and "in chorus" in prompt.lower())
    # Must explicitly forbid the generic phrasing
    assert "never just" in prompt.lower() or "always specify" in prompt.lower()
