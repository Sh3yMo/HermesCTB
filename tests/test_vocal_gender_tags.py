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
    assert result == "male verse, female chorus, duet bridge"


def test_vocal_role_tags_swapped():
    """Generizität: reverse gender mapping (female verse, male chorus) must also work."""
    enh = _make_enhancer()
    lyrics = "[Verse - female]\nline\n\n[Chorus - male]\nline"
    result = enh._build_vocal_role_tags(lyrics)
    assert "female verse" in result
    assert "male chorus" in result


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
    assert result.count("male verse") == 1


# ─── inject_audio_settings ───────────────────────────────────────────────────

def _workflow_with_node_94():
    return {
        "3": {"inputs": {"cfg": 3, "steps": 100, "sampler_name": "dpmpp_sde"}},
        "94": {"inputs": {"tags": "", "lyrics": "", "bpm": 120, "keyscale": "C major",
                          "timesignature": "4", "language": "en", "duration": 60,
                          "cfg_scale": 4, "temperature": 0.85, "top_p": 0.9,
                          "top_k": 0, "min_p": 0}},
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
    assert tags.startswith("male verse")
    assert "female chorus" in tags
    assert "duet bridge" in tags
    assert "pop" in tags
    # Verbose phrasing must be gone
    assert "throughout" not in tags
    assert "consistent voice" not in tags


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


# ─── Fix 14: Encoder parameter injection ─────────────────────────────────────

def _make_enhancer_with_encoder_cfg(encoder_cfg):
    config = {
        "openrouter_api_key": "test",
        "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
        "ace_step_encoder": encoder_cfg,
    }
    return AudioEnhancer(config)


def _minimal_settings():
    s = AudioSettings()
    s.genre = "pop"
    s.caption = "pop"
    s.voice = "any"
    s.duration = 60
    s.structure = ["Verse 1 - male"]
    s.lyrics = {"Verse 1 - male": "line"}
    return s


def test_inject_sets_encoder_params_from_config():
    enh = _make_enhancer_with_encoder_cfg({
        "cfg_scale": 6, "temperature": 0.7, "top_p": 0.85,
        "top_k": 0, "min_p": 0, "ksampler_cfg": 5,
    })
    wf = _workflow_with_node_94()
    result = enh.inject_audio_settings(wf, _minimal_settings())
    assert result["94"]["inputs"]["cfg_scale"] == 6
    assert result["94"]["inputs"]["temperature"] == 0.7
    assert result["94"]["inputs"]["top_p"] == 0.85
    assert result["94"]["inputs"]["top_k"] == 0
    assert result["94"]["inputs"]["min_p"] == 0
    assert result["3"]["inputs"]["cfg"] == 5


def test_inject_uses_workflow_default_when_config_missing():
    enh = _make_enhancer()  # no ace_step_encoder section
    wf = _workflow_with_node_94()
    result = enh.inject_audio_settings(wf, _minimal_settings())
    # Workflow defaults must remain
    assert result["94"]["inputs"]["cfg_scale"] == 4
    assert result["94"]["inputs"]["temperature"] == 0.85
    assert result["94"]["inputs"]["top_p"] == 0.9
    assert result["3"]["inputs"]["cfg"] == 3


def test_inject_partial_encoder_config():
    """Only some keys set in config — others keep workflow default."""
    enh = _make_enhancer_with_encoder_cfg({"cfg_scale": 7})
    wf = _workflow_with_node_94()
    result = enh.inject_audio_settings(wf, _minimal_settings())
    assert result["94"]["inputs"]["cfg_scale"] == 7
    assert result["94"]["inputs"]["temperature"] == 0.85  # untouched
    assert result["3"]["inputs"]["cfg"] == 3  # untouched


# ─── LLM prompt wording ──────────────────────────────────────────────────────

def test_build_vocal_prompt_any_forbids_voice_in_caption():
    """voice='any' voice_rule must FORBID voice gender in caption (auto-injected)."""
    enh = _make_enhancer()
    settings = AudioSettings()
    settings.voice = "any"
    prompt = enh._build_vocal_prompt("summer pop song", "genre: pop", settings)
    # LLM still uses compound section tags
    assert "[verse - male]" in prompt.lower() or "[chorus - female]" in prompt.lower()
    # Caption rule must explicitly exclude voice/gender content
    lower = prompt.lower()
    assert "do not mention voice gender" in lower or "auto-injected" in lower
