#!/usr/bin/env python3
# llm_tester.py
"""
LLM Comparison Tool for Short Film Pipeline agents.
Runs Director / VideoScript / Prompt Agent with different LLMs in parallel
and displays results side-by-side in a browser UI.

Usage:
    py llm_tester.py
    -> opens http://localhost:7799 automatically
"""

import asyncio
import json
import logging
import os
import re
import sys
import time
import traceback
import webbrowser

# Force UTF-8 on Windows stdout/stderr so Unicode chars (→ ← etc.) don't crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")
from pathlib import Path
from threading import Thread
from typing import Optional

import httpx
from flask import Flask, jsonify, request, send_from_directory

# ── Setup ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Add CTB root to path so we can import pipeline modules
CTB_ROOT = Path(__file__).parent
sys.path.insert(0, str(CTB_ROOT))

from config_loader import load_config
from director_agent import DirectorAgent, SceneSegment
from short_film_pipeline import (
    ScreenplayAgent, VideoScriptAgent, LTXPromptAgent,
    _extract_appearance_features,
)
from music_video_pipeline import MusicVideoPrompter, Segment

CONFIG = load_config()
UI_DIR = CTB_ROOT / "llm_tester_ui"
RESULTS_DIR = CTB_ROOT / "LLM Test Results"

# ── Dimension weights for recommendation engine ────────────────────────────
# Weight 2.0 = pipeline-kritisch (Fehler bricht downstream-Schritt)
# Weight 1.8 = strukturfundamental (Basis des gesamten Outputs)
# Weight 1.5 = qualitätswichtig
# Weight 1.2 = gut-zu-haben
# Weight 1.0 = nice-to-have
DIMENSION_WEIGHTS: dict[str, float] = {
    # Director — format_adherence kritisch: Pipeline parst JSON direkt
    "format_adherence":    2.0,
    "story_arc":           1.8,  # Ohne Struktur kein kohärenter Film
    "frame_actions":       1.5,  # Direkte Bildgenerierungs-Eingabe
    "scene_quality":       1.2,
    "env_consistency":     1.0,
    # VideoScript — trigger_injection kritisch: LoRA-Tags fehlen -> falsche Generierung
    "trigger_injection":   2.0,
    "face_handling":       1.8,  # Falsche Gesichtsreferenz -> Artefakte
    "prompt_clarity":      1.5,
    "frame_separation":    1.2,
    "env_continuity":      1.0,
    # PromptAgent — ltx_syntax kritisch: falsches Format -> LTX generiert nichts
    "ltx_syntax":          2.0,
    "protagonist_usage":   1.8,  # Falsche Verwendung -> Charakter-Artefakte
    "camera_description":  1.5,
    "cinematic_quality":   1.2,
    "mood_capture":        1.0,
    # Screenplay — three_act_structure kritisch: Basis für Director-Segmentierung
    "three_act_structure": 2.0,
    "narrative_coherence": 1.8,  # Ohne Logik kein verwertbares Drehbuch
    "cinematic_language":  1.5,  # Muss visuell umsetzbar sein
    "visual_specificity":  1.2,
    "character_depth":     1.0,
    # Songtext — lyrical_alignment kritisch: Sync mit Audio-Content
    "lyrical_alignment":   2.0,
    "image_generability":  1.8,  # Nicht generierbar = nutzloser Prompt
    "mood_accuracy":       1.5,
    "segment_variety":     1.2,
    "visual_creativity":   1.0,
    # ACE Lyrics — tag_syntax kritisch: falsche Tags -> ACEStep generiert nichts
    "tag_syntax":          2.0,
    "rhyme_scheme":        1.8,  # Kein Reim = unbrauchbare Lyrics
    "structure_compliance":1.5,
    "lyrical_quality":     1.2,
    "mood_match":          1.0,
}

ROLE_DIMENSIONS: dict[str, list[str]] = {
    "director":     ["format_adherence", "story_arc", "frame_actions", "scene_quality", "env_consistency"],
    "video_script": ["trigger_injection", "face_handling", "prompt_clarity", "frame_separation", "env_continuity"],
    "prompt_agent": ["ltx_syntax", "protagonist_usage", "camera_description", "cinematic_quality", "mood_capture"],
    "screenplay":   ["three_act_structure", "narrative_coherence", "cinematic_language", "visual_specificity", "character_depth"],
    "songtext":     ["lyrical_alignment", "image_generability", "mood_accuracy", "segment_variety", "visual_creativity"],
    "acelyrics":    ["tag_syntax", "rhyme_scheme", "structure_compliance", "lyrical_quality", "mood_match"],
}

CRITICAL_WEIGHT = 1.8   # Dimensionen ab diesem Gewicht gelten als kritisch
WEAK_THRESHOLD  = 3.0   # Score darunter = schwach in dieser Dimension

# ── Test Fixtures ──────────────────────────────────────────────────────────

TEST_FIXTURES = {
    "beach_sunset": {
        "name": "Beach Sunset — Woman finds washed-up vase",
        "story": (
            "A young woman sits alone on a beach during golden hour, watching waves crash. "
            "A mysterious ornate vase washes up at her feet. She picks it up, examines it, "
            "and as the sun sets completely, a faint glow emanates from within the vase."
        ),
        "style": "Cinematic Realism",
        "target_duration": 30.0,
        "protagonist": (
            "A young woman in her late 20s with long auburn hair tied loosely. "
            "She wears a light linen dress in pale blue and no shoes. "
            "Her expression is contemplative and slightly melancholic."
        ),
        "environment": "golden hour, clear sky, warm directional sunlight, wet sand, amber and blue tones",
    },
    "city_rain": {
        "name": "City Rain — Detective follows a lead",
        "story": (
            "A world-weary detective walks through rain-soaked city streets at night. "
            "He spots a briefcase abandoned in a doorway. Inside he finds only a photograph "
            "of himself from 20 years ago. He looks up — across the street, a figure vanishes into the fog."
        ),
        "style": "Neo-Noir",
        "target_duration": 25.0,
        "protagonist": (
            "A man in his 50s with grey stubble and tired eyes. "
            "He wears a dark trench coat over a rumpled shirt. "
            "His posture is slightly hunched, suggesting years of burden."
        ),
        "environment": "night, overcast sky, cold blue artificial lighting, wet asphalt, deep blue and yellow tones",
    },
    "forest_child": {
        "name": "Forest — Child discovers a hidden door",
        "story": (
            "A curious child wanders into an ancient forest during early morning mist. "
            "Between two massive oak trees she finds a small wooden door set into the ground. "
            "She opens it and peers into golden light below — then something small and luminous floats up."
        ),
        "style": "Fantasy",
        "target_duration": 20.0,
        "protagonist": (
            "A girl around 10 years old with dark curly hair and bright green eyes. "
            "She wears a red knitted sweater and mud-stained jeans with rubber boots. "
            "She carries a small backpack."
        ),
        "environment": "early morning, misty forest, diffused cool light, damp moss and leaves, green and grey tones",
    },
}

# Fixed test segments for VideoScript and Prompt Agent tests
# (used when no Director output is available)
FIXTURE_SEGMENTS_TEMPLATE = [
    {
        "index": 0, "label": "Opening", "duration": 8.0,
        "scene_description": "Woman sits at water's edge as waves roll in gently around her feet.",
        "shot_type": "WS", "camera_move": "dolly_in", "qwen_camera_command": "将镜头切换到全景",
        "camera_extended": False, "lighting": "Golden hour backlight", "cut_type": "cut",
        "mood": "Contemplative, serene", "act": 1, "pattern": "A",
        "start_frame_action": "Wide shot of woman sitting at the shoreline, waves approaching.",
        "end_frame_action": "Camera has moved closer, woman's face becoming more visible.",
        "middle_frame_action": "", "environmental_context": "golden hour, clear sky, warm sunlight, wet sand",
        "protagonist_position": "seated 1m from waterline", "face_visible": True,
        "needs_multiangle": True, "frame_count": 2,
        "ltx_camera_lora": "dolly_smooth_v2", "effect_lora": "golden_hour_v1",
    },
    {
        "index": 1, "label": "Discovery", "duration": 6.0,
        "scene_description": "The vase tumbles in the surf and comes to rest at her feet.",
        "shot_type": "ECU", "camera_move": "static", "qwen_camera_command": "将镜头切换到大特写",
        "camera_extended": False, "lighting": "Warm side light from setting sun", "cut_type": "cut",
        "mood": "Mysterious, wonder", "act": 1, "pattern": "C",
        "start_frame_action": "Extreme close-up of ornate vase resting on wet sand, water receding.",
        "end_frame_action": "", "middle_frame_action": "",
        "environmental_context": "golden hour, clear sky, warm sunlight, wet sand",
        "protagonist_position": "", "face_visible": False,
        "needs_multiangle": False, "frame_count": 1,
        "ltx_camera_lora": "", "effect_lora": "ocean_surf_v1",
    },
]


# ── Test Runner ────────────────────────────────────────────────────────────

class TestRunner:
    """Runs agent tests with a substituted model and returns raw output."""

    def __init__(self, model_id: str, api_key: str, base_url: str):
        self.model_id = model_id
        self.api_key = api_key
        self.base_url = base_url

    def _make_config(self) -> dict:
        """Inject test model into a copy of the real config."""
        cfg = json.loads(json.dumps(CONFIG))
        film_agents = cfg.setdefault("film_agents", {})
        # Inject model into all agent configs
        for agent_key in ("director", "video_script", "ltx_prompt", "screenplay"):
            film_agents.setdefault(agent_key, {})["model"] = self.model_id
        return cfg

    async def run_director(self, fixture_key: str) -> dict:
        fixture = TEST_FIXTURES.get(fixture_key, list(TEST_FIXTURES.values())[0])
        cfg = self._make_config()
        agent = DirectorAgent(cfg)
        t0 = time.time()
        try:
            segments = await agent.create_scene_breakdown(
                story=fixture["story"],
                target_duration=fixture["target_duration"],
                style=fixture["style"],
            )
            elapsed = time.time() - t0
            return {
                "model": self.model_id,
                "role": "director",
                "fixture": fixture_key,
                "elapsed": round(elapsed, 2),
                "output": [s.to_dict() for s in segments],
                "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "director", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": None, "error": str(e),
            }

    async def run_video_script(self, fixture_key: str) -> dict:
        fixture = TEST_FIXTURES.get(fixture_key, list(TEST_FIXTURES.values())[0])
        cfg = self._make_config()
        agent = VideoScriptAgent(cfg)
        t0 = time.time()
        results = []
        try:
            prev = None
            for seg_data in FIXTURE_SEGMENTS_TEMPLATE:
                seg = SceneSegment.from_dict({**seg_data, "audio_clip": "", "prompt": "",
                    "first_frame": "", "last_frame": "", "video_clip": "",
                    "status": "pending",
                    "qwen_description": "", "start_frame_prompt": "", "end_frame_prompt": "",
                    "middle_frame_prompt": "", "middle_frame": "",
                    "low_confidence": False, "validation_score": None})
                seg = await agent.write_segment_prompts(seg, fixture["protagonist"], prev_segment=prev)
                results.append({
                    "index": seg.index,
                    "label": seg.label,
                    "shot_type": seg.shot_type,
                    "face_visible": seg.face_visible,
                    "start_frame_prompt": seg.start_frame_prompt,
                    "end_frame_prompt": seg.end_frame_prompt,
                    "middle_frame_prompt": seg.middle_frame_prompt,
                })
                prev = seg
            return {
                "model": self.model_id, "role": "video_script", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": results, "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "video_script", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": None, "error": str(e),
            }

    async def run_acelyrics(self) -> dict:
        from audio_enhancer import AudioEnhancer, AudioSettings
        cfg = self._make_config()
        cfg["audio_llm_model"] = self.model_id
        cfg["openrouter_api_key"] = self.api_key
        enhancer = AudioEnhancer(cfg)
        settings = AudioSettings(
            type="vocal",
            genre="indie pop",
            duration=180,
            language="en",
            voice="any",
        )
        idea = "loss, nostalgia, rain — melancholic indie pop song about letting go"
        t0 = time.time()
        try:
            result = await enhancer._generate_song_impl(settings, idea)
            output = {
                "caption": result.caption,
                "structure": result.structure,
                "lyrics": result.lyrics,
            }
            return {
                "model": self.model_id, "role": "acelyrics", "fixture": "indie_melancholy",
                "elapsed": round(time.time() - t0, 2), "output": output, "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "acelyrics", "fixture": "indie_melancholy",
                "elapsed": round(time.time() - t0, 2), "output": None, "error": str(e),
            }

    async def run_prompt_agent(self, fixture_key: str) -> dict:
        fixture = TEST_FIXTURES.get(fixture_key, list(TEST_FIXTURES.values())[0])
        cfg = self._make_config()
        agent = LTXPromptAgent(cfg)
        t0 = time.time()
        results = []
        try:
            for seg_data in FIXTURE_SEGMENTS_TEMPLATE:
                seg = SceneSegment.from_dict({**seg_data, "audio_clip": "", "prompt": "",
                    "first_frame": "", "last_frame": "", "video_clip": "",
                    "ltx_camera_lora": "", "effect_lora": "", "status": "pending",
                    "qwen_description": "", "start_frame_prompt": "", "end_frame_prompt": "",
                    "middle_frame_prompt": "", "middle_frame": "",
                    "low_confidence": False, "validation_score": None})
                seg = await agent.process_segment(seg, fixture["protagonist"])
                results.append({
                    "index": seg.index,
                    "label": seg.label,
                    "prompt": seg.prompt,
                    "ltx_camera_lora": seg.ltx_camera_lora,
                    "effect_lora": seg.effect_lora,
                })
            return {
                "model": self.model_id, "role": "prompt_agent", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": results, "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "prompt_agent", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": None, "error": str(e),
            }

    async def run_screenplay(self, fixture_key: str) -> dict:
        fixture = TEST_FIXTURES.get(fixture_key, list(TEST_FIXTURES.values())[0])
        cfg = self._make_config()
        agent = ScreenplayAgent(cfg)
        t0 = time.time()
        try:
            screenplay, protagonist, environment = await agent.write(
                story_idea=fixture["story"],
                target_duration=fixture["target_duration"],
                style=fixture["style"],
            )
            return {
                "model": self.model_id, "role": "screenplay", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2),
                "output": {"screenplay": screenplay, "protagonist": protagonist, "environment": environment},
                "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "screenplay", "fixture": fixture_key,
                "elapsed": round(time.time() - t0, 2), "output": None, "error": str(e),
            }

    async def run_songtext(self) -> dict:
        cfg = self._make_config()
        # Inject model and api_key into music video prompter config keys
        cfg["openrouter_model"] = self.model_id
        cfg["openrouter_api_key"] = self.api_key
        prompter = MusicVideoPrompter(cfg)
        mock_segments = [
            Segment(index=0, start_time=0.0,  end_time=20.0, label="Verse 1", lyrics="Empty streets at midnight, your ghost still lingers here"),
            Segment(index=1, start_time=20.0, end_time=40.0, label="Chorus",  lyrics="I'm drowning in the silence, the echo of your name"),
            Segment(index=2, start_time=40.0, end_time=60.0, label="Verse 2", lyrics="Old photographs on the table, faded at the edges"),
            Segment(index=3, start_time=60.0, end_time=75.0, label="Bridge",  lyrics="Maybe time will wash it clean, maybe not"),
            Segment(index=4, start_time=75.0, end_time=90.0, label="Outro",   lyrics="Streetlights fade to morning, still I wait"),
        ]
        theme = "loss, nostalgia, rain"
        t0 = time.time()
        try:
            prompts = await prompter.generate_segment_prompts(mock_segments, theme)
            output = [
                {"index": seg.index, "label": seg.label, "lyrics": seg.lyrics, "prompt": p}
                for seg, p in zip(mock_segments, prompts)
            ]
            return {
                "model": self.model_id, "role": "songtext", "fixture": "indie_melancholy",
                "elapsed": round(time.time() - t0, 2), "output": output, "error": None,
            }
        except Exception as e:
            return {
                "model": self.model_id, "role": "songtext", "fixture": "indie_melancholy",
                "elapsed": round(time.time() - t0, 2), "output": None,
                "error": str(e) + "\n--- TRACEBACK ---\n" + traceback.format_exc(),
            }


# ── Judge LLM Scoring ──────────────────────────────────────────────────────

JUDGE_MODEL = "openai/gpt-4o-mini"

DIRECTOR_DIMENSIONS    = ["scene_quality", "format_adherence", "story_arc", "frame_actions", "env_consistency"]
VIDEOSCRIPT_DIMENSIONS = ["prompt_clarity", "face_handling", "frame_separation", "trigger_injection", "env_continuity"]
PROMPT_DIMENSIONS      = ["cinematic_quality", "ltx_syntax", "protagonist_usage", "camera_description", "mood_capture"]
SCREENPLAY_DIMENSIONS  = ["narrative_coherence", "character_depth", "cinematic_language", "three_act_structure", "visual_specificity"]
SONGTEXT_DIMENSIONS    = ["lyrical_alignment", "visual_creativity", "mood_accuracy", "image_generability", "segment_variety"]
ACELYRICS_DIMENSIONS   = ["tag_syntax", "rhyme_scheme", "structure_compliance", "lyrical_quality", "mood_match"]

DIMENSION_DESCRIPTIONS = {
    "scene_quality":       "Cinematic creativity, shot type variety, dramatic choices (0-5)",
    "format_adherence":    "All required JSON fields present and valid (0-5)",
    "story_arc":           "Acts 1/2/3 structure, narrative coherence and flow (0-5)",
    "frame_actions":       "Quality of start/end frame action descriptions (0-5)",
    "env_consistency":     "Environmental context captured consistently across segments (0-5)",
    "prompt_clarity":      "Visual specificity, generable image description quality (0-5)",
    "face_handling":       "Correct use of protagonist description — appearance only, not pose (0-5)",
    "frame_separation":    "Start vs end prompts describe meaningfully different moments (0-5)",
    "trigger_injection":   "Each prompt opens with 'Next Scene:' followed by exactly one camera command — no duplicates, no extra 'Next Scene:' prefixes (0-5)",
    "env_continuity":      "Environmental context embedded in each prompt (0-5)",
    "cinematic_quality":   "Cinematic atmosphere, visual richness of the LTX video prompt (0-5)",
    "ltx_syntax":          "Prompt is formatted well for LTX 2.3 video generation (0-5)",
    "protagonist_usage":   "Protagonist description used appropriately without overriding scene (0-5)",
    "camera_description":  "Camera movement described clearly for LTX (0-5)",
    "mood_capture":        "Mood and lighting effectively translated into prompt language (0-5)",
    "narrative_coherence": "Story logic, cause-effect and scene flow (0-5)",
    "character_depth":     "Protagonist and secondary character believability (0-5)",
    "cinematic_language":  "Use of visual, filmic language — not theatrical or literary (0-5)",
    "three_act_structure": "Clear setup, confrontation, resolution (0-5)",
    "visual_specificity":  "Concrete visual detail suitable for directing (0-5)",
    "lyrical_alignment":   "Visual prompts reflect the mood and content of the lyrics (0-5)",
    "visual_creativity":   "Originality and imagination of visual interpretations (0-5)",
    "mood_accuracy":       "Emotional tone matches the song/segment mood (0-5)",
    "image_generability":  "Prompts are concrete and generatable by an image model (0-5)",
    "segment_variety":     "Visual diversity across segments — no repetitive descriptions (0-5)",
    "tag_syntax":          "ACEStep section tags correct: [Verse], [Chorus], optional modifier (0-5)",
    "rhyme_scheme":        "Lyrics have consistent rhyme and rhythm within sections (0-5)",
    "structure_compliance":"Song structure matches requested genre and duration (0-5)",
    "lyrical_quality":     "Lyrical creativity, imagery, emotional resonance (0-5)",
    "mood_match":          "Overall mood and tone match the requested theme/genre (0-5)",
}

CUSTOM_DIMENSIONS = ["correctness", "completeness", "quality", "requirement_adherence"]

DIMENSION_DESCRIPTIONS_CUSTOM = {
    "correctness":           "Does the output correctly solve the request? (0-5)",
    "completeness":          "Is the output complete — nothing important missing? (0-5)",
    "quality":               "Code/text quality: clarity, best practices, style (0-5)",
    "requirement_adherence": "Does the output match all stated requirements? (0-5)",
}


def _detect_language(text: str) -> str:
    t = text.strip()
    if re.search(r'<!DOCTYPE\s+html|<html[\s>]', t, re.IGNORECASE):
        return "html"
    fence = re.search(r'```(\w+)', t)
    if fence:
        lang = fence.group(1).lower()
        if lang in ("python", "py"):                    return "python"
        if lang in ("javascript", "js", "typescript", "ts"): return "javascript"
        if lang == "css":                               return "css"
        if lang in ("html", "htm"):                     return "html"
        if lang in ("bash", "sh", "shell"):             return "shell"
        if lang == "json":                              return "json"
        return lang
    if re.match(r'^(import |from |def |class |#!.*python)', t):
        return "python"
    return "text"


async def run_custom_test(model_id: str, prompt: str, system_prompt: str, api_key: str) -> dict:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    t0 = time.time()
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model_id, "messages": messages},
            )
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            return {
                "model": model_id, "role": "custom",
                "elapsed": round(time.time() - t0, 2),
                "output": {"text": text, "detected_language": _detect_language(text)},
                "error": None,
            }
    except Exception as e:
        return {
            "model": model_id, "role": "custom",
            "elapsed": round(time.time() - t0, 2),
            "output": None, "error": str(e),
        }


async def _score_with_judge_custom(
    output_text: str, prompt: str, criteria: str,
    api_key: str, judge_model: str = JUDGE_MODEL,
) -> dict:
    if not criteria:
        criteria = "Korrektheit, Vollständigkeit, Code-Qualität, Anforderungserfüllung"
    dim_desc = "\n".join(
        f"- {d}: {DIMENSION_DESCRIPTIONS_CUSTOM[d]}" for d in CUSTOM_DIMENSIONS
    )
    system = (
        "You are an expert AI output evaluator. Score the response on each dimension "
        "from 0 to 5 (decimals allowed). Return ONLY a JSON object. No prose."
    )
    user = (
        f"Original prompt:\n{prompt[:500]}\n\n"
        f"Evaluation criteria: {criteria}\n\n"
        f"Scoring dimensions:\n{dim_desc}\n\n"
        f"Response to evaluate:\n{output_text[:3000]}\n\n"
        f"Return JSON: {{{', '.join(f'\"{d}\": <0-5>' for d in CUSTOM_DIMENSIONS)}}}"
    )
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": judge_model,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": 200,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                content = m.group(0)
            scores = json.loads(content)
            return {d: max(0.0, min(5.0, float(scores.get(d, 0)))) for d in CUSTOM_DIMENSIONS}
    except Exception as e:
        logger.warning("Custom judge scoring failed: %s", e)
        return {d: 0.0 for d in CUSTOM_DIMENSIONS}


async def _score_with_judge(role: str, output: dict, api_key: str, judge_model: str = JUDGE_MODEL) -> dict:
    """Ask a judge LLM to score the output on 5 dimensions. Returns {dimension: score}."""
    if role == "director":
        dimensions = DIRECTOR_DIMENSIONS
    elif role == "video_script":
        dimensions = VIDEOSCRIPT_DIMENSIONS
    elif role == "screenplay":
        dimensions = SCREENPLAY_DIMENSIONS
    elif role == "songtext":
        dimensions = SONGTEXT_DIMENSIONS
    elif role == "acelyrics":
        dimensions = ACELYRICS_DIMENSIONS
    else:
        dimensions = PROMPT_DIMENSIONS

    dim_desc = "\n".join(f"- {d}: {DIMENSION_DESCRIPTIONS[d]}" for d in dimensions)
    output_text = json.dumps(output, indent=2, ensure_ascii=False)[:3000]

    system = (
        "You are an expert evaluator of AI-generated film production outputs. "
        "Score the provided output on each dimension from 0 to 5 (decimals allowed). "
        "Return ONLY a JSON object with dimension names as keys and numeric scores as values. "
        "No explanation, no prose."
    )
    user = (
        f"Role being evaluated: {role}\n\n"
        f"Scoring dimensions:\n{dim_desc}\n\n"
        f"Output to score:\n{output_text}\n\n"
        f"Return JSON: {{{', '.join(f'\"{d}\": <0-5>' for d in dimensions)}}}"
    )

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 300,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            match = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if match:
                content = match.group(0)
            scores = json.loads(content)
            if not isinstance(scores, dict):
                raise ValueError(f"Judge returned {type(scores).__name__} instead of dict: {content[:200]}")
            # Ensure all dimensions present, clamp to 0-5
            return {d: max(0.0, min(5.0, float(scores.get(d, 0)))) for d in dimensions}
    except Exception as e:
        logger.warning("Judge scoring failed: %s", e)
        return {d: 0.0 for d in dimensions}


# ── Flask App ──────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(UI_DIR))


@app.route("/")
def index():
    return send_from_directory(str(UI_DIR), "index.html")


@app.route("/test-fixtures")
def get_fixtures():
    return jsonify({
        "fixtures": [{"key": k, "name": v["name"]} for k, v in TEST_FIXTURES.items()]
    })


@app.route("/run", methods=["POST"])
def run_tests():
    """
    POST body: {
        "role": "director" | "video_script" | "prompt_agent",
        "models": ["model/id1", "model/id2", ...],   # up to 4
        "fixture": "beach_sunset",
        "judge": true | false
    }
    Returns: {"results": [{model, role, elapsed, output, error, scores?}, ...]}
    """
    body = request.get_json(force=True) or {}
    role = body.get("role", "director")
    models = (body.get("models") or [])[:4]
    fixture_key = body.get("fixture", "beach_sunset")
    do_judge = body.get("judge", False)
    judge_model = body.get("judge_model", JUDGE_MODEL)
    creative_judge_model = body.get("creative_judge_model", "") or judge_model

    if not models:
        return jsonify({"error": "No models provided"}), 400

    api_key = (
        CONFIG.get("prompt_enhancer", {}).get("openrouter_api_key", "")
        or CONFIG.get("openrouter_api_key", "")
    )

    async def _run_all():
        runners = [TestRunner(m.strip(), api_key, "https://openrouter.ai/api/v1/chat/completions")
                   for m in models if m.strip()]

        tasks = []
        for runner in runners:
            if role == "director":
                tasks.append(runner.run_director(fixture_key))
            elif role == "video_script":
                tasks.append(runner.run_video_script(fixture_key))
            elif role == "screenplay":
                tasks.append(runner.run_screenplay(fixture_key))
            elif role == "songtext":
                tasks.append(runner.run_songtext())
            elif role == "acelyrics":
                tasks.append(runner.run_acelyrics())
            else:
                tasks.append(runner.run_prompt_agent(fixture_key))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                res = {"model": models[i], "role": role, "fixture": fixture_key,
                       "elapsed": 0, "output": None, "error": str(res)}
            if do_judge and res.get("output") and not res.get("error"):
                is_creative = role in ("screenplay", "songtext", "acelyrics")
                model_for_judge = creative_judge_model if is_creative else judge_model
                res["scores"] = await _score_with_judge(role, res["output"], api_key, model_for_judge)
                res["total_score"] = round(sum(res["scores"].values()), 2)
            final.append(res)

        return final

    # Run async in a new event loop (Flask is sync)
    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run_all())
    finally:
        loop.close()

    # Save results to file
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    out_path = RESULTS_DIR / f"llm_test_results_{ts}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Results saved to %s", out_path)

    return jsonify({"results": results, "saved_to": str(out_path)})


@app.route("/ranking")
def get_ranking():
    K = 3
    NEUTRAL = 12.5
    TOP_N = 3

    all_results = []
    # Neuer Ordner + Root für Rückwärtskompatibilität mit alten Ergebnissen
    for search_dir in [RESULTS_DIR, CTB_ROOT]:
        for f in search_dir.glob("llm_test_results_*.json"):
            try:
                all_results.extend(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass

    agg: dict = {}
    for r in all_results:
        if not r.get("total_score") or r.get("error"):
            continue
        role = r.get("role", "unknown")
        model = r.get("model", "unknown")
        agg.setdefault(role, {}).setdefault(model, {
            "sum": 0.0, "count": 0, "dim_sums": {}, "dim_counts": {}
        })
        entry = agg[role][model]
        entry["sum"] += r["total_score"]
        entry["count"] += 1
        for dim, score in (r.get("scores") or {}).items():
            entry["dim_sums"][dim] = entry["dim_sums"].get(dim, 0.0) + score
            entry["dim_counts"][dim] = entry["dim_counts"].get(dim, 0) + 1

    ranking: dict = {}
    for role, models_data in agg.items():
        role_dims = ROLE_DIMENSIONS.get(role, [])
        critical_dims = [d for d in role_dims if DIMENSION_WEIGHTS.get(d, 1.0) >= CRITICAL_WEIGHT]

        entries = []
        for model, d in models_data.items():
            avg = d["sum"] / d["count"]
            bayes = (d["count"] * avg + K * NEUTRAL) / (d["count"] + K)
            dim_avgs = {
                dim: round(d["dim_sums"][dim] / d["dim_counts"][dim], 2)
                for dim in d["dim_sums"] if d["dim_counts"].get(dim, 0) > 0
            }
            # Gewichteter Score über kritische Dimensionen
            w_sum = sum(dim_avgs.get(dim, 0) * DIMENSION_WEIGHTS.get(dim, 1.0) for dim in role_dims if dim in dim_avgs)
            w_total = sum(DIMENSION_WEIGHTS.get(dim, 1.0) for dim in role_dims if dim in dim_avgs)
            weighted_score = round(w_sum / w_total, 2) if w_total > 0 else round(avg / 5.0, 2)
            entries.append({
                "model": model,
                "avg_score": round(avg, 2),
                "count": d["count"],
                "bayesian_score": round(bayes, 2),
                "weighted_score": weighted_score,
                "dim_avgs": dim_avgs,
            })

        entries.sort(key=lambda x: x["bayesian_score"], reverse=True)
        top3 = [dict(e) for e in entries[:TOP_N]]

        # Empfehlung berechnen
        recommendation = None
        if entries:
            top_bayes    = entries[0]
            top_weighted = max(entries, key=lambda x: x["weighted_score"])

            if top_weighted["model"] == top_bayes["model"]:
                weak = [d for d in critical_dims if top_bayes["dim_avgs"].get(d, 5.0) < WEAK_THRESHOLD]
                if weak:
                    dim_labels = ", ".join(d.replace("_", " ") for d in weak)
                    recommendation = {
                        "model": top_bayes["model"],
                        "is_top1": True,
                        "text": f"Platz 1 bestätigt — kritische Schwäche bei: {dim_labels}. Ergebnis sorgfältig prüfen."
                    }
                else:
                    recommendation = {"model": top_bayes["model"], "is_top1": True, "text": None}
            else:
                weak_dims = [d for d in critical_dims if top_bayes["dim_avgs"].get(d, 5.0) < WEAK_THRESHOLD]
                improvements = []
                for d in (weak_dims or critical_dims):
                    b_val = top_bayes["dim_avgs"].get(d, 0.0)
                    w_val = top_weighted["dim_avgs"].get(d, 0.0)
                    if w_val > b_val:
                        improvements.append(f"{d.replace('_', ' ')} ({b_val:.1f} -> {w_val:.1f})")
                top1_short = top_bayes["model"].split("/")[-1]
                rec_short  = top_weighted["model"].split("/")[-1]
                if improvements:
                    text = (
                        f"Trotz höherem Gesamt-Score schwächelt {top1_short} bei kritischen Dimensionen: "
                        f"{', '.join(improvements)}. {rec_short} ist dort konstanter."
                    )
                else:
                    text = (
                        f"{rec_short} erzielt in gewichteten Kernbereichen dieser Rolle "
                        f"({top_weighted['weighted_score']:.2f}) besser als {top1_short} ({top_bayes['weighted_score']:.2f})."
                    )
                recommendation = {"model": top_weighted["model"], "is_top1": False, "text": text}

        # weak_dims pro Eintrag berechnen, dann dim_avgs + weighted_score entfernen
        for e in top3:
            e["weak_dims"] = {
                d: round(e["dim_avgs"].get(d, 5.0), 1)
                for d in critical_dims
                if e["dim_avgs"].get(d, 5.0) < WEAK_THRESHOLD
            }
            e.pop("dim_avgs", None)
            e.pop("weighted_score", None)

        ranking[role] = {"top3": top3, "recommendation": recommendation}

    return jsonify({"ranking": ranking})


@app.route("/run-custom", methods=["POST"])
def run_custom():
    body = request.get_json(force=True) or {}
    prompt = (body.get("prompt") or "").strip()
    system_prompt = (body.get("system_prompt") or "").strip()
    models = (body.get("models") or [])[:4]
    do_judge = body.get("judge", False)
    judge_model = body.get("judge_model", JUDGE_MODEL)
    judge_criteria = (body.get("judge_criteria") or "").strip()

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    if not models:
        return jsonify({"error": "No models provided"}), 400

    api_key = (
        CONFIG.get("prompt_enhancer", {}).get("openrouter_api_key", "")
        or CONFIG.get("openrouter_api_key", "")
    )

    async def _run_all():
        tasks = [run_custom_test(m.strip(), prompt, system_prompt, api_key)
                 for m in models if m.strip()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        final = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                res = {"model": models[i], "role": "custom", "elapsed": 0,
                       "output": None, "error": str(res)}
            if do_judge and res.get("output") and not res.get("error"):
                text = res["output"].get("text", "")
                res["scores"] = await _score_with_judge_custom(
                    text, prompt, judge_criteria, api_key, judge_model
                )
                res["total_score"] = round(sum(res["scores"].values()), 2)
            final.append(res)
        return final

    loop = asyncio.new_event_loop()
    try:
        results = loop.run_until_complete(_run_all())
    finally:
        loop.close()

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = int(time.time())
    (RESULTS_DIR / f"llm_custom_results_{ts}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return jsonify({"results": results})


# ── Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = 7799
    url = f"http://localhost:{port}"
    print(f"\n LLM Tester starting @ {url}\n")

    def _open_browser():
        time.sleep(1.2)
        webbrowser.open(url)

    Thread(target=_open_browser, daemon=True).start()
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
