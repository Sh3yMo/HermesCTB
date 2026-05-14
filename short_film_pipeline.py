# short_film_pipeline.py
"""
Short Film Pipeline — orchestrates Director -> Script -> Prompt -> Keyframe -> Render -> Stitch.
Mirrors the structure of music_video_pipeline.py.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

from director_agent import DirectorAgent, SceneSegment
from keyframe_agent import KeyframeAgent
from prompt_enhancer import enhance_prompt

LORA_PROFILES_PATH = Path(__file__).parent / "lora_profiles.json"
OUTPUT_BASE = Path(__file__).parent / "output" / "short_films"


# ── LoRA Resolution ────────────────────────────────────────────────────────

def _load_profiles() -> dict:
    return json.loads(LORA_PROFILES_PATH.read_text(encoding="utf-8"))


def resolve_ltx_loras(segment: SceneSegment) -> list:
    """
    Build the 4-slot LTX LoRA list for a segment render.
    Slot 1: distilled (always)
    Slot 2: VBVR (always)
    Slot 3: camera LoRA (always)
    Slot 4: effect LoRA (optional, based on scene_description keywords)
    """
    profiles = _load_profiles()
    loras = []

    # Slot 1 — distilled
    base = profiles.get("ltx_base", {})
    if "distilled" in base:
        d = base["distilled"]
        loras.append({"lora": d["file"], "weight": d.get("weight", 1.0), "slot": 1})

    # Slot 2 — VBVR
    if "vbvr" in base:
        v = base["vbvr"]
        loras.append({"lora": v["file"], "weight": v.get("weight", 1.0), "slot": 2})

    # Slot 3 — camera LoRA (extended moves fall back to static)
    camera_key = segment.camera_move if not segment.camera_extended else "static"
    camera_loras = profiles.get("ltx_camera", {})
    if camera_key in camera_loras:
        c = camera_loras[camera_key]
        loras.append({"lora": c["file"], "weight": c.get("weight", 1.0), "slot": 3})
    elif "static" in camera_loras:
        c = camera_loras["static"]
        loras.append({"lora": c["file"], "weight": c.get("weight", 1.0), "slot": 3})

    # Slot 4 — effect LoRA (keyword match, first match wins)
    desc_lower = segment.scene_description.lower()
    for effect_name, effect in profiles.get("ltx_effects", {}).items():
        if any(kw in desc_lower for kw in effect.get("trigger_keywords", [])):
            loras.append({"lora": effect["file"], "weight": effect.get("weight", 0.8), "slot": 4})
            break

    return loras[:4]  # hard cap at 4 slots


# ── Session State ──────────────────────────────────────────────────────────

@dataclass
class FilmSession:
    session_id: str = ""
    mode: str = "story_length"        # "story_length" | "story_audio"
    director_mode: str = "auto"       # "auto" | "review"
    story: str = ""
    screenplay: str = ""
    protagonist_description: str = ""
    environment_context: str = ""
    target_duration: float = 30.0
    audio_path: str = ""
    style: str = "Cinematic Realism"
    character_ref: str = ""
    character_refs: Dict[str, str] = field(default_factory=dict)
    workflow: str = ""                 # FFLF2V or FFLFA2V
    segments: List[Dict[str, Any]] = field(default_factory=list)
    final_video: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "director_mode": self.director_mode,
            "story": self.story,
            "screenplay": self.screenplay,
            "protagonist_description": self.protagonist_description,
            "environment_context": self.environment_context,
            "target_duration": self.target_duration,
            "audio_path": self.audio_path,
            "style": self.style,
            "character_ref": self.character_ref,
            "character_refs": self.character_refs,
            "workflow": self.workflow,
            "segments": self.segments,
            "final_video": self.final_video,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilmSession":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FilmSession":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


# ── Appearance Feature Extraction ──────────────────────────────────────────

_HAIR_KEYWORDS = (
    "hair", "blonde", "brunette", "redhead", "gray hair", "grey hair",
    "black hair", "brown hair", "curly", "wavy", "straight hair", "braid",
    "ponytail", "bun", "short hair", "long hair", "shoulder-length",
)
_CLOTHING_KEYWORDS = (
    "wearing", "dressed", "shirt", "blouse", "jacket", "coat", "dress",
    "skirt", "jeans", "pants", "sweater", "hoodie", "uniform", "suit",
    "vest", "scarf", "hat", "cap", "shoes", "boots",
)

_FACE_KEYWORDS = (
    "eyeliner", "lipstick", "makeup", "mascara", "earring", "hoop", "piercing",
    "skin", "complexion", "eyes", "lips", "eyebrow", "lashes", "smudged",
    "sharp", "contour", "rouge", "blush",
)

def _extract_appearance_features(protagonist_description: str) -> str:
    """
    Extract hair, clothing, and face/makeup descriptors from a protagonist description.

    Used to build an identity anchor for prompts without overriding pose.
    """
    if not protagonist_description:
        return ""

    parts = []
    lower = protagonist_description.lower()

    # Extract sentences/clauses that mention hair, clothing, or face features
    sentences = re.split(r"[.,;]", protagonist_description)
    for sentence in sentences:
        s_lower = sentence.lower()
        if any(kw in s_lower for kw in _HAIR_KEYWORDS + _CLOTHING_KEYWORDS + _FACE_KEYWORDS):
            parts.append(sentence.strip())

    if parts:
        return ", ".join(p for p in parts if p)

    # Fallback: first 80 chars (truncate at word boundary)
    truncated = protagonist_description[:80]
    last_space = truncated.rfind(" ")
    return truncated[:last_space] if last_space > 20 else truncated


# ── Screenplay Agent ───────────────────────────────────────────────────────

class ScreenplayAgent:
    """Writes a full short film screenplay from a theme/story idea (uses Cydonia)."""

    def __init__(self, config: dict):
        agent_cfg = config.get("film_agents", {}).get("screenplay", {})
        self.model = agent_cfg.get("model", "thedrummer/cydonia-24b-v4.1")
        self.max_tokens = agent_cfg.get("max_tokens", 4000)
        self.fallback_models = agent_cfg.get("fallback_models", [])
        self.api_key = config.get("prompt_enhancer", {}).get("openrouter_api_key", "") or config.get("openrouter_api_key", "")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def _call_openrouter(self, messages: list) -> str:
        _max_retries = 5
        _backoff = [30, 60, 120, 300, 600]
        model_candidates = [self.model] + [x for x in self.fallback_models if x and x != self.model]
        for model_idx, current_model in enumerate(model_candidates):
            logger.info("ScreenplayAgent → OpenRouter [%s]", current_model)
            for attempt in range(_max_retries):
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={"model": current_model, "messages": messages, "max_tokens": self.max_tokens, "provider": {"allow_fallbacks": True}}
                    )
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset-requests")
                        try:
                            sleep_s = float(retry_after) if retry_after else float(_backoff[min(attempt, len(_backoff) - 1)])
                        except ValueError:
                            sleep_s = float(_backoff[min(attempt, len(_backoff) - 1)])
                        if attempt < _max_retries - 1:
                            logger.warning("ScreenplayAgent 429 (model=%s, attempt %d/%d), sleeping %.0fs...", current_model, attempt + 1, _max_retries, sleep_s)
                            await asyncio.sleep(sleep_s)
                            continue
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("ScreenplayAgent 429 retries exhausted for '%s', switching to fallback: %s", current_model, model_candidates[model_idx + 1])
                        break
                    elif resp.status_code in (404, 503):
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("ScreenplayAgent %d (model=%s), switching to fallback: %s", resp.status_code, current_model, model_candidates[model_idx + 1])
                            break
                        resp.raise_for_status()
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        choices = data.get("choices") or []
                        if not choices:
                            raise RuntimeError(f"ScreenplayAgent: OpenRouter returned no choices: {str(data)[:400]}")
                        content = choices[0].get("message", {}).get("content")
                        if not content:
                            raise RuntimeError(f"ScreenplayAgent: OpenRouter returned empty content: {str(data)[:400]}")
                        logger.info("ScreenplayAgent ← response %d chars", len(content))
                        return content.strip()
        raise RuntimeError(f"ScreenplayAgent: OpenRouter 429 alle Modelle erschöpft {model_candidates}.")

    async def write(self, story_idea: str, target_duration: float, style: str) -> tuple:
        """
        Write a full screenplay from a story idea.
        Returns (screenplay_text, protagonist_description, environment_context).
        """
        system = (
            "You are a feature film screenwriter. Your task is to write a complete short film screenplay.\n"
            "Requirements:\n"
            "- Write a vivid, scene-by-scene narrative in present tense\n"
            "- Include a clear PROTAGONIST block at the start: describe their appearance, age, clothing in 2-3 sentences\n"
            "- Include an ENVIRONMENT block at the start: comma-separated fixed scene attributes — "
            "specific indoor/outdoor LOCATION TYPE first (e.g. 'apartment kitchen', 'rooftop terrace', 'subway car'), "
            "then: time of day, key light quality, dominant surface material, color palette tendency. "
            "(e.g. 'apartment kitchen, midnight, harsh flickering fluorescent light, ceramic linoleum, neon green and cold grey'). "
            "Use civilian/domestic terms for the location type — never abstract descriptors like 'sterile space'. "
            "Include 2-3 specific physical objects that anchor the space (e.g. 'dirty dishes in sink, stove with pots, cracked tile floor').\n"
            "- No technical camera directions (those are for the director)\n"
            "- Focus on story, character, atmosphere, and emotion\n"
            "- Format: Start with 'PROTAGONIST: [description]' on its own line, "
            "then 'ENVIRONMENT: [attributes]' on its own line, then write the screenplay\n"
            "\nDialogue rules (IMPORTANT):\n"
            "- The protagonist speaks in AT MOST ONE scene — ideally not at all\n"
            "- In all other scenes the protagonist expresses himself ONLY through physical reactions: "
            "facial expressions, body language, gestures, posture\n"
            "- Secondary and one-time characters may speak freely to create story context\n"
            "- No internal monologue or voice-over for the protagonist\n"
            "\nDURATION SCALING RULES — MANDATORY:\n"
            "- Each scene covers ONE moment or action only (no multi-beat scenes)\n"
            "- \u226430s  \u2192 write 2\u20133 scenes, single emotional beat, no subplots\n"
            "- 31\u201360s \u2192 write 3\u20135 scenes, simple setup \u2192 climax \u2192 resolution\n"
            "- 61\u201390s \u2192 write 5\u20137 scenes, one complication allowed\n"
            "- >90s  \u2192 write 6\u201310 scenes, fuller arc permitted\n"
            "- Fewer dense scenes always beat many shallow ones\n"
            "- DO NOT write more scenes than the duration can support\n"
        )
        user = (
            f"Story idea: {story_idea}\n"
            f"Visual style: {style}\n"
            f"Target duration: {target_duration} seconds\n"
            f"Match scene count to the duration scaling rules — prefer fewer dense scenes over many shallow ones.\n\n"
            "Write the complete short film screenplay:"
        )
        raw = await self._call_openrouter([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        # Extract protagonist description and environment context
        protagonist_desc = ""
        environment_ctx = ""
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("PROTAGONIST:") and not protagonist_desc:
                protagonist_desc = stripped.split(":", 1)[-1].strip()
            elif stripped.upper().startswith("ENVIRONMENT:") and not environment_ctx:
                environment_ctx = stripped.split(":", 1)[-1].strip()

        return raw, protagonist_desc, environment_ctx


# ── Frame Skill Base + Implementations ────────────────────────────────────

class FrameSkillBase:
    """Base class for per-frame Qwen IE prompt generation skills."""
    skill_name: str = "base"

    async def generate_frame_prompts(
        self,
        segment: "SceneSegment",
        protagonist_description: str,
        prev_segment: Optional["SceneSegment"],
    ) -> dict:
        """
        Returns dict with keys:
          start_prompt (str), end_prompt (str), middle_prompt (str, optional)
        """
        raise NotImplementedError


class FrameSkill_Qwen(FrameSkillBase):
    """
    Generates per-frame Qwen IE prompts using an LLM.

    Key behaviors vs. old VideoScriptAgent:
    - Returns structured JSON with separate start/end/middle prompts in ONE call
    - Protagonist reference: only hair/clothing features (not pose) when face_visible
    - No protagonist reference at all when face not visible
    - Each prompt embeds environmental_context for consistency
    - Injects prev_segment location context for face-visible shots to maintain scene continuity
    - Combined LoRA trigger words: shot_type trigger + movement modifier from Director
    """
    skill_name = "qwen"

    def __init__(self, config: dict):
        agent_cfg = config.get("film_agents", {}).get("video_script", {})
        self.model = agent_cfg.get("model", "qwen/qwen3.5-flash-02-23")
        self.max_tokens = agent_cfg.get("max_tokens", 1200)
        self.fallback_models = agent_cfg.get("fallback_models", [])
        self.api_key = (
            config.get("prompt_enhancer", {}).get("openrouter_api_key", "")
            or config.get("openrouter_api_key", "")
        )
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def _call_openrouter(self, messages: list) -> str:
        _max_retries = 5
        _backoff = [30, 60, 120, 300, 600]
        model_candidates = [self.model] + [x for x in self.fallback_models if x and x != self.model]
        for model_idx, current_model in enumerate(model_candidates):
            logger.info("FrameSkill_Qwen → OpenRouter [%s]", current_model)
            for attempt in range(_max_retries):
                async with httpx.AsyncClient(timeout=90) as client:
                    resp = await client.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={
                            "model": current_model,
                            "messages": messages,
                            "max_tokens": self.max_tokens,
                            "response_format": {"type": "json_object"},
                            "provider": {"allow_fallbacks": True},
                        }
                    )
                    if resp.status_code == 429:
                        retry_after = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset-requests")
                        try:
                            sleep_s = float(retry_after) if retry_after else float(_backoff[min(attempt, len(_backoff) - 1)])
                        except ValueError:
                            sleep_s = float(_backoff[min(attempt, len(_backoff) - 1)])
                        if attempt < _max_retries - 1:
                            logger.warning("FrameSkill_Qwen 429 (model=%s, attempt %d/%d), sleeping %.0fs...", current_model, attempt + 1, _max_retries, sleep_s)
                            await asyncio.sleep(sleep_s)
                            continue
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("FrameSkill_Qwen 429 retries exhausted for '%s', switching to fallback: %s", current_model, model_candidates[model_idx + 1])
                        break
                    elif resp.status_code in (400, 404, 503):
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("FrameSkill_Qwen %d (model=%s), switching to fallback: %s", resp.status_code, current_model, model_candidates[model_idx + 1])
                            break
                        resp.raise_for_status()
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        choices = data.get("choices") or []
                        if not choices:
                            raise RuntimeError(f"FrameSkill_Qwen: no choices: {str(data)[:400]}")
                        content = choices[0].get("message", {}).get("content")
                        if not content:
                            raise RuntimeError(f"FrameSkill_Qwen: empty content: {str(data)[:400]}")
                        logger.info("FrameSkill_Qwen ← %d chars", len(content))
                        return content.strip()
        raise RuntimeError(f"FrameSkill_Qwen: OpenRouter 429 alle Modelle erschöpft {model_candidates}.")

    async def generate_frame_prompts(
        self,
        segment: SceneSegment,
        protagonist_description: str,
        prev_segment: Optional[SceneSegment],
    ) -> dict:
        cam_cmd = segment.qwen_camera_command or ""
        is_static = segment.camera_move == "static"
        is_three_frame = segment.frame_count >= 3 and bool(segment.middle_frame_action)

        # Protagonist context
        if segment.face_visible and protagonist_description:
            protagonist_ref = _extract_appearance_features(protagonist_description)
            protagonist_ctx = (
                f"Character face+appearance anchor (hair, face features, clothing — NO pose): "
                f"{protagonist_ref}"
            )
        else:
            protagonist_ctx = ""

        # Outfit constraint for non-face-visible shots (prevents drift via continuity frames)
        outfit_ctx = ""
        if not segment.face_visible and protagonist_description:
            outfit_ref = _extract_appearance_features(protagonist_description)
            if outfit_ref:
                outfit_ctx = f"Character outfit/appearance (always present, never change): {outfit_ref}"

        # Location continuity injection for face-visible shots
        location_ctx = ""
        if segment.face_visible and prev_segment and prev_segment.protagonist_position:
            loc_parts = []
            if prev_segment.protagonist_position:
                loc_parts.append(f"protagonist position: {prev_segment.protagonist_position}")
            env = prev_segment.environmental_context or segment.environmental_context
            if env:
                loc_parts.append(f"scene environment: {env}")
            if loc_parts:
                location_ctx = "Continuity from previous shot — " + "; ".join(loc_parts) + "."

        # Environmental context
        env_ctx = segment.environmental_context or ""

        # Frame action descriptions from Director
        start_action = segment.start_frame_action or segment.scene_description
        end_action = segment.end_frame_action or start_action
        middle_action = segment.middle_frame_action or ""

        # Build output schema description
        if is_static:
            schema_desc = (
                'Return JSON: {"start_prompt": "...", "end_prompt": ""}\n'
                "start_prompt: The single keyframe for this static shot.\n"
                "end_prompt: Leave empty string (static shot = no movement)."
            )
        elif is_three_frame:
            schema_desc = (
                'Return JSON: {"start_prompt": "...", "end_prompt": "...", "middle_prompt": "..."}\n'
                "start_prompt: Opening visual state.\n"
                "middle_prompt: Midpoint of camera movement.\n"
                "end_prompt: Final visual state (camera fully moved)."
            )
        else:
            schema_desc = (
                'Return JSON: {"start_prompt": "...", "end_prompt": "..."}\n'
                "start_prompt: Opening visual state (beginning of camera movement).\n"
                "end_prompt: Final visual state (end of camera movement, different from start)."
            )

        system = (
            "You write image-editing prompts for the Qwen Image Edit model used to generate film keyframes.\n"
            "Rules:\n"
            "- ALWAYS start each prompt with 'Next Scene:'\n"
            "- Describe the scene visually: environment, lighting, atmosphere, textures, colors\n"
            "- NEVER use character names — describe characters by visual appearance only\n"
            "- Do NOT describe poses or body positions from the character appearance reference\n"
            "- No dialogue, no camera terminology, no internal directions\n"
            f"- Shot composition: {segment.shot_type} | Mood: {segment.mood} | Lighting: {segment.lighting}\n"
            + (f"- Camera framing command (inject after 'Next Scene:'): {cam_cmd}\n" if cam_cmd else "")
            + (f"- MANDATORY SCENE SETTING — every prompt MUST start with this location before any action description. Never substitute or omit: {env_ctx}\n" if env_ctx else "")
            + (f"- {protagonist_ctx}\n" if protagonist_ctx else "")
            + (f"- {outfit_ctx}\n" if outfit_ctx else "")
            + (f"- {location_ctx}\n" if location_ctx else "")
        )

        user = (
            f"Scene description: {segment.scene_description}\n\n"
            f"Start frame action: {start_action}\n"
            + (f"End frame action: {end_action}\n" if not is_static else "")
            + (f"Middle frame action: {middle_action}\n" if is_three_frame and middle_action else "")
            + f"\n{schema_desc}"
        )

        raw = await self._call_openrouter([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ])

        # Parse JSON response
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: treat entire response as start_prompt
            logger.warning("FrameSkill_Qwen: non-JSON response, falling back")
            parsed = {"start_prompt": raw}

        def _fix_prompt(p: str) -> str:
            """Ensure prompt starts with Next Scene: and has camera command injected."""
            if not p:
                return p
            p = p.strip()
            if not p.lower().startswith("next scene:"):
                p = f"Next Scene: {p}"
            if cam_cmd:
                after = p[len("Next Scene:"):].lstrip()
                if not after.startswith(cam_cmd):
                    p = f"Next Scene: {cam_cmd} {after}"
            return p

        start = _fix_prompt(parsed.get("start_prompt", ""))
        end = _fix_prompt(parsed.get("end_prompt", "")) if not is_static else ""
        middle = _fix_prompt(parsed.get("middle_prompt", ""))

        return {"start_prompt": start, "end_prompt": end, "middle_prompt": middle}


class FrameSkill_Flux2Klein(FrameSkillBase):
    """Placeholder — uses Flux 2 Klein Multi-Image workflow."""
    skill_name = "flux2klein"

    async def generate_frame_prompts(self, segment, protagonist_description, prev_segment) -> dict:
        raise NotImplementedError("Flux2Klein skill not yet implemented")


# ── VideoScriptAgent (skill-based) ────────────────────────────────────────

class VideoScriptAgent:
    """Writes per-frame Qwen IE prompts per segment using a pluggable skill."""

    SKILLS = {
        "qwen": FrameSkill_Qwen,
        "flux2klein": FrameSkill_Flux2Klein,
    }

    def __init__(self, config: dict):
        agent_cfg = config.get("film_agents", {}).get("video_script", {})
        skill_name = agent_cfg.get("skill", "qwen")
        skill_cls = self.SKILLS.get(skill_name, FrameSkill_Qwen)
        self.skill = skill_cls(config)

    async def write_segment_prompts(
        self,
        segment: "SceneSegment",
        protagonist_description: str,
        prev_segment: Optional["SceneSegment"] = None,
    ) -> "SceneSegment":
        """
        Write per-frame prompts for a segment.
        Populates segment.start_frame_prompt, end_frame_prompt, middle_frame_prompt,
        and segment.qwen_description (backward compat = start_frame_prompt).
        """
        result = await self.skill.generate_frame_prompts(segment, protagonist_description, prev_segment)
        segment.start_frame_prompt  = result.get("start_prompt", "")
        segment.end_frame_prompt    = result.get("end_prompt", segment.start_frame_prompt)
        segment.middle_frame_prompt = result.get("middle_prompt", "")
        segment.qwen_description    = segment.start_frame_prompt  # backward compat
        return segment

    # Legacy alias
    async def write_segment_prompt(
        self,
        segment: "SceneSegment",
        protagonist_description: str,
    ) -> "SceneSegment":
        return await self.write_segment_prompts(segment, protagonist_description, prev_segment=None)


# ── Video Skill Base + Implementations ────────────────────────────────────

class VideoSkillBase:
    """Base class for LTX video prompt generation skills."""
    skill_name: str = "base"

    async def generate_video_prompt(
        self,
        segment: "SceneSegment",
        protagonist_description: str,
    ) -> str:
        raise NotImplementedError


class VideoSkill_LTX23(VideoSkillBase):
    """Wraps enhance_prompt() from prompt_enhancer.py for LTX 2.3."""
    skill_name = "ltx23"

    async def generate_video_prompt(self, segment: SceneSegment, protagonist_description: str) -> str:
        return await enhance_prompt(
            segment.scene_description,
            mode="film",
            context={
                "shot_type": segment.shot_type,
                "lighting": segment.lighting,
                "camera_move": segment.camera_move,
                "mood": segment.mood,
                "protagonist": protagonist_description,
                "environmental_context": segment.environmental_context,
            }
        )


class VideoSkill_WAN22(VideoSkillBase):
    """Placeholder for WAN 2.2 video prompt generation."""
    skill_name = "wan22"

    async def generate_video_prompt(self, segment, protagonist_description) -> str:
        raise NotImplementedError("WAN 2.2 skill not yet implemented")


# ── LTXPromptAgent (skill-based, replaces PromptAgent) ────────────────────

class LTXPromptAgent:
    """Generates LTX video prompts and resolves LoRAs using a pluggable video skill."""

    SKILLS = {
        "ltx23": VideoSkill_LTX23,
        "wan22": VideoSkill_WAN22,
    }

    def __init__(self, config: dict):
        agent_cfg = config.get("film_agents", {}).get("ltx_prompt", {})
        skill_name = agent_cfg.get("skill", "ltx23")
        skill_cls = self.SKILLS.get(skill_name, VideoSkill_LTX23)
        self.skill = skill_cls()

    async def process_segment(self, segment: "SceneSegment", protagonist_description: str = "") -> "SceneSegment":
        logger.info("LTXPromptAgent → skill=%s segment %d (%s)", self.skill.skill_name, segment.index, segment.label)
        segment.prompt = await self.skill.generate_video_prompt(segment, protagonist_description)
        logger.info("LTXPromptAgent ← segment %d done (%d chars)", segment.index, len(segment.prompt))

        # Resolve LoRAs and populate segment fields
        loras = resolve_ltx_loras(segment)
        profiles = _load_profiles()
        camera_key = segment.camera_move if not segment.camera_extended else "static"
        cam = profiles.get("ltx_camera", {}).get(camera_key, {})
        segment.ltx_camera_lora = cam.get("file", "")

        slot4 = [l for l in loras if l.get("slot") == 4]
        segment.effect_lora = slot4[0]["lora"] if slot4 else ""
        return segment


# Legacy alias so any code still referencing PromptAgent doesn't break
PromptAgent = LTXPromptAgent


# ── Film Pipeline Orchestrator ─────────────────────────────────────────────

class FilmPipeline:
    """
    Orchestrates the full short film generation pipeline.
    Designed to be called from ctb.py with Telegram callback hooks.
    """

    def __init__(
        self,
        config: dict,
        comfy_caller,
        progress_cb=None,
        normalize_frame_path: Optional[Callable[[str], str]] = None,
        frame_validation_notify_fn=None,
        workflow_preprocessor: Optional[Callable] = None,
    ):
        """
        config: loaded config.json dict
        comfy_caller: async callable(workflow_name, params) -> dict
        progress_cb: optional async callable(text) for Telegram progress messages
        normalize_frame_path: optional callable(path) -> stable_filename for ComfyUI image refs.
            Applied to all keyframe outputs before storing on segment. Prevents pasted/ path issues.
        frame_validation_notify_fn: optional async callable(image_path, reason, artifact_score)
            Called by FrameValidator when artifact_score <= 2. Used for Telegram alerts.
        """
        self.config = config
        self.comfy_caller = comfy_caller
        self.progress_cb = progress_cb or (lambda msg: None)
        self.normalize_frame_path = normalize_frame_path or (lambda p: p)
        self.frame_validation_notify_fn = frame_validation_notify_fn
        self.workflow_preprocessor = workflow_preprocessor or (lambda w: w)
        self.director = DirectorAgent(config)
        self.screenplay_agent = ScreenplayAgent(config)
        self.video_script_agent = VideoScriptAgent(config)
        self.ltx_prompt_agent = LTXPromptAgent(config)

    async def _notify(self, msg: str):
        result = self.progress_cb(msg)
        if hasattr(result, "__await__"):
            await result

    def _get_session_dir(self, session_id: str) -> Path:
        d = OUTPUT_BASE / session_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _normalize(self, path: Optional[str]) -> Optional[str]:
        """Normalize a frame path through the injected normalize_frame_path callable."""
        if not path:
            return path
        try:
            return self.normalize_frame_path(path)
        except Exception as e:
            logger.warning("normalize_frame_path failed for %r: %s", path, e)
            return path

    async def _generate_multicam_refs(self, character_ref_path: str, session_dir: Path) -> Dict[str, str]:
        """Generate 8 perspective refs via F2K9B MCA workflow. Returns angle→path dict."""
        mcfg = self.config.get("film_agents", {}).get("multicam_refs", {})
        comfyui_url = self.config.get("comfyui_url", "http://127.0.0.1:8188")
        workflows_dir = Path(self.config.get("workflows_dir", "./Workflows"))
        workflow_file = workflows_dir / mcfg["workflow"]

        if not workflow_file.exists():
            raise FileNotFoundError(f"multicam_refs workflow not found: {workflow_file}")

        with open(workflow_file, encoding="utf-8") as f:
            workflow = json.load(f)

        workflow = self.workflow_preprocessor(workflow)

        img_bytes = Path(character_ref_path).read_bytes()
        fname = f"mca_ref_{int(time.time())}.png"

        async with httpx.AsyncClient(timeout=30) as client:
            up = await client.post(
                f"{comfyui_url}/upload/image",
                files={"image": (fname, img_bytes, "image/png")},
                data={"overwrite": "true"},
            )
            up.raise_for_status()
            uploaded_name = up.json().get("name", fname)

        workflow[mcfg["input_node"]]["inputs"]["image"] = uploaded_name

        async with httpx.AsyncClient(timeout=15) as client:
            pr = await client.post(
                f"{comfyui_url}/prompt",
                json={"prompt": workflow, "client_id": str(uuid.uuid4())},
            )
            pr.raise_for_status()
            prompt_id = pr.json()["prompt_id"]

        logger.info("[multicam] Submitted workflow prompt_id=%s", prompt_id)

        outputs = None
        for attempt in range(60):
            await asyncio.sleep(3)
            async with httpx.AsyncClient(timeout=10) as client:
                hist = await client.get(f"{comfyui_url}/history/{prompt_id}")
                hist_data = hist.json()
            if prompt_id in hist_data:
                job_outputs = hist_data[prompt_id].get("outputs", {})
                if job_outputs:
                    outputs = job_outputs
                    break
            logger.debug("[multicam] Waiting for outputs (attempt %d/60)...", attempt + 1)

        if not outputs:
            raise Exception(f"[multicam] Timeout: no outputs for prompt {prompt_id}")

        images = outputs.get(mcfg["output_node"], {}).get("images", [])
        angle_order = mcfg["angle_order"]
        refs_dir = session_dir / "character_refs"
        refs_dir.mkdir(exist_ok=True)

        refs: Dict[str, str] = {}
        for idx, angle in enumerate(angle_order):
            if idx >= len(images):
                logger.warning("[multicam] Missing image for angle %s (got %d)", angle, len(images))
                break
            img_info = images[idx]
            async with httpx.AsyncClient(timeout=30) as client:
                view = await client.get(
                    f"{comfyui_url}/view",
                    params={
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                    },
                )
                view.raise_for_status()
            dest = refs_dir / f"{angle}.png"
            dest.write_bytes(view.content)
            refs[angle] = str(dest)
            logger.info("[multicam] Saved %s → %s", angle, dest.name)

        return refs

    async def run(
        self,
        story: str,
        target_duration: float,
        style: str,
        director_mode: str = "auto",
        character_ref: Optional[str] = None,
        audio_path: Optional[str] = None,
        segment_review_cb: Optional[Callable] = None,
        script_review_cb: Optional[Callable] = None,
        screenplay_review_cb: Optional[Callable] = None,
        director_review_cb: Optional[Callable] = None,
        prewritten_screenplay: str = "",
        prewritten_protagonist: str = "",
        prewritten_environment: str = "",
    ) -> str:
        """
        Full pipeline run. Returns path to final_film.mp4.
        segment_review_cb: optional async callable(segment) -> bool
        """
        session_id = f"film_{int(time.time())}"
        session_dir = self._get_session_dir(session_id)
        session = FilmSession(
            session_id=session_id,
            mode="story_audio" if audio_path else "story_length",
            director_mode=director_mode,
            story=story,
            target_duration=target_duration,
            style=style,
            character_ref=character_ref or "",
            audio_path=audio_path or "",
        )

        # Step 1 — Screenplay Agent
        if prewritten_screenplay and prewritten_protagonist:
            screenplay = prewritten_screenplay
            protagonist_description = prewritten_protagonist
            environment_context = prewritten_environment
            logger.info("FilmPipeline: reusing pre-written screenplay (%d chars)", len(screenplay))
        else:
            await self._notify("✍️ Screenplay Agent writing script...")
            screenplay, protagonist_description, environment_context = await self.screenplay_agent.write(
                story, target_duration, style
            )
            logger.info(
                "ScreenplayAgent: %d chars, protagonist: %s, environment: %s",
                len(screenplay), protagonist_description[:80], environment_context[:80],
            )
        session.screenplay = screenplay
        session.protagonist_description = protagonist_description
        session.environment_context = environment_context

        # Step 1b — Screenplay Review
        if screenplay_review_cb:
            await screenplay_review_cb(screenplay, protagonist_description)

        # Step 2 — Audio segmentation (if audio mode)
        audio_segments = None
        if audio_path:
            await self._notify("Splitting audio into segments...")
            audio_segments = self._split_audio(audio_path, session_dir)

        # Step 3 — Director Agent
        await self._notify("🎬 Director Agent planning scene breakdown...")
        segments = await self.director.create_scene_breakdown(
            screenplay, target_duration, style, audio_segments
        )
        if audio_segments:
            for i, seg in enumerate(segments):
                if i < len(audio_segments):
                    seg.audio_clip = audio_segments[i].get("path", "")

        # Normalize Director output: static camera always has frame_count=1
        for seg in segments:
            if seg.camera_move == "static":
                seg.frame_count = 1

        # Step 3a — Environmental context carry-forward
        # Fill any segment where Director didn't set environmental_context from screenplay base
        base_env = environment_context
        for seg in segments:
            if not seg.environmental_context:
                seg.environmental_context = base_env
            else:
                base_env = seg.environmental_context  # Director updated it; adopt for rest

        # Optional Review Mode
        if director_mode == "review":
            await self._notify(self._format_segment_plan(segments))

        # Step 3b — Director Review
        if director_review_cb:
            await director_review_cb(segments, screenplay, protagonist_description)

        # Step 4 — Video Script Agent (per-frame Qwen prompts)
        await self._notify("🎥 Video Script Agent writing keyframe prompts...")
        for i, seg in enumerate(segments):
            prev = segments[i - 1] if i > 0 else None
            seg = await self.video_script_agent.write_segment_prompts(seg, protagonist_description, prev_segment=prev)
            segments[i] = seg

        # Step 5 — LTX Prompt Agent (video prompts)
        await self._notify("🤖 LTX Prompt Agent generating video prompts...")
        for i, seg in enumerate(segments):
            seg = await self.ltx_prompt_agent.process_segment(seg, protagonist_description)
            segments[i] = seg

        session.segments = [s.to_dict() for s in segments]
        session.save(session_dir / "film_session.json")

        # Step 5b — Script Review (optional, before any ComfyUI generation)
        if script_review_cb:
            await script_review_cb(segments, screenplay)

        # Step 5c — Multicam character refs (optional)
        mcfg = self.config.get("film_agents", {}).get("multicam_refs", {})
        if character_ref and mcfg.get("enabled"):
            try:
                await self._notify("📷 Generating character angle refs...")
                session.character_refs = await self._generate_multicam_refs(character_ref, session_dir)
                logger.info("[multicam] %d angle refs generated", len(session.character_refs))
                session.save(session_dir / "film_session.json")
            except Exception as e:
                logger.warning("[multicam] Failed, falling back to single ref: %s", e)
                session.character_refs = {}

        # Step 6 — Keyframe Agent
        keyframe_agent = KeyframeAgent(self.config, str(session_dir))
        await self._notify("Generating keyframes...")
        film_agents_cfg = dict(self.config.get("film_agents", {}))
        if self.frame_validation_notify_fn:
            film_agents_cfg["frame_validation_notify_fn"] = self.frame_validation_notify_fn
        enhancer_cfg = self.config.get("prompt_enhancer", {})

        for i, seg in enumerate(segments):
            prev = segments[i - 1] if i > 0 else None
            hard_cut = (seg.cut_type == "cut" and prev is not None and
                        seg.shot_type in ("EWS", "WS") and prev.shot_type in ("EWS", "WS"))

            ff, lf, mf = await keyframe_agent.generate_keyframes(
                seg, prev, character_ref, hard_cut, self.comfy_caller,
                film_agents_cfg=film_agents_cfg,
                enhancer_cfg=enhancer_cfg,
                character_refs=session.character_refs or None,
            )
            # Normalize paths to prevent pasted/ path issues in subsequent calls
            seg.first_frame  = self._normalize(ff) or ""
            seg.last_frame   = self._normalize(lf) or ""
            seg.middle_frame = self._normalize(mf) or ""

            # Protagonist position carry-forward: adopt from previous if Director didn't set it
            if not seg.protagonist_position and prev and prev.protagonist_position:
                seg.protagonist_position = prev.protagonist_position

            seg.status = "keyframe_ready"
            segments[i] = seg

            if segment_review_cb:
                approved = await segment_review_cb(seg)
                if not approved:
                    await self._notify(f"Regenerating keyframes for segment {i}...")
                    ff, lf, mf = await keyframe_agent.generate_keyframes(
                        seg, prev, character_ref, True, self.comfy_caller,
                        film_agents_cfg=film_agents_cfg,
                        enhancer_cfg=enhancer_cfg,
                        character_refs=session.character_refs or None,
                    )
                    seg.first_frame  = self._normalize(ff) or ""
                    seg.last_frame   = self._normalize(lf) or ""
                    seg.middle_frame = self._normalize(mf) or ""
                    seg.low_confidence = False
                    seg.validation_score = None
                    segments[i] = seg

        session.segments = [s.to_dict() for s in segments]
        session.save(session_dir / "film_session.json")

        # Step 7 — Render Agent
        clips_dir = session_dir / "clips"
        clips_dir.mkdir(exist_ok=True)
        await self._notify("Rendering video segments...")
        for i, seg in enumerate(segments):
            await self._notify(f"  Rendering segment {i + 1}/{len(segments)}: {seg.label}...")
            clip_path = str(clips_dir / f"seg_{i:03d}.mp4")
            loras = resolve_ltx_loras(seg)

            if seg.camera_move == "static":
                workflow = "LTX2.3 - IA2V" if seg.audio_clip else "LTX2.3 - I2V"
            else:
                workflow = "LTX2.3 - FFLFA2V" if seg.audio_clip else "LTX2.3 - FFLF2V"

            render_params = {
                "prompt": seg.prompt,
                "first_frame": seg.first_frame,
                "last_frame": seg.last_frame,
                "audio_clip": seg.audio_clip,
                "loras": loras,
                "duration": seg.duration,
            }

            result = await self.comfy_caller(workflow, render_params)
            seg.video_clip = result.get("output_path", clip_path)
            seg.status = "completed"
            segments[i] = seg
            session.segments = [s.to_dict() for s in segments]
            session.save(session_dir / "film_session.json")

        # Step 8 — Stitch
        await self._notify("Stitching segments...")
        final_path = str(session_dir / "film_final.mp4")
        self._stitch_segments(segments, final_path)
        session.final_video = final_path
        session.save(session_dir / "film_session.json")

        await self._notify("Film complete!")
        return final_path

    async def resume(
        self,
        session_path: str,
        resume_from: str,  # "rerender" | "render" | "keyframes"
        segment_review_cb=None,
    ) -> str:
        """Resume pipeline from existing film_session.json.

        resume_from="rerender"  — keeps keyframes, re-renders ALL clips unconditionally
        resume_from="render"    — skips clips that already exist on disk
        resume_from="keyframes" — regenerates missing keyframes, then renders missing clips
        """
        session = FilmSession.load(Path(session_path))
        session_dir = Path(session_path).parent
        clips_dir = session_dir / "clips"
        clips_dir.mkdir(exist_ok=True)

        segments = [SceneSegment.from_dict(s) for s in session.segments]

        # Normalize: static camera always has frame_count=1
        for seg in segments:
            if seg.camera_move == "static":
                seg.frame_count = 1

        # Reset any interrupted in-progress segments back to last safe checkpoint
        for seg in segments:
            if seg.status == "generating":
                seg.status = "keyframe_ready" if seg.first_frame else "pending"

        await self._notify(f"▶️ Resuming `{session.session_id}` — mode: **{resume_from}**")

        # ── Keyframe stage (only for "keyframes" mode) ────────────────
        if resume_from == "keyframes":
            keyframe_agent = KeyframeAgent(self.config)
            for i, seg in enumerate(segments):
                if seg.status in ("keyframe_ready", "completed") and Path(seg.first_frame or "").exists():
                    await self._notify(f"⏭️ Seg {i + 1}/{len(segments)}: keyframes exist, skipping")
                    continue
                await self._notify(f"🖼️ Seg {i + 1}/{len(segments)}: generating keyframes...")
                prev = segments[i - 1] if i > 0 else None
                hard_cut = prev is not None and prev.cut_type == "cut"
                first_frame, last_frame, middle_frame = await keyframe_agent.generate_keyframes(
                    seg, prev, session.character_ref or None, hard_cut,
                    self.comfy_caller, self.config, self.config,
                )
                seg.first_frame = self.normalize_frame_path(first_frame) if first_frame else first_frame
                seg.last_frame = self.normalize_frame_path(last_frame) if last_frame else last_frame
                seg.middle_frame = self.normalize_frame_path(middle_frame) if middle_frame else middle_frame
                seg.status = "keyframe_ready"
                segments[i] = seg
                if segment_review_cb:
                    await segment_review_cb(seg, i, segments)
                session.segments = [s.to_dict() for s in segments]
                session.save(Path(session_path))

        # ── Render stage ──────────────────────────────────────────────
        for i, seg in enumerate(segments):
            # "rerender" forces re-render; other modes skip existing clips
            if resume_from != "rerender" and seg.video_clip and Path(seg.video_clip).exists():
                await self._notify(f"⏭️ Seg {i + 1}/{len(segments)}: clip exists, skipping")
                continue
            await self._notify(f"🎬 Seg {i + 1}/{len(segments)}: rendering clip...")

            if seg.camera_move == "static":
                workflow = "LTX2.3 - IA2V" if seg.audio_clip else "LTX2.3 - I2V"
            else:
                workflow = "LTX2.3 - FFLFA2V" if seg.audio_clip else "LTX2.3 - FFLF2V"

            render_params = {
                "prompt": seg.prompt,
                "first_frame": seg.first_frame,
                "last_frame": seg.last_frame,
                "duration": seg.duration,
                "loras": resolve_ltx_loras(seg),
            }
            if seg.audio_clip:
                render_params["audio_clip"] = seg.audio_clip

            result = await self.comfy_caller(workflow, render_params)
            clip_path = str(clips_dir / f"seg_{i:03d}.mp4")
            Path(result["output_path"]).rename(clip_path)
            seg.video_clip = clip_path
            seg.status = "completed"
            segments[i] = seg
            session.segments = [s.to_dict() for s in segments]
            session.save(Path(session_path))

        # ── Stitch ────────────────────────────────────────────────────
        await self._notify("Stitching segments...")
        final_path = str(session_dir / "film_final.mp4")
        self._stitch_segments(segments, final_path)
        session.final_video = final_path
        session.save(Path(session_path))
        await self._notify("Film complete!")
        return final_path

    def _split_audio(self, audio_path: str, session_dir: Path) -> list:
        """Split audio into segments using MV pipeline's segment_audio."""
        from music_video_pipeline import segment_audio
        segs_dir = session_dir / "segments"
        segs_dir.mkdir(exist_ok=True)
        segments = segment_audio(audio_path, str(segs_dir))
        return [{"path": seg.audio_clip, "start": seg.start_time, "end": seg.end_time, "duration": seg.end_time - seg.start_time} for seg in segments]

    def _format_segment_plan(self, segments) -> str:
        lines = ["Director's Segment Plan:\n"]
        for seg in segments:
            ext = " [text-only camera]" if seg.camera_extended else ""
            lines.append(
                f"{seg.index + 1}. {seg.label} ({seg.duration:.0f}s)\n"
                f"  Shot: {seg.shot_type} | Camera: {seg.camera_move}{ext} | Frames: {seg.frame_count}\n"
                f"  Lighting: {seg.lighting} | Cut: {seg.cut_type} | Face: {seg.face_visible}\n"
                f"  Start: {seg.start_frame_action}\n"
                f"  End:   {seg.end_frame_action}\n"
            )
        return "\n".join(lines)

    def _stitch_segments(self, segments, output: str):
        """Concatenate video clips via ffmpeg concat demuxer."""
        clips = [s.video_clip for s in segments if s.video_clip and Path(s.video_clip).exists()]
        if not clips:
            raise RuntimeError("No video clips to stitch")
        list_file = Path(output).parent / "concat_list.txt"
        list_file.write_text("\n".join(f"file '{c}'" for c in clips), encoding="utf-8")
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy", output
        ], check=True)
