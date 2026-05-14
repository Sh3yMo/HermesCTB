# director_agent.py
"""
Director Agent — generates a structured scene breakdown from a story idea.
Uses film_grammar.md as system context to produce cinematically sound segment plans.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

FILM_GRAMMAR_PATH = Path(__file__).parent / "film_grammar.md"
PREFS_PATH = Path(__file__).parent / "director_preferences.json"
MAX_PREFERENCES = 30

VALID_PRIMARY_MOVES = {"static", "dolly_in", "dolly_out", "dolly_left", "dolly_right", "jib_up", "jib_down"}
VALID_EXTENDED_MOVES = {"pan_left", "pan_right", "tilt_up", "tilt_down"}

QWEN_COMMAND_MAP = {
    "dolly_in":    "将镜头设置为推镜头",
    "dolly_out":   "将镜头设置为拉镜头",
    "dolly_left":  "将镜头向左移动",
    "dolly_right": "将镜头向右移动",
    "jib_up":      "将镜头向上移动",
    "jib_down":    "将镜头向下移动",
    "static":      "",
    "pan_left":    "将镜头向左旋转45度",
    "pan_right":   "将镜头向右旋转45度",
    "tilt_up":     "将镜头向上移动",
    "tilt_down":   "将镜头向下移动",
}

# Maps shot type → Qwen multi-angle LoRA framing command.
# These are shot COMPOSITION triggers, not camera movement (which belongs to LTX).
QWEN_SHOT_TYPE_MAP = {
    "EWS": "将镜头切换到大全景",   # Extreme Wide Shot
    "WS":  "将镜头切换到全景",     # Wide Shot
    "MS":  "将镜头切换到中景",     # Medium Shot
    "MCU": "将镜头切换到中近景",   # Medium Close-Up
    "CU":  "将镜头切换到特写",     # Close-Up
    "ECU": "将镜头切换到大特写",   # Extreme Close-Up
    "OTS": "将镜头切换到过肩镜头", # Over-the-Shoulder
    "POV": "将镜头切换到主观视角", # Point of View
    "INS": "将镜头切换到插入镜头", # Insert shot
}

# Full-name aliases so the map works even if LLM returns long-form shot type names
_SHOT_TYPE_ALIASES = {
    "EXTREME WIDE SHOT": "EWS",
    "WIDE SHOT": "WS",
    "MEDIUM SHOT": "MS",
    "MEDIUM CLOSE-UP": "MCU",
    "MEDIUM CLOSE UP": "MCU",
    "CLOSE-UP": "CU",
    "CLOSE UP": "CU",
    "EXTREME CLOSE-UP": "ECU",
    "EXTREME CLOSE UP": "ECU",
    "OVER THE SHOULDER": "OTS",
    "OVER-THE-SHOULDER": "OTS",
    "POINT OF VIEW": "POV",
    "INSERT": "INS",
    "INSERT SHOT": "INS",
}


def _normalize_shot_type(raw: str) -> str:
    """Return uppercase abbreviation for a shot type, handling long-form names."""
    upper = raw.strip().upper()
    return _SHOT_TYPE_ALIASES.get(upper, upper)


@dataclass
class SceneSegment:
    index: int
    label: str
    duration: float
    scene_description: str
    shot_type: str
    camera_move: str
    qwen_camera_command: str
    camera_extended: bool
    lighting: str
    cut_type: str
    mood: str
    act: int
    pattern: str
    audio_clip: str = ""
    qwen_description: str = ""
    prompt: str = ""
    first_frame: str = ""
    last_frame: str = ""
    video_clip: str = ""
    ltx_camera_lora: str = ""
    effect_lora: str = ""
    status: str = "pending"
    low_confidence: bool = False
    validation_score: Optional[int] = None
    # Frame action descriptors — set by Director
    start_frame_action: str = ""      # visual state at frame 0
    end_frame_action: str = ""        # visual state at last frame (empty = same as start for static)
    middle_frame_action: str = ""     # midpoint for 3-keyframe shots
    # Per-frame Qwen prompts — set by VideoScriptAgent skill
    start_frame_prompt: str = ""
    end_frame_prompt: str = ""
    middle_frame_prompt: str = ""
    middle_frame: str = ""            # output path for middle keyframe
    # Scene continuity — set by Director, carried forward by FilmPipeline
    environmental_context: str = ""   # fixed scene attrs: sky, light quality, time, surface
    protagonist_position: str = ""    # spatial location (e.g. "1m from waterline, seated")
    face_visible: bool = True         # does shot show protagonist's face (incl. 3/4 + side profile)
    needs_multiangle: bool = True     # should Multi-Angle LoRA be active
    frame_count: int = 2              # 1=static, 2=movement, 3=complex move

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneSegment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class DirectorAgent:
    def __init__(self, config: dict):
        agent_cfg = config.get("film_agents", {}).get("director", {})
        self.model = agent_cfg.get("model", "qwen/qwen3-14b")
        self.max_tokens = agent_cfg.get("max_tokens", 6000)
        self.fallback_models = agent_cfg.get("fallback_models", [])
        self.api_key = config.get("prompt_enhancer", {}).get("openrouter_api_key", "") or config.get("openrouter_api_key", "")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self._grammar: str = ""

    def _load_grammar(self) -> str:
        if not self._grammar:
            self._grammar = FILM_GRAMMAR_PATH.read_text(encoding="utf-8")
        return self._grammar

    def _load_preferences(self) -> List[Dict]:
        if PREFS_PATH.exists():
            try:
                return json.loads(PREFS_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_preference(self, rule: str, source: str = "") -> None:
        prefs = self._load_preferences()
        prefs.append({"rule": rule, "source": source, "created_at": datetime.utcnow().isoformat()})
        if len(prefs) > MAX_PREFERENCES:
            prefs = prefs[-MAX_PREFERENCES:]
        PREFS_PATH.write_text(json.dumps(prefs, indent=2, ensure_ascii=False), encoding="utf-8")

    async def _call_openrouter(self, messages: list, model: str = None, max_tokens: int = None) -> str:
        _model = model or self.model
        _max_retries = 5
        _backoff = [30, 60, 120, 300, 600]
        model_candidates = [_model] + [x for x in self.fallback_models if x and x != _model]
        for model_idx, current_model in enumerate(model_candidates):
            logger.info("DirectorAgent → OpenRouter [%s] (%d messages)", current_model, len(messages))
            for attempt in range(_max_retries):
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        self.base_url,
                        headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                        json={
                            "model": current_model,
                            "messages": messages,
                            "max_tokens": max_tokens or self.max_tokens,
                            "reasoning": {"effort": "none"},
                            "provider": {"ignore": ["parasail"], "allow_fallbacks": True},
                        }
                    )
                    if resp.status_code == 429:
                        retry_after_header = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset-requests")
                        if retry_after_header:
                            try:
                                sleep_s = float(retry_after_header)
                            except ValueError:
                                sleep_s = float(_backoff[min(attempt, len(_backoff) - 1)])
                        else:
                            sleep_s = float(_backoff[min(attempt, len(_backoff) - 1)])
                        if attempt < _max_retries - 1:
                            logger.warning("DirectorAgent 429 (model=%s, attempt %d/%d), sleeping %.0fs...", current_model, attempt + 1, _max_retries, sleep_s)
                            await asyncio.sleep(sleep_s)
                            continue
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("DirectorAgent 429 retries exhausted for '%s', switching to fallback: %s", current_model, model_candidates[model_idx + 1])
                        break
                    elif resp.status_code in (404, 503):
                        if model_idx < len(model_candidates) - 1:
                            logger.warning("DirectorAgent %d (model=%s), switching to fallback: %s", resp.status_code, current_model, model_candidates[model_idx + 1])
                            break
                        resp.raise_for_status()
                    else:
                        resp.raise_for_status()
                        data = resp.json()
                        choices = data.get("choices") or []
                        if not choices:
                            raise RuntimeError(f"OpenRouter returned no choices: {str(data)[:400]}")
                        content = choices[0].get("message", {}).get("content")
                        if not content:
                            raise RuntimeError(f"OpenRouter returned empty content: {str(data)[:400]}")
                        logger.info("DirectorAgent ← response %d chars", len(content))
                        return content.strip()
        raise RuntimeError(f"OpenRouter 429: alle Modelle erschöpft {model_candidates}.")

    async def create_scene_breakdown(
        self,
        story: str,
        target_duration: float,
        style: str,
        audio_segments: Optional[List[Dict]] = None,
    ) -> List["SceneSegment"]:
        """Generate a structured list of SceneSegments from a story idea."""
        grammar = self._load_grammar()

        if audio_segments:
            segment_count = len(audio_segments)
            segment_durations = [s["duration"] for s in audio_segments]
            timing_note = (
                f"Audio mode: you MUST generate exactly {segment_count} segments. "
                f"Durations (seconds): {segment_durations}"
            )
        else:
            timing_note = (
                f"Story mode: Determine scene count and individual durations yourself based on "
                f"cinematic storytelling needs. Quick action cuts: 2-3s. Normal scenes: 4-8s. "
                f"Dramatic/slow scenes: up to 30s max per segment. "
                f"Total of all durations must not exceed {target_duration}s."
            )

        system_prompt = f"""You are a professional film director. You must use the Film Grammar rulebook below to plan every shot.

{grammar}

---
OUTPUT RULES:
- Return ONLY a valid JSON array of segment objects. No prose, no markdown fences.
- Segments are numbered starting from 1. The first segment must have index 1. Do not reserve any segment index for opening cards or protagonist shots.
- Each segment must have ALL these fields:
  index, label, duration, scene_description, shot_type, camera_move,
  qwen_camera_command, camera_extended, lighting, cut_type, mood, act, pattern,
  start_frame_action, end_frame_action, middle_frame_action,
  environmental_context, protagonist_position, face_visible, needs_multiangle, frame_count
- camera_move must be one of: static, dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, pan_left, pan_right, tilt_up, tilt_down
- camera_extended must be true ONLY for pan_left, pan_right, tilt_up, tilt_down
- qwen_camera_command: leave empty — set automatically by the pipeline from shot_type
- act: 1, 2, or 3
- pattern: letter A-G matching the scene pattern from Section 7
- Apply the Director Decision Protocol (Section 12) for every segment

NEW FRAME ACTION FIELDS:
- start_frame_action: One sentence describing the opening visual moment of this segment (what we see at frame 0)
- end_frame_action: One sentence describing the final visual moment (what we see at the last frame). Leave empty string for static shots where start = end.
- middle_frame_action: One sentence for midpoint of complex 3-keyframe shots. Leave empty string otherwise.
- frame_count: 1 for static camera, 2 for standard camera moves (dolly/jib), 3 for complex compound moves
- face_visible: true ONLY if the protagonist's own face is visible (front, 3/4, or side profile views count). Set false for ALL shots where only secondary or other characters appear — even if their faces are clearly visible — and for back shots, EWS with tiny figure, and insert/object shots.
- needs_multiangle: true whenever a specific shot composition must be enforced (MS, MCU, CU, ECU, OTS, POV shots). false only for shots where framing is secondary (EWS, WS inserts).
- environmental_context: Specific indoor/outdoor LOCATION TYPE first (e.g. "apartment kitchen", "rooftop terrace", "subway car"), then: time of day, key light quality, surface material, dominant color palette. Carry the exact location type from the screenplay ENVIRONMENT block — never rephrase or omit it. Example: "apartment kitchen, 3 AM, harsh flickering fluorescent, cracked ceramic tile, sickly yellow and cold white".
- protagonist_position: Protagonist's spatial relationship to the environment in this shot (e.g. "seated 1m from waterline", "standing center-frame against the cliff", "walking along the shoreline"). Empty for shots where protagonist is not visible.

TRIGGER WORD COMBINATIONS (for scene_description):
You may combine Qwen Multi-Angle LoRA framing triggers with directional modifiers.
Example: for a medium shot filmed from behind at 45° right, describe "medium shot from behind, camera angled 45° right" — the pipeline translates this to combined trigger words.
Be creative: describe shot angle, distance, and camera orientation for unique compositions.
In scene_description, reuse the exact location type from environmental_context — never rephrase it (e.g. "apartment kitchen" must not become "sterile kitchen", "clinical space", or any other reinterpretation)."""

        prefs = self._load_preferences()
        if prefs:
            system_prompt += "\n\n## User Preferences (learned from feedback — apply always):\n"
            system_prompt += "\n".join(f"- {p['rule']}" for p in prefs)

        user_msg = (
            f"Story: {story}\n\n"
            f"Visual style: {style}\n"
            f"Target total duration: {target_duration} seconds\n"
            f"{timing_note}\n\n"
            "Generate the scene breakdown JSON array:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        last_err: Exception | None = None
        for attempt in range(1, 3):  # up to 2 attempts
            raw = await self._call_openrouter(messages)
            try:
                return self._parse_breakdown(raw)
            except ValueError as e:
                last_err = e
                logger.warning("DirectorAgent: JSON parse failed (attempt %d/2): %s", attempt, e)
        raise last_err

    def _parse_breakdown(self, raw: str) -> List["SceneSegment"]:
        """Parse LLM JSON response into SceneSegment list."""
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        if raw.endswith("```"):
            raw = raw[:-3].strip()
        # Extract JSON array boundaries robustly (strip leading/trailing prose)
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1 and end > start:
            raw = raw[start:end + 1]
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"Director Agent: invalid JSON from LLM ({e}). Raw (first 400): {raw[:400]}") from e
        if isinstance(data, dict):
            # LLM sometimes wraps array in {"segments": [...]} or similar
            for key in ("segments", "scenes", "breakdown", "scene_breakdown"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError(f"Director Agent: expected JSON array, got {type(data).__name__}. Raw (first 400): {raw[:400]}")

        segments = []
        for item in data:
            move = item.get("camera_move", "static").lower().replace("-", "_").replace(" ", "_")
            if move not in VALID_PRIMARY_MOVES | VALID_EXTENDED_MOVES:
                move = "static"

            # Shot type drives Qwen framing command; camera_move drives LTX animation only
            shot_type = _normalize_shot_type(item.get("shot_type", ""))
            item["shot_type"] = shot_type  # normalise to abbreviation in the segment too
            item["qwen_camera_command"] = QWEN_SHOT_TYPE_MAP.get(shot_type, "")

            item["camera_move"] = move
            item["camera_extended"] = move in VALID_EXTENDED_MOVES
            item.setdefault("audio_clip", "")
            item.setdefault("prompt", "")
            item.setdefault("first_frame", "")
            item.setdefault("last_frame", "")
            item.setdefault("video_clip", "")
            item.setdefault("ltx_camera_lora", "")
            item.setdefault("effect_lora", "")
            item.setdefault("status", "pending")
            item.setdefault("start_frame_action", "")
            item.setdefault("end_frame_action", "")
            item.setdefault("middle_frame_action", "")
            item.setdefault("start_frame_prompt", "")
            item.setdefault("end_frame_prompt", "")
            item.setdefault("middle_frame_prompt", "")
            item.setdefault("middle_frame", "")
            item.setdefault("environmental_context", "")
            item.setdefault("protagonist_position", "")
            item.setdefault("face_visible", True)
            item.setdefault("needs_multiangle", True)
            item.setdefault("frame_count", 2)

            seg = SceneSegment.from_dict(item)
            segments.append(seg)

        # Renormalize indices to start at 1 (LLM sometimes starts at 2)
        if segments and segments[0].index != 1:
            offset = segments[0].index - 1
            for seg in segments:
                seg.index -= offset

        return segments

    async def refine_segment(
        self,
        segment: "SceneSegment",
        feedback: str,
        all_segments: List["SceneSegment"],
        screenplay: str,
        protagonist_description: str,
    ) -> "SceneSegment":
        """Refine a single segment based on natural language feedback. Returns updated SceneSegment."""
        grammar = self._load_grammar()
        prefs = self._load_preferences()

        context_lines = []
        for s in all_segments:
            if s.index != segment.index:
                context_lines.append(f"  Seg {s.index}: {s.shot_type}, {s.camera_move}, {s.duration:.1f}s, {s.mood}")

        system_prompt = f"""You are a professional film director refining a single scene based on user feedback.
Use the Film Grammar rulebook to ensure the result is cinematically sound.

{grammar}

---
OUTPUT RULES:
- Return ONLY a single valid JSON object (NOT an array). No prose, no markdown fences.
- The object must have ALL required fields (same as before — keep unchanged fields identical to input).
- camera_move must be one of: static, dolly_in, dolly_out, dolly_left, dolly_right, jib_up, jib_down, pan_left, pan_right, tilt_up, tilt_down
- camera_extended must be true ONLY for pan_left, pan_right, tilt_up, tilt_down
- qwen_camera_command: leave empty — set automatically by the pipeline
- Apply user feedback faithfully while preserving narrative continuity with surrounding scenes."""

        if prefs:
            system_prompt += "\n\n## User Preferences (apply always):\n"
            system_prompt += "\n".join(f"- {p['rule']}" for p in prefs)

        seg_json = json.dumps(segment.to_dict(), ensure_ascii=False, indent=2)
        other_segs = "\n".join(context_lines) if context_lines else "  (no other segments)"

        user_msg = (
            f"Screenplay excerpt:\n{screenplay[:1500]}\n\n"
            f"Protagonist: {protagonist_description}\n\n"
            f"Other segments (for continuity):\n{other_segs}\n\n"
            f"Current segment to refine:\n{seg_json}\n\n"
            f"User feedback: {feedback}\n\n"
            "Return the refined segment as a single JSON object:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]

        raw = await self._call_openrouter(messages, max_tokens=2000)
        # Normalize: wrap in array so _parse_breakdown can handle it
        raw_stripped = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        if raw_stripped.startswith("{"):
            raw_stripped = f"[{raw_stripped}]"
        segments = self._parse_breakdown(raw_stripped)
        updated = segments[0]
        # Preserve index and pipeline-set fields
        updated.index = segment.index
        updated.audio_clip = segment.audio_clip
        updated.status = segment.status
        return updated

    async def extract_preference(
        self,
        original_segment: "SceneSegment",
        feedback: str,
    ) -> Optional[str]:
        """Extract a general reusable rule from specific edit feedback. Returns rule string or None."""
        user_msg = (
            f"Original segment: shot_type={original_segment.shot_type}, "
            f"camera_move={original_segment.camera_move}, "
            f"scene='{original_segment.scene_description[:200]}'\n\n"
            f"User feedback: {feedback}\n\n"
            "Extract ONE concise general rule (max 1 sentence, English). "
            "Return only the rule text. If no general rule applies, return the word null."
        )
        messages = [
            {"role": "system", "content": (
                "You are a film director learning from user feedback. "
                "Extract a general, reusable rule from specific scene feedback. "
                "The rule should apply to future films, not just this one scene."
            )},
            {"role": "user", "content": user_msg},
        ]
        try:
            raw = await self._call_openrouter(messages, max_tokens=100)
            raw = raw.strip().strip('"').strip("'")
            if raw.lower() in ("null", "none", "n/a", ""):
                return None
            return raw
        except Exception as e:
            logger.warning("extract_preference failed: %s", e)
            return None
