# keyframe_agent.py
"""
Keyframe Agent — generates first/last frame pairs per segment via Qwen IE Rapid.
Manages character consistency (Next Scene LoRA) and angle changes (Multi-Angle LoRA).
"""

import json
from pathlib import Path
from typing import Dict, Optional

from director_agent import SceneSegment
from frame_validator import FrameValidator, ValidationContext

QWEN_WORKFLOW_NAME = "Qwen IE Rapid"

# ── Multicam angle selection ────────────────────────────────────────────────

_SHOT_TO_ANGLE: Dict[str, str] = {
    "ECU": "close_up", "CU": "close_up", "MCU": "close_up", "BCU": "close_up",
    "WS": "wide", "EWS": "wide", "MS": "wide", "MLS": "wide",
    "OTS": "right_45",
}

_POSITION_TO_ANGLE: Dict[str, str] = {
    "left profile": "left_profile",   "profile left": "left_profile",
    "right profile": "right_profile", "profile right": "right_profile",
    "turned left": "left_45",         "45 left": "left_45",
    "turned right": "right_45",       "45 right": "right_45",
    "aerial": "aerial",               "bird": "aerial",    "overhead": "aerial",
    "low angle": "low_angle",         "worm": "low_angle",
}


def _select_angle(segment: "SceneSegment") -> str:
    shot = (segment.shot_type or "").upper().strip()
    position = (segment.protagonist_position or "").lower()
    if shot in _SHOT_TO_ANGLE:
        return _SHOT_TO_ANGLE[shot]
    for key, angle in _POSITION_TO_ANGLE.items():
        if key in position:
            return angle
    return "close_up"

LORA_PROFILES_PATH = Path(__file__).parent / "lora_profiles.json"


def _load_keyframe_loras() -> dict:
    profiles = json.loads(LORA_PROFILES_PATH.read_text(encoding="utf-8"))
    return profiles.get("keyframe_loras", {})


def build_qwen_lora_list(has_character: bool, needs_multiangle: bool = True) -> list:
    """Build the LoRA list for a Qwen IE Rapid call.

    Both LoRAs gate on has_character: Next Scene enforces character continuity,
    Multi-Angle enforces shot composition framing every time (not only on angle changes).
    """
    if not has_character:
        return []
    loras = _load_keyframe_loras()
    result = []
    if "qwen_next_scene" in loras:
        result.append({
            "lora": loras["qwen_next_scene"]["file"],
            "weight": loras["qwen_next_scene"]["weight"]
        })
    if needs_multiangle and "qwen_multiangle" in loras:
        result.append({
            "lora": loras["qwen_multiangle"]["file"],
            "weight": loras["qwen_multiangle"]["weight"]
        })
    return result


def _build_qwen_prompt(segment: "SceneSegment") -> str:
    """Build a Qwen prompt. Uses VideoScriptAgent output if available, falls back to rule-based construction."""
    # If VideoScriptAgent wrote a prompt, use it directly (it already starts with "Next Scene:")
    if segment.qwen_description:
        return segment.qwen_description

    # Fallback: rule-based construction
    parts = ["Next Scene:"]
    if segment.qwen_camera_command:
        parts.append(segment.qwen_camera_command)
    parts.append(segment.scene_description)
    if segment.lighting:
        parts.append(segment.lighting)
    if segment.mood:
        parts.append(segment.mood)
    return " ".join(parts)


class KeyframeAgent:
    def __init__(self, config: dict, session_dir: str):
        self.config = config
        self.session_dir = Path(session_dir)
        self.keyframe_dir = self.session_dir / "keyframes"
        self.keyframe_dir.mkdir(parents=True, exist_ok=True)

    # ── Frame count ────────────────────────────────────────────────────────

    def _determine_frame_count(self, segment: SceneSegment) -> int:
        """1 for static, otherwise use segment.frame_count (Director-set, 2 or 3)."""
        if segment.camera_move == "static":
            return 1
        return max(1, segment.frame_count)

    # ── Reference selection ────────────────────────────────────────────────

    def _get_frame_input(
        self,
        segment: SceneSegment,
        prev_segment: Optional[SceneSegment],
        hard_cut: bool,
        character_ref: Optional[str],
        is_start: bool,
        character_refs: Optional[Dict[str, str]] = None,
    ) -> Optional[str]:
        """
        Determine which image to use as reference for a single frame generation call.

        Face-visible shots: always use clean character_ref for BOTH start and end frames
        so protagonist identity is anchored to the master reference, not a generated frame.
        If character_refs (multicam) is provided, select the angle matching shot/position.

        Non-face shots: start uses legacy continuity logic; end uses prev last_frame.
        """
        if segment.face_visible:
            # Anchor to master reference — use angle-specific ref if available
            if character_refs:
                angle = _select_angle(segment)
                return character_refs.get(angle) or character_ref
            return character_ref  # may be None if no ref provided

        # Non-face shot — continuity from previous segment is safe
        if prev_segment is None:
            return character_ref
        if hard_cut or not prev_segment.last_frame:
            return character_ref  # hard cut or no prev frame → fall back to master reference
        if is_start:
            return prev_segment.last_frame
        # End frame: same continuity source
        return prev_segment.last_frame

    # Legacy alias used by existing callers
    def get_first_frame_input(
        self,
        segment: SceneSegment,
        prev_segment: Optional[SceneSegment],
        hard_cut: bool,
        character_ref: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=True)

    # ── Single-frame generation ────────────────────────────────────────────

    async def _generate_single_frame(
        self,
        prompt: str,
        frame_input: Optional[str],
        loras: list,
        comfy_caller,
        film_agents_cfg: Optional[dict],
        enhancer_cfg: Optional[dict],
        segment: SceneSegment,
    ) -> Optional[str]:
        """Call Qwen IE Rapid once with a specific prompt and input image."""
        params: dict = {
            "TextEncodeQwenImageEditPlus": {"prompt": prompt},
            "loras": loras,
        }
        if frame_input:
            params["Load Image"] = {"image": frame_input}

        async def _generate() -> tuple:
            result = await comfy_caller(QWEN_WORKFLOW_NAME, params)
            if not isinstance(result, dict):
                raise TypeError(
                    f"KeyframeAgent: comfy_caller returned {type(result).__name__}, expected dict. "
                    f"Segment index: {segment.index}, workflow: {QWEN_WORKFLOW_NAME}"
                )
            qwen_output = result.get("output_path")
            actual_first = frame_input if frame_input else qwen_output
            # paths[0] must be a local file path for the validator to score.
            # qwen_output is always a full absolute path; frame_input may be a
            # bare ComfyUI input filename (not readable locally).  So always put
            # qwen_output first so the validator reads the right file.
            return qwen_output, actual_first

        validator = FrameValidator(film_agents_cfg or {}, enhancer_cfg or {})
        ctx = ValidationContext(
            scene_description=segment.scene_description,
            shot_type=segment.shot_type,
            lighting=segment.lighting,
            mood=segment.mood,
        )
        ff, lf, vresult = await validator.validate_and_maybe_retry(_generate, ctx)

        if vresult is not None and not vresult.passed:
            segment.low_confidence = True
            segment.validation_score = vresult.score
            print(
                f"[FrameValidator] Segment {segment.index} BELOW threshold "
                f"(score={vresult.score}/5): {vresult.reason}"
            )

        # _generate returns (qwen_output, actual_first); ff = qwen_output
        return ff  # ff = qwen_output (the newly generated frame)

    # ── Prompt selection per frame ─────────────────────────────────────────

    @staticmethod
    def _pick_frame_prompt(segment: SceneSegment, frame: str) -> str:
        """
        Pick the appropriate per-frame prompt.
        frame: "start" | "end" | "middle" | "only"
        Falls back to qwen_description if per-frame prompts not set.
        """
        if frame in ("start", "only"):
            return segment.start_frame_prompt or segment.qwen_description or _build_qwen_prompt(segment)
        if frame == "end":
            return segment.end_frame_prompt or segment.start_frame_prompt or segment.qwen_description or _build_qwen_prompt(segment)
        if frame == "middle":
            return segment.middle_frame_prompt or segment.start_frame_prompt or segment.qwen_description or _build_qwen_prompt(segment)
        return segment.qwen_description or _build_qwen_prompt(segment)

    # ── Main entry point ───────────────────────────────────────────────────

    async def generate_keyframes(
        self,
        segment: SceneSegment,
        prev_segment: Optional[SceneSegment],
        character_ref: Optional[str],
        hard_cut: bool,
        comfy_caller,
        film_agents_cfg: Optional[dict] = None,
        enhancer_cfg: Optional[dict] = None,
        character_refs: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """
        Generate keyframes for a segment.
        Returns (first_frame_path, last_frame_path, middle_frame_path_or_None).

        - frame_count=1 (static): single frame, returned as both first and last
        - frame_count=2: separate start + end frames
        - frame_count=3: start + middle + end frames
        """
        has_character = character_ref is not None
        loras = build_qwen_lora_list(has_character, needs_multiangle=True)
        frame_count = self._determine_frame_count(segment)

        kw = dict(
            loras=loras, comfy_caller=comfy_caller,
            film_agents_cfg=film_agents_cfg, enhancer_cfg=enhancer_cfg,
            segment=segment,
        )

        if frame_count == 1:
            # Static shot: one frame serves as both first and last
            ref = self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=True, character_refs=character_refs)
            prompt = self._pick_frame_prompt(segment, "only")
            ff = await self._generate_single_frame(prompt, ref, **kw)
            return ff, ff, None

        elif frame_count >= 3 and segment.middle_frame_action:
            # 3-keyframe: start → middle → end
            start_ref = self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=True,  character_refs=character_refs)
            end_ref   = self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=False, character_refs=character_refs)
            ff = await self._generate_single_frame(self._pick_frame_prompt(segment, "start"), start_ref, **kw)
            mf = await self._generate_single_frame(self._pick_frame_prompt(segment, "middle"), start_ref, **kw)
            lf = await self._generate_single_frame(self._pick_frame_prompt(segment, "end"),   end_ref,   **kw)
            return ff, lf, mf

        else:
            # 2-keyframe: start + end
            start_ref = self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=True,  character_refs=character_refs)
            end_ref   = self._get_frame_input(segment, prev_segment, hard_cut, character_ref, is_start=False, character_refs=character_refs)
            ff = await self._generate_single_frame(self._pick_frame_prompt(segment, "start"), start_ref, **kw)
            lf = await self._generate_single_frame(self._pick_frame_prompt(segment, "end"),   end_ref,   **kw)
            return ff, lf, None

    # Legacy alias for backward compatibility with any external callers
    async def generate_frame_pair(
        self,
        segment: SceneSegment,
        prev_segment: Optional[SceneSegment],
        character_ref: Optional[str],
        hard_cut: bool,
        comfy_caller,
        film_agents_cfg: Optional[dict] = None,
        enhancer_cfg: Optional[dict] = None,
        character_refs: Optional[Dict[str, str]] = None,
    ) -> tuple:
        """Legacy wrapper — returns (first_frame, last_frame) only."""
        ff, lf, _mf = await self.generate_keyframes(
            segment, prev_segment, character_ref, hard_cut, comfy_caller,
            film_agents_cfg=film_agents_cfg, enhancer_cfg=enhancer_cfg,
            character_refs=character_refs,
        )
        return ff, lf
