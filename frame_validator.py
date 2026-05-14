# frame_validator.py
"""
Vision LLM post-generation frame validator.
Checks generated images against scene descriptions via OpenRouter.
Supports auto-retry (keep best-scoring result) or flag-and-continue mode.
"""

import asyncio
import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Awaitable, Optional

import httpx


@dataclass
class ValidationContext:
    scene_description: str
    shot_type: str = ""
    lighting: str = ""
    mood: str = ""


@dataclass
class ValidationResult:
    score: int        # 1-5
    reason: str
    passed: bool      # score >= threshold
    attempts: int


_SCORE_RE = re.compile(r'"score"\s*:\s*(\d)')
_ARTIFACT_RE = re.compile(r'"artifact_score"\s*:\s*(\d)')
_MATCH_RE = re.compile(r'"match_score"\s*:\s*(\d)')

_PROMPT_TEMPLATE = (
    "Evaluate this AI-generated image on TWO dimensions and reply ONLY with compact JSON.\n"
    "1. ARTIFACT: Are there visible generation artifacts? "
    "(anatomical distortions, extra limbs/faces, blob growths, melting features, "
    "corrupted textures, duplicate body parts, impossible anatomy) "
    "— score 1=severe artifacts, 5=perfectly clean image\n"
    "2. MATCH: Does the image match the scene description? "
    "Description: {scene_description} | Shot: {shot_type} | "
    "Lighting: {lighting} | Mood: {mood} "
    "— score 1=no match, 5=perfect match\n"
    'Reply ONLY with compact JSON: {{"artifact_score": N, "match_score": N, "reason": "..."}}'
)


class FrameValidator:
    def __init__(self, cfg: dict, enhancer_cfg: dict):
        self.enabled = bool(cfg.get("frame_validation_enabled", False))
        self.threshold = int(cfg.get("frame_validation_threshold", 3))
        self.max_retries = int(cfg.get("frame_validation_max_retries", 2))
        self.auto_retry = bool(cfg.get("frame_validation_auto_retry", True))
        self.notify_fn = cfg.get("frame_validation_notify_fn")  # Optional async callable(path, reason, score)
        self.retry_delay = float(cfg.get("frame_validation_retry_delay", 10.0))

        model = str(cfg.get("frame_validation_model", "")).strip()
        if not model:
            model = str(enhancer_cfg.get("vision_model", "qwen/qwen-2.5-vl-7b-instruct")).strip()
        self.model = model

        fallbacks = enhancer_cfg.get("vision_fallback_models", [])
        self.fallback_models = [m for m in (fallbacks or []) if m and m != self.model]

        self.api_key = str(enhancer_cfg.get("openrouter_api_key", "")).strip()
        self.base_url = str(
            enhancer_cfg.get("openrouter_base_url", "https://openrouter.ai/api/v1/chat/completions")
        ).strip()
        self.max_tokens = int(enhancer_cfg.get("vision_max_tokens", 150))
        self.image_detail = str(enhancer_cfg.get("vision_image_detail", "high")).strip()
        self.timeout = int(enhancer_cfg.get("request_timeout_seconds", 30))

    async def validate_and_maybe_retry(
        self,
        generate_fn: Callable[[], Awaitable[tuple]],
        context: ValidationContext,
    ) -> tuple:
        """
        Call generate_fn() once (or up to max_retries+1 times if validation fails).
        Returns (first_frame_path, last_frame_path, ValidationResult | None).
        ValidationResult is None when validation is disabled.
        """
        if not self.enabled:
            paths = await generate_fn()
            return paths[0], paths[1], None

        best_result: Optional[ValidationResult] = None
        best_paths: Optional[tuple] = None
        attempts = 0

        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                paths = await generate_fn()
            except Exception as e:
                if best_paths is not None:
                    print(f"[FrameValidator] Retry {attempt} generation failed: {e}")
                    if attempt > self.max_retries:
                        print(f"[FrameValidator] Max retries exhausted after generation failure, keeping best prior result")
                        break
                    print(f"[FrameValidator] Will retry again in {self.retry_delay}s... (attempt {attempt}/{self.max_retries + 1})")
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise  # First attempt failed — nothing to fall back to
            result = await self._score_image_path(paths[0], context, attempt)

            if best_result is None or result.score > best_result.score:
                best_result = result
                best_paths = paths

            if result.score >= self.threshold:
                break

            if not self.auto_retry or attempt > self.max_retries:
                break

            # Give ComfyUI time to release VRAM before re-submitting
            print(f"[FrameValidator] Score {result.score}/5 below threshold {self.threshold}, "
                  f"retrying in {self.retry_delay}s... (attempt {attempt}/{self.max_retries + 1})")
            await asyncio.sleep(self.retry_delay)

        return best_paths[0], best_paths[1], best_result

    async def score_image_bytes(
        self,
        img_bytes: bytes,
        context: ValidationContext,
        image_path: Optional[str] = None,
    ) -> ValidationResult:
        """Validate raw image bytes (used for T2I Point B in ctb.py)."""
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return await self._score_b64(b64, context, attempt=1, image_path=image_path)

    async def _score_image_path(
        self,
        image_path: str,
        context: ValidationContext,
        attempt: int,
    ) -> ValidationResult:
        img_bytes = Path(image_path).read_bytes()
        b64 = base64.b64encode(img_bytes).decode("ascii")
        return await self._score_b64(b64, context, attempt, image_path=image_path)

    async def _score_b64(
        self,
        b64: str,
        context: ValidationContext,
        attempt: int,
        image_path: Optional[str] = None,
    ) -> ValidationResult:
        prompt_text = _PROMPT_TEMPLATE.format(
            scene_description=context.scene_description[:400],
            shot_type=context.shot_type,
            lighting=context.lighting,
            mood=context.mood,
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": self.image_detail,
                        },
                    },
                ],
            }
        ]

        models_to_try = [self.model] + self.fallback_models
        last_error: Optional[Exception] = None

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for model_name in models_to_try:
                try:
                    print(f"[FrameValidator] Vision try: {model_name} (attempt {attempt})")
                    resp = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": model_name,
                            "messages": messages,
                            "max_tokens": self.max_tokens,
                        },
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        or ""
                    )
                    score, artifact_score, reason = self._parse_response(text)
                    passed = score >= self.threshold
                    if artifact_score <= 2:
                        print(f"[FrameValidator] ARTIFACT DETECTED (artifact={artifact_score}/5): {reason}")
                        if self.notify_fn and image_path:
                            try:
                                await self.notify_fn(image_path, reason, artifact_score)
                            except Exception as ne:
                                print(f"[FrameValidator] notify_fn failed: {ne}")
                    print(f"[FrameValidator] score={score}/5 (artifact={artifact_score} match={score}+) passed={passed}: {reason}")
                    return ValidationResult(score=score, reason=reason, passed=passed, attempts=attempt)
                except Exception as e:
                    print(f"[FrameValidator] Vision failed ({model_name}): {e}")
                    last_error = e
                    continue

        # All models failed — pass through with neutral score to avoid blocking pipeline
        print(f"[FrameValidator] All models failed, passing through. Last error: {last_error}")
        return ValidationResult(score=self.threshold, reason="validation unavailable", passed=True, attempts=attempt)

    @staticmethod
    def _parse_response(text: str) -> tuple:
        """Parse VLM response JSON. Returns (combined_score, artifact_score, reason).
        combined_score = min(artifact_score, match_score) so a severe artifact always fails.
        Backwards-compatible: if only 'score' is present, uses it for both dimensions.
        """
        try:
            data = json.loads(text.strip())
            reason = str(data.get("reason", ""))
            if "artifact_score" in data or "match_score" in data:
                artifact_score = max(1, min(5, int(data.get("artifact_score", 3))))
                match_score = max(1, min(5, int(data.get("match_score", 3))))
                combined = min(artifact_score, match_score)
                return combined, artifact_score, reason
            # Legacy single-score format
            score = max(1, min(5, int(data.get("score", 3))))
            return score, score, reason
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        # Regex fallbacks
        ma = _ARTIFACT_RE.search(text)
        mm = _MATCH_RE.search(text)
        if ma and mm:
            a = max(1, min(5, int(ma.group(1))))
            m = max(1, min(5, int(mm.group(1))))
            return min(a, m), a, "(parse error)"
        ms = _SCORE_RE.search(text)
        if ms:
            s = max(1, min(5, int(ms.group(1))))
            return s, s, "(parse error)"
        return 3, 3, "(unparseable response)"
