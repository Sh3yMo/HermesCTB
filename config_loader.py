import json
import os
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "telegram_token": "",
    "allowed_user_ids": [],
    "comfyui_url": "http://127.0.0.1:8188",
    "workflows_dir": "./Workflows",
    "comfy_view_timeout_seconds": 60,
    "comfy_view_retries": 3,
    "telegram_send_read_timeout_seconds": 600,
    "telegram_send_write_timeout_seconds": 600,
    "telegram_send_connect_timeout_seconds": 30,
    "telegram_send_pool_timeout_seconds": 30,
    "telegram_send_retries": 2,
    "telegram_send_retry_delay_seconds": 3,
    "comfy_recovery": {
        "enabled": False,
        "restart_command": "",
        "max_restarts_per_job": 1,
        "healthcheck_timeout_seconds": 120,
        "healthcheck_interval_seconds": 2,
        "queue_error_threshold": 8,
        "job_timeout_seconds": 2400,
        "history_retry_attempts": 60,
        "history_retry_delay_seconds": 3,
        "slowdown_detection": {
            "enabled": False,
            "threshold_multiplier": 3.0,
            "grace_period_seconds": 120,
            "min_steps_before_detection": 3,
        },
    },
    "lora_automation": {
        "enabled": False,
        "workflow_name_contains": [],
        "motion_map": {},
        "pattern_map": {},
    },
    "film_agents": {
        "frame_validation_enabled": False,
        "frame_validation_model": "",
        "frame_validation_threshold": 3,
        "frame_validation_max_retries": 2,
        "frame_validation_auto_retry": True,
        "multicam_refs": {
            "enabled": False,
            "workflow": "F2K9B MCA.json",
            "input_node": "76",
            "output_node": "9",
            "angle_order": [
                "close_up", "wide", "right_45", "right_profile",
                "aerial", "low_angle", "left_45", "left_profile"
            ],
        },
    },
    "prompt_enhancer": {
        "enabled": True,
        "openrouter_enabled": True,
        "openrouter_api_key": "",
        "openrouter_model": "qwen/qwen3-next-80b-a3b-instruct:free",
        "fallback_models": ["openrouter/free"],
        "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions",
        "request_timeout_seconds": 30,
        "max_retries": 3,
        "retry_backoff_seconds": [1, 3, 8],
        "disable_reasoning": True,
        "reasoning_effort": "none",
        "reasoning_exclude": True,
        "translate_on_local_fallback": True,
        "translation_max_tokens": 180,
        "enhancement_max_tokens": 700,
        "cinematic_writing_mode": "strict_cinematic",
        "vision_enabled": False,
        "vision_model": "qwen/qwen-2.5-vl-7b-instruct",
        "vision_fallback_models": [],
        "vision_max_tokens": 500,
        "vision_image_detail": "high",
        "unusable_output_failures_for_cooldown": 1,
        "unusable_output_cooldown_seconds": 900,
        "whisper_enabled": True,
        "whisper_api_key": "",
        "whisper_model": "whisper-large-v3-turbo",
        "whisper_endpoint": "https://api.groq.com/openai/v1/audio/transcriptions",
        "idea_mode_enabled": True,
        "idea_mode_ask_questions": False,
        "idea_mode_max_questions": 2,
        "audio_llm_model": "qwen/qwen3.5-plus-02-15",
        "genre_validation_model": "qwen/qwen3-next-80b-a3b-instruct:free",
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str = "config.json") -> dict:
    path = Path(config_path)
    if not path.exists():
        return deepcopy(DEFAULT_CONFIG)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    config = _deep_merge(DEFAULT_CONFIG, raw)
    config["allowed_user_ids"] = [int(x) for x in config.get("allowed_user_ids", [])]

    # Keep the runtime secret out of config.json entirely: if the key is blank
    # (template / deployment), fall back to the OPENROUTER_API_KEY env var.
    enhancer = config.setdefault("prompt_enhancer", {})
    if not enhancer.get("openrouter_api_key"):
        enhancer["openrouter_api_key"] = os.environ.get("OPENROUTER_API_KEY", "")

    return config
