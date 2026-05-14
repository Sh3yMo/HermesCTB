import json
from copy import deepcopy
from pathlib import Path


DEFAULT_REGISTRY = {
    "categories": {
        "txt2img": {
            "name": "🖼️ Text → Image",
            "description": "Generate images from text",
            "needs_input_image": False,
            "needs_prompt": True,
            "needs_resolution": True,
            "workflows": ["Flux Dev", "Z-Image"],
        },
        "img2img": {
            "name": "🎨 Image → Image",
            "description": "Edit or transform images",
            "needs_input_image": True,
            "needs_prompt": True,
            "needs_resolution": False,
            "workflows": ["Pony Cosplay-v2", "Pony Cosplay-v3", "Qwen IE Rapid", "Flux2 Klein M-I Edit"],
        },
        "img2vid": {
            "name": "🎬 Image → Video",
            "description": "Create videos from images",
            "needs_input_image": True,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": True,
            "workflows": ["LTX2.3 - I2V", "LTX2.3 - FFLF2V", "LTX2.3 (4.2) - I2V", "LTX2.3 (4.2) - FFLF2V", "LTX2.3 (4.2) - FFMFLF2V"],
        },
        "txt2vid": {
            "name": "📹 Text → Video",
            "description": "Generate videos from text",
            "needs_input_image": False,
            "needs_prompt": True,
            "needs_resolution": True,
            "needs_duration": True,
            "workflows": ["LTX2.3 - T2V", "LTX2.3 (4.2) - T2V"],
        },
        "vid2vid": {
            "name": "🎞️ Video → Video",
            "description": "Extend or transform videos",
            "needs_input_image": False,
            "needs_input_video": True,
            "needs_input_audio": False,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": True,
            "prompt_hint": (
                "📹 **Video2Video Workflow**\n\n"
                "**Prompt format:**\n"
                "1. Describe the original video\n"
                "2. Add: '*the video continues with*'\n"
                "3. Describe the continuation\n\n"
                "**Example:**\n"
                "`a woman plays guitar. the video continues with zooming out showing the full band`\n\n"
                "💡 **Note:** Clip length = original + continuation"
            ),
            "workflows": ["LTX2.3 - V2V", "LTX2.3 (4.2) - V2V"],
        },
        "va2vid": {
            "name": "🎵🎞️ Video+Audio → Video",
            "description": "Transform a video with audio guidance",
            "needs_input_image": False,
            "needs_input_video": True,
            "needs_input_audio": True,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": True,
            "prompt_hint": (
                "🎵 **Video+Audio→Video Workflow**\n\n"
                "💡 **Note:** Video length is auto-matched to audio length\n\n"
                "Describe what should happen in the video:"
            ),
            "workflows": ["LTX2.3 (4.2) - VA2V"],
        },
        "ia2vid": {
            "name": "🎵🖼️ Image+Audio → Video",
            "description": "Create videos from image and audio",
            "needs_input_image": True,
            "needs_input_video": False,
            "needs_input_audio": True,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": False,
            "prompt_hint": (
                "🎵 **Image+Audio→Video Workflow**\n\n"
                "💡 **Note:** Video length is auto-matched to audio length (max 15s)\n\n"
                "Describe what should happen in the video:"
            ),
            "workflows": ["LTX2.3 - IA2V", "LTX2.3 - FFLFA2V", "LTX2.3 (4.2) - IA2V", "LTX2.3 (4.2) - FFMFLFA2V"],
        },
        "ta2vid": {
            "name": "🎵📹 Text+Audio → Video",
            "description": "Generate videos from text with audio",
            "needs_input_image": False,
            "needs_input_video": False,
            "needs_input_audio": True,
            "needs_prompt": True,
            "needs_resolution": True,
            "needs_duration": False,
            "prompt_hint": (
                "🎵 **Text+Audio→Video Workflow**\n\n"
                "💡 **Note:** Video length is auto-matched to audio length (max 60s)\n\n"
                "Describe what should happen in the video:"
            ),
            "workflows": ["LTX2.3 - TA2V", "LTX2.3 (4.2) - TA2V"],
        },
        "tl2vid": {
            "name": "🎬📷 Text+LastFrame → Video",
            "description": "Generate videos from text with a guided last frame",
            "needs_input_image": True,
            "needs_input_video": False,
            "needs_input_audio": False,
            "needs_prompt": True,
            "needs_resolution": True,
            "needs_duration": True,
            "prompt_hint": (
                "📷 **Text+LastFrame→Video Workflow**\n\n"
                "💡 **Note:** The uploaded image will be used as the last frame of the video\n\n"
                "Describe what should happen in the video:"
            ),
            "workflows": ["LTX2.3 - TLF2V"],
        },
        "tal2vid": {
            "name": "🎵📷 Text+Audio+LastFrame → Video",
            "description": "Generate videos from text and audio with a guided last frame",
            "needs_input_image": True,
            "needs_input_video": False,
            "needs_input_audio": True,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": False,
            "prompt_hint": (
                "🎵📷 **Text+Audio+LastFrame→Video Workflow**\n\n"
                "💡 **Note:** Video length is auto-matched to audio length. The uploaded image will be used as the last frame\n\n"
                "Describe what should happen in the video:"
            ),
            "workflows": ["LTX2.3 - TALF2V"],
        },
        "audio": {
            "name": "🎵 Audio",
            "description": "Generate music and audio",
            "needs_input_image": False,
            "needs_input_video": False,
            "needs_input_audio": False,
            "needs_prompt": True,
            "needs_resolution": False,
            "needs_duration": False,
            "workflows": ["ACE-Step 1.5"],
        },
    },
    "workflow_options": {
        "Flux Dev": {
            "resolution": {"mode": "width_height", "node_id": "27"},
        },
        "Z-Image": {
            "resolution": {"mode": "aspect_ratio_label", "node_id": "28"},
        },
        "LTX2.3 - T2V": {
            "resolution": {"mode": "longer_edge_portrait", "node_id": "864", "portrait_node_id": "1577"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 - I2V": {
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 - V2V": {
            "duration": {"node_id": "971", "mode": "xi_xf"},
        },
        "LTX2.3 - IA2V": {
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 - TA2V": {
            "resolution": {"mode": "longer_edge_portrait", "node_id": "864", "portrait_node_id": "1577"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 - FFLF2V": {
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "first_frame", "label": "🖼️ First Frame", "type": "image", "node_id": "149", "required": True, "modes": ["single"]},
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 - FFLFA2V": {
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "first_frame", "label": "🖼️ First Frame", "type": "image", "node_id": "149", "required": True, "modes": ["single"]},
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 - TLF2V": {
            "resolution": {"mode": "longer_edge_portrait", "node_id": "864", "portrait_node_id": "1577"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 - TALF2V": {
            "resolution": {"mode": "longer_edge_portrait", "node_id": "864", "portrait_node_id": "1577"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 (4.2) - T2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 (4.2) - TA2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
        },
        "LTX2.3 (4.2) - I2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 (4.2) - IA2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
        },
        "LTX2.3 (4.2) - V2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
        },
        "LTX2.3 (4.2) - VA2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
        },
        "LTX2.3 (4.2) - FFMFLF2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "first_frame", "label": "🖼️ First Frame", "type": "image", "node_id": "149", "required": True, "modes": ["single"]},
                    {"id": "middle_frame", "label": "🖼️ Middle Frame", "type": "image", "node_id": "1705", "required": False, "modes": ["single"]},
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 (4.2) - FFMFLFA2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "first_frame", "label": "🖼️ First Frame", "type": "image", "node_id": "149", "required": True, "modes": ["single"]},
                    {"id": "middle_frame", "label": "🖼️ Middle Frame", "type": "image", "node_id": "1705", "required": False, "modes": ["single"]},
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "LTX2.3 (4.2) - FFLF2V": {
            "resolution": {"mode": "aspect_ratio_v2", "combo_node_id": "1774", "slider_node_id": "1688", "scale_node_id": "1775:1772"},
            "duration": {"node_id": "196", "mode": "xi_xf"},
            "input_schema": {
                "enabled": True,
                "modes": ["single"],
                "unused_slot_strategy": "keep_existing",
                "slots": [
                    {"id": "first_frame", "label": "🖼️ First Frame", "type": "image", "node_id": "149", "required": True, "modes": ["single"]},
                    {"id": "last_frame", "label": "🖼️ Last Frame", "type": "image", "node_id": "786", "required": True, "modes": ["single"]},
                ],
            },
        },
        "Flux2 Klein M-I Edit": {
            "input_schema": {
                "enabled": True,
                "modes": ["single", "multi"],
                "upscale_toggle": True,
                "unused_slot_strategy": "repeat_primary",
                "slots": [
                    {"id": "single_base", "label": "Single Base Image", "type": "image", "node_id": "54", "required": True, "modes": ["single"]},
                    {"id": "multi_image_1", "label": "Multi Image 1", "type": "image", "node_id": "33", "required": True, "modes": ["multi"]},
                    {"id": "multi_image_2", "label": "Multi Image 2", "type": "image", "node_id": "34", "required": False, "modes": ["multi"]},
                    {"id": "multi_image_3", "label": "Multi Image 3", "type": "image", "node_id": "36", "required": False, "modes": ["multi"]},
                    {"id": "multi_image_4", "label": "Multi Image 4", "type": "image", "node_id": "45", "required": False, "modes": ["multi"]},
                    {"id": "multi_pose", "label": "Pose Image (Optional)", "type": "image", "node_id": "37", "required": False, "modes": ["multi"]},
                ],
                "output_profiles": {
                    "single": {
                        "upscale_off": ["19"],
                        "upscale_off_disable_class_types": [
                            "SeedVR2VideoUpscaler",
                            "SeedVR2LoadDiTModel",
                            "SeedVR2LoadVAEModel",
                            "Image Comparer (rgthree)",
                            "LayerUtility: PurgeVRAM",
                        ],
                        "upscale_on": ["70"],
                        "upscale_on_node_input_overrides": [
                            {"node_id": "28", "inputs": {"resolution": 2560, "max_resolution": 2560}},
                            {"node_id": "73", "inputs": {"resolution": 2560, "max_resolution": 2560}},
                        ],
                    },
                    "multi": {
                        "upscale_off": ["80"],
                        "upscale_off_disable_class_types": [
                            "SeedVR2VideoUpscaler",
                            "SeedVR2LoadDiTModel",
                            "SeedVR2LoadVAEModel",
                            "Image Comparer (rgthree)",
                            "LayerUtility: PurgeVRAM",
                        ],
                        "upscale_on": ["25"],
                        "upscale_on_node_input_overrides": [
                            {"node_id": "28", "inputs": {"resolution": 2560, "max_resolution": 2560}},
                            {"node_id": "73", "inputs": {"resolution": 2560, "max_resolution": 2560}},
                        ],
                    },
                },
            },
        },
    },
    "no_prompt_workflows": ["Pony Cosplay-v2", "Pony Cosplay-v3"],
}


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_registry(registry_path: str = "workflow_registry.json") -> dict:
    path = Path(registry_path)
    if not path.exists():
        return deepcopy(DEFAULT_REGISTRY)

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    return _deep_merge(DEFAULT_REGISTRY, raw)


REGISTRY = load_registry()
WORKFLOW_CATEGORIES = REGISTRY.get("categories", {})
WORKFLOW_OPTIONS = REGISTRY.get("workflow_options", {})
NO_PROMPT_WORKFLOWS = set(REGISTRY.get("no_prompt_workflows", []))
WORKFLOW_GROUPS = REGISTRY.get("groups", {})
