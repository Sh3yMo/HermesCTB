"""Tests for PromptRelay smart-node prompt injection in comfyui.inject_prompt."""

from comfyui import build_smart_prompt, has_relay_smart_node, inject_prompt


def test_build_smart_prompt_empty_list():
    assert build_smart_prompt([]) == ""


def test_build_smart_prompt_skips_blank_beats():
    assert build_smart_prompt(["", "  ", None] * 0 + ["", "real"]) == "Scene 1:\nreal"


def test_build_smart_prompt_block_syntax_multi():
    beats = ["man sits on bench", "panda climbs lap", "panda leaps away"]
    out = build_smart_prompt(beats)
    assert out == (
        "Scene 1:\n"
        "man sits on bench\n"
        "\n"
        "Scene 2:\n"
        "panda climbs lap\n"
        "\n"
        "Scene 3:\n"
        "panda leaps away"
    )


def test_inject_prompt_targets_smart_node_smart_prompt_field():
    wf = {
        "608": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": "", "global_prompt": "", "epsilon": 0.001},
            "_meta": {"title": "Prompt Relay Encode (Smart) positive"},
        }
    }
    out = inject_prompt(wf, "Scene 1:\nfoo", global_prompt="cinematic anchor")
    assert out["608"]["inputs"]["smart_prompt"] == "Scene 1:\nfoo"
    assert out["608"]["inputs"]["global_prompt"] == "cinematic anchor"


def test_inject_prompt_smart_node_without_global_keeps_existing_global():
    wf = {
        "608": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": "", "global_prompt": "preset"},
            "_meta": {"title": "positive"},
        }
    }
    inject_prompt(wf, "new beats")  # no global_prompt arg
    assert wf["608"]["inputs"]["global_prompt"] == "preset"
    assert wf["608"]["inputs"]["smart_prompt"] == "new beats"


def test_inject_prompt_skips_negative_titled_smart_node():
    wf = {
        "608": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": "", "global_prompt": ""},
            "_meta": {"title": "Smart positive"},
        },
        "609": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": "", "global_prompt": ""},
            "_meta": {"title": "Smart negative"},
        },
    }
    inject_prompt(wf, "pos text", global_prompt="g")
    assert wf["608"]["inputs"]["smart_prompt"] == "pos text"
    assert wf["608"]["inputs"]["global_prompt"] == "g"
    assert wf["609"]["inputs"]["smart_prompt"] == ""
    assert wf["609"]["inputs"]["global_prompt"] == ""


def test_inject_prompt_clip_text_encode_backward_compat():
    wf = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["146", 0]},
            "_meta": {"title": "CLIP Text Encode positive"},
        }
    }
    inject_prompt(wf, "hello")
    assert wf["1"]["inputs"]["text"] == "hello"


def test_has_relay_smart_node_detects_smart_encoder():
    wf = {
        "608": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": ""},
            "_meta": {"title": "positive"},
        }
    }
    assert has_relay_smart_node(wf) is True


def test_has_relay_smart_node_false_on_legacy_workflow():
    wf = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["x", 0]},
            "_meta": {"title": "positive"},
        },
        "2": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["x", 0], "vae": ["y", 0]},
            "_meta": {"title": "decode"},
        },
    }
    assert has_relay_smart_node(wf) is False


def test_has_relay_smart_node_ignores_non_dict_entries():
    wf = {
        "ignored_meta": "not a node dict",
        "608": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": ""},
        },
    }
    assert has_relay_smart_node(wf) is True


def test_inject_prompt_mixed_workflow_prefers_positives_of_each_kind():
    wf = {
        "1": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["x", 0]},
            "_meta": {"title": "positive clip"},
        },
        "2": {
            "class_type": "PromptRelaySmartEncode",
            "inputs": {"smart_prompt": "", "global_prompt": ""},
            "_meta": {"title": "positive smart"},
        },
    }
    inject_prompt(wf, "shared text", global_prompt="anchor")
    assert wf["1"]["inputs"]["text"] == "shared text"
    assert wf["2"]["inputs"]["smart_prompt"] == "shared text"
    assert wf["2"]["inputs"]["global_prompt"] == "anchor"
