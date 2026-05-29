"""Bug #6 fix verification: SlowdownMonitor absolute step-time threshold.

The relative threshold (`dur > avg * 3.0`) cannot detect a CONSTANT slowdown
because `avg ≈ current` — it only fires on graduals. The absolute ceiling
catches initial CPU-fallback (every step takes 130s+/it from step 1).

These tests drive `_on_progress` synthetically with fixed step-time gaps and
check whether `_slowdown_detected` flips.
"""
import os
import sys
import time
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import comfyui  # noqa: E402
import comfyui_patches  # noqa: E402  — applies the monkey-patch on import


def _make_monitor():
    """Fresh monitor with the patch already applied (via import side-effect)."""
    return comfyui.SlowdownMonitor("test-prompt")


def _feed_constant_steps(monitor, dur_s: float, n_steps: int):
    """Replay `n_steps` progress callbacks spaced `dur_s` apart.

    We override `time.time` inside the patched _on_progress by advancing a
    counter and patching the module-level reference both call sites use.
    """
    base = 1_000_000.0
    fake_now = [base]

    def fake_time():
        return fake_now[0]

    # Both the original (in comfyui) and the patch (in comfyui_patches) call
    # time.time(); patching both modules' time symbol covers both.
    with patch.object(comfyui.time, "time", fake_time), \
         patch.object(comfyui_patches.time, "time", fake_time):
        for i in range(n_steps):
            monitor._on_progress(value=i + 1, max_steps=100)
            fake_now[0] += dur_s


def test_absolute_threshold_fires_on_constant_cpu_fallback():
    """130s/step from step 1 must trip the absolute ceiling (cap=120s).
    The relative threshold cannot detect this (avg ≈ current → never 3x)."""
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 120.0), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 3), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", True):
        _feed_constant_steps(monitor, dur_s=130.0, n_steps=5)
    assert monitor.slowdown_detected, "absolute threshold must fire at 130s/step (cap=120s)"


def test_absolute_threshold_does_not_fire_on_legit_long_ia2v():
    """100s/step is in the legit IA2V range (precedent: 108s/it on long segs).
    Must NOT trip — would be false-positive."""
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 120.0), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 3), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", True):
        _feed_constant_steps(monitor, dur_s=100.0, n_steps=5)
    assert not monitor.slowdown_detected, "100s/step is below cap — must not fire"


def test_absolute_threshold_does_not_fire_on_normal_speed():
    """Normal generation (~2s/step) far below cap — must not fire."""
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 120.0), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 3), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", True):
        _feed_constant_steps(monitor, dur_s=2.0, n_steps=10)
    assert not monitor.slowdown_detected, "2s/step normal speed — must not fire"


def test_absolute_threshold_respects_min_steps():
    """Even if step time exceeds cap, must wait ABSOLUTE_MIN_STEPS before firing.

    Fix 19: absolute check uses comfyui_patches.ABSOLUTE_MIN_STEPS (own knob),
    not comfyui.COMFY_SLOWDOWN_MIN_STEPS (that one governs the relative check).
    """
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 120.0), \
         patch.object(comfyui_patches, "ABSOLUTE_MIN_STEPS", 5), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 99), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", True):
        _feed_constant_steps(monitor, dur_s=130.0, n_steps=3)
        assert not monitor.slowdown_detected, "below ABSOLUTE_MIN_STEPS (5) — must not fire yet"
        _feed_constant_steps(monitor, dur_s=130.0, n_steps=3)
        assert monitor.slowdown_detected, "after ABSOLUTE_MIN_STEPS reached — must fire"


def test_absolute_fires_faster_than_relative_min_steps():
    """Fix 19 core: absolute CPU-fallback fires at ABSOLUTE_MIN_STEPS=2 even
    while the relative MIN_STEPS=3 hasn't been reached. The 339s/it LTX
    CPU-fallback scenario — must trip after 2 step-durations, not 3+."""
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 80.0), \
         patch.object(comfyui_patches, "ABSOLUTE_MIN_STEPS", 2), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 3), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", True):
        # 3 progress callbacks → 2 step durations → reaches ABSOLUTE_MIN_STEPS=2
        _feed_constant_steps(monitor, dur_s=339.0, n_steps=3)
        assert monitor.slowdown_detected, "339s/it must fire at ABSOLUTE_MIN_STEPS=2"


def test_apply_config_loads_absolute_min_steps():
    cfg = {"comfy_recovery": {"slowdown_detection": {
        "absolute_min_steps_before_detection": 4,
        "per_workflow_overrides": {"LTXV": 80.0},
    }}}
    comfyui_patches.apply_config(cfg)
    assert comfyui_patches.ABSOLUTE_MIN_STEPS == 4
    comfyui_patches.ABSOLUTE_MIN_STEPS = 2  # restore default for other tests


def test_absolute_threshold_disabled_when_enabled_false():
    """SLOWDOWN_ENABLED=False disables both relative AND absolute paths."""
    monitor = _make_monitor()
    with patch.object(comfyui, "COMFY_SLOWDOWN_ABSOLUTE_SEC", 120.0), \
         patch.object(comfyui, "COMFY_SLOWDOWN_MIN_STEPS", 3), \
         patch.object(comfyui, "COMFY_SLOWDOWN_ENABLED", False):
        _feed_constant_steps(monitor, dur_s=300.0, n_steps=10)
    assert not monitor.slowdown_detected, "disabled monitor must never fire"


# ---------------------------------------------------------------------------
# Fix 12: per-workflow threshold detection
# ---------------------------------------------------------------------------

def _wf(class_types: list[str]) -> dict:
    """Build a minimal workflow dict with the given node class_types."""
    return {str(i): {"class_type": ct, "inputs": {}} for i, ct in enumerate(class_types)}


def test_detect_threshold_for_ace_step_workflow():
    comfyui_patches.PER_WORKFLOW_THRESHOLDS = {
        "TextEncodeAceStepAudio": 10.0,
        "AceStep": 10.0,
        "LTXV": 80.0,
    }
    wf = _wf(["KSampler", "TextEncodeAceStepAudio1.5", "VAEDecode"])
    assert comfyui_patches.detect_threshold_for_workflow(wf) == 10.0


def test_detect_threshold_for_ltx_workflow():
    comfyui_patches.PER_WORKFLOW_THRESHOLDS = {
        "AceStep": 10.0,
        "LTXV": 80.0,
    }
    wf = _wf(["KSampler", "LTXVScheduler", "LTXVAudioVAEEncode"])
    assert comfyui_patches.detect_threshold_for_workflow(wf) == 80.0


def test_detect_threshold_unknown_workflow_returns_none():
    comfyui_patches.PER_WORKFLOW_THRESHOLDS = {
        "AceStep": 10.0,
        "LTXV": 80.0,
    }
    wf = _wf(["KSampler", "VAEDecode", "SomeRandomNode"])
    assert comfyui_patches.detect_threshold_for_workflow(wf) is None


def test_detect_threshold_empty_overrides_returns_none():
    comfyui_patches.PER_WORKFLOW_THRESHOLDS = {}
    wf = _wf(["TextEncodeAceStepAudio1.5"])
    assert comfyui_patches.detect_threshold_for_workflow(wf) is None


def test_apply_config_loads_per_workflow_overrides():
    cfg = {
        "comfy_recovery": {
            "slowdown_detection": {
                "per_workflow_overrides": {
                    "AceStep": 10.0,
                    "LTXV": 80.0,
                    "Flux2": 30.0,
                }
            }
        }
    }
    comfyui_patches.apply_config(cfg)
    assert comfyui_patches.PER_WORKFLOW_THRESHOLDS == {
        "AceStep": 10.0, "LTXV": 80.0, "Flux2": 30.0,
    }
    assert comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC == 0.0


def test_apply_config_legacy_single_value_when_no_overrides():
    cfg = {
        "comfy_recovery": {
            "slowdown_detection": {"absolute_step_seconds": 75.0}
        }
    }
    comfyui_patches.apply_config(cfg)
    assert comfyui_patches.PER_WORKFLOW_THRESHOLDS == {}
    assert comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC == 75.0
