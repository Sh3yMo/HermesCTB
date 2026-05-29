"""Runtime patches for comfyui module.

WHY THIS EXISTS: an OS-level lock on `comfyui.py` (Windows Defender + Docker-
WSL2 bind-mount handles) prevented direct edits during a debug session, so the
Bug #6 fix (absolute step-time threshold in SlowdownMonitor) is applied as a
monkey-patch here instead. Imported by `api.py` after `init_config`.

The patch adds:
1. `COMFY_SLOWDOWN_ABSOLUTE_SEC` config global (default 0 = disabled).
2. Per-workflow `absolute_step_seconds` override loaded from
   `comfy_recovery.slowdown_detection.per_workflow_overrides` in config.json.
3. Wrapped `SlowdownMonitor._on_progress` that fires _slowdown_detected when
   a single step exceeds the absolute ceiling — catches initial CPU-fallback
   that the relative `threshold_multiplier * avg` rule cannot detect when
   slowdown is constant from step 1 (avg ≈ current → never triggers).
4. Wrapped `queue_and_wait_with_recovery` that scans the workflow's class_types,
   selects the matching per-workflow threshold, applies it to the global for
   the duration of the queue, and restores the previous value on exit.

Once `comfyui.py` becomes writable again, fold this into the module and
delete this file.
"""
from __future__ import annotations

import sys
import time
from typing import Optional

import comfyui

# ---------------------------------------------------------------------------
# Config global + loader
# ---------------------------------------------------------------------------

# Default 0 disables absolute detection until per-workflow override fires.
comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = 0.0

# class_type substring → absolute_step_seconds. First substring match wins.
PER_WORKFLOW_THRESHOLDS: dict[str, float] = {}

# Fix 19: absolute CPU-fallback detection needs FEWER samples than the relative
# check. A constant CPU-fallback is slow from step 1 — no avg-baseline needed.
# Default 2 (fire on the 2nd step duration). Relative check keeps MIN_STEPS=3.
ABSOLUTE_MIN_STEPS: int = 2


def apply_config(config: dict) -> None:
    """Read per_workflow_overrides from config and write the module globals.

    Call this once after `comfyui.init_config(config)` in api.py.
    """
    global PER_WORKFLOW_THRESHOLDS, ABSOLUTE_MIN_STEPS
    rv = config.get("comfy_recovery", {})
    sd = rv.get("slowdown_detection", {})
    overrides = sd.get("per_workflow_overrides", {}) or {}
    PER_WORKFLOW_THRESHOLDS = {str(k): float(v) for k, v in overrides.items()}
    ABSOLUTE_MIN_STEPS = int(sd.get("absolute_min_steps_before_detection", ABSOLUTE_MIN_STEPS))
    # Legacy single value is honoured as a final fallback if the dict is empty.
    legacy = sd.get("absolute_step_seconds")
    if legacy is not None and not PER_WORKFLOW_THRESHOLDS:
        comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = float(legacy)
    else:
        comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = 0.0


def detect_threshold_for_workflow(workflow: dict) -> Optional[float]:
    """Scan a ComfyUI workflow dict for known class_type substrings and return
    the matching `absolute_step_seconds`. Returns None when nothing matches.

    Substring match (case-sensitive) lets one config entry cover families:
    "LTXV" matches `LTXVScheduler`, `LTXVAudioVAEEncode`, etc.
    """
    if not isinstance(workflow, dict) or not PER_WORKFLOW_THRESHOLDS:
        return None
    class_types: set[str] = set()
    for node in workflow.values():
        if isinstance(node, dict):
            ct = node.get("class_type")
            if isinstance(ct, str):
                class_types.add(ct)
    for needle, threshold in PER_WORKFLOW_THRESHOLDS.items():
        for ct in class_types:
            if needle in ct:
                return threshold
    return None


# ---------------------------------------------------------------------------
# Monkey-patch SlowdownMonitor._on_progress
# ---------------------------------------------------------------------------

_orig_on_progress = comfyui.SlowdownMonitor._on_progress


def _patched_on_progress(self, value: int, max_steps: int) -> None:
    """Wrap the original to ADD an absolute step-time check.

    Original behaviour (relative threshold) runs first; if it didn't trigger,
    we apply the absolute check. Both share `self._slowdown_detected` and the
    grace period, so the existing restart pipeline picks up either trigger.
    """
    # Snapshot last-step-ts BEFORE original mutates it, so we can compute the
    # same `dur` the original uses for the threshold check.
    with self._lock:
        prev_ts = self._last_step_ts
        already_detected = self._slowdown_detected

    _orig_on_progress(self, value, max_steps)

    if already_detected:
        return  # original already fired, nothing more to do

    if prev_ts is None:
        return  # first sample — no step duration yet

    now = time.time()
    dur = now - prev_ts
    abs_cap = comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC

    with self._lock:
        if self._slowdown_detected:
            return  # original triggered after we released the lock
        if (
            comfyui.COMFY_SLOWDOWN_ENABLED
            and abs_cap > 0
            and len(self.step_times) >= ABSOLUTE_MIN_STEPS
            and dur > abs_cap
        ):
            self._slowdown_detected = True
            self._slowdown_ts = now
            print(
                f"[SlowdownMonitor] ABSOLUTE threshold tripped: "
                f"step {value}/{max_steps} took {dur:.1f}s > cap {abs_cap:.1f}s "
                f"(constant slowdown / likely CPU-fallback)"
            )


comfyui.SlowdownMonitor._on_progress = _patched_on_progress


# ---------------------------------------------------------------------------
# Wrap queue_and_wait_with_recovery: apply per-workflow threshold
# ---------------------------------------------------------------------------

_orig_queue_and_wait = comfyui.queue_and_wait_with_recovery


async def _patched_queue_and_wait_with_recovery(workflow: dict, job_timeout=None):
    """Apply the per-workflow absolute_step_seconds for the duration of one
    queue. Snapshots the previous value and restores it on exit so unrelated
    queues are not affected.
    """
    detected = detect_threshold_for_workflow(workflow)
    previous = comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC
    if detected is None:
        # No match — disable absolute check (relative still applies) and log.
        comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = 0.0
        class_types = sorted({
            n.get("class_type", "?") for n in workflow.values()
            if isinstance(n, dict) and "class_type" in n
        })
        print(
            f"[slowdown] no per_workflow_overrides match for workflow "
            f"class_types={class_types[:8]}{'...' if len(class_types) > 8 else ''} "
            f"— absolute check DISABLED for this queue (relative still active)"
        )
    else:
        comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = detected
        print(f"[slowdown] using absolute_step_seconds={detected:.1f}s for this queue")
    try:
        return await _orig_queue_and_wait(workflow, job_timeout)
    finally:
        comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = previous


comfyui.queue_and_wait_with_recovery = _patched_queue_and_wait_with_recovery


def _rebind_in_callers() -> None:
    """Re-bind the patched function in modules that already imported the
    original via `from comfyui import queue_and_wait_with_recovery`. Necessary
    because that import grabs a reference to the original function object;
    patching the comfyui module attribute alone does not update the caller's
    local name.
    """
    for mod_name in list(sys.modules.keys()):
        mod = sys.modules.get(mod_name)
        if mod is None or mod is comfyui:
            continue
        cur = getattr(mod, "queue_and_wait_with_recovery", None)
        if cur is _orig_queue_and_wait:
            setattr(mod, "queue_and_wait_with_recovery", _patched_queue_and_wait_with_recovery)


_rebind_in_callers()
