"""Runtime patches for comfyui module.

WHY THIS EXISTS: an OS-level lock on `comfyui.py` (Windows Defender + Docker-
WSL2 bind-mount handles) prevented direct edits during a debug session, so the
Bug #6 fix (absolute step-time threshold in SlowdownMonitor) is applied as a
monkey-patch here instead. Imported by `api.py` after `init_config`.

The patch adds:
1. `COMFY_SLOWDOWN_ABSOLUTE_SEC` config global (default 120s, configurable via
   `comfy_recovery.slowdown_detection.absolute_step_seconds` in config.json).
2. Wrapped `SlowdownMonitor._on_progress` that fires _slowdown_detected when
   a single step exceeds the absolute ceiling — catches initial CPU-fallback
   that the relative `threshold_multiplier * avg` rule cannot detect when
   slowdown is constant from step 1 (avg ≈ current → never triggers).

Once `comfyui.py` becomes writable again, fold this into the module and
delete this file.
"""
from __future__ import annotations

import time

import comfyui

# ---------------------------------------------------------------------------
# Config global + loader
# ---------------------------------------------------------------------------

# Default mirrors the value in the in-source comment.
comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = 120.0


def apply_config(config: dict) -> None:
    """Read absolute_step_seconds from config and write the comfyui global.

    Call this once after `comfyui.init_config(config)` in api.py.
    """
    rv = config.get("comfy_recovery", {})
    sd = rv.get("slowdown_detection", {})
    comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC = float(
        sd.get("absolute_step_seconds", comfyui.COMFY_SLOWDOWN_ABSOLUTE_SEC)
    )


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
            and len(self.step_times) >= comfyui.COMFY_SLOWDOWN_MIN_STEPS
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
