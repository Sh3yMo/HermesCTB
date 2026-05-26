"""FFmpeg shared-DLL bootstrap for torchcodec/demucs on Windows.

torchcodec 0.13+ requires the FFmpeg shared DLLs (avcodec, avformat, ...) on
the OS loader path. Python 3.8+ on Windows ignores PATH for native DLLs by
default — only directories registered via os.add_dll_directory() are searched.

This module locates the Gyan.FFmpeg.Shared install directory (winget package)
and registers it once at import. Idempotent — safe to import multiple times.

Import order: this module MUST be imported BEFORE torchcodec/torchaudio/demucs.
"""
from __future__ import annotations

import os
import sys
import glob

_INITIALIZED = False


def _find_ffmpeg_shared_bin() -> str | None:
    """Locate the Gyan.FFmpeg.Shared bin folder (contains avcodec-XX.dll etc.).

    Returns None if not found.
    """
    # 1. Explicit override via env var
    env_path = os.environ.get("FFMPEG_SHARED_BIN")
    if env_path and os.path.isdir(env_path):
        return env_path

    # 2. winget Gyan.FFmpeg.Shared default install path
    candidates = []
    localappdata = os.environ.get("LOCALAPPDATA")
    if localappdata:
        pattern = os.path.join(
            localappdata, "Microsoft", "WinGet", "Packages",
            "Gyan.FFmpeg.Shared_*", "ffmpeg-*-full_build-shared", "bin"
        )
        candidates.extend(glob.glob(pattern))

    # 3. Pick the newest by mtime (handles version upgrades)
    for path in sorted(candidates, key=os.path.getmtime, reverse=True):
        if os.path.isfile(os.path.join(path, "avcodec-62.dll")) or \
           any(f.startswith("avcodec-") for f in os.listdir(path)):
            return path
    return None


def init() -> bool:
    """Register the FFmpeg shared-DLL directory. Returns True on success."""
    global _INITIALIZED
    if _INITIALIZED:
        return True
    if sys.platform != "win32":
        _INITIALIZED = True
        return True
    if not hasattr(os, "add_dll_directory"):
        _INITIALIZED = True
        return True
    bin_dir = _find_ffmpeg_shared_bin()
    if not bin_dir:
        return False
    try:
        os.add_dll_directory(bin_dir)
        # Also prepend to PATH for any subprocess (demucs CLI) that inherits env
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        _INITIALIZED = True
        return True
    except (OSError, FileNotFoundError):
        return False


# Auto-init on import — callers just `import _ffmpeg_init` before torchcodec.
init()
