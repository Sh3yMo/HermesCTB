"""Write a secret-free snapshot of config.json to config.backup.json.

config.json is gitignored (holds the real OpenRouter key). This script copies it,
blanks every secret-bearing value, and writes the result to the TRACKED
config.backup.json so the current settings are versioned on GitHub. On restore,
copy config.backup.json -> config.json and paste the API key back in.

Run manually (`py -3 scripts/backup_config.py`) or automatically via the
pre-commit hook (.git/hooks/pre-commit), which regenerates + stages the backup
on every commit.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "config.json")
DST = os.path.join(ROOT, "config.backup.json")

# A key is a secret if its (lowercased) name matches any of these.
_SECRET_RE = re.compile(r"(api_key|token|secret|password)$|^.*_key$|^telegram_token$")


def _is_secret_key(key: str) -> bool:
    k = key.lower()
    return bool(_SECRET_RE.search(k))


def _sanitize(obj):
    if isinstance(obj, dict):
        return {
            k: ("" if (_is_secret_key(k) and isinstance(v, (str, int, float)))
                else _sanitize(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def main() -> int:
    if not os.path.exists(SRC):
        print(f"[backup_config] {SRC} not found — nothing to back up (skip).")
        return 0
    try:
        with open(SRC, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[backup_config] cannot parse config.json: {e!r}")
        return 1
    sanitized = _sanitize(cfg)
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"[backup_config] wrote {DST} (secrets blanked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
