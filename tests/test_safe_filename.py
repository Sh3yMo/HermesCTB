"""_safe_filename helper: slugify, sanitize, length cap, empty fallback."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_helper():
    # api imports FastAPI heavy stack on module load — pull the symbol.
    from api import _safe_filename
    return _safe_filename


def test_simple_artist_title():
    sf = _load_helper()
    assert sf("Neon Vipers", "Midnight Run") == "Neon Vipers - Midnight Run"


def test_strips_filesystem_unsafe_chars():
    sf = _load_helper()
    out = sf("AC/DC", "Back: In Black?")
    assert "/" not in out
    assert ":" not in out
    assert "?" not in out
    assert "ACDC" in out
    assert "Back" in out and "Black" in out


def test_collapses_whitespace():
    sf = _load_helper()
    assert sf("  Spaced   Out  ", "Multi   Word") == "Spaced Out - Multi Word"


def test_apostrophe_and_hyphen_preserved():
    sf = _load_helper()
    out = sf("D'Angelo", "Brown-Sugar")
    assert "'" in out
    assert "-" in out


def test_empty_parts_skipped():
    sf = _load_helper()
    assert sf("", "Title") == "Title"
    assert sf("Artist", "") == "Artist"
    assert sf("   ", "Title") == "Title"


def test_returns_empty_when_all_unusable():
    sf = _load_helper()
    assert sf("") == ""
    assert sf("", "") == ""
    assert sf("   ", "  ") == ""
    # Pure emoji input strips to nothing → empty (caller must fall back).
    assert sf("🎵🎶") == ""


def test_unicode_letters_preserved_via_word_class():
    sf = _load_helper()
    out = sf("Björk", "Hyperballad")
    # \w in re.UNICODE keeps ö
    assert "Björk" in out


def test_length_cap_default_80():
    sf = _load_helper()
    long_artist = "A" * 60
    long_title = "B" * 60
    out = sf(long_artist, long_title)
    assert len(out) <= 80


def test_length_cap_custom():
    sf = _load_helper()
    out = sf("ABCDEFGH", "12345", max_len=10)
    assert len(out) == 10


def test_strips_trailing_dots_and_dashes():
    sf = _load_helper()
    # Trailing dots break Windows file-explorer; helper trims them.
    assert sf("Band.", "Title-") == "Band - Title"
