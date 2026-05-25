"""
Audio Enhancer for CTB — ACE-Step 1.5 music generation.

Handles caption building, lyrics assembly, genre validation,
Qwen3.5 Plus integration, and Inspire Me randomization.
"""

import json
import random
import logging
import bisect
import difflib
import re
import httpx
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# ─── Constants ───────────────────────────────────────────────

DEFAULT_STRUCTURE = [
    "Intro", "Verse 1", "Pre-Chorus", "Chorus",
    "Verse 2", "Bridge", "Chorus", "Outro"
]

AVAILABLE_BLOCKS = [
    "Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge",
    "Outro", "Instrumental", "Guitar Solo", "Piano Interlude",
    "Build", "Drop", "Breakdown", "Fade Out",
]

INSTRUMENTAL_BLOCKS = {
    "Instrumental", "Guitar Solo",
    "Piano Interlude", "Build", "Drop", "Breakdown", "Fade Out",
}

BPM_PRESETS = [60, 80, 100, 120, 140, 160]

KEY_PRESETS = {
    "major": {
        "C major":  {"emoji": "☀️",  "mood": "Pure, neutral, clear"},
        "D major":  {"emoji": "🎉", "mood": "Triumphant, festive"},
        "E major":  {"emoji": "💪", "mood": "Powerful, radiant"},
        "F major":  {"emoji": "🌿", "mood": "Peaceful, pastoral"},
        "G major":  {"emoji": "😊", "mood": "Cheerful, optimistic"},
        "A major":  {"emoji": "✨", "mood": "Warm, bright"},
        "Bb major": {"emoji": "🎭", "mood": "Majestic, noble"},
    },
    "minor": {
        "A minor":  {"emoji": "😢", "mood": "Melancholic, reflective"},
        "B minor":  {"emoji": "🌙", "mood": "Dark, mysterious"},
        "C minor":  {"emoji": "💔", "mood": "Dramatic, tragic"},
        "D minor":  {"emoji": "🌧️", "mood": "Sad, serious"},
        "E minor":  {"emoji": "🥀", "mood": "Tender, wistful"},
        "F minor":  {"emoji": "😰", "mood": "Gloomy, tense"},
        "G minor":  {"emoji": "🖤", "mood": "Somber, passionate"},
    },
}

TIME_SIGNATURES = ["4", "3", "6", "2"]

# Display labels for time signatures (node value -> human label)
TIME_SIGNATURE_LABELS = {"4": "4/4", "3": "3/4", "6": "6/8", "2": "2/4"}

DURATION_PRESETS = [60, 120, 180, 240]

TOP_LANGUAGES = [
    "en", "de", "es", "fr", "ja",
    "ko", "zh", "pt", "it", "ar",
    "tr", "ru",
]

# Display labels for languages (ISO code -> human label)
LANGUAGE_LABELS = {
    "en": "English", "de": "German", "es": "Spanish", "fr": "French",
    "ja": "Japanese", "ko": "Korean", "zh": "Chinese", "pt": "Portuguese",
    "it": "Italian", "ar": "Arabic", "tr": "Turkish", "ru": "Russian",
    "nl": "Dutch", "pl": "Polish", "vi": "Vietnamese", "cs": "Czech",
    "fa": "Persian", "id": "Indonesian", "uk": "Ukrainian", "hu": "Hungarian",
    "sv": "Swedish", "ro": "Romanian", "el": "Greek",
}

FORMATTING_TIPS = (
    "💡 Formatting tips:\n"
    "• UPPERCASE = shouting/high intensity\n"
    "• Extended vowels: \"Feeeling so aliiive\" (unstable, use cautiously)\n"
    "• (parentheses) = background vocals\n"
    "• Keep lines 6-10 syllables for natural flow\n"
    "• Add style hints: [Chorus - anthemic]"
)


# ─── Normalization Helpers ────────────────────────────────────

# All valid keyscale values accepted by ComfyUI
VALID_KEYSCALES = {
    "C major", "C minor", "C# major", "C# minor", "D major", "D minor",
    "D# major", "D# minor", "Eb major", "Eb minor", "E major", "E minor",
    "F major", "F minor", "F# major", "F# minor", "G major", "G minor",
    "G# major", "G# minor", "Ab major", "Ab minor", "A major", "A minor",
    "A# major", "A# minor", "Bb major", "Bb minor", "B major", "B minor",
    "Cb major", "Cb minor", "Db major", "Db minor", "Gb major", "Gb minor",
}

# Map "4/4" -> "4", "3/4" -> "3", "6/8" -> "6", "2/4" -> "2"
_TS_SLASH_MAP = {"4/4": "4", "3/4": "3", "6/8": "6", "2/4": "2"}


def _normalize_key(raw) -> str:
    """Normalize key string to ComfyUI format (e.g. 'C Major' -> 'C major')."""
    raw = str(raw).strip()
    parts = raw.split()
    if len(parts) == 2:
        normalized = f"{parts[0]} {parts[1].lower()}"
        if normalized in VALID_KEYSCALES:
            return normalized
    if raw in VALID_KEYSCALES:
        return raw
    return raw


def _normalize_time_signature(raw) -> str:
    """Normalize time signature to ComfyUI format (e.g. '4/4' -> '4')."""
    raw = str(raw).strip()
    if raw in _TS_SLASH_MAP:
        return _TS_SLASH_MAP[raw]
    if raw in {"2", "3", "4", "6"}:
        return raw
    return "4"


# ─── Duration Estimation ─────────────────────────────────────

def _count_syllables(text: str) -> int:
    """Count syllables via vowel-group heuristic."""
    text = re.sub(r'[^a-zA-Z ]', '', text).lower()
    count = 0
    for word in text.split():
        groups = re.findall(r'[aeiouy]+', word)
        count += max(1, len(groups))
    return count


def estimate_duration_from_lyrics(lyrics_text: str, bpm: int = 120, time_signature: str = "4") -> int:
    """Estimate song duration in seconds from assembled lyrics + tempo.

    Each section with lyrics → syllable-count / syllables-per-beat → seconds.
    Each instrumental section → 4 bars default.
    Adds 15 % padding, rounds to nearest 5 s, clamps 30–240 s.
    """
    beats_per_bar = int(time_signature) if str(time_signature).isdigit() else 4
    bpm_safe = max(60, min(200, bpm))
    beat_dur = 60.0 / bpm_safe

    sections = re.split(r'\n(?=\[)', lyrics_text.strip())
    total = 0.0

    for section in sections:
        lines = [l.strip() for l in section.split('\n')]
        lyric_lines = [l for l in lines if l and not l.startswith('[')]
        if not lyric_lines:
            # Instrumental block: default 4 bars
            total += beat_dur * beats_per_bar * 4
        else:
            syllables = sum(_count_syllables(l) for l in lyric_lines)
            # ~2 syllables per beat at 120 BPM; scales lightly with tempo
            syl_per_beat = max(1.0, min(3.0, bpm_safe / 65.0))
            total += (syllables / syl_per_beat) * beat_dur

    total *= 1.15  # breathing room / fills
    return max(30, min(240, int(round(total / 5)) * 5))


# ─── Data Classes ────────────────────────────────────────────

@dataclass
class AudioSettings:
    """Stores all audio generation settings."""
    type: str = "vocal"              # vocal / instrumental
    mode: str = "enhanced"           # manual / enhanced / inspire
    voice: str = "any"               # male / female / any
    genre: Optional[str] = None
    bpm: Optional[int] = None
    key: Optional[str] = None
    time_signature: str = "4"
    duration: Optional[int] = None
    language: str = "en"
    structure: List[str] = field(default_factory=lambda: list(DEFAULT_STRUCTURE))
    lyrics: Dict[str, str] = field(default_factory=dict)
    caption: str = ""
    current_block_index: int = 0
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type, "mode": self.mode, "voice": self.voice, "genre": self.genre,
            "bpm": self.bpm, "key": self.key, "time_signature": self.time_signature,
            "duration": self.duration, "language": self.language,
            "structure": self.structure, "lyrics": dict(self.lyrics),
            "caption": self.caption, "current_block_index": self.current_block_index,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AudioSettings":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def get_current_block(self) -> Optional[str]:
        """Get the current block name for lyrics input."""
        if self.current_block_index < len(self.structure):
            return self.structure[self.current_block_index]
        return None

    def get_unique_lyrics_blocks(self) -> List[str]:
        """Get blocks that need unique lyrics input (skip repeats)."""
        seen = set()
        unique = []
        for block in self.structure:
            # Normalize: "Chorus" repeated uses same lyrics
            base_name = block
            if base_name not in seen:
                seen.add(base_name)
                unique.append(base_name)
        return unique

    def is_instrumental_block(self, block_name: str) -> bool:
        """Check if a block is instrumental (no lyrics needed)."""
        base = block_name.split(" ")[0] if " " in block_name else block_name
        # Check exact match and base match
        return block_name in INSTRUMENTAL_BLOCKS or base in INSTRUMENTAL_BLOCKS

    def assemble_lyrics(self) -> str:
        """Assemble the final lyrics string from structure + lyrics dict."""
        parts = []
        for block in self.structure:
            tag = f"[{block}]"
            lyrics_text = self.lyrics.get(block, "")
            if lyrics_text:
                parts.append(f"{tag}\n{lyrics_text}")
            else:
                parts.append(tag)
        return "\n\n".join(parts)

    def export_lyrics(self, audio_output_path: str) -> Optional[str]:
        """Save lyrics as a .txt file next to the audio file.

        The lyrics file has the same name as the audio file with '_lyrics.txt' suffix.
        Structure tags like [Verse 1] are included.

        Returns the lyrics file path, or None if no lyrics to export.
        """
        if self.type == "instrumental" or not self.lyrics:
            return None

        # Check if there are any non-empty lyrics
        has_content = any(text.strip() for text in self.lyrics.values())
        if not has_content:
            return None

        # Build lyrics text with structure tags
        lines = []
        for block in self.structure:
            tag = f"[{block}]"
            text = self.lyrics.get(block, "").strip()
            lines.append(tag)
            if text:
                lines.append(text)
            lines.append("")  # blank line between blocks

        lyrics_text = "\n".join(lines).strip()

        # Determine output path
        audio_path = Path(audio_output_path)
        lyrics_path = audio_path.parent / f"{audio_path.stem}_lyrics.txt"

        lyrics_path.parent.mkdir(parents=True, exist_ok=True)
        lyrics_path.write_text(lyrics_text, encoding="utf-8")
        print(f"Lyrics exported to: {lyrics_path}")
        return str(lyrics_path)

    def build_caption_from_settings(self) -> str:
        """Build caption string from set audio settings (Manual mode)."""
        parts = []
        if self.genre:
            parts.append(self.genre)
        if self.key:
            mood = ""
            for group in KEY_PRESETS.values():
                if self.key in group:
                    mood = group[self.key]["mood"].lower()
                    break
            if mood:
                parts.append(mood)
        if self.bpm:
            parts.append(f"{self.bpm} BPM")
        if self.voice != "any" and self.type == "vocal":
            parts.append(f"{self.voice} vocals")
        if self.type == "instrumental":
            parts.append("instrumental")
        return ", ".join(parts) if parts else "music"

    def settings_summary(self) -> str:
        """Human-readable summary of current settings."""
        lines = []
        if self.type == "vocal":
            voice_label = {"male": "Male", "female": "Female", "any": "Any"}.get(self.voice, self.voice)
            lines.append(f"🎤 Voice: {voice_label}")
        if self.genre:
            lines.append(f"🎸 Genre: {self.genre}")
        if self.bpm:
            lines.append(f"🥁 BPM: {self.bpm}")
        if self.key:
            lines.append(f"🎵 Key: {self.key}")
        lines.append(f"🎼 Time: {TIME_SIGNATURE_LABELS.get(self.time_signature, self.time_signature)}")
        if self.duration:
            lines.append(f"⏱️ Duration: {self.duration}s")
        lines.append(f"🌐 Language: {LANGUAGE_LABELS.get(self.language, self.language)}")
        return " | ".join(lines)


# ─── AudioEnhancer Class ─────────────────────────────────────

class AudioEnhancer:
    """Handles audio generation logic for ACE-Step 1.5 workflows."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.openrouter_api_key = str(config.get("openrouter_api_key", "")).strip()
        self.openrouter_base_url = str(config.get("openrouter_base_url",
            "https://openrouter.ai/api/v1/chat/completions")).strip()

        # Qwen3.5 Plus for Enhanced/Inspire modes
        self.qwen_plus_model = str(config.get("audio_llm_model",
            "qwen/qwen3.5-plus-02-15")).strip()

        # Cheap/free model for genre validation
        self.genre_validation_model = str(config.get("genre_validation_model",
            "qwen/qwen3-next-80b-a3b-instruct:free")).strip()

        self.request_timeout = int(config.get("request_timeout_seconds", 30))
        self.song_generation_timeout = max(self.request_timeout, 90)

        # Load genre vocabulary
        self.genres_vocab: set = set()
        self._load_genres_vocab()

    def _load_genres_vocab(self):
        """Load genre vocabulary from genres_vocab.txt."""
        vocab_path = Path(__file__).parent / "genres_vocab.txt"
        if vocab_path.exists():
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.genres_vocab = {
                    line.strip().lower()
                    for line in f if line.strip()
                }
            logger.info(f"Loaded {len(self.genres_vocab)} genres from vocabulary")
        else:
            logger.warning(f"Genre vocabulary not found at {vocab_path}")

    # ─── Genre Validation ────────────────────────────────────

    async def validate_genre(self, user_input: str) -> Dict[str, Any]:
        """
        Validate genre input against vocabulary using local fuzzy matching.
        Returns: {"valid": bool, "genre": str, "suggestion": str|None}
        """
        normalized = user_input.strip().lower()

        # Exact match
        if normalized in self.genres_vocab:
            return {"valid": True, "genre": user_input.strip(), "suggestion": None}

        if not self.genres_vocab:
            # No vocabulary loaded — accept as-is
            return {"valid": True, "genre": user_input.strip(), "suggestion": None}

        # Fuzzy match against the full vocabulary
        # difflib handles typos like "gagster rap" -> "gangster rap" well
        matches = difflib.get_close_matches(normalized, self.genres_vocab, n=3, cutoff=0.7)
        if matches:
            # Pick the best match (first is highest similarity)
            best = matches[0]
            return {"valid": False, "genre": best, "suggestion": best}

        # No close match — accept as-is (ACE-Step handles unknown genres gracefully,
        # and descriptive inputs like "happy summer music" get resolved during caption generation)
        return {"valid": True, "genre": user_input.strip(), "suggestion": None}

    # ─── OpenRouter Helper ───────────────────────────────────

    async def _call_openrouter(self, prompt: str, model: str = None,
                                max_tokens: int = 2000,
                                system_prompt: str = None,
                                disable_reasoning: bool = False,
                                timeout: int = None) -> str:
        """Call OpenRouter API with given prompt."""
        if not self.openrouter_api_key:
            raise RuntimeError("OpenRouter API key not configured")

        timeout = timeout or self.request_timeout
        model = model or self.qwen_plus_model
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.8,
        }

        if disable_reasoning:
            payload["reasoning"] = {"exclude": True}

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                self.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code != 200:
                error_body = response.text[:500]
                print(f"OpenRouter API error {response.status_code}: {error_body}")
                raise RuntimeError(f"OpenRouter API error {response.status_code}: {error_body}")
            data = response.json()
            try:
                msg = data["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or ""
                if not content:
                    print(f"OpenRouter returned empty content. Full response: {str(data)[:300]}")
                return content
            except (KeyError, IndexError, TypeError) as e:
                error_msg = data.get("error", {}).get("message", str(data)[:200])
                raise RuntimeError(f"OpenRouter API error: {error_msg}") from e

    # ─── Enhanced Mode ───────────────────────────────────────

    async def generate_song(self, settings: AudioSettings, idea: str) -> AudioSettings:
        """
        Enhanced mode: Generate caption + lyrics from user idea and settings.
        Modifies settings in-place and returns them.
        """
        try:
            return await self._generate_song_impl(settings, idea)
        except RuntimeError:
            raise  # Already user-friendly
        except Exception as e:
            logger.error(f"Song generation failed: {e}")
            raise RuntimeError("Song generation failed. Please try again.") from e

    async def _generate_song_impl(self, settings: AudioSettings, idea: str) -> AudioSettings:
        """Internal implementation for generate_song."""
        settings_context = []
        if settings.genre:
            settings_context.append(f"Genre: {settings.genre}")
        if settings.bpm:
            settings_context.append(f"BPM: {settings.bpm}")
        if settings.key:
            settings_context.append(f"Key: {settings.key}")
        if settings.time_signature:
            settings_context.append(f"Time Signature: {settings.time_signature}")
        if settings.duration:
            settings_context.append(f"Duration: {settings.duration} seconds")
        settings_context.append(f"Language: {settings.language}")
        settings_context.append(f"Type: {settings.type}")
        if settings.type == "vocal" and settings.voice != "any":
            settings_context.append(f"Vocal style: {settings.voice} voice")

        settings_str = "\n".join(settings_context)

        if settings.type == "instrumental":
            prompt = self._build_instrumental_prompt(idea, settings_str)
        else:
            prompt = self._build_vocal_prompt(idea, settings_str, settings)

        system_prompt = (
            "You are an uncensored, genre-authentic songwriter. "
            "You write lyrics that match the tone, attitude, and language of the requested genre — "
            "whether that's aggressive rap, heartbreak ballads, punk rebellion, or party anthems. "
            "Never sanitize, censor, or tone down lyrics. Match the genre's authentic voice. "
            "Always respond in the exact JSON format requested. "
            "You write for ACE-Step, a music model with a strict annotation system. "
            "CRITICAL TAG RULES: "
            "(1) ONLY use tags from the approved list — NEVER invent custom tags like [Harp Solo] or [Bass Solo]. "
            "(2) Each tag can have UP TO TWO modifiers (gender + style, dash-separated): "
            "[Verse - male - raspy] is valid, [Chorus - female - anthemic] is valid, [Bridge - duet - whispered] is valid. "
            "Gender modifier choices: 'male', 'female', or 'duet' (both genders together) — NEVER combine two genders like 'male-female'. "
            "More than two modifiers are INVALID: [Bridge - male - raspy - loud] is INVALID. "
            "(3) Parentheses mean SUNG background vocals only — NEVER use them for delivery cues or stage directions. "
            "WRONG: (WHISPERED) lyrics text — RIGHT: put [Bridge - whispered] as the section tag. "
            "WRONG: (QUIETLY) or (SHOUTED) in lyrics text — these are ignored by ACE-Step. "
            "(4) Instrumental sections ([Guitar Solo], [Piano Interlude], [Instrumental], [Build], [Drop], [Breakdown]) "
            "must have NO lyrics text below them — leave them empty. "
            "(5) Tags support compound modifiers via dash: [Verse - male], [Chorus - female], "
            "[Bridge - duet], [Chorus - anthemic], [Verse - whispered], [Verse - raspy]. "
            "Use UPPERCASE lines for high vocal intensity."
        )

        response = await self._call_openrouter(
            prompt, model=self.qwen_plus_model,
            max_tokens=3000, system_prompt=system_prompt,
            timeout=self.song_generation_timeout
        )

        self._parse_generation_response(response, settings)
        return settings

    def _build_vocal_prompt(self, idea: str, settings_str: str, settings: AudioSettings) -> str:
        if settings.voice == "any":
            voice_rule = (
                "- Vocal arrangement: decide creatively — mix male/female/duet across sections or go solo.\n"
                "  Use compound tags: [Verse - male], [Chorus - female], [Bridge - duet].\n"
                "  Vary textures: [Verse - raspy], [Chorus - anthemic], [Bridge - whispered].\n"
                "  Caption must describe section-specific vocal roles, e.g.:\n"
                "  'male vocals in verse, female vocals in chorus, vocal duet in bridge,\n"
                "   alternating male and female lead vocals'.\n"
                "  Never just 'male and female vocals' — always specify which section each voice sings."
            )
        elif settings.voice == "male":
            voice_rule = (
                "- Vocal: solo male throughout. Keep male voice consistent in all section tags.\n"
                "  Vary textures to avoid monotony: [Verse - raspy], [Chorus - powerful], [Bridge - whispered].\n"
                "  Use (backing vocals) in parentheses for depth. Caption must include 'male vocal'."
            )
        else:
            voice_rule = (
                "- Vocal: solo female throughout. Keep female voice consistent in all section tags.\n"
                "  Vary textures to avoid monotony: [Verse - breathy], [Chorus - powerful], [Bridge - whispered].\n"
                "  Use (harmonies) in parentheses for depth. Caption must include 'female vocal'."
            )

        return (
            f"Create a complete song based on this idea:\n\"{idea}\"\n\n"
            f"Current settings:\n{settings_str}\n\n"
            f"Respond in this exact JSON format:\n"
            f'{{\n'
            f'  "caption": "genre, era, instruments, vocal characteristics, timbre, production style, mood",\n'
            f'  "lyrics": "[Intro]\\n\\n[Verse 1 - male]\\nlyrics here...\\n\\n[Chorus - female]\\nlyrics...",\n'
            f'  "bpm": 120,\n'
            f'  "key": "C major",\n'
            f'  "time_signature": "4",\n'
            f'  "duration": 180,\n'
            f'  "genre": "resolved genre tag from vocabulary"\n'
            f'}}\n\n'
            f"Rules:\n"
            f"- Caption: richly describe the sound — include:\n"
            f"  • genre and era (e.g. '90s grunge', '80s synth-pop', 'modern trap', 'contemporary R&B')\n"
            f"  • instruments\n"
            f"  • vocal characteristics (female vocal / male vocal / breathy / powerful / falsetto / raspy / choir)\n"
            f"  • timbre (warm / bright / crisp / punchy / lush / raw)\n"
            f"  • production style (lo-fi / studio-polished / bedroom pop / live recording)\n"
            f"  • mood\n"
            f"  Caption and lyrics tags MUST be consistent — no conflicts between them.\n"
            f"{voice_rule}\n"
            f"- Structure tags — ONLY use tags from this exact list (never invent new ones):\n"
            f"  Basic: [Intro], [Verse 1], [Pre-Chorus], [Chorus], [Final Chorus], [Bridge], [Outro]\n"
            f"  Vocal compound (ONE modifier max): [Verse - male], [Chorus - female], [Bridge - duet],\n"
            f"                  [Verse - whispered], [Chorus - anthemic], [Chorus - powerful],\n"
            f"                  [Verse - raspy], [Verse - breathy], [Chorus - choir], [Verse - falsetto]\n"
            f"  Dynamic (no lyrics below): [Build], [Drop], [Breakdown]\n"
            f"  Instrumental (no lyrics below): [Guitar Solo], [Piano Interlude], [Instrumental]\n"
            f"  Special (no lyrics below): [Fade Out], [Silence]\n"
            f"- INVALID tag examples — DO NOT use: [Harp Solo], [Bass Solo], [Flute Solo],\n"
            f"  [Bridge - whisper - male] (two modifiers), [Verse - male - raspy] (two modifiers)\n"
            f"- When vocal style conflicts with one-modifier rule (e.g. 'duet chorus but man whispers'):\n"
            f"  → Use the DELIVERY modifier in the tag: [Chorus - whispered]. Describe the arrangement in caption.\n"
            f"  → NEVER add a second modifier. NEVER use (WHISPERED) in the lyrics text as a workaround.\n"
            f"  → Put the whisper modifier on the CORRECT section (the one the user asked to be whispered).\n"
            f"- Parentheses () = SUNG background vocals/harmonies ONLY.\n"
            f"  NEVER use parentheses for delivery cues, instrument cues, or stage directions.\n"
            f"  Wrong: (WHISPERED) line here — Right: use [Bridge - whispered] as the section tag.\n"
            f"  Wrong: (DRUMS KICK IN) — parentheses are only for sung harmonies like (echoes, fades away).\n"
            f"- Instrumental/dynamic sections MUST be empty — no text below the tag.\n"
            f"- For an instrumental intro, use [Intro] with no text, or place [Instrumental] first.\n"
            f"- Use UPPERCASE lines for high-intensity vocal moments.\n"
            f"- If the idea mentions an instrumental solo or break, use the closest standard tag.\n"
            f"- CRITICAL: Match lyrics length AND section count to duration! Too many lyrics or\n"
            f"  sections = ACE-Step crams/garbles them and barely any of it is actually sung.\n"
            f"  Approximate total sung lines by duration (at ~120 BPM):\n"
            f"  20-30s: ~4-6 lines, 1-2 SUNG sections only (e.g. 1 short verse, or 1 chorus)\n"
            f"  45s: ~8-10 lines, ≤3 sung sections (short verse + chorus)\n"
            f"  60s: ~12-16 lines total (intro + verse + chorus + short outro)\n"
            f"  90s: ~18-26 lines total (intro + verse + chorus + verse + chorus + outro)\n"
            f"  120s: ~24-32 lines total (intro + 2 verses + 2 choruses + bridge + outro)\n"
            f"  180s: ~36-48 lines total (full structure with multiple verses)\n"
            f"  240s: ~48-64 lines total (extended structure, 4+ verses, multiple chorus repeats, solos)\n"
            f"  At higher BPM (140+), you can fit ~20% more lines. At lower BPM (80-), use ~20% fewer.\n"
            f"  Instrumental/solo sections take time but need no lyrics — count them as ~4-8 lines of time.\n"
            f"- HARD RULE (never violate): output at most as many sections as the duration can\n"
            f"  realistically sing at normal pace. ≤30s → max 1-2 sung sections (+ optional empty\n"
            f"  [Intro]/[Outro]); ≤45s → max 3 sung sections; ≤60s → no more than intro+verse+\n"
            f"  chorus+outro. Do NOT emit a full multi-section song (verse/pre-chorus/chorus/\n"
            f"  verse/bridge/final-chorus) unless duration ≥ 120s. Keep the song's natural\n"
            f"  structure (intro / breaks / outro are good) — just scale the NUMBER of sections\n"
            f"  to the duration so every section is actually rendered.\n"
            f"- Keep verses to 4 lines max, choruses to 3-4 lines. Fewer lines is safer than too many.\n"
            f"- ALWAYS use a rhyme scheme — genre doesn't matter. Default AABB (couplets) or ABAB.\n"
            f"  Line-ending words must rhyme. Never write free verse unless the user explicitly requests it.\n"
            f"- Only fill in bpm/key/time_signature/duration/genre if not already set above\n"
            f"- If genre was given as a description, resolve it to specific genre tags\n"
            f"- Write lyrics in {settings_str.split('Language: ')[-1].split(chr(10))[0]}\n"
            f"- Keep lyric lines 6-10 syllables\n"
            f"- Respond ONLY with valid JSON, no markdown, no explanation"
        )

    def _build_instrumental_prompt(self, idea: str, settings_str: str) -> str:
        return (
            f"Create an instrumental music description based on this idea:\n\"{idea}\"\n\n"
            f"Current settings:\n{settings_str}\n\n"
            f"Respond in this exact JSON format:\n"
            f'{{\n'
            f'  "caption": "genre, mood, instruments, production description",\n'
            f'  "lyrics": "[Instrumental]",\n'
            f'  "bpm": 120,\n'
            f'  "key": "C major",\n'
            f'  "time_signature": "4",\n'
            f'  "duration": 180,\n'
            f'  "genre": "resolved genre tag"\n'
            f'}}\n\n'
            f"Rules:\n"
            f"- Caption: describe the instrumental sound in detail\n"
            f"- Lyrics: use [Instrumental] or structure tags like [Intro - ambient], [Main Theme - piano]\n"
            f"- Only fill in bpm/key/time_signature/duration/genre if not already set above\n"
            f"- Respond ONLY with valid JSON, no markdown, no explanation"
        )

    def _parse_generation_response(self, response: str, settings: AudioSettings):
        """Parse Qwen3.5 Plus JSON response into AudioSettings."""
        # Strip markdown code fences if present
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse LLM response as JSON: {text[:200]}")
            # Fallback: use response as caption
            settings.caption = response.strip()[:500]
            return

        settings.caption = data.get("caption", settings.caption)

        # Parse lyrics into structure + lyrics dict
        raw_lyrics = data.get("lyrics", "")
        if raw_lyrics:
            self._parse_lyrics_into_settings(raw_lyrics, settings)

        # Fill unset metadata
        if not settings.bpm and data.get("bpm"):
            settings.bpm = int(data["bpm"])
        if not settings.key and data.get("key"):
            settings.key = _normalize_key(data["key"])
        if not settings.duration and data.get("duration"):
            settings.duration = int(data["duration"])
        if data.get("time_signature"):
            settings.time_signature = _normalize_time_signature(data["time_signature"])
        if data.get("genre") and not settings.genre:
            settings.genre = data["genre"]

    def _parse_lyrics_into_settings(self, raw_lyrics: str, settings: AudioSettings):
        """Parse raw lyrics string with structure tags into settings."""
        structure = []
        lyrics = {}
        current_tag = None
        current_lines = []

        for line in raw_lyrics.split("\n"):
            tag_match = re.match(r'^\[(.+?)\]\s*$', line.strip())
            if tag_match:
                # Save previous block
                if current_tag is not None:
                    lyrics[current_tag] = "\n".join(current_lines).strip()
                current_tag = tag_match.group(1)
                structure.append(current_tag)
                current_lines = []
            elif current_tag is not None:
                current_lines.append(line)

        # Save last block
        if current_tag is not None:
            lyrics[current_tag] = "\n".join(current_lines).strip()

        settings.structure = structure
        settings.lyrics = lyrics

    # ─── Inspire Me ──────────────────────────────────────────

    def randomize_unset_settings(self, settings: AudioSettings):
        """Randomize any unset audio settings for Inspire Me mode."""
        if not settings.genre:
            # Pick a random genre from vocabulary
            if self.genres_vocab:
                settings.genre = random.choice(list(self.genres_vocab))

        if not settings.bpm:
            settings.bpm = random.choice(BPM_PRESETS)

        if not settings.key:
            all_keys = list(KEY_PRESETS["major"].keys()) + list(KEY_PRESETS["minor"].keys())
            settings.key = random.choice(all_keys)

        if not settings.duration:
            settings.duration = random.choice(DURATION_PRESETS)

        # Language: keep as-is (default English)
        # Time signature: not randomized — 4/4 default is correct for most genres

    async def inspire_me(self, settings: AudioSettings) -> AudioSettings:
        """
        Inspire Me mode: randomize unset settings, then generate via Qwen.
        """
        self.randomize_unset_settings(settings)

        # Generate an idea prompt from the randomized settings
        idea = (
            f"Create a {settings.genre} song in {settings.key} at {settings.bpm} BPM. "
            f"Be creative and surprising with the theme and lyrics."
        )

        return await self.generate_song(settings, idea)

    # ─── Cover / Analyze Existing Song ──────────────────────

    async def restructure_analyzed_song(
        self,
        tags: str,
        bpm: int,
        keyscale: str,
        lyrics_raw: str,
        caption: str,
        duration: int,
        user_instruction: str = "",
    ) -> "AudioSettings":
        """
        Takes raw Music Analyzer Lite outputs and uses LLM to restructure
        the raw lyrics transcription into ACEStep-format with section tags.
        Returns a populated AudioSettings ready for workflow injection.
        """
        settings = AudioSettings()
        settings.bpm = bpm
        settings.key = _normalize_key(keyscale)
        settings.duration = duration

        is_instrumental = (
            not lyrics_raw.strip()
            or lyrics_raw.strip().lower() in ("[instrumental]", "instrumental")
        )
        if is_instrumental:
            settings.type = "instrumental"
            settings.structure = ["Instrumental"]
            settings.lyrics = {}
            settings.caption = caption or tags
            if tags and not settings.genre:
                settings.genre = tags.split(",")[0].strip()
            return settings

        settings.type = "vocal"

        if user_instruction:
            system_prompt = (
                "You are a music editor. Given a raw audio transcript and musical analysis of an "
                "existing song, your task is to format the lyrics into ACEStep-compatible structure "
                "and improve the caption. "
                "The user has provided specific instructions that OVERRIDE default preservation rules — "
                "follow them exactly, including translation, voice changes, or lyric modifications. "
                "Always respond in the exact JSON format requested. No markdown, no explanation."
            )
        else:
            system_prompt = (
                "You are a music editor. Given a raw audio transcript and musical analysis of an "
                "existing song, your task is to format the lyrics into ACEStep-compatible structure "
                "and improve the caption. "
                "CRITICAL: Preserve the original lyrics content — do NOT rewrite or change the words. "
                "Only add structure tags, performance tags, and fix obvious transcription errors. "
                "Always respond in the exact JSON format requested. No markdown, no explanation."
            )

        preserve_rule = (
            "- Preserve original lyrics word for word EXCEPT where user instructions say otherwise\n"
            if user_instruction else
            "- Preserve ALL original lyrics word for word\n"
        )

        user_instruction_block = (
            f"USER INSTRUCTIONS (highest priority — override default rules where needed):\n"
            f"{user_instruction}\n\n"
            if user_instruction else ""
        )

        task_intro = (
            "Format and modify this song transcript for ACEStep per the user instructions below. "
            "The user instructions take FULL PRIORITY — translate, rewrite, or change sections as instructed. "
            "Do not let preservation rules block user-requested changes.\n\n"
            if user_instruction else
            "Format this existing song transcript into ACEStep structure.\n\n"
        )

        prompt = (
            task_intro
            + user_instruction_block
            + f"Raw lyrics (from audio transcription — may contain errors):\n{lyrics_raw}\n\n"
            f"Musical analysis:\n"
            f"- Tags: {tags}\n"
            f"- BPM: {bpm}\n"
            f"- Key: {keyscale}\n"
            f"- Duration: {duration}s\n"
            f"- Caption: {caption}\n\n"
            f"Respond with this exact JSON:\n"
            f'{{\n'
            f'  "caption": "genre, era, instruments, vocal characteristics, production style, mood",\n'
            f'  "lyrics": "[Verse 1 - female]\\nline1\\nline2\\n\\n[Chorus - anthemic]\\nline1\\nline2",\n'
            f'  "language": "en",\n'
            f'  "time_signature": "4",\n'
            f'  "genre": "resolved genre tag"\n'
            f'}}\n\n'
            f"Rules:\n"
            + preserve_rule
            + f"- Add structure tags based on natural song flow: "
            f"[Intro], [Verse 1], [Verse 2], [Pre-Chorus], [Chorus], [Bridge], [Outro]\n"
            f"- If the same lines repeat, tag them as [Chorus]\n"
            f"- For silent/instrumental sections, use [Instrumental] or [Bridge] with no text below\n"
            f"- Detect the actual language of the lyrics for the 'language' field (ISO code: 'en', 'de', 'es', etc.)\n"
            f"- Caption: improve based on tags and analysis, describe the actual sound\n"
            f"- ONLY use structure tags from this list: "
            f"[Intro], [Verse 1], [Verse 2], [Pre-Chorus], [Chorus], [Final Chorus], [Bridge], [Outro], "
            f"[Instrumental], [Guitar Solo], [Piano Interlude], [Build], [Drop], [Breakdown], [Fade Out], [Silence]\n"
            f"- Section tag modifiers (up to 2 per tag, dash-separated): gender AND/OR vocal style.\n"
            f"  Gender modifiers: 'male', 'female', or 'duet' (both genders singing together).\n"
            f"  NEVER combine two genders like 'male-female' — use 'duet' instead.\n"
            f"  Add gender ONLY when clearly stated in caption/tags ('female vocal', 'male singer', 'duet'). If ambiguous, omit.\n"
            f"  Vocal style modifiers (based on LYRIC TEXT, not overall caption energy):\n"
            f"  * whispered — intro/outro with short repeated phrases, single words, or vocal sounds "
            f"(e.g. 'Oh. Oh. Oh.', 'mmm', 'yeah yeah'); tender bridge moments\n"
            f"  * raspy — lines with raw desperation, urgency, repeated pleading\n"
            f"  * spoken — prose-style lines that read as speech, not song\n"
            f"  * anthemic — big climactic chorus meant to be sung loudly\n"
            f"  * choir — ensemble of multiple voices singing together as a group; use for chorus/bridge sections meant to sound like a choir\n"
            f"  * falsetto — high-register thin voice; use sparingly\n"
            f"  Combine gender + style when both fit: [Verse - male - raspy], [Chorus - female - anthemic], [Bridge - duet - whispered].\n"
            f"  Style alone also valid: [Intro - whispered], [Chorus - anthemic], [Chorus - choir], [Final Chorus - choir].\n"
            f"  CORRECT: [Intro - whispered], [Bridge - duet - whispered], [Chorus - female - anthemic], [Verse - male - raspy], [Chorus - choir - whispered], [Final Chorus - choir]\n"
            f"  WRONG: [Chorus - male-female - whispered] — use 'duet' not 'male-female'\n"
            f"  WRONG: [Intro]\\n[whispered]\\nlyrics — NEVER put a modifier on its own line\n"
            f"- Background vocals / harmonies: use parentheses inline in the lyric line: "
            f"  We rise together (together)\n"
            f"- High intensity / shouting: use UPPERCASE on that specific line: WE WILL ROCK YOU\n"
            f"- Do NOT invent tags not listed here. No [anthem], [adlib], [falsetto], [harmony] tags.\n"
            f"- Respond ONLY with valid JSON"
        )

        response = await self._call_openrouter(
            prompt,
            model=self.qwen_plus_model,
            max_tokens=3000,
            system_prompt=system_prompt,
            timeout=self.song_generation_timeout,
        )

        logger.info(f"[cover] raw LLM response:\n{response}")
        self._parse_restructure_response(response, settings, tags)
        logger.info(f"[cover] structure: {settings.structure}")
        logger.info(f"[cover] lyrics keys: {list(settings.lyrics.keys())}")
        logger.info(f"[cover] assembled:\n{settings.assemble_lyrics()}")
        return settings

    def _parse_restructure_response(
        self, response: str, settings: "AudioSettings", raw_tags: str
    ):
        """Parse LLM restructuring response into settings."""
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse restructure response: {text[:200]}")
            settings.structure = ["Verse 1"]
            settings.lyrics = {"Verse 1": response.strip()[:1000]}
            return

        if data.get("caption"):
            settings.caption = data["caption"]
        if data.get("genre"):
            settings.genre = data["genre"]
        elif raw_tags:
            settings.genre = raw_tags.split(",")[0].strip()
        if data.get("language"):
            settings.language = data["language"]
        if data.get("time_signature"):
            settings.time_signature = _normalize_time_signature(data["time_signature"])

        raw_lyrics = data.get("lyrics", "")
        if raw_lyrics:
            self._parse_lyrics_into_settings(raw_lyrics, settings)

    # ─── Workflow Injection ──────────────────────────────────

    def _build_vocal_role_tags(self, assembled_lyrics: str) -> str:
        """Scan lyrics for gender-tagged sections; return explicit role description for the tags field."""
        import re
        pattern = re.compile(r'\[([^\]]+)\]')
        role_map: dict = {}
        for match in pattern.finditer(assembled_lyrics):
            tag = match.group(1).lower()
            parts = [p.strip() for p in tag.split('-')]
            if len(parts) >= 2:
                section = parts[0].split()[0]  # "verse 1" → "verse"
                for part in parts[1:]:
                    if part in ('male', 'female', 'duet'):
                        role_map.setdefault(section, part)
                        break
        if not role_map:
            return ""
        desc_parts = [f"{gender} vocals in {section}" for section, gender in role_map.items()]
        return ", ".join(desc_parts) + ", alternating male and female lead vocals"

    def inject_audio_settings(self, workflow: Dict, settings: AudioSettings) -> Dict:
        """
        Inject audio settings into ACE-Step 1.5 workflow JSON.
        Node 94 = TextEncodeAceStepAudio1.5
        Node 98 = EmptyAceStep1.5LatentAudio (duration)
        """
        # Node 94: Text encoding parameters
        if "94" in workflow:
            node_94 = workflow["94"]["inputs"]
            if settings.genre:
                caption_rest = settings.caption or ""
                if "," in caption_rest:
                    caption_rest = caption_rest.split(",", 1)[1].strip()
                effective_tags = f"{settings.genre}, {caption_rest}" if caption_rest else settings.genre
            else:
                effective_tags = settings.caption or ""
            assembled = settings.assemble_lyrics()
            vocal_role_desc = self._build_vocal_role_tags(assembled)
            if vocal_role_desc:
                effective_tags = f"{vocal_role_desc}, {effective_tags}"
            node_94["tags"] = effective_tags
            node_94["lyrics"] = assembled

            if settings.bpm:
                node_94["bpm"] = settings.bpm
            if settings.key:
                node_94["keyscale"] = settings.key
            if settings.time_signature:
                node_94["timesignature"] = settings.time_signature
            if settings.language:
                node_94["language"] = settings.language

            # Sync duration in text encoder too
            if settings.duration:
                node_94["duration"] = settings.duration

        # Node 98: Duration (latent audio length)
        if "98" in workflow and settings.duration:
            workflow["98"]["inputs"]["seconds"] = settings.duration

        return workflow
