"""Music Video Producer Style Director.

Loads empirically researched producer profiles (data/mv_producer_profiles.json),
classifies sub-genre + per-section sentiment via a cheap LLM call, picks a
producer profile + applies sub-genre modifiers, builds a per-segment shot plan
with variety guardrails, and renders a director brief that plan_segments()
injects into its system prompt.

Designed to fail soft: any exception in the pipeline yields an empty director
brief, restoring pre-director behavior. Stage F/G/H/I/J fixes remain untouched.
"""

from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable, Optional

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_PATH = Path(__file__).parent / "data" / "mv_producer_profiles.json"

SHOT_CODES = ["ECU", "CU", "MCU", "MS", "MLS", "LS", "WS"]
CLOSEUP_SHOTS = {"ECU", "CU"}
ESTABLISHING_SHOTS = {"WS", "LS"}

SENTIMENT_LABELS = {
    "sad", "melancholic", "intimate", "anthemic", "celebratory",
    "aggressive", "tense", "calm", "romantic", "neutral",
}

# Mood inferred from musical key when LLM sentiment unavailable.
# Source: audio_enhancer.KEY_PRESETS.
KEY_TO_MOOD_FALLBACK = {
    "A minor": "melancholic", "A maj": "celebratory",
    "B minor": "sad", "B maj": "celebratory",
    "C minor": "sad", "C maj": "celebratory",
    "D minor": "sad", "D maj": "anthemic",
    "E minor": "melancholic", "E maj": "anthemic",
    "F minor": "tense", "F maj": "celebratory",
    "G minor": "melancholic", "G maj": "anthemic",
}


class MVDirector:
    def __init__(self, profiles_path: Optional[Path | str] = None) -> None:
        path = Path(profiles_path) if profiles_path else DEFAULT_PROFILES_PATH
        with open(path, "r", encoding="utf-8") as fh:
            self._data = json.load(fh)
        self._producers = [p for p in self._data.get("producers", []) if p.get("enabled", True)]
        self._sub_genre_modifiers = self._data.get("sub_genre_modifiers", {})
        self._universal_forbidden = self._data.get("universal_forbidden_by_genre", {})
        if not self._producers:
            raise ValueError("mv_producer_profiles.json contained no enabled producers")

    # ── Producer selection ─────────────────────────────────────────

    def select_producer_profile(
        self,
        genre: str,
        sub_genre: Optional[str],
        mood: Optional[str],
        song_seed: Optional[int | str] = None,
    ) -> dict:
        """Pick a producer profile matching genre + sub_genre.

        Selection precedence:
          1. sub_genre_modifiers.preferred_producers, intersected with genre.
          2. Profiles whose sub_genres include the sub_genre tag.
          3. All profiles for the genre.
        Deterministic per song via seeded random.
        """
        genre_norm = (genre or "").lower().strip().replace(" ", "_").replace("-", "_")
        genre_canon = self._canonicalise_genre(genre_norm)
        sub_norm = (sub_genre or "").lower().strip().replace(" ", "_").replace("-", "_") or None

        genre_candidates = [p for p in self._producers if genre_canon in p.get("genres", [])]
        if not genre_candidates:
            logger.warning("MVDirector: no profile for genre=%s, falling back to first", genre)
            return dict(self._producers[0])

        preferred_ids: list[str] = []
        if sub_norm:
            mod = self._sub_genre_modifiers.get(sub_norm, {})
            preferred_ids = list(mod.get("preferred_producers", []))

        if preferred_ids:
            pool = [p for p in genre_candidates if p["id"] in preferred_ids]
            if pool:
                return dict(self._seeded_choice(pool, song_seed))

        if sub_norm:
            pool = [p for p in genre_candidates if sub_norm in p.get("sub_genres", [])]
            if pool:
                return dict(self._seeded_choice(pool, song_seed))

        return dict(self._seeded_choice(genre_candidates, song_seed))

    def apply_sub_genre_modifiers(self, profile: dict, sub_genre: Optional[str]) -> dict:
        """Return a copy of profile with sub-genre modifiers merged in."""
        if not sub_genre:
            return profile
        mod = self._sub_genre_modifiers.get(sub_genre.lower().strip(), {})
        if not mod:
            return profile

        out = json.loads(json.dumps(profile))  # deep copy
        if "closeup_freq_mult" in mod:
            out["closeup_frequency"] = min(0.95, out.get("closeup_frequency", 0.30) * mod["closeup_freq_mult"])
        if "closeup_freq_floor" in mod:
            out["closeup_frequency"] = max(out["closeup_frequency"], mod["closeup_freq_floor"])
        if "edit_tempo_override" in mod:
            out["edit_tempo"] = mod["edit_tempo_override"]
        if mod.get("film_grain_force"):
            out["film_grain"] = True
        if "lighting_bias" in mod:
            out.setdefault("lighting_recipes", []).insert(0, mod["lighting_bias"])
        for fb in mod.get("forbid_extra", []):
            out.setdefault("_extra_forbidden", []).append(fb)
        if "force_settings" in mod:
            out.setdefault("_forced_settings", []).extend(mod["force_settings"])
        out["_applied_sub_genre"] = sub_genre
        out["_modifier_aggression"] = mod.get("aggression_mult", 1.0)
        out["_modifier_intimacy"] = mod.get("intimacy_mult", 1.0)
        return out

    # ── Sub-genre + sentiment classifier ────────────────────────────

    async def classify_sub_genre_and_sentiment(
        self,
        lyrics: dict | str,
        genre: str,
        key: Optional[str],
        tempo: Optional[float],
        aligned_sections: Optional[list[dict]] = None,
        text_caller: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ) -> dict:
        """Classify song-level sub-genre + per-section sentiment.

        text_caller: async fn (system_prompt, user_prompt) -> response text.
                     Pass the existing OpenRouter wrapper (Stage J text_model).
                     If None, falls back to rule-based key/tempo classification.
        Returns: {sub_genre: str|None, per_section_sentiment: [{section_idx,label}]}
        """
        section_count = len(aligned_sections) if aligned_sections else self._count_sections(lyrics)

        if text_caller is None:
            return self._fallback_classify(genre, key, tempo, section_count, aligned_sections)

        sections_repr = self._aligned_sections_repr(aligned_sections, lyrics)
        system = (
            "You classify a song's sub-genre and the emotional sentiment of each section. "
            "Return ONLY valid JSON, no prose. Schema: "
            '{"sub_genre": "<one tag from list or null>", '
            '"per_section_sentiment": [{"section_idx": <int>, "label": "<sentiment>"}, ...]}. '
            f"Sentiment labels MUST be one of: {sorted(SENTIMENT_LABELS)}. "
            "Sub-genre MUST be one of: gangster_rap, drill, luxury_pop_rap, conscious, "
            "alternative_hip_hop, trap, soundcloud_rap, mumble_rap, rnb, power_ballad, "
            "grunge, post_punk, industrial_rock, dance_pop, dark_pop, or null if uncertain."
        )
        user = (
            f"Genre: {genre}\nKey: {key}\nTempo: {tempo}\n"
            f"Sections ({section_count}):\n{sections_repr}\n"
            "Classify."
        )
        try:
            raw = await text_caller(system, user)
            parsed = self._parse_json_block(raw)
            return self._validate_classification(parsed, section_count)
        except Exception as exc:
            logger.warning("MVDirector: LLM sentiment failed (%s), falling back", exc)
            return self._fallback_classify(genre, key, tempo, section_count, aligned_sections)

    def _fallback_classify(
        self,
        genre: str,
        key: Optional[str],
        tempo: Optional[float],
        section_count: int,
        aligned_sections: Optional[list[dict]],
    ) -> dict:
        sub_genre = self._guess_sub_genre_from_genre(genre)
        base_mood = KEY_TO_MOOD_FALLBACK.get((key or "").strip(), None)
        if base_mood is None:
            if tempo and tempo >= 120:
                base_mood = "anthemic"
            elif tempo and tempo <= 80:
                base_mood = "melancholic"
            else:
                base_mood = "neutral"

        per_section = []
        for idx in range(section_count):
            label = base_mood
            if aligned_sections and idx < len(aligned_sections):
                sec_label = (aligned_sections[idx].get("section_label") or "").lower()
                if "chorus" in sec_label and base_mood in {"neutral", "melancholic"}:
                    label = "anthemic" if base_mood == "neutral" else "sad"
                elif "intro" in sec_label or "outro" in sec_label:
                    label = "intimate" if base_mood in {"sad", "melancholic"} else "calm"
            per_section.append({"section_idx": idx, "label": label})
        return {"sub_genre": sub_genre, "per_section_sentiment": per_section}

    # ── Shot plan builder ────────────────────────────────────────────

    def build_shot_plan(
        self,
        aligned_sections: list[dict],
        profile: dict,
        sentiment: dict,
        song_seed: Optional[int | str] = None,
    ) -> list[dict]:
        """Per aligned section, sample a shot type honoring distribution and
        variety guardrails:
          - no two consecutive sections share shot_type
          - close-up quota = ceil(closeup_frequency * N)
          - at least one establishing shot in first 3 sections
          - sad/melancholic chorus segments forced to CU/ECU
        """
        rng = random.Random(self._seed_value(song_seed))
        distribution = profile.get("shot_distribution") or self._uniform_distribution()
        closeup_freq = profile.get("closeup_frequency", 0.30)
        n = len(aligned_sections)
        if n == 0:
            return []

        section_sentiment = {
            entry["section_idx"]: entry["label"]
            for entry in sentiment.get("per_section_sentiment", [])
        }

        required_closeups = max(1, int(round(closeup_freq * n)))
        plan: list[dict] = []
        used_count = {s: 0 for s in SHOT_CODES}
        last_shot: Optional[str] = None
        establishing_placed = False

        for idx, sec in enumerate(aligned_sections):
            label = (sec.get("section_label") or sec.get("label") or "").lower()
            role = sec.get("role") or sec.get("portrait_role") or "story"
            mood_tag = section_sentiment.get(idx, "neutral")

            forced: Optional[str] = None

            if not establishing_placed and idx < 3:
                if "intro" in label or idx == 0:
                    forced = "WS" if rng.random() < 0.7 else "LS"
                    establishing_placed = True

            if "chorus" in label and mood_tag in {"sad", "melancholic", "intimate"}:
                forced = "ECU" if rng.random() < 0.3 else "CU"

            if mood_tag == "aggressive" and "chorus" in label and forced is None:
                forced = "MCU" if rng.random() < 0.5 else "CU"

            remaining = n - idx
            remaining_closeups_needed = max(
                0, required_closeups - (used_count["CU"] + used_count["ECU"])
            )
            if remaining_closeups_needed >= remaining and forced not in CLOSEUP_SHOTS:
                forced = "CU"

            if forced is not None:
                shot = forced
            else:
                shot = self._sample_shot(rng, distribution, last_shot)

            if last_shot == shot:
                if shot in CLOSEUP_SHOTS:
                    shot = "ECU" if last_shot == "CU" else "CU"
                else:
                    shot = self._sample_shot(rng, distribution, last_shot, avoid=last_shot)

            lighting = self._pick_for_mood(profile, mood_tag, key="mood_mapping", fallback_key="lighting_recipes", rng=rng)
            motion = self._seeded_pick(profile.get("motion_preference") or ["static_locked"], rng)
            framing_note = self._framing_note(shot, role, mood_tag, label, profile)

            forbidden = list(profile.get("forbidden_for_genres", {}).values())
            forbidden_flat = [f for sub in forbidden for f in sub] + profile.get("_extra_forbidden", [])

            plan.append({
                "section_idx": idx,
                "section_label": sec.get("section_label") or sec.get("label") or f"Section_{idx}",
                "role": role,
                "shot_type": shot,
                "motion": motion,
                "lighting": lighting,
                "mood_tag": mood_tag,
                "framing_note": framing_note,
                "forbidden": forbidden_flat,
            })
            used_count[shot] = used_count.get(shot, 0) + 1
            last_shot = shot

        return plan

    # ── Director brief renderer ──────────────────────────────────────

    def render_director_brief(
        self,
        profile: dict,
        shot_plan: list[dict],
        song_genre: str,
    ) -> str:
        """Format profile + shot_plan as a numbered RULES block for LLM system prompt."""
        lines: list[str] = []
        lines.append(f"Producer style reference: **{profile.get('name')}** — {profile.get('ethos_oneliner','')}")
        lines.append(f"Edit tempo: {profile.get('edit_tempo')}  |  Film grain: {profile.get('film_grain')}")
        lines.append(f"Color palette: {', '.join(profile.get('color_palette', [])[:6])}")
        lines.append(f"Lighting recipes: {', '.join(profile.get('lighting_recipes', [])[:5])}")
        if profile.get("_forced_settings"):
            lines.append(f"Required settings: {', '.join(profile['_forced_settings'])}")

        lines.append("")
        lines.append("PER-SEGMENT SHOT DIRECTIVES (you MUST follow these):")
        lines.append("| idx | section | role | shot | motion | lighting | mood | framing |")
        for entry in shot_plan:
            lines.append(
                f"| {entry['section_idx']} | {entry['section_label']} | {entry['role']} "
                f"| {entry['shot_type']} | {entry['motion']} | {entry['lighting']} "
                f"| {entry['mood_tag']} | {entry['framing_note']} |"
            )

        forbidden_song_level = self._collect_forbidden(profile, song_genre)
        if forbidden_song_level:
            lines.append("")
            lines.append("FORBIDDEN VISUAL ELEMENTS for this song (do NOT depict any of these):")
            lines.append(", ".join(sorted(set(forbidden_song_level))))

        lines.append("")
        lines.append("HARD RULES:")
        lines.append("1. Use the exact shot type given for each segment in the directive table above.")
        lines.append("2. No two consecutive segments may share the same shot type.")
        n = len(shot_plan)
        cu_required = sum(1 for e in shot_plan if e["shot_type"] in CLOSEUP_SHOTS)
        lines.append(f"3. Close-up quota: at least {cu_required} of {n} segments use CU or ECU on the vocalist's face.")
        lines.append("4. At least one of the first three segments MUST be a wide or long shot (establishing).")
        lines.append("5. Sad/melancholic chorus segments MUST show vocalist emotion via CU/ECU (tear, closed eyes, lip tremor, hands to face).")
        if profile.get("story_archetypes"):
            lines.append(
                f"6. Story segments MUST match producer archetypes: {', '.join(profile['story_archetypes'])}."
            )
        lines.append("7. Honor the FORBIDDEN list — those elements may NEVER appear in any segment.")

        return "\n".join(lines)

    # ── Helpers ─────────────────────────────────────────────────────

    def _canonicalise_genre(self, genre_norm: str) -> str:
        if any(k in genre_norm for k in ("hip_hop", "hiphop", "rap", "trap", "drill")):
            return "hip_hop"
        if any(k in genre_norm for k in ("rock", "metal", "punk", "grunge")):
            return "rock"
        if any(k in genre_norm for k in ("pop", "rnb", "soul", "dance", "edm", "electronic")):
            return "pop"
        return "pop"

    def _guess_sub_genre_from_genre(self, genre: str) -> Optional[str]:
        g = (genre or "").lower()
        order = [
            ("gangster", "gangster_rap"), ("drill", "drill"), ("trap", "trap"),
            ("conscious", "conscious"), ("mumble", "mumble_rap"),
            ("luxury", "luxury_pop_rap"), ("rnb", "rnb"), ("r&b", "rnb"),
            ("ballad", "power_ballad"), ("grunge", "grunge"),
            ("post-punk", "post_punk"), ("post punk", "post_punk"),
            ("industrial", "industrial_rock"), ("dance pop", "dance_pop"),
            ("dark pop", "dark_pop"), ("synth", "dance_pop"),
        ]
        for needle, tag in order:
            if needle in g:
                return tag
        return None

    def _count_sections(self, lyrics: Any) -> int:
        if isinstance(lyrics, dict):
            return max(1, len(lyrics))
        if isinstance(lyrics, str):
            tags = re.findall(r"\[([^\]]+)\]", lyrics)
            return max(1, len(tags))
        if isinstance(lyrics, list):
            return max(1, len(lyrics))
        return 1

    def _aligned_sections_repr(self, aligned: Optional[list[dict]], lyrics: Any) -> str:
        if aligned:
            rows = []
            for idx, sec in enumerate(aligned):
                label = sec.get("section_label") or sec.get("label") or f"Section_{idx}"
                role = sec.get("role") or sec.get("portrait_role") or "story"
                text = (sec.get("text") or sec.get("lyrics") or "")[:140]
                rows.append(f"  [{idx}] {label} ({role}): {text}")
            return "\n".join(rows)
        if isinstance(lyrics, dict):
            return "\n".join(f"  [{i}] {k}: {str(v)[:140]}" for i, (k, v) in enumerate(lyrics.items()))
        return str(lyrics)[:1000]

    def _parse_json_block(self, raw: str) -> dict:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("no JSON found in LLM response")
        return json.loads(match.group(0))

    def _validate_classification(self, data: dict, n: int) -> dict:
        sub_genre = data.get("sub_genre")
        if sub_genre is not None and not isinstance(sub_genre, str):
            sub_genre = None
        sentiment_in = data.get("per_section_sentiment") or []
        out_sentiment = []
        for entry in sentiment_in:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("section_idx")
            lbl = entry.get("label")
            if not isinstance(idx, int) or not (0 <= idx < n):
                continue
            if not isinstance(lbl, str) or lbl not in SENTIMENT_LABELS:
                lbl = "neutral"
            out_sentiment.append({"section_idx": idx, "label": lbl})
        if len(out_sentiment) < n:
            covered = {e["section_idx"] for e in out_sentiment}
            for i in range(n):
                if i not in covered:
                    out_sentiment.append({"section_idx": i, "label": "neutral"})
        out_sentiment.sort(key=lambda e: e["section_idx"])
        return {"sub_genre": sub_genre, "per_section_sentiment": out_sentiment}

    def _seed_value(self, seed: Optional[int | str]) -> int:
        if seed is None:
            return 0
        if isinstance(seed, int):
            return seed
        return abs(hash(str(seed))) % (2**31)

    def _seeded_choice(self, pool: list, seed: Optional[int | str]):
        if not pool:
            raise ValueError("empty pool for seeded_choice")
        rng = random.Random(self._seed_value(seed))
        return rng.choice(pool)

    def _seeded_pick(self, pool: Iterable[str], rng: random.Random) -> str:
        items = list(pool)
        if not items:
            return ""
        return rng.choice(items)

    def _sample_shot(
        self,
        rng: random.Random,
        distribution: dict[str, float],
        last_shot: Optional[str],
        avoid: Optional[str] = None,
    ) -> str:
        weighted = [(s, max(0.0, distribution.get(s, 0.0))) for s in SHOT_CODES]
        if avoid:
            weighted = [(s, w if s != avoid else 0.0) for s, w in weighted]
        if last_shot:
            weighted = [(s, w * (0.4 if s == last_shot else 1.0)) for s, w in weighted]
        total = sum(w for _, w in weighted)
        if total <= 0:
            return rng.choice([s for s in SHOT_CODES if s != avoid] or SHOT_CODES)
        r = rng.random() * total
        cum = 0.0
        for s, w in weighted:
            cum += w
            if r <= cum:
                return s
        return weighted[-1][0]

    def _uniform_distribution(self) -> dict[str, float]:
        return {s: 1.0 / len(SHOT_CODES) for s in SHOT_CODES}

    def _pick_for_mood(
        self,
        profile: dict,
        mood_tag: str,
        key: str,
        fallback_key: str,
        rng: random.Random,
    ) -> str:
        mood_map = profile.get(key) or {}
        if mood_tag in mood_map and mood_map[mood_tag]:
            return self._seeded_pick(mood_map[mood_tag], rng)
        fallback = profile.get(fallback_key) or []
        if fallback:
            return self._seeded_pick(fallback, rng)
        return "natural_light"

    def _framing_note(self, shot: str, role: str, mood: str, label: str, profile: dict) -> str:
        is_vocal = role in {"male", "female", "duet"}
        if shot == "ECU" and is_vocal and mood in {"sad", "melancholic"}:
            return "tight on vocalist eye or single tear, slow micro-expression"
        if shot == "ECU" and is_vocal and mood == "intimate":
            return "macro on lips mid-lyric, soft focus"
        if shot == "CU" and is_vocal and mood in {"sad", "melancholic"}:
            return "vocalist face filling frame, eyes closed, hands to chest"
        if shot == "CU" and is_vocal and mood == "aggressive":
            return "vocalist face, jaw clenched, hard rim light"
        if shot in {"WS", "LS"} and "chorus" in label:
            return f"wide framing showing full body + environment, {profile.get('opening_pattern','establishing')[:60]}"
        if shot == "MS" and is_vocal:
            return "vocalist waist-up performing to camera"
        if shot == "MCU" and is_vocal:
            return "vocalist shoulders-up, environment context visible"
        if not is_vocal:
            return f"story tableau honoring archetype: {(profile.get('story_archetypes') or ['narrative'])[0]}"
        return f"{shot} framing matching mood {mood}"

    def _collect_forbidden(self, profile: dict, song_genre: str) -> list[str]:
        out: list[str] = []
        for fb_genre, fb_list in (profile.get("forbidden_for_genres") or {}).items():
            if fb_genre == song_genre:
                out.extend(fb_list)
        out.extend(profile.get("_extra_forbidden", []))
        universal = self._universal_forbidden.get(self._canonicalise_genre(song_genre.lower()), [])
        out.extend(universal)
        return out


__all__ = ["MVDirector", "DEFAULT_PROFILES_PATH"]
