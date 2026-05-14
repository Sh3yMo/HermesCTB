from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
import asyncio
import logging
import time
import re
import json
import base64

import requests
import httpx

logger = logging.getLogger(__name__)


DIRECTOR_PRESETS = {
    "none": {"label": "No Director", "hint": "neutral cinematic look, no franchise style cues, balanced color grading"},
    "kurosawa": {"label": "Akira Kurosawa", "hint": "epic widescreen compositions, weather-as-emotion tendency (atmospheric elements reinforce mood), samurai-era texture, bold graphic framing, Rashomon / Seven Samurai grandeur"},
    "blade_runner": {"label": "Blade Runner", "hint": "neo-noir atmosphere, wet reflective surfaces tendency, neon haze, blue-orange split-toning, anamorphic lens flares"},
    "nolan": {"label": "Christopher Nolan", "hint": "epic scale, practical realism, high contrast, IMAX large-format feel, deep focus clarity"},
    "fincher": {"label": "David Fincher", "hint": "cool desaturated palette, precise framing, psychological mood, underexposed shadows, Kodak Vision3 500T aesthetic"},
    "villeneuve": {"label": "Denis Villeneuve", "hint": "minimalist grandeur, atmospheric scale, desaturated earth tones, large-format sensor clarity, Roger Deakins-inspired natural light"},
    "documentary": {"label": "Documentary", "hint": "observational verite, handheld realism, minimal color grading, photojournalistic authenticity"},
    "del_toro": {"label": "Guillermo del Toro", "hint": "dark fairy-tale gothic, amber and teal palette, ornate creature design, baroque production detail, Pan's Labyrinth magical realism"},
    "hollywood_blockbuster": {"label": "Hollywood Blockbuster", "hint": "polished high production value, dynamic three-point lighting, teal-orange color grade, sharp detail"},
    "indie_film": {"label": "Indie Film", "hint": "organic naturalistic visuals, intimate realism, Kodak Portra 400 color science, available-light aesthetic"},
    "cameron": {"label": "James Cameron", "hint": "photorealistic spectacle, deep ocean blue palette, cutting-edge VFX realism, epic action scale, Terminator / Avatar technological grandeur"},
    "matrix": {"label": "Matrix", "hint": "cyberpunk digital aesthetic, desaturated green-shifted color tendency, high-contrast dystopian noir, digital rain visual language"},
    "miyazaki": {"label": "Miyazaki / Ghibli", "hint": "Studio Ghibli hand-painted watercolor aesthetic, lush nature detail, whimsical warmth, soft pastel skies, expressive character animation, nostalgic wonder"},
    "tarantino": {"label": "Quentin Tarantino", "hint": "retro stylization, bold saturation, punchy cinematic energy, Kodak Ektachrome warmth, wide-angle compositions"},
    "ridley_scott": {"label": "Ridley Scott", "hint": "atmospheric smoke and haze, industrial texture, blue-steel color grade, massive scale architecture, Alien / Gladiator visual density"},
    "kubrick": {"label": "Stanley Kubrick", "hint": "one-point perspective symmetry, cold clinical precision, wide-angle distortion, stark overhead lighting, unsettling stillness, 2001 / Shining aesthetic"},
    "spielberg": {"label": "Steven Spielberg", "hint": "warm golden Amblin glow, awe-struck low-angle reveals, sweeping John Williams-style orchestral grandeur, lens flares, sentimental Americana"},
    "malick": {"label": "Terrence Malick", "hint": "golden-hour natural light preference, flowing Steadicam organic movement, whispered voiceover intimacy, wide-angle nature reverence, Tree of Life transcendence"},
    "burton": {"label": "Tim Burton", "hint": "gothic expressionist whimsy, heavily desaturated muted palette with selective color pops, exaggerated angular architecture, pale skin deep shadows, Danny Elfman carnival mood"},
    "wes_anderson": {"label": "Wes Anderson", "hint": "symmetrical framing, pastel palette, whimsical vintage tone, Fuji Pro 400H color science"},
    "wong_kar_wai": {"label": "Wong Kar-wai", "hint": "smeared neon reflections, step-printed motion blur, melancholic saturated reds and greens, intimate handheld framing, In the Mood for Love romantic longing"},
    "snyder": {"label": "Zack Snyder", "hint": "speed-ramped slow motion, hyper-stylized desaturated palette, dramatic backlit silhouettes, painterly comic-book compositions, 300 / ZSJL aesthetic"},
}


FILMSTYLE_PRESETS = {
    "none": {"label": "No Film Style", "hint": "no specific film style applied, standard cinematic rendering"},
    # ── 3D Animation ──
    "cartoon_2d": {"label": "2D Cartoon", "hint": "flat 2D cartoon animation, bold outlines, solid color fills, exaggerated squash-and-stretch motion, classic western animation style"},
    "disney_3d": {"label": "Disney 3D", "hint": "Disney 3D animation style, soft rounded character designs, magical glowing particle effects, lush detailed environments, warm fairy-tale color palette"},
    "dreamworks": {"label": "DreamWorks", "hint": "DreamWorks-style 3D CGI rendering, exaggerated proportions, dynamic action poses, sharp comedic timing in expressions, slightly edgier and bolder colors than Pixar"},
    "pixar": {"label": "Pixar", "hint": "Pixar-style 3D CGI rendering, subsurface scattering on skin, expressive character design, vibrant saturated palette, clean geometric environments, warm storytelling aesthetic"},
    # ── Anime ──
    "anime_cel": {"label": "Cel-Shaded Anime", "hint": "traditional cel-shaded anime rendering, flat color fills with sharp hard-edged shadows, minimal gradients, bold black outlines, stylized non-photorealistic look"},
    "anime_chibi": {"label": "Chibi / SD Anime", "hint": "super-deformed chibi anime proportions, oversized heads, tiny bodies, exaggerated cute expressions, pastel soft colors, comedic and adorable aesthetic"},
    "anime_classic": {"label": "Classic 90s Anime", "hint": "1990s anime aesthetic, slightly grainy cel animation texture, warm analog color palette, hand-drawn linework imperfections, Cowboy Bebop / Akira era visual style"},
    "anime_modern": {"label": "Modern Anime", "hint": "modern digital anime style, clean sharp linework, vivid colors, detailed light shafts and lens flares, Makoto Shinkai-inspired atmospheric backgrounds with photorealistic skies"},
    # ── Artistic ──
    "comic_book": {"label": "Comic Book / Graphic Novel", "hint": "comic book panel aesthetic, bold ink outlines, halftone dot shading, vibrant pop colors, dynamic action lines, Sin City / Spider-Verse inspired"},
    "oil_painting": {"label": "Oil Painting", "hint": "classical oil painting in motion, visible thick brush strokes, rich pigment colors, Renaissance chiaroscuro lighting, canvas texture visible"},
    "pixel_art": {"label": "Pixel Art", "hint": "retro pixel-art animation style, limited 16-bit color palette, visible square pixels, sprite-based character design, nostalgic 90s video game aesthetic"},
    "rotoscope": {"label": "Rotoscope Animation", "hint": "rotoscoped animation over live-action, fluid painterly outlines tracing real motion, A Scanner Darkly / Waking Life shifting reality aesthetic"},
    "ukiyo_e": {"label": "Ukiyo-e / Japanese Woodblock", "hint": "traditional Japanese woodblock print style, flat perspective, bold black outlines, limited earth-tone palette with indigo and vermillion, wave and nature motifs, Hokusai aesthetic"},
    "watercolor": {"label": "Watercolor Painting", "hint": "animated watercolor painting aesthetic, soft bleeding edges, visible paper texture, translucent color washes, impressionistic brush strokes"},
    # ── Classic & Genre ──
    "film_noir_classic": {"label": "Classic Film Noir", "hint": "black and white high-contrast cinematography, Venetian blind shadow patterns, cigarette smoke haze, 1940s detective genre aesthetic"},
    "gothic_horror": {"label": "Gothic Horror", "hint": "deep crimson and black palette, flickering warm practical light sources, creeping fog tendency, Hammer Horror atmosphere, baroque decay, oppressive shadow density"},
    "grindhouse": {"label": "Grindhouse / B-Movie", "hint": "scratched 35mm exploitation film aesthetic, missing frames, cigarette burn reel marks, oversaturated colors, cheap practical effects look, 70s drive-in cinema feel"},
    "neo_noir": {"label": "Neo-Noir", "hint": "modern noir cinematography, wet reflective surface tendency, neon color spill, deep shadows with selective color pops, high contrast urban aesthetic, Drive / Collateral mood"},
    "silent_film": {"label": "Silent Film Era", "hint": "vintage 1920s silent film aesthetic, black and white with sepia tint, film grain and scratches, exaggerated theatrical performances, title card style"},
    "western": {"label": "Western / Spaghetti", "hint": "sun-bleached warm earth-tone palette, dust particle haze, extreme close-up eye details, Ennio Morricone tension, Leone wide-shot standoff compositions, 70mm anamorphic scope"},
    # ── Contemporary ──
    "drone_aerial": {"label": "Drone / Aerial", "hint": "high-altitude drone cinematography, sweeping bird's-eye establishing shots, vast landscape scale, smooth gimbal-stabilized gliding motion", "camera_locked": True},
    "mockumentary": {"label": "Mockumentary", "hint": "documentary-style talking head framing, candid zoom-ins on reactions, fourth-wall breaking glances, The Office / Parks and Rec naturalistic comedy style"},
    "music_video": {"label": "Music Video", "hint": "stylized rapid-cut montage, bold color grading shifts, performance-stage lighting, lens flares, high-fashion editorial framing, MTV aesthetic"},
    "underwater": {"label": "Underwater", "hint": "submerged aquatic cinematography, light caustics dancing on surfaces, floating particle debris, blue-green color cast, slow fluid motion"},
    # ── Found Footage / POV ──
    "bodycam": {"label": "Bodycam / Police", "hint": "chest-mounted bodycam perspective, wide-angle fisheye distortion, harsh flashlight illumination, shaky movement, timestamp and badge number overlay", "camera_locked": True},
    "found_footage": {"label": "Found Footage", "hint": "handheld shaky cam, low-fi video texture, timestamp overlay, VHS tracking artifacts, Blair Witch / Cloverfield aesthetic"},
    "pov_subjective": {"label": "POV / Subjective", "hint": "first-person immersive perspective throughout, body-mounted camera sway, hands/arms occasionally visible in frame, Hardcore Henry style", "camera_locked": True},
    # ── Retro & Cyberpunk ──
    "cyberpunk": {"label": "Cyberpunk", "hint": "futuristic high-tech aesthetic, holographic UI elements, magenta-cyan neon haze, chrome and glass surface reflections, Blade Runner 2049 / Ghost in the Shell dystopian tech noir"},
    "steampunk": {"label": "Steampunk", "hint": "Victorian-era brass and copper machinery, clockwork gears, steam vents, sepia-toned warm palette, dirigibles and goggles, ornate industrial fantasy aesthetic"},
    "super8": {"label": "Super 8mm Film", "hint": "Super 8mm home movie aesthetic, heavy film grain, warm amber color cast, light leaks, slight overexposure, vintage 1970s family footage feel"},
    "synthwave": {"label": "Synthwave / Retrowave", "hint": "80s retro-futuristic neon grid landscape, hot pink and electric blue palette, chrome reflections, sunset gradient sky, VHS scanline overlay, outrun aesthetic"},
    "vhs_horror": {"label": "VHS Horror", "hint": "degraded VHS horror aesthetic, heavy tracking glitches, washed-out colors, sudden static bursts, found-tape dread, Ring / V/H/S style"},
    "vhs_retro": {"label": "VHS / Retro", "hint": "analog VHS tape aesthetic, color bleeding, scan-line noise, tracking distortion, 4:3 aspect ratio feel, 80s/90s home video warmth"},
    # ── Stop Motion ──
    "claymation": {"label": "Claymation", "hint": "clay figure animation, visible fingerprint indentations on surfaces, plasticine material texture, slightly jerky frame-by-frame motion, Wallace & Gromit style"},
    "stop_motion": {"label": "Stop Motion", "hint": "stop-motion puppet animation look, visible material textures (felt, clay, wood), subtle frame stutter, miniature set design, Laika / Aardman aesthetic"},
    # ── Surveillance & Broadcast ──
    "news_broadcast": {"label": "News Broadcast", "hint": "live television news broadcast aesthetic, lower-third chyron graphics, flat studio lighting, shoulder-camera field reporting, 24-hour news channel visual language"},
    "screenlife": {"label": "Screenlife / Desktop", "hint": "entire frame is a computer or phone screen, cursor movements, app windows, video calls, notification pop-ups, Unfriended / Searching digital-native storytelling", "camera_locked": True},
    "surveillance": {"label": "Surveillance / Dashcam", "hint": "fixed wide-angle security camera perspective, low resolution grain, timestamp overlay, static unmanned framing", "camera_locked": True},
    "thermal_imaging": {"label": "Thermal / Infrared", "hint": "false-color thermal imaging palette, heat signatures visible on bodies, cold environment contrast, military/scientific observation aesthetic"},
    "war_footage": {"label": "War Documentary Footage", "hint": "embedded combat camera aesthetic, desaturated gritty palette, shaky handheld urgency, dust and debris particles, Black Hawk Down / Hurt Locker realism"},
}


CAMERA_PRESETS = {
    "extreme_close_up": "extreme close-up macro shot, 100mm macro lens equivalent, razor-thin depth of field on fine detail",
    "close_up": "close-up portrait framing, 85mm f/1.4 lens equivalent, shallow depth of field with creamy bokeh",
    "medium_shot": "medium shot from waist up, 50mm standard lens equivalent, balanced depth of field",
    "full_shot": "full body shot, 35mm lens equivalent, moderate depth of field showing full subject and environment context",
    "wide_shot": "wide establishing shot, 24mm lens equivalent, deep focus showing subject within environment",
    "extreme_wide": "extreme wide cinematic landscape shot, 16mm ultra-wide lens equivalent, deep focus, environmental dominance",
    "top_down": "top-down overhead camera angle, bird's-eye perspective, geometric composition emphasis",
    "low_angle": "low-angle heroic perspective, worm's-eye view, subject appears powerful and dominant",
    "high_angle": "high-angle vulnerable perspective, looking down at subject, diminishing scale",
    "dutch_angle": "dutch-angle tilted horizon for tension and instability, dynamic diagonal composition",
    "over_shoulder": "over-the-shoulder composition, 50mm equivalent, foreground bokeh framing with selective focus on subject",
    "pov": "first-person POV subjective framing, wide-angle immersive perspective",
}


LIGHT_PRESETS = {
    "natural": "natural daylight lighting, open shade softness, balanced color temperature",
    "golden_hour": "golden hour warm sunlight, low sun angle casting long shadows, warm skin tones with gentle shadow gradation",
    "blue_hour": "blue hour twilight tone, cool ambient fill, deep blue sky gradient with residual warm horizon light",
    "three_point": "classic three-point studio lighting: key light at 45 degrees, fill light opposite, rim light for subject separation",
    "dramatic": "dramatic Rembrandt chiaroscuro, strong key light from 45 degrees creating triangle shadow, deep contrast ratio",
    "backlit": "strong backlit silhouette edges, rim light outlining subject contours, lens flare potential, exposed-for-highlights look",
    "neon": "colored neon practical lighting, mixed color temperature, cyberpunk-influenced RGB ambient spill",
    "volumetric": "volumetric god-rays and atmospheric beams, visible light shafts through haze/fog/dust particles",
    "soft": "soft diffused beauty lighting, large soft source overhead, minimal shadows, flattering even illumination",
    "low_key": "low-key dark cinematic lighting, predominantly shadow with selective highlights, noir exposure style",
    "high_key": "high-key bright airy lighting, minimal shadows, clean luminous aesthetic, overexposed background",
    "practical": "practical in-scene light sources only, motivated lighting from visible lamps/candles/screens, naturalistic falloff",
}


MOOD_PRESETS = {
    "epic": "epic, grand, heroic tone",
    "mysterious": "mysterious suspenseful tone",
    "melancholic": "melancholic reflective tone",
    "intense": "intense dramatic tone",
    "peaceful": "peaceful calm tone",
    "ominous": "ominous unsettling tone",
    "romantic": "romantic intimate tone",
    "action": "high-energy action tone",
    "horror": "disturbing horror tone",
    "whimsical": "playful whimsical tone",
}

MOTION_PRESETS = {
    "none": "no motion preset; follow user-described movement only, otherwise keep natural subtle motion",
    "static": "locked static camera, no intentional movement",
    "push_in": "slow cinematic push-in toward the subject",
    "orbit": "smooth orbital camera move around the subject",
    "dolly_left": "controlled dolly move to camera-left",
    "fly_over": "forward fly-over move passing above/over the subject",
    "handheld": "subtle handheld micro-shake, documentary realism",
    "shoulder_reveal": "start behind the subject shoulder, then glide forward to reveal the scene ahead",
    "flyover_reveal": "forward fly-over across/above subject with clear background reveal timing",
    "push_through": "camera pushes through foreground layers/obstacles into the main scene reveal",
    "crane_up_reveal": "crane upward move that progressively reveals larger environment scale",
    "parallax_slide": "lateral slide with strong foreground/background parallax depth reveal",
}

CINEMATIC_PATTERN_PRESETS = {
    "none": "no fixed beat pattern; keep progression natural and coherent",
    "dialogue_coverage": "conversation rhythm within a single shot: camera gently drifts between speakers, subtle focus shifts emphasize reactions, steady framing holds the spatial relationship",
    "action_choreo": "continuous action energy: camera tracks movement with fluid urgency, maintains spatial clarity throughout the choreography, dynamic pacing within one unbroken take",
    "impact_emphasis": "building anticipation through slow deliberate approach, sharp sudden motion burst on impact, settling into aftermath stillness",
    "one_take_flow": "single continuous take: smooth unbroken camera movement with organic transitions, no hard cuts or reframing, flowing spatial exploration",
    "epic_reveal": "gradual scale revelation: start tight on subject, camera slowly pulls back or rises to unveil the full environment, culminating in grand wide perspective",
    "suspense_build": "tension ramp: restrained minimal motion gradually escalating to faster urgent movement, building unease through pacing acceleration",
    "martial_arts_coverage": "steady tracking of fighters maintaining spatial clarity, fluid camera following strike-and-response rhythms, readable body mechanics throughout continuous motion",
    "impact_cutaway": "sustained focus on action with momentary lingering on impact details: strike connects, camera holds briefly on contact point, then follows through to aftermath",
    "bullet_time_beat": "deliberate slow fluid motion during key moment, emphasizing weight and detail of the action, graceful deceleration then smooth return to normal speed",
    "fight_one_take": "continuous unbroken fight choreography: camera orbits and weaves around combatants, maintaining spatial logic and action readability in one flowing take",
    "sports_broadcast_fight": "clear stable framing of combat with consistent spatial orientation, wide enough to read the full exchange, steady tracking that prioritizes action clarity over style",
}


@dataclass
class EnhancementResult:
    final_prompt: str
    used_llm: bool
    used_fallback: bool
    status_message: str
    route_steps: list[str]


class PromptEnhancer:
    def __init__(self, config: Dict[str, Any]):
        self.enabled = bool(config.get("enabled", True))
        self.openrouter_enabled = bool(config.get("openrouter_enabled", True))
        self.openrouter_api_key = str(config.get("openrouter_api_key", "")).strip()
        self.openrouter_model = str(
            config.get("openrouter_model", "qwen/qwen3-next-80b-a3b-instruct:free")
        ).strip()
        self.openrouter_base_url = str(
            config.get("openrouter_base_url", "https://openrouter.ai/api/v1/chat/completions")
        ).strip()
        self.request_timeout_seconds = int(config.get("request_timeout_seconds", 30))
        self.max_retries = int(config.get("max_retries", 5))
        self.retry_backoff_seconds = config.get("retry_backoff_seconds", [30, 60, 120, 300])
        self.fallback_models = config.get("fallback_models", ["openrouter/free"])
        self.disable_reasoning = bool(config.get("disable_reasoning", True))
        self.reasoning_effort = str(config.get("reasoning_effort", "none")).strip()
        self.reasoning_exclude = bool(config.get("reasoning_exclude", True))
        self.translate_on_local_fallback = bool(config.get("translate_on_local_fallback", True))
        self.translation_max_tokens = int(config.get("translation_max_tokens", 180))
        self.enhancement_max_tokens = int(config.get("enhancement_max_tokens", 700))
        self.cinematic_writing_mode = self._normalize_cinematic_mode(
            str(config.get("cinematic_writing_mode", "strict_cinematic")).strip().lower()
        )
        self.vision_enabled = bool(config.get("vision_enabled", False))
        self.vision_model = str(
            config.get("vision_model", "qwen/qwen-2.5-vl-7b-instruct")
        ).strip()
        self.vision_fallback_models = config.get("vision_fallback_models", [])
        self.vision_max_tokens = int(config.get("vision_max_tokens", 500))
        self.vision_image_detail = str(config.get("vision_image_detail", "high")).strip()
        self.unusable_output_failures_for_cooldown = int(
            config.get("unusable_output_failures_for_cooldown", 1)
        )
        self.unusable_output_cooldown_seconds = int(
            config.get("unusable_output_cooldown_seconds", 900)
        )
        if not isinstance(self.retry_backoff_seconds, list) or not self.retry_backoff_seconds:
            self.retry_backoff_seconds = [30, 60, 120, 300]
        if not isinstance(self.fallback_models, list):
            self.fallback_models = ["openrouter/free"]
        if not isinstance(self.vision_fallback_models, list):
            self.vision_fallback_models = []
        # Idea Mode
        self.idea_mode_enabled: bool = config.get("idea_mode_enabled", True)
        self.idea_mode_ask_questions: bool = config.get("idea_mode_ask_questions", False)
        self.idea_mode_max_questions: int = config.get("idea_mode_max_questions", 2)
        self._model_unusable_failures: Dict[str, int] = {}
        self._model_cooldown_until: Dict[str, float] = {}

    def _normalize_cinematic_mode(self, mode: str) -> str:
        legacy_map = {
            "balanced": "strict_cinematic",
            "rich": "cinematic",
        }
        normalized = legacy_map.get(str(mode).strip().lower(), str(mode).strip().lower())
        if normalized not in {"strict", "cinematic", "strict_cinematic"}:
            normalized = "strict_cinematic"
        return normalized

    def _resolve_cinematic_writing_mode(self, selections: Dict[str, Any] | None = None) -> str:
        mode = str((selections or {}).get("cinematic_writing_mode", self.cinematic_writing_mode)).strip().lower()
        return self._normalize_cinematic_mode(mode)

    def _cinematic_writing_instruction(self, mode: str, model_family: str = "generic") -> str:
        """Return cinematic writing instruction adapted per model family."""
        if model_family == "ltx":
            if mode == "strict":
                return (
                    "Writing mode STRICT: keep phrasing concise and technical. "
                    "Short direct sentences, minimal adjectives, chronological order. "
                    "Prioritize clarity over atmosphere."
                )
            if mode == "cinematic":
                return (
                    "Writing mode CINEMATIC: use expressive cinematic prose with vivid atmosphere. "
                    "Describe light, texture, and mood evocatively — like a film critic narrating a scene. "
                    "Maintain chronological flow within the single paragraph."
                )
            return (
                "Writing mode STRICT_CINEMATIC: chronological and precise, but with cinematic film language. "
                "Keep events in clear temporal order while using evocative visual descriptions. "
                "Balance technical precision with filmic readability."
            )
        if model_family == "flux":
            if mode == "strict":
                return (
                    "Writing mode STRICT: front-load subject and action, minimal adjectives. "
                    "Keep it concise and direct — every word should serve the image."
                )
            if mode == "cinematic":
                return (
                    "Writing mode CINEMATIC: use rich visual language and atmospheric description. "
                    "Evoke mood through specific material textures, light quality, and color."
                )
            return (
                "Writing mode STRICT_CINEMATIC: front-load key elements but describe them with film language. "
                "Concise yet evocative."
            )
        if model_family == "pony":
            if mode == "strict":
                return (
                    "Writing mode STRICT: favor descriptive tags over full sentences. "
                    "Keep it compact — tag-like phrasing separated by commas."
                )
            if mode == "cinematic":
                return (
                    "Writing mode CINEMATIC: mix natural language with tags for richer description. "
                    "Use descriptive phrases for mood and atmosphere between tags."
                )
            return (
                "Writing mode STRICT_CINEMATIC: balanced tag + descriptive phrase style. "
                "Core elements as tags, atmosphere as short descriptive phrases."
            )
        # Generic fallback
        if mode == "strict":
            return "Writing mode STRICT: concise, technical, plain phrasing."
        if mode == "cinematic":
            return "Writing mode CINEMATIC: expressive cinematic prose with vivid atmosphere."
        return "Writing mode STRICT_CINEMATIC: structural clarity with cinematic wording."

    @staticmethod
    def _infer_speaker(context: str) -> str | None:
        """Infer likely speaker gender from surrounding text using word-boundary matching."""
        c = context.lower()
        female_markers = [
            r"\bdie frau\b", r"\bfrau\b", r"\bsie\b", r"\bshe\b",
            r"\bwoman\b", r"\bfemale\b", r"\bihr\b", r"\bihre\b",
        ]
        male_markers = [
            r"\bder mann\b", r"\bmann\b", r"\ber\b", r"\bhe\b",
            r"\bman\b", r"\bmale\b", r"\bsein\b", r"\bseine\b",
        ]
        female_hits = sum(1 for m in female_markers if re.search(m, c))
        male_hits = sum(1 for m in male_markers if re.search(m, c))
        if female_hits > male_hits and female_hits > 0:
            return "woman"
        if male_hits > female_hits and male_hits > 0:
            return "man"
        return None

    def _is_vision_template(self, workflow_name: str) -> bool:
        template_key = self._select_template_key(workflow_name)
        return template_key in {"video_i2v", "video_ia2v", "video_v2v"}

    def _bytes_to_data_url(self, image_bytes: bytes) -> str:
        if not image_bytes:
            raise RuntimeError("Empty image bytes for vision analysis.")
        mime = "image/png"
        if image_bytes[:3] == b"\xff\xd8\xff":
            mime = "image/jpeg"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            mime = "image/webp"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    def _extract_json_object(self, text: str) -> Dict[str, Any]:
        raw = (text or "").strip()
        if not raw:
            return {}
        if raw.startswith("```"):
            # Remove opening triple-backtick fence (with optional language tag) and closing fence.
            raw = re.sub(r"^```(?:json)?\s*", "", raw, count=1)
            raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            snippet = raw[start : end + 1]
            try:
                data = json.loads(snippet)
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}

    def _build_source_visual_context(self, data: Dict[str, Any]) -> str:
        fields = [
            ("visual_style", "Visual style"),
            ("subject", "Subject"),
            ("clothing_or_nudity", "Clothing/Nudity"),
            ("pose_and_body_position", "Pose/Body"),
            ("interaction", "Interaction"),
            ("shot_type", "Shot type"),
            ("camera_angle", "Camera angle"),
            ("lighting", "Lighting"),
            ("background_and_setting", "Background/Setting"),
        ]
        parts: list[str] = []
        for key, label in fields:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(f"{label}: {value.strip()}")
        return " | ".join(parts).strip()

    def _analyze_source_image_with_openrouter(
        self,
        image_bytes: bytes,
        workflow_name: str,
        user_prompt: str,
        selections: Dict[str, Any],
    ) -> tuple[str, list[str]]:
        route_steps: list[str] = []
        model_candidates = [self.vision_model] + [
            x for x in self.vision_fallback_models if x and x != self.vision_model
        ]
        if not model_candidates:
            raise RuntimeError("No vision model configured.")

        image_data_url = self._bytes_to_data_url(image_bytes)
        last_error = None
        for model_name in model_candidates:
            try:
                print(f"[Enhancer] Vision try: {model_name}")
                route_steps.append(f"Vision try: {model_name}")
                payload = {
                    "model": model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You analyze source frames for image-to-video workflows. "
                                "Return only valid compact JSON with keys: "
                                "visual_style, subject, clothing_or_nudity, pose_and_body_position, "
                                "interaction, shot_type, camera_angle, lighting, background_and_setting."
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"Workflow: {workflow_name}\n"
                                        f"User prompt intent: {user_prompt}\n"
                                        f"Selected camera preset: {selections.get('camera')}\n"
                                        f"Selected motion preset: {selections.get('motion')}\n"
                                        "Analyze the source image faithfully. Do not invent details."
                                    ),
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": image_data_url,
                                        "detail": self.vision_image_detail,
                                    },
                                },
                            ],
                        },
                    ],
                    "temperature": 0.2,
                    "max_tokens": self.vision_max_tokens,
                }
                reasoning_opts = self._build_reasoning_options()
                if reasoning_opts:
                    payload["reasoning"] = reasoning_opts
                data = self._request_with_retries(
                    payload=payload,
                    model_name=model_name,
                    request_label="vision-analysis",
                )
                text, finish_reason = self._extract_text_and_meta(data)
                if not text:
                    raise RuntimeError(
                        f"No text content from vision model (finish_reason={finish_reason!r})."
                    )
                parsed = self._extract_json_object(text)
                context_text = self._build_source_visual_context(parsed)
                if not context_text:
                    raise RuntimeError("Vision model returned empty/invalid analysis JSON.")
                print(f"[Enhancer] Vision success: {model_name}")
                route_steps.append(f"Vision success: {model_name}")
                return context_text, route_steps
            except Exception as e:
                last_error = e
                print(f"[Enhancer] Vision failed ({model_name}): {e}")
                route_steps.append(f"Vision failed: {model_name} ({e})")

        raise RuntimeError(f"Vision analysis failed. Last error: {last_error}")

    def _build_reasoning_options(self) -> Dict[str, Any] | None:
        if not self.disable_reasoning:
            return None
        effort = (self.reasoning_effort or "none").strip()
        # OpenRouter guidance: use one control strategy; effort=none is best-effort "no thinking".
        return {
            "effort": effort,
            "exclude": bool(self.reasoning_exclude),
        }

    def _protect_dialogue_tokens(self, text: str) -> tuple[str, Dict[str, str]]:
        if not text:
            return "", {}
        token_map: Dict[str, str] = {}
        pattern = re.compile(r'"([^"\n]{1,600})"|„([^“\n]{1,600})“|“([^”\n]{1,600})”')
        index = 0

        def _repl(match: re.Match) -> str:
            nonlocal index
            index += 1
            token = f"[DIALOGUE_{index}]"
            token_map[token] = match.group(0)
            return token

        protected = pattern.sub(_repl, text)
        return protected, token_map

    def _restore_dialogue_tokens(self, text: str, token_map: Dict[str, str]) -> str:
        restored = text or ""
        for token, raw in token_map.items():
            restored = restored.replace(token, raw)
        return restored

    def _is_prompt_usable(self, text: str) -> bool:
        if not text:
            return False
        cleaned = " ".join(text.split()).strip()
        if len(cleaned) < 60:
            return False
        words = cleaned.split()
        if len(words) < 12:
            return False
        tail = words[-1].lower().strip(".,;:!?")
        if tail in {"a", "an", "the", "on", "in", "at", "to", "with", "of", "for", "from", "by", "and", "or", "as"}:
            return False
        if self._looks_truncated(cleaned):
            return False
        return True

    def _looks_truncated(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return True
        if stripped.endswith("..."):
            return True
        tail = stripped.split()[-1].lower().strip(".,;:!?")
        if tail in {"a", "an", "the", "on", "in", "at", "to", "with", "of", "for", "from", "by", "and", "or", "as"}:
            return True
        return False

    def _build_ltx_directives(
        self,
        template_key: str,
        style_hint: str,
        film_style_key: str,
        film_style_hint: str,
        camera_hint: str,
        light_hint: str,
        mood_hint: str,
        motion_hint: str,
        pattern_hint: str,
        cinematic_mode: str,
        duration_seconds,
        timeline_mode: bool,
        storytelling: bool,
        source_visual_context: str,
        source_video_seconds,
        workflow_name: str,
        selections: Dict[str, Any],
    ) -> dict:
        """Build LTX-2/2.3 specific directives.

        LTX requires: single flowing paragraph, natural language, max 200 words,
        chronological description, no labels/sections, start directly with action.
        """
        # Build a compact context block for the LLM (not sent to ComfyUI directly)
        context_parts = [f"scene action: {{user_prompt}}"]
        context_parts.append(f"camera: {camera_hint}")
        context_parts.append(f"lighting: {light_hint}")
        context_parts.append(f"mood: {mood_hint}")
        if film_style_key != "none":
            context_parts.append(f"FILM STYLE (primary visual medium — highest priority): {film_style_hint}")
            context_parts.append(f"director color/composition hints (adapt WITHIN the film style): {style_hint}")
        else:
            context_parts.append(f"visual style: {style_hint}")
        context_parts.append(f"motion: {motion_hint}")
        context_parts.append(f"beat pattern: {pattern_hint}")

        # Local fallback: weave into a single flowing paragraph
        local_sections = []
        if film_style_key != "none":
            local_sections.append(f"In {film_style_hint},")
        local_sections.extend([
            "{user_prompt}.",
            f"Shot with {camera_hint}, {light_hint}.",
        ])
        if film_style_key != "none":
            local_sections.append(f"The mood is {mood_hint}, with {style_hint} color grading adapted to the {film_style_hint} medium.")
        else:
            local_sections.append(f"The mood is {mood_hint}, with {style_hint}.")
        if motion_hint and "none" not in motion_hint.lower():
            local_sections.append(f"Camera moves with {motion_hint}.")

        # Type-specific guards for local fallback
        if template_key == "video_i2v":
            local_sections.insert(0, "The scene from the source image comes to life.")
        elif template_key == "video_v2v":
            local_sections.insert(0, "The video continues with")
            if isinstance(source_video_seconds, (int, float)) and source_video_seconds > 0:
                local_sections.append(
                    f"Continuation starts after the {float(source_video_seconds):.1f}s source clip."
                )
        elif template_key == "video_ia2v":
            local_sections.insert(0, "The scene from the source image animates to match the audio.")

        # LLM instructions — core LTX prompting rules
        llm_base = (
            "CRITICAL FORMAT RULES FOR LTX VIDEO MODEL: "
            "1) Output MUST be a SINGLE FLOWING PARAGRAPH in natural English. "
            "No bullet points, no labels, no section headers, no line breaks. "
            "2) STRICT 200 WORD LIMIT — count carefully, never exceed 200 words. "
            "3) Start DIRECTLY with the main action — no preamble. "
            "4) Describe events CHRONOLOGICALLY as they unfold in time. "
            "5) WEAVE all technical details (camera, lighting, style, mood) naturally into the scene description. "
            "Write like a cinematographer describing a continuous shot, not like a form with fields. "
            "6) Use specific film language that LTX responds to: "
            "film stock references (Kodak 2383, Fujifilm Provia), camera systems (ARRI Alexa), "
            "lens focal lengths (24mm, 50mm, 85mm), aperture (f/1.4, f/2.8), "
            "shutter language (180-degree shutter), color grading terms (teal-orange, desaturated). "
            "7) Include quality guardrails at the end: 'No distortion, no jitter, no artifacts.' "
            "8) STYLE CONSISTENCY: If a visual style or film style is specified (e.g., anime, found footage, VHS), "
            "it MUST apply from the VERY FIRST FRAME and remain CONSISTENT throughout the entire video. "
            "NEVER describe a style as appearing, shifting, or transitioning mid-scene. "
            "State the style upfront as the rendering medium, not as an event. "
            "Example WRONG: 'The scene shifts into anime style.' "
            "Example CORRECT: 'In cel-shaded anime style, a group of women explores a bunker.' "
            "9) NEVER invent style transitions, visual shifts, or aesthetic changes that the user did not request. "
            "The visual style is a constant rendering property, not a narrative element. "
            "10) PRIORITY RULE: If both a FILM STYLE and a DIRECTOR style are specified, "
            "the FILM STYLE defines the base rendering medium (e.g., found footage, anime, VHS) and MUST be preserved. "
            "The director style provides color grading and composition hints that must be ADAPTED WITHIN the film style. "
            "Example: 'Found Footage + Ridley Scott' = handheld shaky cam with low-fi texture AND blue-steel color grade with atmospheric haze. "
            "NEVER let a director style replace or override the film style's core visual characteristics. "
            "11) SCENE PRESERVATION: The user's prompt defines the SCENE (location, setting, objects, characters). "
            "Director style defines the LOOK (color grading, lighting approach, composition style). "
            "When director hints mention scene elements (e.g., 'industrial texture', 'massive architecture'), "
            "interpret them as VISUAL TREATMENT (gritty texture, grand framing) NOT as location or setting changes. "
            "NEVER replace, modify, or override the user's described scene, location, or setting based on director hints. "
            "Example: User says 'bunker' + Director says 'industrial texture' = a bunker rendered with gritty industrial visual treatment, NOT a factory. "
            "12) NIGHT-VISION CONTEXT RULE: If the film style is 'found footage', 'surveillance', or 'war footage' "
            "AND the lighting is dark/low-key (e.g., low_key, volumetric in dark setting, or user prompt implies darkness/night), "
            "apply full night-vision camera aesthetic: monochrome green-phosphor tint over the ENTIRE image, "
            "all light sources and eyes appear as bright overexposed white-green flares, "
            "infrared retroreflective eye-shine on any visible eyes (pupils glow bright white against green), "
            "CCD noise grain, IR illuminator hotspot. "
            "If the scene is daytime or well-lit, do NOT apply night-vision — keep natural colors for the respective style instead. "
        )

        # Type-specific LLM instructions
        if template_key == "video_t2v":
            llm_instructions = (
                llm_base +
                "This is a TEXT-TO-VIDEO generation. Build the scene from scratch. "
                "Describe subject, action, environment, then camera movement as one continuous shot description."
            )
        elif template_key == "video_i2v":
            llm_instructions = (
                llm_base +
                "This is IMAGE-TO-VIDEO. The source image is the starting frame. "
                "PRESERVE the source image's composition, subject identity, location, and framing. "
                "Describe how the scene ANIMATES from this still frame — what moves, what changes. "
                "Do NOT re-describe or replace the scene. Only describe the motion and progression."
            )
        elif template_key == "video_v2v":
            llm_instructions = (
                llm_base +
                "This is VIDEO-TO-VIDEO continuation. "
                "Start with 'the video continues with' and then describe ONLY new actions. "
                "Do NOT recap or re-describe what happened in the source clip. "
                "Maintain continuity of identity, environment, and motion trajectory."
            )
        elif template_key == "video_ia2v":
            llm_instructions = (
                llm_base +
                "This is IMAGE+AUDIO-TO-VIDEO. Audio synchronization is the highest priority. "
                "Describe motion that aligns with audio beats and timing. "
                "Keep camera movement subtle to avoid breaking audio sync. "
                "PRESERVE source image composition and subject identity."
            )
        else:
            llm_instructions = llm_base

        # Add context for LLM
        llm_instructions += (
            f"\n\nContext to weave into the paragraph:\n" +
            "\n".join(f"- {p}" for p in context_parts)
        )

        if source_visual_context and template_key in {"video_i2v", "video_ia2v", "video_v2v"}:
            local_sections.append(source_visual_context)
            llm_instructions += (
                f"\nSource visual context (preserve as hard constraints): {source_visual_context}"
            )

        llm_instructions += (
            " Never swap character identity, gender, or role mapping from the user prompt. "
            "If the prompt explicitly distinguishes actors (e.g., woman/man, she/he), preserve that mapping exactly."
        )

        llm_instructions += f" {self._cinematic_writing_instruction(cinematic_mode, 'ltx')}"

        if storytelling:
            llm_instructions += " Storytelling mode is ON; maintain narrative progression coherently within the single paragraph."

        if timeline_mode:
            duration_text = (
                f"{int(duration_seconds)}s"
                if isinstance(duration_seconds, (int, float)) and duration_seconds > 0
                else "unknown duration"
            )
            llm_instructions += (
                f" Timeline mode is ON (target: {duration_text}). "
                "Embed temporal cues naturally in the flowing text (e.g., 'after a moment', 'then', 'as the scene progresses'). "
                "Do NOT use explicit timestamps like '0.0-2.5s'. Keep it as natural temporal flow within the single paragraph."
            )
            if template_key == "video_v2v" and isinstance(source_video_seconds, (int, float)) and source_video_seconds > 0:
                llm_instructions += (
                    f" Source clip is {float(source_video_seconds):.2f}s; continuation begins after it."
                )

        return {
            "template_key": template_key,
            "template_label": self._template_label(template_key),
            "camera_hint": camera_hint,
            "motion_hint": motion_hint,
            "pattern_hint": pattern_hint,
            "timeline_mode": timeline_mode,
            "duration_seconds": duration_seconds,
            "source_video_seconds": source_video_seconds,
            "local_sections": local_sections,
            "llm_instructions": llm_instructions,
            "cinematic_writing_mode": cinematic_mode,
            "model_family": "ltx",
        }

    def _build_flux_directives(
        self,
        template_key: str,
        style_hint: str,
        film_style_key: str,
        film_style_hint: str,
        camera_hint: str,
        light_hint: str,
        mood_hint: str,
        workflow_name: str,
        selections: Dict[str, Any],
    ) -> dict:
        """Build Flux-specific directives.

        Flux requires: natural language paragraphs, 30-80 words sweet spot,
        camera/lens language as first-class rendering instructions,
        NO negative prompts, NO quality boosters. Front-load important elements.
        """
        source_visual_context = str(selections.get("_source_visual_context", "")).strip()

        # Local fallback: concise flowing paragraph
        local_sections = []
        if film_style_key != "none":
            local_sections.append(f"In {film_style_hint},")
        local_sections.extend([
            "{user_prompt},",
            f"shot with {camera_hint},",
            f"{light_hint},",
        ])
        if film_style_key != "none":
            local_sections.append(f"{style_hint} color grading adapted to the film style, {mood_hint}.")
        else:
            local_sections.append(f"{style_hint}, {mood_hint}.")

        # Check if this is an image editing workflow (Flux2 Klein)
        is_edit = "EDIT" in (workflow_name or "").upper() or "KLEIN" in (workflow_name or "").upper()

        llm_instructions = (
            "CRITICAL FORMAT RULES FOR FLUX MODEL: "
            "1) Output a SINGLE FLOWING PARAGRAPH in natural English. "
            "No bullet points, no labels, no section headers. "
            "2) TARGET 30-80 WORDS — this is the sweet spot for Flux. Never exceed 100 words. "
            "3) FRONT-LOAD the most important elements: main subject and action come first. "
            "4) Camera and lens language IS understood as rendering instructions by Flux — "
            "use focal lengths (24mm, 50mm, 85mm), aperture (f/1.4, f/2.8), "
            "camera references (shot on ARRI Alexa, Hasselblad medium format), "
            "film stocks (Kodak Portra 400, Fujifilm Pro 400H). These control the visual output. "
            "5) DO NOT use quality boosters like 'masterpiece', 'best quality', '4k', 'highly detailed'. "
            "Flux ignores or is harmed by these. "
            "6) DO NOT include negative language ('no artifacts', 'without distortion'). "
            "Flux does not support negative prompts. Instead use POSITIVE alternatives "
            "(e.g., 'sharp focus' instead of 'not blurry'). "
            "7) Use specific visual descriptions: materials, textures, color grading terms. "
            "8) SCENE PRESERVATION: The user's prompt defines the SCENE (location, setting, objects). "
            "Director style defines the LOOK (color grading, lighting, composition). "
            "When director hints mention scene elements, interpret them as visual treatment, NOT location changes. "
            "NEVER replace the user's described scene based on director hints. "
            "9) NIGHT-VISION CONTEXT RULE: If the film style is 'found footage', 'surveillance', or 'war footage' "
            "AND the scene is dark/nighttime, apply night-vision camera look: monochrome green-phosphor tint, "
            "eyes show bright white-green retroreflective IR shine, light sources overexpose to white-green flares, "
            "CCD noise grain. If daytime or well-lit, do NOT apply night-vision — use natural colors for the style."
        )

        if is_edit:
            llm_instructions += (
                " This is an IMAGE EDITING workflow. The user is modifying an existing image. "
                "Focus the prompt on WHAT SHOULD CHANGE while preserving the rest. "
                "Be precise about the edit target."
            )

        # Context for LLM
        context_parts = [
            f"scene: {{user_prompt}}",
            f"camera/lens: {camera_hint}",
            f"lighting: {light_hint}",
            f"mood: {mood_hint}",
        ]
        if film_style_key != "none":
            context_parts.append(f"FILM STYLE (primary visual medium — highest priority): {film_style_hint}")
            context_parts.append(f"director color/composition (adapt WITHIN film style): {style_hint}")
        else:
            context_parts.append(f"visual style: {style_hint}")
        if source_visual_context:
            context_parts.append(f"source visual context: {source_visual_context}")
            llm_instructions += (
                f"\nSource visual context (preserve as constraints): {source_visual_context}"
            )

        llm_instructions += (
            "\n\nContext to weave into the paragraph:\n" +
            "\n".join(f"- {p}" for p in context_parts)
        )

        return {
            "template_key": template_key,
            "template_label": self._template_label(template_key),
            "camera_hint": camera_hint,
            "motion_hint": "",
            "pattern_hint": "",
            "timeline_mode": False,
            "duration_seconds": None,
            "source_video_seconds": None,
            "local_sections": local_sections,
            "llm_instructions": llm_instructions,
            "cinematic_writing_mode": self._resolve_cinematic_writing_mode(selections),
            "model_family": "flux",
        }

    def _build_pony_directives(
        self,
        template_key: str,
        style_hint: str,
        film_style_key: str,
        film_style_hint: str,
        camera_hint: str,
        light_hint: str,
        mood_hint: str,
        workflow_name: str,
        selections: Dict[str, Any],
    ) -> dict:
        """Build Pony/Z-Image specific directives.

        Pony V6 XL / CyberRealistic Pony requires: mandatory quality score prefix,
        hybrid tag + natural language, CLIP skip 2, 77 token limit (~71 usable).
        Camera terms are weak on base Pony but better on CyberRealistic.
        """
        source_visual_context = str(selections.get("_source_visual_context", "")).strip()
        is_z_image = "Z-IMAGE" in (workflow_name or "").upper() or "ZIMAGE" in (workflow_name or "").upper()

        # Local fallback: Pony score prefix + concise description
        score_prefix = "score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up"
        local_sections = [f"{score_prefix},"]
        if film_style_key != "none":
            local_sections.append(f"{film_style_hint},")
        local_sections.extend([
            "{user_prompt},",
            f"{light_hint},",
            f"{mood_hint},",
            f"{style_hint}.",
        ])
        if is_z_image:
            # CyberRealistic Pony handles camera terms better
            local_sections.insert(2, f"{camera_hint},")

        llm_instructions = (
            "CRITICAL FORMAT RULES FOR PONY DIFFUSION / CYBERREALISTIC PONY MODEL: "
            "1) Output MUST start with the quality score chain: "
            "'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up' — "
            "this is MANDATORY, never omit it. "
            "2) After the scores, write a CONCISE description mixing natural language with descriptive tags. "
            "3) STRICT 70 TOKEN LIMIT after the score prefix — keep it very concise. "
            "This means roughly 50-60 words maximum INCLUDING the score prefix. "
            "4) DO NOT use quality boosters like 'masterpiece', 'best quality', 'highly detailed'. "
            "The score system replaces these entirely. "
            "5) For camera/framing: use descriptive terms like 'close-up', 'wide shot', "
            "'dramatic lighting', 'shallow depth of field' rather than specific f-stops or focal lengths. "
        )

        if is_z_image:
            llm_instructions += (
                "6) This is CyberRealistic Pony (photorealistic finetune). "
                "Camera terms like 'DSLR photo', 'bokeh', 'macro photography' work well here. "
                "Specific focal lengths (85mm, 50mm) and camera brands have moderate effect. "
            )
        else:
            llm_instructions += (
                "6) This model is art/illustration focused. "
                "Avoid specific camera hardware terms (f-stops, focal lengths, camera brands). "
                "Use art-style terms: 'detailed background', 'dramatic lighting', 'soft shading'. "
            )

        llm_instructions += (
            "7) DO NOT include negative prompt content in the output. "
            "Pony does not need negatives with proper score tags. "
            "8) Use commas to separate concepts, not periods or complex sentences. "
            "9) SCENE PRESERVATION: The user's prompt defines the SCENE (location, setting, objects). "
            "Director style defines the LOOK. When director hints mention scene elements, "
            "interpret them as visual treatment, NOT location changes. "
            "10) NIGHT-VISION CONTEXT RULE: If film style is 'found footage', 'surveillance', or 'war footage' "
            "AND scene is dark/nighttime, add night-vision tags: monochrome green tint, glowing eyes, IR eye-shine, CCD noise. "
            "If daytime or well-lit, do NOT add night-vision tags."
        )

        # Context for LLM
        context_parts = [
            f"scene: {{user_prompt}}",
            f"framing: {camera_hint}",
            f"lighting: {light_hint}",
            f"mood: {mood_hint}",
        ]
        if film_style_key != "none":
            context_parts.append(f"FILM STYLE (primary visual medium — highest priority): {film_style_hint}")
            context_parts.append(f"director color/composition (adapt WITHIN film style): {style_hint}")
        else:
            context_parts.append(f"style: {style_hint}")
        if source_visual_context:
            context_parts.append(f"source visual context: {source_visual_context}")
            llm_instructions += (
                f"\nSource visual context (preserve as constraints): {source_visual_context}"
            )

        llm_instructions += (
            "\n\nContext to weave into the output:\n" +
            "\n".join(f"- {p}" for p in context_parts)
        )

        return {
            "template_key": template_key,
            "template_label": self._template_label(template_key),
            "camera_hint": camera_hint,
            "motion_hint": "",
            "pattern_hint": "",
            "timeline_mode": False,
            "duration_seconds": None,
            "source_video_seconds": None,
            "local_sections": local_sections,
            "llm_instructions": llm_instructions,
            "cinematic_writing_mode": self._resolve_cinematic_writing_mode(selections),
            "model_family": "pony",
        }

    def _detect_model_family(self, workflow_name: str) -> str:
        """Detect which model family a workflow targets."""
        wf_upper = (workflow_name or "").upper()
        if "LTX" in wf_upper:
            return "ltx"
        if "FLUX" in wf_upper:
            return "flux"
        if "PONY" in wf_upper or "Z-IMAGE" in wf_upper or "ZIMAGE" in wf_upper:
            return "pony"
        if "QWEN" in wf_upper:
            return "qwen"
        return "generic"

    def _select_template_key(self, workflow_name: str) -> str:
        wf_upper = (workflow_name or "").upper()
        if "IA2V" in wf_upper or "IA2VID" in wf_upper or ("AUDIO" in wf_upper and "IMAGE" in wf_upper):
            return "video_ia2v"
        if "V2V" in wf_upper:
            return "video_v2v"
        if "I2V" in wf_upper:
            return "video_i2v"
        if "T2V" in wf_upper:
            return "video_t2v"
        return "image_default"

    def _template_label(self, template_key: str) -> str:
        labels = {
            "image_default": "Image Default",
            "video_t2v": "Video T2V",
            "video_i2v": "Video I2V",
            "video_v2v": "Video V2V",
            "video_ia2v": "Video IA2V",
        }
        return labels.get(template_key, template_key)

    def _normalize_v2v_continuation_prompt(self, workflow_name: str, user_prompt: str) -> tuple[str, bool]:
        """Ensure V2V prompts explicitly start as continuation instructions."""
        if self._select_template_key(workflow_name) != "video_v2v":
            return user_prompt, False
        text = (user_prompt or "").strip()
        if not text:
            return "the video continues with cinematic continuation from the source clip.", True
        lowered = text.lower().strip()
        markers = (
            "the video continues with",
            "video continues with",
            "continue with",
            "continues with",
        )
        if any(lowered.startswith(m) for m in markers):
            return text, False
        return f"the video continues with {text}", True

    def _strip_v2v_prefix(self, text: str) -> str:
        return re.sub(
            r"^\s*(the video continues with|video continues with|continue with|continues with)\s*[:,-]?\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        ).strip()

    def _remove_inline_v2v_prefix_repetitions(self, text: str) -> str:
        """Remove repeated inline continuation cue fragments inside V2V body."""
        cleaned = re.sub(
            r"\b(the video continues with|video continues with|continue with|continues with)\b\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
        return cleaned

    def _merge_untimed_v2v_text_into_first_timeline(self, text: str) -> str:
        """
        If V2V continuation has untimed prose before the first timeline segment,
        merge that prose into the first segment so no action is lost.
        """
        continuation = (text or "").strip()
        if not continuation:
            return continuation

        first_ts = re.search(r"(\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)", continuation)
        if not first_ts:
            return continuation

        prefix = continuation[: first_ts.start()].strip(" .\n\t")
        if not prefix:
            return continuation

        timeline_part = continuation[first_ts.start():]
        first_seg_match = re.match(
            r"(?P<head>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)\s*(?P<body>.*?)(?=(?:\s+\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)|$)",
            timeline_part,
            flags=re.DOTALL,
        )
        if not first_seg_match:
            return continuation

        head = first_seg_match.group("head")
        body = (first_seg_match.group("body") or "").strip()
        merged_body = f"{prefix}. {body}".strip()
        merged_body = re.sub(r"\s{2,}", " ", merged_body)
        merged_first = f"{head} {merged_body}".strip()
        rebuilt = merged_first + timeline_part[first_seg_match.end():]
        return rebuilt.strip()

    def _strip_untimed_prefix_before_first_timeline(self, text: str) -> tuple[str, bool]:
        """
        If timeline segments already exist, drop untimed prose before the first segment.
        This prevents recap/setup blobs from polluting the first beat.
        """
        body = (text or "").strip()
        if not body:
            return body, False
        first_ts = re.search(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:", body)
        if not first_ts:
            return body, False
        if first_ts.start() == 0:
            return body, False
        trimmed = body[first_ts.start():].strip()
        return trimmed, trimmed != body

    def _format_timeline_second(self, value: float) -> str:
        value = max(0.0, float(value))
        if float(int(value)) == value:
            return f"{value:.1f}".rstrip("0").rstrip(".") + ".0"
        return f"{value:.1f}"

    def _has_timeline_segments(self, text: str) -> bool:
        return bool(re.search(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:", text or ""))

    def _build_v2v_timeline_from_prose(
        self,
        prose: str,
        start_seconds: float,
        end_seconds: float,
    ) -> str:
        """
        Build a minimal timeline from prose when model output omitted explicit time segments.
        """
        text = re.sub(r"\s+", " ", (prose or "").strip())
        if not text:
            text = "continuation action unfolds naturally."
        if end_seconds <= start_seconds:
            end_seconds = start_seconds + 8.0

        blocks = self._extract_event_blocks_from_prose(text)
        if not blocks:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
            if not sentences:
                sentences = [text]
            if len(sentences) == 1:
                blocks = [sentences[0]]
            elif len(sentences) == 2:
                blocks = [sentences[0], sentences[1]]
            else:
                first = " ".join(sentences[:1]).strip()
                second = " ".join(sentences[1:2]).strip()
                third = " ".join(sentences[2:]).strip()
                blocks = [b for b in (first, second, third) if b]

        total = end_seconds - start_seconds
        if len(blocks) == 1:
            cuts = [start_seconds, end_seconds]
        elif len(blocks) == 2:
            cuts = [start_seconds, start_seconds + total * 0.45, end_seconds]
        elif len(blocks) == 4:
            cuts = [
                start_seconds,
                start_seconds + total * 0.25,
                start_seconds + total * 0.5,
                start_seconds + total * 0.75,
                end_seconds,
            ]
        else:
            cuts = [
                start_seconds,
                start_seconds + total * 0.33,
                start_seconds + total * 0.66,
                end_seconds,
            ]

        parts = []
        for idx, block in enumerate(blocks):
            s = cuts[idx]
            e = cuts[idx + 1]
            s_txt = self._format_timeline_second(round(s, 1))
            e_txt = self._format_timeline_second(round(max(e, s + 0.1), 1))
            parts.append(f"{s_txt}-{e_txt}s: {block}")
        return " ".join(parts).strip()

    def _extract_event_blocks_from_prose(self, text: str) -> list[str]:
        """
        Generic event extraction for V2V prose:
        - split by strong progression markers
        - keep dialogue-centric clauses grouped
        - cap to 4 beats
        """
        prose = (text or "").strip()
        if not prose:
            return []

        normalized = re.sub(r"\s+", " ", prose)
        marker_pattern = re.compile(
            r"\b(then|after(?:wards)?|after\s+that|suddenly|finally|thereafter|next)\b",
            flags=re.IGNORECASE,
        )

        parts: list[str] = []
        last = 0
        for m in marker_pattern.finditer(normalized):
            chunk = normalized[last:m.start()].strip(" ,.")
            if chunk:
                parts.append(chunk)
            last = m.start()
        tail = normalized[last:].strip(" ,.")
        if tail:
            parts.append(tail)

        if not parts:
            parts = [normalized]

        # Secondary split for very long chunks.
        refined: list[str] = []
        for p in parts:
            if len(p) > 220:
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
                if sents:
                    refined.extend(sents)
                else:
                    refined.append(p)
            else:
                refined.append(p)

        # Merge tiny fragments into neighbors.
        merged: list[str] = []
        for chunk in refined:
            if merged and len(chunk.split()) <= 4:
                merged[-1] = f"{merged[-1]}. {chunk}".strip()
            else:
                merged.append(chunk)

        # Limit to max 4 blocks by merging overflow into the last block.
        if len(merged) > 4:
            head = merged[:3]
            tail_block = " ".join(merged[3:]).strip()
            merged = [*head, tail_block]

        # Drop near-duplicate event blocks (keep first occurrence).
        def _tokenize(s: str) -> set[str]:
            return set(re.findall(r"[a-zA-Z']+", s.lower()))

        deduped: list[str] = []
        seen_token_sets: list[set[str]] = []
        for block in merged:
            cur_tokens = _tokenize(block)
            if len(cur_tokens) < 4:
                deduped.append(block)
                seen_token_sets.append(cur_tokens)
                continue
            is_duplicate = False
            for prev_tokens in seen_token_sets:
                if not prev_tokens:
                    continue
                overlap = len(cur_tokens & prev_tokens)
                denom = max(1, min(len(cur_tokens), len(prev_tokens)))
                if (overlap / denom) >= 0.72:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(block)
                seen_token_sets.append(cur_tokens)

        merged = deduped

        # Clean punctuation spacing.
        cleaned = []
        for b in merged:
            t = re.sub(r"\s{2,}", " ", b).strip()
            t = re.sub(r"\s+([,.;:!?])", r"\1", t)
            if t:
                cleaned.append(t)
        return cleaned

    def _offset_v2v_timeline_segments(self, text: str, offset_seconds: float) -> tuple[str, bool]:
        """
        Shift timeline windows by source clip length so V2V uses absolute/global timing.
        Only applies when segments appear to start from relative time near 0.
        """
        body = (text or "").strip()
        if not body or offset_seconds <= 0:
            return body, False

        pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s?\s*:")
        matches = list(pattern.finditer(body))
        if not matches:
            return body, False

        try:
            first_start = float(matches[0].group(1))
        except Exception:
            return body, False

        # If timeline already appears absolute (starts at/after source end), do not shift again.
        if first_start >= max(0.0, float(offset_seconds) - 0.05):
            return body, False

        def _replace(m: re.Match) -> str:
            start = float(m.group(1)) + float(offset_seconds)
            end = float(m.group(2)) + float(offset_seconds)
            return f"{self._format_timeline_second(start)}-{self._format_timeline_second(end)}s:"

        shifted = pattern.sub(_replace, body)
        return shifted, shifted != body

    def _rescale_and_clamp_v2v_timeline(
        self,
        text: str,
        start_floor_seconds: float,
        target_end_seconds: float,
    ) -> tuple[str, bool]:
        """
        Rescale timeline windows to fit into [start_floor_seconds, target_end_seconds],
        then clamp to bounds while preserving chronological order.
        """
        body = (text or "").strip()
        start_floor = float(start_floor_seconds)
        target_end = float(target_end_seconds)
        if not body or target_end <= start_floor:
            return body, False

        pattern = re.compile(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)s?\s*:")
        matches = list(pattern.finditer(body))
        if not matches:
            return body, False

        starts: list[float] = []
        ends: list[float] = []
        for m in matches:
            starts.append(float(m.group(1)))
            ends.append(float(m.group(2)))
        original_last_end = max(ends)
        first_start = max(start_floor, min(starts))

        # Already within bounds: clamp tiny over/under-shoot only.
        if original_last_end <= target_end + 1e-6:
            changed = False

            def _clamp_only(m: re.Match) -> str:
                nonlocal changed
                s = max(start_floor, min(float(m.group(1)), target_end))
                e = max(s, min(float(m.group(2)), target_end))
                repl = f"{self._format_timeline_second(s)}-{self._format_timeline_second(e)}s:"
                if repl != m.group(0):
                    changed = True
                return repl

            clamped = pattern.sub(_clamp_only, body)
            return clamped, changed

        denom = max(1e-6, original_last_end - first_start)
        span_target = max(0.2, target_end - first_start)
        scale = span_target / denom

        changed = False
        prev_end = first_start
        idx = 0

        def _rescale(m: re.Match) -> str:
            nonlocal changed, prev_end, idx
            idx += 1
            s_raw = float(m.group(1))
            e_raw = float(m.group(2))

            s_new = first_start + (s_raw - first_start) * scale
            e_new = first_start + (e_raw - first_start) * scale

            s_new = max(start_floor, min(s_new, target_end))
            e_new = max(s_new, min(e_new, target_end))

            # Keep monotonic and avoid zero-length segments when possible.
            s_new = max(s_new, prev_end if idx > 1 else first_start)
            if e_new < s_new:
                e_new = s_new
            if e_new == s_new and e_new < target_end:
                e_new = min(target_end, s_new + 0.1)

            prev_end = e_new
            repl = f"{self._format_timeline_second(s_new)}-{self._format_timeline_second(e_new)}s:"
            if repl != m.group(0):
                changed = True
            return repl

        adjusted = pattern.sub(_rescale, body)
        return adjusted, changed

    def _dedupe_v2v_timeline_dialogue_quotes(self, text: str) -> tuple[str, bool]:
        """
        Remove duplicate quoted dialogue across timeline segments.
        Keeps the latest occurrence (later segment wins) to preserve chronological progression.
        """
        body = (text or "").strip()
        if not body:
            return body, False

        seg_pattern = re.compile(
            r"(?P<head>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)\s*(?P<content>.*?)(?=(?:\s+\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)|$)",
            flags=re.DOTALL,
        )
        segments = list(seg_pattern.finditer(body))
        if not segments:
            return body, False

        quote_pattern = re.compile(r'"([^"]+)"')
        seen_quotes: set[str] = set()
        changed = False
        rebuilt: list[tuple[str, str]] = []

        # Walk from first to last => earlier beats keep their quotes.
        for seg in segments:
            head = seg.group("head")
            content = seg.group("content")
            current = content

            def _replace_quote(m: re.Match) -> str:
                nonlocal changed
                q_norm = re.sub(r"\s+", " ", (m.group(1) or "").strip().lower())
                if not q_norm:
                    return m.group(0)
                if q_norm in seen_quotes:
                    changed = True
                    return ""
                seen_quotes.add(q_norm)
                return m.group(0)

            current = quote_pattern.sub(_replace_quote, current)
            current = re.sub(r"\s{2,}", " ", current).strip()
            current = re.sub(r"\s+([,.;:!?])", r"\1", current)
            current = re.sub(r"([,;:])\s*([,;:])+", r"\1 ", current)

            rebuilt.append((head, current))
        out_parts = []
        for head, content in rebuilt:
            if content:
                out_parts.append(f"{head} {content}")
            else:
                out_parts.append(f"{head} action continues naturally.")
        out = " ".join(out_parts).strip()
        return out, changed

    def _strip_inline_timeline_markers_in_segments(self, text: str) -> tuple[str, bool]:
        """
        Remove inline relative timeline fragments inside segment bodies like ', 0.0-2.5s.'
        while keeping the segment headers intact.
        """
        body = (text or "").strip()
        if not body:
            return body, False

        seg_pattern = re.compile(
            r"(?P<head>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)\s*(?P<content>.*?)(?=(?:\s+\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)|$)",
            flags=re.DOTALL,
        )
        segments = list(seg_pattern.finditer(body))
        if not segments:
            return body, False

        changed = False
        rebuilt: list[str] = []
        inline_ts = re.compile(r"(?:[,;]?\s*)\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s\.?")
        for seg in segments:
            head = seg.group("head")
            content = seg.group("content")
            new_content = inline_ts.sub("", content)
            new_content = re.sub(r"\s{2,}", " ", new_content).strip()
            new_content = re.sub(r"\s+([,.;:!?])", r"\1", new_content)
            new_content = re.sub(r"([,;:])\s*([,;:])+", r"\1 ", new_content)
            if new_content != content:
                changed = True
            if not new_content:
                new_content = "action continues naturally."
            rebuilt.append(f"{head} {new_content}")

        return " ".join(rebuilt).strip(), changed

    def _normalize_v2v_continuation_prose(self, text: str) -> str:
        """
        Convert arbitrary continuation text (with/without timeline markup) into stable prose.
        - removes timeline headers and inline time fragments
        - removes repeated continuation cue words
        - de-duplicates repeated quotes (first occurrence kept)
        """
        prose = (text or "").strip()
        if not prose:
            return prose

        prose = self._remove_inline_v2v_prefix_repetitions(self._strip_v2v_prefix(prose))
        prose = re.sub(r"\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:\s*", "", prose)
        prose = re.sub(r"(?:[,;]?\s*)\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s\.?", "", prose)

        seen_quotes: set[str] = set()
        def _dedupe_quote(m: re.Match) -> str:
            q = (m.group(1) or "").strip()
            qn = re.sub(r"\s+", " ", q.lower())
            if not qn:
                return m.group(0)
            if qn in seen_quotes:
                return ""
            seen_quotes.add(qn)
            return m.group(0)

        prose = re.sub(r'"([^"]+)"', _dedupe_quote, prose)
        prose = re.sub(r"\s{2,}", " ", prose).strip()
        prose = re.sub(r"\s+([,.;:!?])", r"\1", prose)
        prose = re.sub(r"([,;:])\s*([,;:])+", r"\1 ", prose)
        if not prose:
            prose = "continuation action unfolds naturally."
        return prose

    def _polish_v2v_timeline_language(self, timeline_text: str) -> str:
        """
        Light language polish for timeline segment bodies only.
        Keeps timing and role mapping intact; only reduces stiff repetition.
        """
        text = (timeline_text or "").strip()
        if not text:
            return text

        seg_pattern = re.compile(
            r"(?P<head>\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)\s*(?P<content>.*?)(?=(?:\s+\d+(?:\.\d+)?\s*-\s*\d+(?:\.\d+)?s?\s*:)|$)",
            flags=re.DOTALL,
        )

        out_parts: list[str] = []
        for seg in seg_pattern.finditer(text):
            head = seg.group("head")
            content = (seg.group("content") or "").strip()
            if not content:
                out_parts.append(f"{head} action continues naturally.")
                continue

            # Clean spacing/punctuation and capitalize sentence starts after separators.
            content = re.sub(r"\s{2,}", " ", content).strip()
            content = re.sub(r"\s+([,.;:!?])", r"\1", content)
            content = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), content)
            if content and content[0].islower():
                content = content[0].upper() + content[1:]

            out_parts.append(f"{head} {content}")

        if out_parts:
            return " ".join(out_parts).strip()
        return text

    def _enforce_v2v_dialogue_speaker_map(
        self,
        timeline_text: str,
        quote_speaker_map: Dict[str, str] | None,
    ) -> tuple[str, bool]:
        """
        Enforce that known quoted lines stay with the mapped speaker (man/woman).
        Only performs targeted subject swaps around quote-introducing verbs.
        """
        text = (timeline_text or "").strip()
        if not text or not quote_speaker_map:
            return text, False

        changed = False
        out = text
        verbs = r"(asks|asked|says|said|speaks|spoke|tells|told|murmurs|whispers|shouts|yells)"

        for quote_raw, speaker in quote_speaker_map.items():
            if not quote_raw or speaker not in {"man", "woman"}:
                continue
            q = re.escape(quote_raw)
            if speaker == "man":
                # wrong: woman/she + verb + "quote" -> man + verb + "quote"
                p1 = re.compile(rf"\b(the woman|she)\b(\s+{verbs}\s+\"{q}\")", flags=re.IGNORECASE)
                new_out, n = p1.subn(r"the man\2", out)
                if n > 0:
                    changed = True
                    out = new_out
                p2 = re.compile(rf"\b(\"{q}\"[^\n\r]{{0,80}}\bby\s+)(the woman|she)\b", flags=re.IGNORECASE)
                new_out, n = p2.subn(r"\1the man", out)
                if n > 0:
                    changed = True
                    out = new_out
            else:
                # wrong: man/he + verb + "quote" -> woman + verb + "quote"
                p1 = re.compile(rf"\b(the man|he)\b(\s+{verbs}\s+\"{q}\")", flags=re.IGNORECASE)
                new_out, n = p1.subn(r"the woman\2", out)
                if n > 0:
                    changed = True
                    out = new_out
                p2 = re.compile(rf"\b(\"{q}\"[^\n\r]{{0,80}}\bby\s+)(the man|he)\b", flags=re.IGNORECASE)
                new_out, n = p2.subn(r"\1the woman", out)
                if n > 0:
                    changed = True
                    out = new_out

        return out, changed

    def _ensure_v2v_structure_in_final_prompt(
        self,
        workflow_name: str,
        prompt_text: str,
        selections: Dict[str, Any],
    ) -> tuple[str, bool]:
        """Guarantee V2V final prompt order: source summary -> continuation cue -> continuation."""
        if self._select_template_key(workflow_name) != "video_v2v":
            return prompt_text, False
        text = (prompt_text or "").strip()
        if not text:
            return "the video continues with cinematic continuation from the source clip.", True
        source_visual_context = str((selections or {}).get("_source_visual_context", "")).strip()
        timeline_mode = bool((selections or {}).get("timeline_mode", False))
        source_video_seconds = (selections or {}).get("_source_video_seconds")
        duration_seconds = (selections or {}).get("_duration_seconds")
        lowered = text.lower()
        continuation_source = text
        if "the video continues with" in lowered:
            parts = re.split(r"\bthe video continues with\b", text, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                continuation_source = parts[1].strip()

        continuation = self._normalize_v2v_continuation_prose(continuation_source)
        if timeline_mode:
            start_floor = float(source_video_seconds) if isinstance(source_video_seconds, (int, float)) else 0.0
            end_target = float(duration_seconds) if isinstance(duration_seconds, (int, float)) and duration_seconds > start_floor else (start_floor + 12.0)
            continuation = self._build_v2v_timeline_from_prose(continuation, start_floor, end_target)
            continuation = self._polish_v2v_timeline_language(continuation)
            quote_speaker_map = (selections or {}).get("_dialogue_quote_speaker_map")
            continuation, _ = self._enforce_v2v_dialogue_speaker_map(continuation, quote_speaker_map)

        if source_visual_context:
            structured = (
                f"source clip summary: {source_visual_context}. "
                f"the video continues with {continuation}"
            )
            return structured, structured.strip() != text.strip()
        normalized = f"the video continues with {continuation}"
        return normalized, normalized.strip() != text.strip()

    def _build_template_directives(self, workflow_name: str, selections: Dict[str, Any]) -> dict:
        style_key = selections.get("style", "none")
        film_style_key = selections.get("film_style", "none")
        camera_key = selections.get("camera")
        light_key = selections.get("light")
        mood_key = selections.get("mood")
        motion_key = selections.get("motion", "none")
        pattern_key = selections.get("pattern", "none")
        storytelling = bool(selections.get("storytelling", False))
        source_visual_context = str(selections.get("_source_visual_context", "")).strip()
        source_video_seconds = selections.get("_source_video_seconds")

        style_hint = DIRECTOR_PRESETS.get(style_key, DIRECTOR_PRESETS["none"])["hint"]
        film_style_data = FILMSTYLE_PRESETS.get(film_style_key, FILMSTYLE_PRESETS["none"])
        film_style_hint = film_style_data["hint"]
        # If film style locks camera perspective, ignore user camera preset
        if film_style_data.get("camera_locked"):
            camera_hint = film_style_hint  # film style hint already describes the camera
        else:
            camera_hint = CAMERA_PRESETS.get(camera_key, "cinematic medium framing")
        light_hint = LIGHT_PRESETS.get(light_key, "balanced cinematic lighting")
        mood_hint = MOOD_PRESETS.get(mood_key, "coherent cinematic atmosphere")
        motion_hint = MOTION_PRESETS.get(
            motion_key,
            "prefer explicit motion from user prompt; if absent, keep subtle cinematic movement",
        )
        pattern_hint = CINEMATIC_PATTERN_PRESETS.get(
            pattern_key,
            "no fixed beat pattern; keep progression natural and coherent",
        )

        template_key = self._select_template_key(workflow_name)
        model_family = self._detect_model_family(workflow_name)
        cinematic_mode = self._resolve_cinematic_writing_mode(selections)
        duration_seconds = selections.get("_duration_seconds")
        timeline_mode = bool(selections.get("timeline_mode", False)) and template_key != "image_default"
        if film_style_key != "none":
            base_common = [
                f"FILM STYLE (primary visual medium): {film_style_hint}",
                f"Director color/composition (adapt within film style): {style_hint}",
                f"Lighting: {light_hint}",
                f"Mood: {mood_hint}",
            ]
        else:
            base_common = [
                f"Visual style: {style_hint}",
                f"Lighting: {light_hint}",
                f"Mood: {mood_hint}",
            ]

        # ── LTX-specific prompt building ──
        # LTX-2/2.3 requires: single flowing paragraph, natural language,
        # max 200 words, chronological, no labels/sections, start with action.
        if model_family == "ltx" and template_key != "image_default":
            return self._build_ltx_directives(
                template_key=template_key,
                style_hint=style_hint,
                film_style_key=film_style_key,
                film_style_hint=film_style_hint,
                camera_hint=camera_hint,
                light_hint=light_hint,
                mood_hint=mood_hint,
                motion_hint=motion_hint,
                pattern_hint=pattern_hint,
                cinematic_mode=cinematic_mode,
                duration_seconds=duration_seconds,
                timeline_mode=timeline_mode,
                storytelling=storytelling,
                source_visual_context=source_visual_context,
                source_video_seconds=source_video_seconds,
                workflow_name=workflow_name,
                selections=selections,
            )

        # ── Flux-specific prompt building ──
        # Flux: natural language paragraph, 30-80 words, camera/lens as rendering
        # instructions, NO negatives, NO quality boosters.
        if model_family == "flux":
            return self._build_flux_directives(
                template_key=template_key,
                style_hint=style_hint,
                film_style_key=film_style_key,
                film_style_hint=film_style_hint,
                camera_hint=camera_hint,
                light_hint=light_hint,
                mood_hint=mood_hint,
                workflow_name=workflow_name,
                selections=selections,
            )

        # ── Pony / Z-Image specific prompt building ──
        # Pony V6 XL: mandatory score prefix, hybrid tags + NL, 77 token limit,
        # CLIP skip 2, camera terms weak (better on CyberRealistic).
        if model_family == "pony":
            return self._build_pony_directives(
                template_key=template_key,
                style_hint=style_hint,
                film_style_key=film_style_key,
                film_style_hint=film_style_hint,
                camera_hint=camera_hint,
                light_hint=light_hint,
                mood_hint=mood_hint,
                workflow_name=workflow_name,
                selections=selections,
            )

        if template_key == "image_default":
            local_sections = [
                "Template: IMAGE",
                # Layer 1: Subject
                "Subject and scene: {user_prompt}",
                # Layer 2: Environment
                "Environment treatment: establish setting, background depth, and atmospheric conditions.",
                # Layer 3: Lighting
                f"Lighting setup: {light_hint}",
                # Layer 4: Technical
                f"Camera and lens: {camera_hint}",
                f"Motion intent (optional): {motion_hint}",
                f"Cinematic beat pattern (optional): {pattern_hint}",
                # Layer 5: Style
                f"Visual style: {style_hint}",
                f"Mood: {mood_hint}",
                "Quality: highly detailed, coherent composition, sharp subject focus, professional photography grade.",
                "If user mentions camera motion, treat it as optional for still image composition.",
            ]
            llm_instructions = (
                "Use IMAGE template with 5-layer prompt structure: "
                "1) Subject (primary focus, details, expression, pose, materials/textures) "
                "2) Environment (location, background treatment, atmospheric conditions) "
                "3) Lighting (source, direction, quality, color temperature) "
                "4) Technical (camera perspective, focal length effect, depth of field) "
                "5) Style (genre, color grading, post-processing aesthetic). "
                "Use specific photography terminology: actual lens focal lengths, "
                "aperture-style depth of field descriptions, named lighting setups (Rembrandt, butterfly, split). "
                "Camera preset is framing guidance, not timeline motion. "
                "Treat cinematic pattern as stylistic rhythm guidance, not literal timeline edits."
            )
        elif template_key == "video_t2v":
            local_sections = [
                "Template: VIDEO T2V",
                # Layer 1: Subject
                "Scene premise and subject: {user_prompt}",
                # Layer 2: Environment
                "Environment: establish setting, background depth, atmospheric conditions for scene context.",
                # Layer 3: Lighting
                f"Lighting setup: {light_hint}",
                # Layer 4: Technical (start frame + motion)
                f"Start frame composition: {camera_hint}",
                f"Preferred motion profile: {motion_hint}",
                f"Beat structure: {pattern_hint}",
                "Primary camera motion: prioritize explicit motion from the user prompt; "
                "if absent, use subtle cinematic forward push.",
                "Subject motion: natural, readable movement.",
                "Environment motion: support scene realism and depth.",
                # Layer 5: Style
                f"Visual style: {style_hint}",
                f"Mood: {mood_hint}",
                "Continuity guard: keep motion coherent, avoid abrupt reframing.",
            ]
            llm_instructions = (
                "Use VIDEO T2V template with layered start-frame composition: "
                "1) Subject (primary focus, appearance, pose, action) "
                "2) Environment (location, background, atmosphere) "
                "3) Lighting (setup, direction, color temperature) "
                "4) Technical (camera framing as start frame, then motion as timeline progression) "
                "5) Style (visual aesthetic, color grading, mood). "
                "Separate start-frame composition from camera motion timeline. "
                "Use specific photography terminology for the start frame: lens focal lengths, "
                "depth of field, named lighting setups. "
                "If camera angle conflicts with user motion, keep angle as start frame and motion as primary timeline instruction."
            )
        elif template_key == "video_i2v":
            local_sections = [
                "Template: VIDEO I2V",
                "Source image continuity: preserve identity/composition from input image.",
                "Source environment lock: preserve original location/background unless explicitly changed by user.",
                "Scene directive: {user_prompt}",
                f"Start frame composition: {camera_hint}",
                f"Preferred motion profile: {motion_hint}",
                f"Beat structure: {pattern_hint}",
                "Primary camera motion: prioritize explicit motion from user prompt; "
                "if absent, keep subtle and stable.",
                "Micro-motion: facial/body/environment details, no random large reframing.",
                *base_common,
                "Continuity guard: preserve source framing unless user explicitly requests reframing.",
                "I2V guard: treat prompt as animation of the uploaded still frame, not a full re-staging.",
                "I2V guard: do not replace location, outfit, subject count, or core composition unless explicitly requested.",
            ]
            llm_instructions = (
                "Use VIDEO I2V template. Preserve source image identity and composition. "
                "Motion should evolve from source frame, not replace it. "
                "Do not invent a new location/background/environment unless user explicitly requests it. "
                "Treat the input image as locked canon for subject, outfit, scene, and framing baseline. "
                "Only add motion, timing, and small action progressions that are consistent with that start frame. "
                "Use specific photography and cinematography terminology for lighting and atmosphere descriptions."
            )
        elif template_key == "video_v2v":
            local_sections = [
                "Template: VIDEO V2V",
                "Source video continuity: preserve original style/identity.",
                "Source environment lock: preserve original location/background unless explicitly changed by user.",
                "Continuation directive: {user_prompt}",
                f"Start continuation framing: {camera_hint}",
                f"Preferred motion profile: {motion_hint}",
                f"Beat structure: {pattern_hint}",
                "Camera motion evolution: continue naturally from source clip.",
                "Temporal pacing: smooth progression for edit-friendly transitions.",
                *base_common,
                "Continuity guard: no abrupt jumps in framing, identity, or motion trajectory.",
                "V2V guard: continuation-only actions. Do not recap or restage actions/dialogue already present in source clip.",
                "V2V guard: first timeline beat should begin with the first new action, not static source-scene restatement.",
            ]
            if isinstance(source_video_seconds, (int, float)) and source_video_seconds > 0:
                local_sections.append(
                    f"Source clip length reference: {float(source_video_seconds):.2f}s (continuation starts after source clip end)."
                )
            llm_instructions = (
                "Use VIDEO V2V template. Treat prompt as continuation instruction. "
                "Preserve source video continuity and produce transition-friendly progression. "
                "Do not invent a new location/background/environment unless user explicitly requests it. "
                "Keep final prompt order as: source clip summary first, then the phrase 'the video continues with', then continuation action. "
                "Do not recap or restage source-clip actions/dialogue in continuation beats. "
                "Start the first continuation beat with a new user-requested action, not a static setup restatement."
            )
        else:
            local_sections = [
                "Template: VIDEO IA2V",
                "Source image continuity: preserve identity/composition from input image.",
                "Source environment lock: preserve original location/background unless explicitly changed by user.",
                "Audio sync priority: lip/mouth/body timing follows audio beats.",
                "Scene directive: {user_prompt}",
                f"Start frame composition: {camera_hint}",
                f"Preferred motion profile: {motion_hint}",
                f"Beat structure: {pattern_hint}",
                "Primary camera motion: keep subtle unless explicitly requested.",
                "Expression and gesture changes must align to narration/speech timing.",
                *base_common,
                "Sync guard: avoid desync and avoid aggressive reframing that hurts audio sync.",
            ]
            llm_instructions = (
                "Use VIDEO IA2V template. Audio sync has higher priority than camera flourish. "
                "Keep motion stable and timing-aligned with speech/music. "
                "Do not invent a new location/background/environment unless user explicitly requests it."
            )

        if source_visual_context and template_key in {"video_i2v", "video_ia2v", "video_v2v"}:
            local_sections.append(f"Source visual context lock: {source_visual_context}")
            llm_instructions += (
                " Apply the provided source visual context as hard continuity constraints unless user explicitly asks to change them."
            )
        llm_instructions += (
            " Never swap character identity, gender, or role mapping from the user prompt. "
            "If the prompt explicitly distinguishes actors (e.g., woman/man, she/he), preserve that mapping exactly. "
            "If source visual analysis conflicts with explicit user role/gender instructions, prioritize the user's explicit mapping. "
            "Do not introduce additional speaking characters not explicitly present in the user prompt."
        )
        llm_instructions += f" {self._cinematic_writing_instruction(cinematic_mode, model_family)}"

        if storytelling:
            local_sections.append(
                "Storytelling intent: continue narrative naturally from current context with clear progression."
            )
            llm_instructions += " Storytelling mode is ON; keep narrative progression coherent."

        if timeline_mode:
            duration_text = (
                f"{int(duration_seconds)}s"
                if isinstance(duration_seconds, (int, float)) and duration_seconds > 0
                else "unknown duration"
            )
            local_sections.append(
                "Timeline mode: split sequence into 3-4 chronological beats with explicit timing windows."
            )
            local_sections.append(
                f"Timeline duration target: {duration_text}"
            )
            llm_instructions += (
                " Timeline mode is ON. Create a chronological beat plan with explicit timing windows "
                "(e.g., 0.0-2.5s, 2.5-6.0s, 6.0-10.0s). Use max 4 segments. "
                "Allocate longer spans to complex actions and shorter spans to transitions/impacts. "
                "Do not overlap contradictory actions in the same time window. "
                "If prompt contains only one simple action, keep timeline minimal (1-2 segments)."
            )
            if template_key == "video_v2v":
                llm_instructions += (
                    " For V2V: use absolute/global timeline timestamps. "
                    "Continuation beats must start at or after the source clip end timestamp. "
                    "Never assign timeline windows to events/dialogue that belong to the source clip."
                )
                if isinstance(source_video_seconds, (int, float)) and source_video_seconds > 0:
                    llm_instructions += (
                        f" Source clip reference length is {float(source_video_seconds):.2f}s; "
                        "first continuation timestamp should be at or after this value."
                    )
                    if isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
                        llm_instructions += (
                            f" Keep continuation timeline within global target end {float(duration_seconds):.2f}s."
                        )

        if "QWEN" in (workflow_name or "").upper():
            local_sections.append("Image edit instruction: preserve key identity/features unless explicitly changed.")
            llm_instructions += " For Qwen image-edit workflows, preserve identity unless user requests change."

        return {
            "template_key": template_key,
            "template_label": self._template_label(template_key),
            "camera_hint": camera_hint,
            "motion_hint": motion_hint,
            "pattern_hint": pattern_hint,
            "timeline_mode": timeline_mode,
            "duration_seconds": duration_seconds,
            "source_video_seconds": source_video_seconds,
            "local_sections": local_sections,
            "llm_instructions": llm_instructions,
            "cinematic_writing_mode": cinematic_mode,
        }

    def _is_model_in_cooldown(self, model_name: str) -> bool:
        cooldown_until = self._model_cooldown_until.get(model_name, 0.0)
        return cooldown_until > time.time()

    def _register_unusable_output(self, model_name: str):
        failures = self._model_unusable_failures.get(model_name, 0) + 1
        self._model_unusable_failures[model_name] = failures
        threshold = max(1, self.unusable_output_failures_for_cooldown)
        if failures >= threshold:
            until = time.time() + max(60, self.unusable_output_cooldown_seconds)
            self._model_cooldown_until[model_name] = until
            print(
                f"[Enhancer] Model cooldown activated for {model_name} "
                f"({self.unusable_output_cooldown_seconds}s, failures={failures})"
            )

    def _register_model_success(self, model_name: str):
        self._model_unusable_failures.pop(model_name, None)
        self._model_cooldown_until.pop(model_name, None)

    def idea_to_prompt(
        self,
        idea_text: str,
        workflow_name: str,
        selections: dict | None = None,
        conversation_history: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> tuple[str, list[str]]:
        """Convert a loose idea into a production-ready prompt via LLM."""
        route_steps: list[str] = ["Idea Mode: converting idea to prompt"]
        selections = dict(selections or {})
        directives = self._build_template_directives(workflow_name, selections)
        route_steps.append(f"Template: {directives['template_label']}")

        # Collect non-default preset context
        preset_lines: list[str] = []
        preset_keys = ["style", "film_style", "camera", "motion", "light", "mood", "pattern"]
        for key in preset_keys:
            val = selections.get(key)
            if val and str(val).lower() not in ("none", "default", ""):
                preset_lines.append(f"{key}: {val}")
        preset_context = "\n".join(preset_lines) if preset_lines else "No presets selected."

        # Build system prompt
        base_system = self._system_prompt_for_model(directives)
        idea_instruction = (
            "The user is describing a rough IDEA or concept, not a finished prompt. "
            "Your job is to transform this idea into a polished, production-ready prompt "
            "that follows all template and style guidelines. "
            "Be creative in filling gaps — add specific visual details, camera work, "
            "lighting, and mood where the user left them vague.\n\n"
        )

        if self.idea_mode_ask_questions:
            idea_instruction += (
                "If the idea is too vague to produce a good prompt, you MAY ask up to "
                f"{self.idea_mode_max_questions} short clarifying questions. "
                "Prefix your output with [QUESTION] if you are asking questions, "
                "or [PROMPT] if you are providing the final prompt. "
                "Only ask questions when truly necessary — prefer generating a prompt.\n\n"
            )
        else:
            idea_instruction += (
                "ALWAYS generate a prompt. Never ask clarifying questions. "
                "If the idea is vague, use your creativity to fill in the gaps "
                "with compelling visual details.\n"
                "Output ONLY the final prompt text, no prefix, no markdown.\n\n"
            )

        system_content = base_system + idea_instruction

        # Build user content
        user_content = (
            f"Template: {directives['template_label']}\n"
            f"Workflow: {workflow_name}\n"
            f"Active presets:\n{preset_context}\n\n"
            f"User idea: {idea_text}\n"
        )

        # Build messages
        messages: list[dict] = [{"role": "system", "content": system_content}]
        if conversation_history:
            messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_content})

        payload_template: dict = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.enhancement_max_tokens,
        }
        reasoning_options = self._build_reasoning_options()
        if reasoning_options:
            payload_template["reasoning"] = reasoning_options

        # Model fallback chain
        model_candidates = [self.openrouter_model] + [
            x for x in self.fallback_models if x and x != self.openrouter_model
        ]

        last_error = None
        for model_name in model_candidates:
            if self._is_model_in_cooldown(model_name):
                remaining = int(self._model_cooldown_until.get(model_name, time.time()) - time.time())
                route_steps.append(f"Skipping model in cooldown: {model_name} ({max(1, remaining)}s left)")
                continue
            try:
                print(f"[Enhancer] Idea mode trying model: {model_name}")
                route_steps.append(f"Trying model: {model_name}")
                data = self._request_with_retries(
                    payload=dict(payload_template),
                    model_name=model_name,
                    request_label="idea_to_prompt",
                )
                content, finish_reason = self._extract_text_and_meta(data)
                if not content:
                    self._register_unusable_output(model_name)
                    raise RuntimeError(
                        f"No text content from model (finish_reason={finish_reason!r})."
                    )
                self._register_model_success(model_name)
                route_steps.append(f"Model success: {model_name}")
                return content.strip(), route_steps
            except Exception as e:
                last_error = e
                route_steps.append(f"Model failed: {model_name} ({e})")
                print(f"[Enhancer] Idea mode model failed ({model_name}): {e}")

        raise RuntimeError(
            f"All OpenRouter model attempts failed for idea_to_prompt. Last error: {last_error}. "
            f"Route: {' | '.join(route_steps)}"
        )

    def refine_idea_prompt(
        self,
        current_prompt: str,
        feedback: str,
        workflow_name: str,
        selections: dict | None = None,
        original_idea: str = "",
    ) -> tuple[str, list[str]]:
        """Refine an existing prompt based on user feedback."""
        route_steps: list[str] = ["Idea Mode: refining prompt with feedback"]
        selections = dict(selections or {})
        directives = self._build_template_directives(workflow_name, selections)
        route_steps.append(f"Template: {directives['template_label']}")

        base_system = self._system_prompt_for_model(directives)
        refine_instruction = (
            "The user previously described an idea and received a generated prompt. "
            "Now they are providing feedback to refine it. "
            "Apply their feedback to improve the prompt while keeping its strengths. "
            "Output ONLY the refined prompt text, no prefix, no markdown, no explanations.\n\n"
        )
        system_content = base_system + refine_instruction

        user_content = (
            f"Template: {directives['template_label']}\n"
            f"Workflow: {workflow_name}\n\n"
        )
        if original_idea:
            user_content += f"Original idea: {original_idea}\n\n"
        user_content += (
            f"Current prompt:\n{current_prompt}\n\n"
            f"User feedback: {feedback}\n"
        )

        messages: list[dict] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

        payload_template: dict = {
            "messages": messages,
            "temperature": 0.6,
            "max_tokens": self.enhancement_max_tokens,
        }
        reasoning_options = self._build_reasoning_options()
        if reasoning_options:
            payload_template["reasoning"] = reasoning_options

        # Model fallback chain
        model_candidates = [self.openrouter_model] + [
            x for x in self.fallback_models if x and x != self.openrouter_model
        ]

        last_error = None
        for model_name in model_candidates:
            if self._is_model_in_cooldown(model_name):
                remaining = int(self._model_cooldown_until.get(model_name, time.time()) - time.time())
                route_steps.append(f"Skipping model in cooldown: {model_name} ({max(1, remaining)}s left)")
                continue
            try:
                print(f"[Enhancer] Idea refine trying model: {model_name}")
                route_steps.append(f"Trying model: {model_name}")
                data = self._request_with_retries(
                    payload=dict(payload_template),
                    model_name=model_name,
                    request_label="refine_idea_prompt",
                )
                content, finish_reason = self._extract_text_and_meta(data)
                if not content:
                    self._register_unusable_output(model_name)
                    raise RuntimeError(
                        f"No text content from model (finish_reason={finish_reason!r})."
                    )
                self._register_model_success(model_name)
                route_steps.append(f"Model success: {model_name}")
                return content.strip(), route_steps
            except Exception as e:
                last_error = e
                route_steps.append(f"Model failed: {model_name} ({e})")
                print(f"[Enhancer] Idea refine model failed ({model_name}): {e}")

        raise RuntimeError(
            f"All OpenRouter model attempts failed for refine_idea_prompt. Last error: {last_error}. "
            f"Route: {' | '.join(route_steps)}"
        )

    def enhance_prompt(
        self,
        user_prompt: str,
        workflow_name: str,
        selections: Dict[str, Any],
        duration_seconds: int | None = None,
        source_image_bytes: bytes | None = None,
        inspire_mode: bool = False,
    ) -> EnhancementResult:
        if inspire_mode:
            selections = dict(selections or {})
            selections["_inspire_mode"] = True
            if duration_seconds is not None:
                selections["_duration_seconds"] = duration_seconds
            route_steps: list[str] = ["Inspire Me: generating prompt from presets"]
            directives = self._build_template_directives(workflow_name, selections)
            route_steps.append(f"Template: {directives['template_label']}")
            if directives.get("timeline_mode"):
                dur = directives.get("duration_seconds")
                if isinstance(dur, (int, float)) and dur > 0:
                    route_steps.append(f"Timeline mode: ON ({int(dur)}s)")
            # Vision analysis for inspire mode
            if (
                self.openrouter_enabled
                and self.openrouter_api_key
                and self.vision_enabled
                and self._is_vision_template(workflow_name)
                and source_image_bytes
            ):
                try:
                    visual_context, vision_steps = self._analyze_source_image_with_openrouter(
                        source_image_bytes, workflow_name, "", selections,
                    )
                    selections["_source_visual_context"] = visual_context
                    route_steps.extend(vision_steps)
                except Exception as e:
                    route_steps.append(f"Vision analysis failed: {e}")
            # LLM generation
            if self.openrouter_enabled and self.openrouter_api_key:
                try:
                    final_prompt, llm_steps = self._enhance_with_openrouter(
                        "", workflow_name, selections,
                    )
                    route_steps.extend(llm_steps)
                    return EnhancementResult(
                        final_prompt, True, False,
                        "Inspire Me: creative prompt generated by LLM",
                        route_steps,
                    )
                except Exception as e:
                    route_steps.append(f"LLM inspire failed: {e}")
            route_steps.append("Inspire Me requires OpenRouter — no local fallback")
            return EnhancementResult(
                "cinematic scene", False, False,
                "Inspire Me failed: OpenRouter unavailable",
                route_steps,
            )
        user_prompt, prefixed_v2v = self._normalize_v2v_continuation_prompt(workflow_name, user_prompt)
        selections = dict(selections or {})
        selections["_dialogue_quote_speaker_map"] = self._extract_dialogue_quote_speaker_map(user_prompt)
        if duration_seconds is not None:
            selections["_duration_seconds"] = duration_seconds
        route_steps: list[str] = []
        # Early dialogue detection — show before template/timeline steps so it's visible in [:4] preview
        _early_dialogue_map = self._protect_dialogue_tokens(user_prompt)[1]
        _early_speaker_constraints = self._extract_dialogue_speaker_constraints(user_prompt)
        if _early_dialogue_map:
            route_steps.append(f"🔒 Dialogue locked: {len(_early_dialogue_map)} quote(s) preserved")
        if _early_speaker_constraints:
            route_steps.append(f"🔒 Speaker locks: {len(_early_speaker_constraints)}")
        directives = self._build_template_directives(workflow_name, selections)
        route_steps.append(f"Template: {directives['template_label']}")
        if prefixed_v2v:
            route_steps.append("V2V prefix auto-inserted: 'the video continues with ...'")
        if directives.get("timeline_mode"):
            duration_seconds_resolved = directives.get("duration_seconds")
            if isinstance(duration_seconds_resolved, (int, float)) and duration_seconds_resolved > 0:
                route_steps.append(f"Timeline mode: ON ({int(duration_seconds_resolved)}s)")
            else:
                route_steps.append("Timeline mode: ON")
        if directives.get("template_key") == "video_v2v":
            source_video_seconds = directives.get("source_video_seconds")
            if isinstance(source_video_seconds, (int, float)) and source_video_seconds > 0:
                route_steps.append(f"V2V source clip length: {float(source_video_seconds):.2f}s")
        if (
            self.openrouter_enabled
            and self.openrouter_api_key
            and self.vision_enabled
            and self._is_vision_template(workflow_name)
            and source_image_bytes
        ):
            try:
                visual_context, vision_steps = self._analyze_source_image_with_openrouter(
                    source_image_bytes,
                    workflow_name,
                    user_prompt,
                    selections,
                )
                selections["_source_visual_context"] = visual_context
                route_steps.extend(vision_steps)
            except Exception as e:
                route_steps.append(f"Vision analysis failed: {e}")
                print(f"[Enhancer] Vision analysis failed: {e}")
        if not self.enabled:
            local_prompt = self._build_local_prompt(user_prompt, workflow_name, selections)
            route_steps.append("Enhancer disabled")
            route_steps.append("Used local template")
            return EnhancementResult(
                local_prompt,
                False,
                True,
                "Enhancer disabled, using local format.",
                route_steps,
            )

        if self.openrouter_enabled and self.openrouter_api_key:
            try:
                llm_prompt, llm_route_steps = self._enhance_with_openrouter(
                    user_prompt, workflow_name, selections
                )
                route_steps.extend(llm_route_steps)
                llm_prompt, enforced_v2v_structure = self._ensure_v2v_structure_in_final_prompt(
                    workflow_name, llm_prompt, selections
                )
                if enforced_v2v_structure:
                    route_steps.append("V2V prompt structure enforced in final prompt.")
                return EnhancementResult(
                    llm_prompt,
                    True,
                    False,
                    "LLM enhancement complete.",
                    route_steps,
                )
            except Exception as e:
                route_steps.append(f"LLM path failed: {e}")
                print(f"[Enhancer] OpenRouter failed, falling back to local template: {e}")

        fallback_source_text = user_prompt
        if self.translate_on_local_fallback and self.openrouter_enabled and self.openrouter_api_key:
            try:
                translated_text, translation_steps = self._translate_text_with_openrouter(user_prompt)
                route_steps.extend(translation_steps)
                if translated_text:
                    fallback_source_text = translated_text
            except Exception as e:
                route_steps.append(f"Fallback translation failed: {e}")
                print(f"[Enhancer] Fallback translation failed: {e}")

        local_prompt = self._build_local_prompt(fallback_source_text, workflow_name, selections)
        local_prompt, enforced_v2v_structure = self._ensure_v2v_structure_in_final_prompt(
            workflow_name, local_prompt, selections
        )
        if enforced_v2v_structure:
            route_steps.append("V2V prompt structure enforced in local fallback prompt.")
        route_steps.append("Used local template fallback")
        return EnhancementResult(
            local_prompt,
            False,
            True,
            "LLM unavailable, local fallback used.",
            route_steps,
        )

    def _build_local_prompt(self, user_prompt: str, workflow_name: str, selections: Dict[str, Any]) -> str:
        directives = self._build_template_directives(workflow_name, selections)
        sections = []
        for line in directives["local_sections"]:
            sections.append(line.replace("{user_prompt}", user_prompt.strip()))
        return " ".join(sections)

    def _enhance_with_openrouter(
        self,
        user_prompt: str,
        workflow_name: str,
        selections: Dict[str, Any],
    ) -> tuple[str, list[str]]:
        route_steps: list[str] = []
        dialogue_speaker_constraints = self._extract_dialogue_speaker_constraints(user_prompt)
        protected_prompt, dialogue_token_map = self._protect_dialogue_tokens(user_prompt)
        if dialogue_token_map:
            route_steps.append(f"Protected dialogue tokens: {len(dialogue_token_map)}")
        if dialogue_speaker_constraints:
            route_steps.append(f"Dialogue speaker locks: {len(dialogue_speaker_constraints)}")
        model_candidates = [self.openrouter_model] + [
            x for x in self.fallback_models if x and x != self.openrouter_model
        ]

        last_error = None
        for model_name in model_candidates:
            if self._is_model_in_cooldown(model_name):
                remaining = int(self._model_cooldown_until.get(model_name, time.time()) - time.time())
                route_steps.append(f"Skipping model in cooldown: {model_name} ({max(1, remaining)}s left)")
                continue
            try:
                print(f"[Enhancer] Trying OpenRouter model: {model_name}")
                route_steps.append(f"Trying model: {model_name}")
                payload = self._build_enhancement_payload(
                    user_prompt=protected_prompt,
                    workflow_name=workflow_name,
                    selections=selections,
                    model_name=model_name,
                    dialogue_speaker_constraints=dialogue_speaker_constraints,
                )
                data = self._request_with_retries(
                    payload=payload,
                    model_name=model_name,
                    request_label="enhancement",
                )
                content, finish_reason = self._extract_text_and_meta(data)
                if not content:
                    self._register_unusable_output(model_name)
                    raise RuntimeError(
                        f"No text content from model (finish_reason={finish_reason!r})."
                    )
                content = self._restore_dialogue_tokens(content, dialogue_token_map)
                if str(finish_reason).lower() == "length" and self._looks_truncated(content):
                    self._register_unusable_output(model_name)
                    raise RuntimeError(
                        "Enhancer output hit token limit and appears truncated "
                        f"(finish_reason={finish_reason!r}, len={len(content)})."
                    )
                if not self._is_prompt_usable(content):
                    self._register_unusable_output(model_name)
                    raise RuntimeError(
                        f"Enhancer output too short/incomplete (finish_reason={finish_reason!r}, len={len(content)})."
                    )
                if str(finish_reason).lower() == "length":
                    route_steps.append(
                        f"Model response finished with length but accepted: {model_name}"
                    )
                self._register_model_success(model_name)
                route_steps.append(f"Model success: {model_name}")
                return content, route_steps
            except Exception as e:
                last_error = e
                route_steps.append(f"Model failed: {model_name} ({e})")
                print(f"[Enhancer] Model failed ({model_name}): {e}")

        raise RuntimeError(
            f"All OpenRouter model attempts failed. Last error: {last_error}. "
            f"Route: {' | '.join(route_steps)}"
        )

    def _translate_text_with_openrouter(self, source_text: str) -> tuple[str, list[str]]:
        route_steps: list[str] = []
        protected_text, dialogue_token_map = self._protect_dialogue_tokens(source_text)
        if dialogue_token_map:
            route_steps.append(f"Fallback translation protected dialogue: {len(dialogue_token_map)}")
        model_candidates = [self.openrouter_model] + [
            x for x in self.fallback_models if x and x != self.openrouter_model
        ]

        payload_template = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate the user text into natural, concise English for image/video generation. "
                        "Preserve intent and key details. Output only translated text. "
                        "Never translate or alter [DIALOGUE_N] tokens; keep them exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": protected_text,
                },
            ],
            "temperature": 0.2,
            "max_tokens": self.translation_max_tokens,
        }
        reasoning_options = self._build_reasoning_options()
        if reasoning_options:
            payload_template["reasoning"] = reasoning_options

        last_error = None
        for model_name in model_candidates:
            if self._is_model_in_cooldown(model_name):
                remaining = int(self._model_cooldown_until.get(model_name, time.time()) - time.time())
                route_steps.append(f"Fallback translation skip cooldown: {model_name} ({max(1, remaining)}s left)")
                continue
            try:
                route_steps.append(f"Fallback translation try: {model_name}")
                data = self._request_with_retries(
                    payload=dict(payload_template),
                    model_name=model_name,
                    request_label="translation",
                )
                content, finish_reason = self._extract_text_and_meta(data)
                if content:
                    content = self._restore_dialogue_tokens(content, dialogue_token_map)
                    if str(finish_reason).lower() == "length" and self._looks_truncated(content):
                        self._register_unusable_output(model_name)
                        route_steps.append(
                            f"Fallback translation truncated: {model_name} (finish_reason={finish_reason!r})"
                        )
                        continue
                    self._register_model_success(model_name)
                    route_steps.append(f"Fallback translation success: {model_name}")
                    return content, route_steps
                self._register_unusable_output(model_name)
                route_steps.append(
                    f"Fallback translation empty: {model_name} (finish_reason={finish_reason!r})"
                )
            except Exception as e:
                last_error = e
                route_steps.append(f"Fallback translation failed: {model_name} ({e})")

        raise RuntimeError(f"No translation response from fallback models. Last error: {last_error}")

    def _system_prompt_for_model(self, directives: dict) -> str:
        """Return model-family-specific system prompt for the LLM."""
        family = directives.get("model_family", "generic")
        if family == "ltx":
            return (
                "You are an expert cinematic prompt engineer for the LTX-Video AI model. "
                "You produce a SINGLE FLOWING PARAGRAPH of max 200 words in natural English. "
                "No labels, no sections, no bullet points. Start directly with the action. "
                "Weave camera, lighting, style, and mood naturally into the scene description. "
                "Use film-industry terminology: film stocks, lens focal lengths, aperture, shutter angle. "
                "Output only the final prompt text, no markdown, no explanations.\n\n"
            )
        if family == "flux":
            return (
                "You are an expert prompt engineer for the Flux image generation model by Black Forest Labs. "
                "You produce a CONCISE NATURAL LANGUAGE PARAGRAPH of 30-80 words. "
                "Front-load the most important elements (subject, action first). "
                "Camera and lens specifications (focal lengths, f-stops, camera brands, film stocks) are "
                "understood by Flux as RENDERING INSTRUCTIONS, not scene objects — use them freely. "
                "NEVER use quality boosters (masterpiece, best quality, 4k, highly detailed). "
                "NEVER use negative language (no artifacts, without distortion). "
                "Output only the final prompt text, no markdown, no explanations.\n\n"
            )
        if family == "pony":
            return (
                "You are an expert prompt engineer for Pony Diffusion / CyberRealistic Pony models. "
                "You MUST start every prompt with the quality score chain: "
                "'score_9, score_8_up, score_7_up, score_6_up, score_5_up, score_4_up' — "
                "this is mandatory and must never be omitted. "
                "After scores, write a concise comma-separated description mixing tags and natural language. "
                "Keep total output under 70 tokens (~50-60 words including scores). "
                "NEVER use quality boosters (masterpiece, best quality). The score system replaces them. "
                "Use art/photography terms for framing, not specific f-stops or focal lengths. "
                "Output only the final prompt text, no markdown, no explanations.\n\n"
            )
        # Generic fallback
        return (
            "You are an expert cinematic prompt engineer for AI image and video generation. "
            "You master photography terminology (focal lengths, aperture, depth of field, "
            "named lighting setups like Rembrandt/butterfly/split) and cinematic language. "
            "Convert user intent into one high-quality English production prompt. "
            "Use specific, concrete visual descriptions — never vague terms. "
            "Output only the final prompt text, no markdown, no explanations.\n\n"
        )

    def _build_enhancement_payload(
        self,
        user_prompt: str,
        workflow_name: str,
        selections: Dict[str, Any],
        model_name: str,
        dialogue_speaker_constraints: list[str] | None = None,
    ) -> dict:
        directives = self._build_template_directives(workflow_name, selections)
        style_key = selections.get("style", "none")
        camera_key = selections.get("camera")
        light_key = selections.get("light")
        mood_key = selections.get("mood")
        motion_key = selections.get("motion")
        pattern_key = selections.get("pattern", "none")
        storytelling = bool(selections.get("storytelling", False))
        timeline_mode = bool(directives.get("timeline_mode", False))
        duration_seconds = directives.get("duration_seconds")
        source_video_seconds = directives.get("source_video_seconds")
        source_visual_context = str(selections.get("_source_visual_context", "")).strip()
        cinematic_writing_mode = directives.get("cinematic_writing_mode", self.cinematic_writing_mode)
        speaker_constraints_text = ""
        if dialogue_speaker_constraints:
            speaker_constraints_text = "\n".join(f"- {line}" for line in dialogue_speaker_constraints)

        inspire_mode = bool(selections.get("_inspire_mode", False))

        if inspire_mode:
            user_content = (
                f"Template: {directives['template_label']}\n"
                f"Workflow: {workflow_name}\n"
                f"MODE: INSPIRE ME — No user prompt provided. Generate an original, vivid scene from scratch.\n\n"
                f"Director style preset: {style_key}\n"
                f"Film style preset: {selections.get('film_style', 'none')}\n"
                f"Camera preset: {camera_key}\n"
                f"Motion preset: {motion_key}\n"
                f"Cinematic pattern preset: {pattern_key}\n"
                f"Light preset: {light_key}\n"
                f"Mood preset: {mood_key}\n"
                f"Storytelling mode: {'on' if storytelling else 'off'}\n\n"
                f"Timeline mode: {'on' if timeline_mode else 'off'}\n"
                f"Video duration seconds (if known): {duration_seconds}\n\n"
                f"Source video duration seconds (V2V, if known): {source_video_seconds}\n\n"
                f"Source visual context (if available): {source_visual_context}\n\n"
                f"Cinematic writing mode: {cinematic_writing_mode}\n\n"
                "Requirements:\n"
                "1) Invent an original, compelling scene that naturally embodies ALL selected presets.\n"
                "2) Be concrete and visually specific — use photography terminology (e.g., 'shallow depth of field with f/1.8 bokeh' not 'blurry background', 'Rembrandt lighting' not 'dramatic light').\n"
                "3) Include camera framing, lighting setup, and mood as integrated visual description, not separate labels.\n"
                "4) Keep it concise enough for practical prompting.\n"
                "5) Use cinematic pattern preset as beat/choreography guidance when applicable.\n"
                "6) For video templates: separate start-frame composition vs camera motion timeline.\n"
                "7) If angle and motion conflict, keep angle as start frame and motion as primary timeline behavior.\n"
                "8) Output one clean production prompt body, no section headers like 'Timeline:' or 'Start Frame:'.\n"
                "9) Apply cinematic writing mode while keeping continuity and timeline constraints intact.\n"
                "10) Describe materials, textures, and surface qualities with physical specificity (e.g., 'brushed titanium', 'weathered leather', 'wet cobblestone reflections').\n"
                "11) Ensure lighting direction and shadow descriptions are physically consistent.\n"
                "12) The scene must feel like a real film moment — not generic or abstract. Include specific characters, settings, and actions.\n"
            )
            inspire_system_suffix = (
                "The user has NOT provided a scene description. You must CREATE an original, vivid scene "
                "from scratch that naturally embodies the selected director style, film style, camera, lighting, "
                "mood, and motion presets. Be creative, surprising, and cinematic. "
                "Each generation should feel unique and different.\n\n"
            )
        else:
            user_content = (
                f"Template: {directives['template_label']}\n"
                f"Workflow: {workflow_name}\n"
                f"User prompt (can be German): {user_prompt}\n"
                f"Director style preset: {style_key}\n"
                f"Film style preset: {selections.get('film_style', 'none')}\n"
                f"Camera preset: {camera_key}\n"
                f"Motion preset: {motion_key}\n"
                f"Cinematic pattern preset: {pattern_key}\n"
                f"Light preset: {light_key}\n"
                f"Mood preset: {mood_key}\n"
                f"Storytelling mode: {'on' if storytelling else 'off'}\n\n"
                f"Timeline mode: {'on' if timeline_mode else 'off'}\n"
                f"Video duration seconds (if known): {duration_seconds}\n\n"
                f"Source video duration seconds (V2V, if known): {source_video_seconds}\n\n"
                f"Source visual context (if available): {source_visual_context}\n\n"
                f"Cinematic writing mode: {cinematic_writing_mode}\n\n"
                f"Dialogue speaker lock (if available):\n{speaker_constraints_text}\n\n"
                "Requirements:\n"
                "1) Keep intent, translate/correct to natural English.\n"
                "2) Be concrete and visually specific — use photography terminology (e.g., 'shallow depth of field with f/1.8 bokeh' not 'blurry background', 'Rembrandt lighting' not 'dramatic light').\n"
                "3) Include camera framing, lighting setup, and mood as integrated visual description, not separate labels.\n"
                "4) Keep it concise enough for practical prompting.\n"
                "5) Never translate or alter any [DIALOGUE_N] token; keep the token exactly and place it naturally where speech is described.\n"
                "6) Use cinematic pattern preset as beat/choreography guidance when applicable.\n"
                "7) For video templates: separate start-frame composition vs camera motion timeline.\n"
                "8) If angle and motion conflict, keep angle as start frame and motion as primary timeline behavior.\n"
                "9) Output one clean production prompt body, no section headers like 'Timeline:' or 'Start Frame:'.\n"
                "10) Apply cinematic writing mode while keeping continuity and timeline constraints intact.\n"
                "11) Do not swap gender/role assignment between characters. Preserve explicit she/he, woman/man mapping.\n"
                "12) For V2V continuation: do not repeat or retell source-clip events/dialogue. Timeline must only cover newly generated continuation actions.\n"
                "13) If dialogue speaker lock is provided, each [DIALOGUE_N] line must be spoken by the mapped speaker exactly.\n"
                "14) Describe materials, textures, and surface qualities with physical specificity (e.g., 'brushed titanium', 'weathered leather', 'wet cobblestone reflections').\n"
                "15) Ensure lighting direction and shadow descriptions are physically consistent.\n"
            )
            inspire_system_suffix = ""

        result = {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        self._system_prompt_for_model(directives) +
                        inspire_system_suffix +
                        f"{directives['llm_instructions']}"
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": 0.8 if inspire_mode else 0.5,
            "max_tokens": self.enhancement_max_tokens,
        }
        reasoning_opts = self._build_reasoning_options()
        if reasoning_opts:
            result["reasoning"] = reasoning_opts
        return result

    def _extract_dialogue_speaker_constraints(self, original_prompt: str) -> list[str]:
        """Infer likely speaker mapping for quoted lines to reduce role/gender swaps."""
        text = original_prompt or ""
        quote_iter = list(re.finditer(r'"([^"]+)"', text))
        constraints: list[str] = []
        if not quote_iter:
            return constraints

        for idx, m in enumerate(quote_iter, start=1):
            left_context = text[max(0, m.start() - 180): m.start()]
            speaker = self._infer_speaker(left_context)
            if speaker:
                constraints.append(f"[DIALOGUE_{idx}] must be spoken by {speaker}.")
        return constraints

    def _extract_dialogue_quote_speaker_map(self, original_prompt: str) -> Dict[str, str]:
        """
        Build quote->speaker map from the raw user prompt.
        Speaker values: 'man' or 'woman'.
        """
        text = original_prompt or ""
        quote_iter = list(re.finditer(r'"([^"]+)"', text))
        result: Dict[str, str] = {}
        if not quote_iter:
            return result

        for m in quote_iter:
            quote_text = (m.group(1) or "").strip()
            if not quote_text:
                continue
            left_context = text[max(0, m.start() - 220): m.start()]
            speaker = self._infer_speaker(left_context)
            if speaker:
                result[quote_text] = speaker
        return result

    def _request_with_retries(
        self,
        payload: Dict[str, Any],
        model_name: str,
        request_label: str,
    ) -> Dict[str, Any]:
        payload = dict(payload)
        payload["model"] = model_name
        payload.setdefault("provider", {"allow_fallbacks": True})

        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Sh3yMo/CTB",
            "X-Title": "CTB Prompt Enhancer",
        }

        attempts = max(1, self.max_retries)
        for attempt_index in range(attempts):
            response = requests.post(
                self.openrouter_base_url,
                json=payload,
                headers=headers,
                timeout=self.request_timeout_seconds,
            )

            if response.status_code == 429:
                response_text = response.text.strip()
                print(
                    f"[Enhancer] OpenRouter 429 ({request_label}, model={model_name}, attempt={attempt_index + 1}/{attempts}): "
                    f"{response_text}"
                )
                if attempt_index < attempts - 1:
                    backoff_index = min(attempt_index, len(self.retry_backoff_seconds) - 1)
                    default_sleep = float(self.retry_backoff_seconds[backoff_index])
                    retry_after_header = (
                        response.headers.get("Retry-After")
                        or response.headers.get("x-ratelimit-reset-requests")
                    )
                    if retry_after_header:
                        try:
                            sleep_seconds = float(retry_after_header)
                        except ValueError:
                            sleep_seconds = default_sleep
                    else:
                        sleep_seconds = default_sleep
                    print(
                        f"[Enhancer] Backoff {sleep_seconds}s before retry "
                        f"(model={model_name}, next_attempt={attempt_index + 2}/{attempts})"
                    )
                    time.sleep(sleep_seconds)
                    continue
                raise RuntimeError(
                    f"OpenRouter rate limit (429) for model '{model_name}' during {request_label} after {attempts} attempts. "
                    f"Response: {response_text[:800]}"
                )

            if response.status_code >= 400:
                response_text = response.text.strip()
                raise RuntimeError(
                    f"OpenRouter HTTP {response.status_code} for model '{model_name}'. "
                    f"Response: {response_text[:800]}"
                )

            data = response.json()
            return data

        raise RuntimeError(f"Unexpected retry loop exit for model '{model_name}' ({request_label}).")

    def _extract_text_and_meta(self, data: Dict[str, Any]) -> tuple[str, str]:
        finish_reason = ""
        # Common OpenAI/OpenRouter shape
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            finish_reason = str(choices[0].get("finish_reason", ""))
            msg = choices[0].get("message", {})
            content = msg.get("content")
            text = self._normalize_content(content)
            if text:
                return text, finish_reason

            # Some providers return text at choice-level
            alt_text = choices[0].get("text")
            if isinstance(alt_text, str) and alt_text.strip():
                return alt_text.strip(), finish_reason

        # Some providers expose output text at root
        root_content = data.get("output_text")
        if isinstance(root_content, str) and root_content.strip():
            return root_content.strip(), finish_reason

        return "", finish_reason

    def _extract_text_content(self, data: Dict[str, Any]) -> str:
        text, finish_reason = self._extract_text_and_meta(data)
        if text:
            return text
        raise RuntimeError(f"OpenRouter returned no parseable text content. Raw: {str(data)[:900]}")

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                text_candidate = item.get("text")
                if isinstance(text_candidate, str) and text_candidate.strip():
                    parts.append(text_candidate.strip())
            return " ".join(parts).strip()
        return ""


# ---------------------------------------------------------------------------
# Film-mode prompt enhancement
# ---------------------------------------------------------------------------

FILM_ENHANCE_SYSTEM = """You are an expert cinematographer writing prompts for an AI video model (LTX2.3).
Your task: expand the given scene description into a rich, detailed video generation prompt (80-120 words).

Rules:
- Begin with the dominant visual: what fills the frame first
- Describe lighting quality precisely (hard/soft, direction, colour temperature, shadows)
- Include camera behaviour: movement type, speed, angle
- Describe subject action and micro-details (texture, material, atmosphere)
- End with the emotional register of the shot
- Do NOT include dialogue or sound
- Do NOT use vague words like "beautiful" or "cinematic" — be specific about what the camera sees

Context you will receive: shot_type, lighting, camera_move, mood.
Weave these into the prompt naturally — do not list them as metadata."""


async def enhance_prompt(prompt: str, mode: str = "t2v", context: dict = None):
    """Module-level entry point for prompt enhancement.

    Handles 'film' mode directly with a targeted OpenRouter call.
    For 't2v' mode (used in tests), also calls OpenRouter with a simpler system prompt
    so tests can compare verbosity without spinning up the full PromptEnhancer class.
    """
    cfg = json.loads((Path(__file__).parent / "config.json").read_text())
    api_key = cfg["prompt_enhancer"]["openrouter_api_key"]
    model = cfg.get("film_agents", {}).get("prompt", {}).get("model", "qwen/qwen3-4b")

    if mode == "film":
        ctx = context or {}
        shot_type = ctx.get("shot_type", "MS")
        lighting = ctx.get("lighting", "natural light")
        camera_move = ctx.get("camera_move", "static")
        mood = ctx.get("mood", "neutral")
        protagonist = ctx.get("protagonist", "")
        protagonist_line = f"Protagonist appearance: {protagonist}\n" if protagonist else ""
        user_msg = (
            f"Scene: {prompt}\n\n"
            f"{protagonist_line}"
            f"Shot type: {shot_type}\n"
            f"Lighting: {lighting}\n"
            f"Camera move: {camera_move}\n"
            f"Mood: {mood}\n\n"
            "Write the enhanced video prompt:"
        )
        system = FILM_ENHANCE_SYSTEM
    else:
        # t2v fallback: short enhancement for comparison purposes
        user_msg = f"Enhance this video prompt briefly (30-50 words): {prompt}"
        system = "You are a video prompt writer. Return an enhanced prompt. Be concise."

    fallback_models = cfg.get("film_agents", {}).get("ltx_prompt", {}).get("fallback_models", [])
    model_candidates = [model] + [x for x in fallback_models if x and x != model]
    _max_retries = 5
    _backoff = [30, 60, 120, 300, 600]
    for model_idx, current_model in enumerate(model_candidates):
        for attempt in range(_max_retries):
            if attempt == 0:
                logger.info("PromptEnhancer → OpenRouter [%s]", current_model)
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": current_model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user_msg},
                        ],
                        "max_tokens": 300,
                        "provider": {"allow_fallbacks": True},
                    },
                )
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After") or resp.headers.get("x-ratelimit-reset-requests")
                try:
                    sleep_s = float(retry_after) if retry_after else float(_backoff[min(attempt, len(_backoff) - 1)])
                except ValueError:
                    sleep_s = float(_backoff[min(attempt, len(_backoff) - 1)])
                if attempt < _max_retries - 1:
                    logger.warning(
                        "enhance_prompt 429 (model=%s, attempt %d/%d), sleeping %.0fs...",
                        current_model, attempt + 1, _max_retries, sleep_s,
                    )
                    await asyncio.sleep(sleep_s)
                    continue
                if model_idx < len(model_candidates) - 1:
                    logger.warning(
                        "enhance_prompt 429 retries exhausted for '%s', switching to fallback: %s",
                        current_model, model_candidates[model_idx + 1],
                    )
                break
            elif resp.status_code in (400, 404, 503):
                if model_idx < len(model_candidates) - 1:
                    logger.warning(
                        "enhance_prompt %d (model=%s), switching to fallback: %s",
                        resp.status_code, current_model, model_candidates[model_idx + 1],
                    )
                    break
                resp.raise_for_status()
            else:
                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise RuntimeError(f"OpenRouter returned no choices: {str(data)[:400]}")
                content = choices[0].get("message", {}).get("content")
                if not content:
                    raise RuntimeError(f"OpenRouter returned empty content: {str(data)[:400]}")
                return content.strip()
    raise RuntimeError(f"enhance_prompt: OpenRouter alle Modelle erschöpft {model_candidates}.")
