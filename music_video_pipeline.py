"""
Music Video Pipeline for CTB — segment-based music video generation.

Splits audio into segments, generates video clips per segment via IA2V
ComfyUI workflows, and assembles clips into a final music video.
"""

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

# Default segmentation settings
DEFAULT_FALLBACK_SEGMENT_LENGTH = 10  # seconds
DEFAULT_MAX_SEGMENT_DURATION = 15     # seconds, subdivide longer segments
DEFAULT_CROSSFADE_DURATION = 0.5      # seconds


@dataclass
class Segment:
    index: int
    start_time: float
    end_time: float
    label: str = ""
    lyrics: str = ""
    audio_clip: str = ""
    prompt: str = ""
    enriched_prompt: str = ""
    video_clip: str = ""
    end_frame: str = ""
    end_frame_comfy: str = ""  # ComfyUI filename for the end frame
    transition: str = "cut"  # "cut" or "crossfade"
    status: str = "pending"  # "pending", "generating", "completed"

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "label": self.label,
            "lyrics": self.lyrics,
            "audio_clip": self.audio_clip,
            "prompt": self.prompt,
            "enriched_prompt": self.enriched_prompt,
            "video_clip": self.video_clip,
            "end_frame": self.end_frame,
            "end_frame_comfy": self.end_frame_comfy,
            "transition": self.transition,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Segment":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MVSession:
    audio_path: str = ""
    start_image_path: str = ""
    workflow: str = ""
    prompt_mode: str = ""          # "quick" or "creative"
    global_prompt: str = ""        # Quick Mode only
    theme: str = ""                # Creative Mode only
    scene_anchor: str = ""         # Constant character/scene description for visual consistency
    start_mode: str = ""           # "ta2v" = first segment via TA2V workflow, no start image needed
    resolution: dict = field(default_factory=dict)  # e.g. {"dimensions": "1344 x 768 (landscape)"}
    frame_analysis: bool = True    # Vision-LLM enrichment on/off
    auto_continue: bool = False    # Skip per-segment review and generate all automatically
    segments: List[Segment] = field(default_factory=list)
    crossfade_duration: float = DEFAULT_CROSSFADE_DURATION
    segment_max_duration: float = DEFAULT_MAX_SEGMENT_DURATION
    fallback_segment_length: float = DEFAULT_FALLBACK_SEGMENT_LENGTH
    output_path: str = ""
    current_segment_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "start_image_path": self.start_image_path,
            "workflow": self.workflow,
            "prompt_mode": self.prompt_mode,
            "global_prompt": self.global_prompt,
            "theme": self.theme,
            "scene_anchor": self.scene_anchor,
            "start_mode": self.start_mode,
            "resolution": self.resolution,
            "frame_analysis": self.frame_analysis,
            "auto_continue": self.auto_continue,
            "segments": [s.to_dict() for s in self.segments],
            "crossfade_duration": self.crossfade_duration,
            "segment_max_duration": self.segment_max_duration,
            "fallback_segment_length": self.fallback_segment_length,
            "output_path": self.output_path,
            "current_segment_index": self.current_segment_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MVSession":
        segments_data = data.pop("segments", [])
        session = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        session.segments = [Segment.from_dict(s) for s in segments_data]
        return session

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "MVSession":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def completed_count(self) -> int:
        return sum(1 for s in self.segments if s.status == "completed")

    def next_pending(self) -> Optional[Segment]:
        for s in self.segments:
            if s.status == "pending":
                return s
        return None


def parse_lyrics_file(lyrics_path: str) -> List[Dict[str, Any]]:
    """Parse a lyrics file with [Tag] structure into labeled blocks.

    Returns list of dicts: {"label": str, "lyrics": str}
    """
    if not os.path.exists(lyrics_path):
        return []

    with open(lyrics_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = []
    parts = re.split(r'\[([^\]]+)\]', content)

    i = 1
    while i < len(parts) - 1:
        label = parts[i].strip()
        lyrics = parts[i + 1].strip()
        blocks.append({"label": label, "lyrics": lyrics})
        i += 2

    return blocks


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-show_entries",
            "format=duration", "-of", "csv=p=0", audio_path
        ],
        capture_output=True, text=True
    )
    return float(result.stdout.strip())


def segment_audio(
    audio_path: str,
    output_dir: str,
    lyrics_path: Optional[str] = None,
    fallback_length: float = DEFAULT_FALLBACK_SEGMENT_LENGTH,
    max_duration: float = DEFAULT_MAX_SEGMENT_DURATION,
) -> List[Segment]:
    """Split audio into segments based on lyrics structure or fixed length."""
    total_duration = get_audio_duration(audio_path)
    blocks = []

    if lyrics_path:
        blocks = parse_lyrics_file(lyrics_path)

    if blocks:
        segments = _segment_from_blocks(blocks, total_duration, max_duration)
    else:
        segments = _segment_fixed_length(total_duration, fallback_length)

    os.makedirs(output_dir, exist_ok=True)
    for seg in segments:
        clip_path = os.path.join(output_dir, f"seg_{seg.index:03d}.wav")
        _extract_audio_clip(audio_path, seg.start_time, seg.end_time, clip_path)
        seg.audio_clip = clip_path

    return segments


def _segment_from_blocks(
    blocks: List[Dict[str, Any]],
    total_duration: float,
    max_duration: float,
) -> List[Segment]:
    """Create segments from lyrics blocks, distributing time proportionally."""
    n = len(blocks)
    if n == 0:
        return []

    block_duration = total_duration / n
    segments = []
    idx = 0

    for i, block in enumerate(blocks):
        start = i * block_duration
        end = min((i + 1) * block_duration, total_duration)

        if (end - start) > max_duration:
            sub_count = int((end - start) / max_duration) + 1
            sub_len = (end - start) / sub_count
            for j in range(sub_count):
                sub_start = start + j * sub_len
                sub_end = min(start + (j + 1) * sub_len, total_duration)
                label = f"{block['label']} ({j+1}/{sub_count})" if sub_count > 1 else block["label"]
                segments.append(Segment(
                    index=idx,
                    start_time=round(sub_start, 3),
                    end_time=round(sub_end, 3),
                    label=label,
                    lyrics=block["lyrics"] if j == 0 else "",
                ))
                idx += 1
        else:
            segments.append(Segment(
                index=idx,
                start_time=round(start, 3),
                end_time=round(end, 3),
                label=block["label"],
                lyrics=block["lyrics"],
            ))
            idx += 1

    return segments


def _segment_fixed_length(total_duration: float, segment_length: float) -> List[Segment]:
    """Create segments of fixed length."""
    segments = []
    idx = 0
    t = 0.0
    while t < total_duration:
        end = min(t + segment_length, total_duration)
        segments.append(Segment(
            index=idx,
            start_time=round(t, 3),
            end_time=round(end, 3),
            label=f"Segment {idx + 1}",
        ))
        idx += 1
        t = end
    return segments


def _extract_audio_clip(audio_path: str, start: float, end: float, output_path: str):
    """Extract a clip from audio using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ss", str(start), "-to", str(end),
            "-c", "copy", output_path,
        ],
        capture_output=True, check=True,
    )


def extract_last_frame(video_path: str, output_path: str) -> str:
    """Extract the last frame from a video file using ffmpeg.

    Applies a light sharpen + contrast/saturation boost to counteract
    the accumulated quality drift caused by repeated H.264 encode/decode cycles.

    Returns the output_path on success.
    """
    # Seek slightly earlier (-0.5s) to land on an I-frame rather than a P-frame,
    # which reduces block artifacts.  The vf filters compensate for lossy drift:
    #   unsharp: mild sharpening (luma only, gentle)
    #   eq: slight contrast + saturation boost to counter fading
    subprocess.run(
        [
            "ffmpeg", "-y", "-sseof", "-0.5", "-i", video_path,
            "-frames:v", "1", "-update", "1",
            "-vf", "unsharp=3:3:0.5:3:3:0.0,eq=contrast=1.05:saturation=1.08",
            output_path,
        ],
        capture_output=True, check=True,
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Failed to extract last frame from {video_path}")
    return output_path


def extract_motion_frames(video_path: str, output_dir: str, num_frames: int = 3, lookback_seconds: float = 2.0) -> List[str]:
    """Extract multiple frames from the last N seconds of a video for motion analysis.

    Returns list of frame paths ordered oldest → newest.
    """
    # Get video duration
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        duration = lookback_seconds

    os.makedirs(output_dir, exist_ok=True)
    frames = []
    for i in range(num_frames):
        # Distribute timestamps evenly across last `lookback_seconds`
        offset = lookback_seconds * (i / max(num_frames - 1, 1))
        ts = max(0.0, duration - lookback_seconds + offset)
        out_path = os.path.join(output_dir, f"motion_frame_{i:02d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(ts), "-i", video_path,
             "-frames:v", "1", "-update", "1", out_path],
            capture_output=True, check=True,
        )
        if os.path.exists(out_path):
            frames.append(out_path)

    return frames


def assemble_video(
    session: MVSession,
    output_path: str,
) -> str:
    """Assemble all completed segments into a final music video.

    Applies per-segment transitions (cut or crossfade) and overlays
    the original audio track for seamless sound.

    Returns the output path.
    """
    completed = [s for s in session.segments if s.status == "completed" and s.video_clip]
    if not completed:
        raise ValueError("No completed segments to assemble")

    if len(completed) == 1:
        seg = completed[0]
        audio = seg.audio_clip if seg.audio_clip and os.path.exists(seg.audio_clip) else session.audio_path
        _mux_audio(seg.video_clip, audio, output_path)
        return output_path

    with tempfile.TemporaryDirectory(prefix="ctb_mv_assembly_") as tmp_dir:
        has_crossfade = any(
            s.transition == "crossfade" and session.crossfade_duration > 0
            for s in completed[:-1]
        )
        temp_video = os.path.join(tmp_dir, "assembled_no_audio.mp4")

        if not has_crossfade:
            # Concat demuxer: robust against differing resolutions/pixel formats
            filelist_path = os.path.join(tmp_dir, "filelist.txt")
            with open(filelist_path, "w", encoding="utf-8") as f:
                for seg in completed:
                    f.write(f"file '{seg.video_clip.replace(chr(92), '/')}'\n")
            cmd = [
                "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", filelist_path,
                "-c:v", "libx264", "-preset", "fast", "-r", "24", "-vsync", "cfr",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                temp_video,
            ]
            subprocess.run(cmd, capture_output=True, check=True)
        else:
            # Detect resolution from first clip for normalization
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", completed[0].video_clip],
                capture_output=True, text=True,
            )
            try:
                w, h = probe.stdout.strip().split(",")
                scale_filter = f"scale={w}:{h},setsar=1"
            except ValueError:
                scale_filter = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"

            inputs = []
            for seg in completed:
                inputs.extend(["-i", seg.video_clip])

            # Normalize all inputs first, then apply transitions
            norm_parts = [f"[{i}:v]{scale_filter}[n{i}]" for i in range(len(completed))]
            current_label = "[n0]"
            trans_parts = []
            running_duration = completed[0].duration

            for i in range(1, len(completed)):
                prev_seg = completed[i - 1]
                next_seg = completed[i]
                next_input = f"[n{i}]"

                if prev_seg.transition == "crossfade" and session.crossfade_duration > 0:
                    cf_dur = session.crossfade_duration
                    offset = max(0, running_duration - cf_dur)
                    out_label = f"[cf{i}]"
                    trans_parts.append(
                        f"{current_label}{next_input}xfade=transition=fade:"
                        f"duration={cf_dur}:offset={offset}{out_label}"
                    )
                    current_label = out_label
                    running_duration = offset + next_seg.duration
                else:
                    out_label = f"[cc{i}]"
                    trans_parts.append(
                        f"{current_label}{next_input}concat=n=2:v=1:a=0{out_label}"
                    )
                    current_label = out_label
                    running_duration += next_seg.duration

            filter_complex = ";".join(norm_parts + trans_parts)
            cmd = ["ffmpeg", "-y"] + inputs + [
                "-filter_complex", filter_complex, "-map", current_label,
                "-c:v", "libx264", "-preset", "fast", "-r", "24", "-vsync", "cfr",
                temp_video,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        # Build audio: use individual segment clips if available for perfect sync,
        # otherwise fall back to the full original audio.
        has_audio_clips = all(s.audio_clip and os.path.exists(s.audio_clip) for s in completed)
        if has_audio_clips:
            audio_source = _assemble_audio(completed, session.crossfade_duration, tmp_dir)
        else:
            audio_source = session.audio_path

        _mux_audio(temp_video, audio_source, output_path)

    return output_path


def _assemble_audio(segments: list, crossfade_duration: float, tmp_dir: str) -> str:
    """Concatenate individual segment audio clips with matching transitions."""
    if len(segments) == 1:
        return segments[0].audio_clip

    audio_inputs = []
    for seg in segments:
        audio_inputs.extend(["-i", seg.audio_clip])

    filter_parts = []
    current_label = "[0:a]"

    for i in range(1, len(segments)):
        prev_seg = segments[i - 1]
        next_input = f"[{i}:a]"
        out_label = f"[a{i}]"
        if prev_seg.transition == "crossfade" and crossfade_duration > 0:
            filter_parts.append(
                f"{current_label}{next_input}acrossfade=d={crossfade_duration}:c1=tri:c2=tri{out_label}"
            )
        else:
            filter_parts.append(
                f"{current_label}{next_input}concat=n=2:v=0:a=1{out_label}"
            )
        current_label = out_label

    audio_out = os.path.join(tmp_dir, "assembled_audio.wav")
    cmd = ["ffmpeg", "-y"] + audio_inputs + [
        "-filter_complex", ";".join(filter_parts),
        "-map", current_label,
        audio_out,
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return audio_out


def _mux_audio(video_path: str, audio_path: str, output_path: str):
    """Mux video with audio track."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            output_path,
        ],
        capture_output=True, check=True,
    )


class MusicVideoPrompter:
    """Handles LLM prompt generation and frame analysis for music video segments."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config.get("openrouter_api_key", "")
        self.text_model = config.get("openrouter_model", "qwen/qwen3.5-flash-02-23")
        self.vision_model = config.get("vision_model", "qwen/qwen-2.5-vl-7b-instruct")
        self.vision_fallbacks = config.get("vision_fallback_models", [])
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_tokens = int(config.get("enhancement_max_tokens", 1200))
        self.disable_reasoning = config.get("disable_reasoning", True)

    async def _call_openrouter(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call OpenRouter API. Returns response text."""
        model = model or self.text_model
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if self.disable_reasoning:
            payload["reasoning"] = {"effort": "none"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(self.base_url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except Exception as e:
            print(f"OpenRouter API error ({model}): {e}")
            return ""

        choices = data.get("choices", [])
        if not choices:
            return ""
        result = choices[0].get("message", {}).get("content", "").strip()
        return result

    async def generate_segment_prompts(
        self,
        segments: List[Segment],
        theme: str,
    ) -> List[str]:
        """Generate a visual prompt for each segment based on theme, lyrics, and position."""
        segment_descriptions = []
        total = len(segments)
        for i, seg in enumerate(segments):
            position = "opening" if i == 0 else "closing" if i == total - 1 else f"middle ({i+1}/{total})"
            desc = f"Segment {i+1}/{total} ({position})"
            if seg.label:
                desc += f" — {seg.label}"
            if seg.lyrics:
                desc += f"\nLyrics: {seg.lyrics[:200]}"
            segment_descriptions.append(desc)

        segment_list = "\n---\n".join(segment_descriptions)

        system_prompt = (
            "You are a music video director. Generate a visual prompt for each segment "
            "of a music video. Each prompt should describe what happens visually in that "
            "segment — camera angles, lighting, subjects, motion, mood. Keep each prompt "
            "under 100 words. Match the visual mood to the lyrics and song position. "
            "Ensure visual continuity between segments.\n\n"
            "IMPORTANT: If a segment has lyrics, you MUST include the exact lyrics text "
            "in double quotes within the prompt (e.g. She sings \"verse lyrics here\"). "
            "This is required so the video model recognizes the sung/spoken words.\n\n"
            "IMPORTANT: Always write all prompts in English, regardless of the language of the theme or lyrics.\n\n"
            "Return ONLY a JSON array of strings, one prompt per segment. No other text."
        )

        user_prompt = (
            f"Visual theme/style: {theme}\n\n"
            f"Segments:\n{segment_list}\n\n"
            f"Generate {total} visual prompts as a JSON array of strings."
        )

        response = await self._call_openrouter(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
        )

        try:
            # Strip markdown code fences, then extract first [ ... ] block
            cleaned = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.strip(), flags=re.MULTILINE).strip()
            start = cleaned.find('[')
            end = cleaned.rfind(']')
            if start != -1 and end > start:
                prompts = json.loads(cleaned[start:end + 1])
                if isinstance(prompts, list):
                    # Pad or trim to match segment count
                    while len(prompts) < len(segments):
                        prompts.append(theme)
                    return [str(p) for p in prompts[:len(segments)]]
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Could not parse LLM prompt response ({e}), using theme as fallback")
            print(f"Full response:\n{response}")
            return [theme] * len(segments)

        print(f"Warning: No JSON array found in LLM response, using theme as fallback")
        print(f"Full response:\n{response}")
        return [theme] * len(segments)

    async def analyze_frame(self, frame_path: str) -> str:
        """Analyze a single video frame. Kept for backwards compatibility — delegates to analyze_frames."""
        return await self.analyze_frames([frame_path])

    async def analyze_frames(self, frame_paths: List[str]) -> str:
        """Analyze a sequence of frames from the end of a video clip.

        Focuses on motion direction, lighting, and subject positions to improve
        continuity with the next segment.
        """
        if not frame_paths:
            return ""

        content: List[Dict[str, Any]] = []

        if len(frame_paths) == 1:
            instruction = (
                "Describe this video frame for visual continuity with the next shot. "
                "Focus on: subject position and pose, lighting color and direction, "
                "camera angle, background. Under 80 words. Output ONLY the description."
            )
        else:
            instruction = (
                f"These {len(frame_paths)} frames are from the END of a video clip, ordered oldest to newest. "
                "Describe for visual continuity: "
                "1) Subject position and pose in the LAST frame. "
                "2) Motion direction (what is moving and in which direction). "
                "3) Lighting: color, direction, and any changes across the frames. "
                "4) Camera angle. "
                "Be specific about motion direction (e.g. 'arm moving upward', 'camera pushing in'). "
                "Under 100 words. Output ONLY the description."
            )

        content.append({"type": "text", "text": instruction})
        for path in frame_paths:
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages = [{"role": "user", "content": content}]

        models_to_try = [self.vision_model] + self.vision_fallbacks
        for model in models_to_try:
            try:
                return await self._call_openrouter(messages, model=model, max_tokens=250)
            except Exception as e:
                print(f"Vision model {model} failed: {e}")
                continue

        return ""

    async def enrich_prompt(self, original_prompt: str, frame_description: str, scene_anchor: str = "") -> str:
        """Enrich a segment prompt with visual details from the previous frame."""
        if not frame_description:
            return original_prompt

        anchor_instruction = ""
        if scene_anchor:
            anchor_instruction = (
                f"CONSTANT ELEMENTS (must always be preserved, never changed by frame details): {scene_anchor}\n\n"
            )

        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You refine video generation prompts for visual continuity between video segments. "
                        "Given an original prompt and a description of the previous segment's end frames, "
                        "merge them into one cohesive prompt. Keep the original intent but add continuity. "
                        "CRITICAL RULES: "
                        "1. Continue any motion in the SAME direction (e.g. if arm was moving upward, it continues upward). "
                        "2. Preserve the exact lighting color and direction from the previous frame. "
                        "3. Match the subject's pose and position from the last frame as the starting state. "
                        "4. Constant elements listed by the user must ALWAYS appear, unchanged. "
                        "Under 120 words. Always write in English. Output ONLY the refined prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{anchor_instruction}"
                        f"Original prompt: {original_prompt}\n\n"
                        f"Previous frame: {frame_description}\n\n"
                        f"Refined prompt:"
                    ),
                },
            ],
            max_tokens=300,
        )
        return response if response else original_prompt

    async def generate_start_image_prompt(self, theme: str, genre: str = "") -> str:
        """Generate a T2I prompt for the music video's start image."""
        context = f"Theme: {theme}"
        if genre:
            context += f"\nGenre: {genre}"

        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a text-to-image prompt for the opening frame of a music video. "
                        "The image should set the visual tone for the entire video. "
                        "Include subject, setting, lighting, color palette, and mood. "
                        "Under 80 words. Always write in English. Output ONLY the prompt."
                    ),
                },
                {"role": "user", "content": context},
            ],
            max_tokens=200,
        )
        return response if response else theme

    async def extract_scene_anchor(self, theme: str) -> str:
        """Extract constant visual elements from a theme description for use as scene anchor."""
        response = await self._call_openrouter(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the constant visual elements from a music video theme description. "
                        "List only the elements that must remain identical in every video segment: "
                        "character appearance, clothing/outfit (or lack thereof), props, and core setting. "
                        "Ignore dynamic elements like lighting changes, camera angles, or timed events. "
                        "IMPORTANT: Always respond in English regardless of the input language. "
                        "Write a single concise sentence in English. Output ONLY the anchor sentence."
                    ),
                },
                {"role": "user", "content": f"Theme: {theme}"},
            ],
            max_tokens=120,
        )
        return response if response else theme
