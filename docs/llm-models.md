# LLM Model Routing

Diese Pipeline nutzt **vier spezialisierte LLM-Modelle**, nicht ein einziges.
Alle laufen über OpenRouter (`https://openrouter.ai/api/v1/chat/completions`).
Konfiguriert im Block `prompt_enhancer` in `config.json`.

## Modell-zu-Task Mapping

| Task | Funktion / Datei | Config-Key (unter `prompt_enhancer`) | Default-Wert |
|------|------------------|--------------------------------------|--------------|
| Lyrics-Generierung + Song-Enhancement | `AudioEnhancer.__init__` (`audio_enhancer.py:376`) | `audio_llm_model` | `google/gemma-4-31b-it` |
| Video-Segment-Prompts + Frame-Variant-Prompts (MCA Startframes) | `MusicVideoPrompter.generate_segment_prompts` (`music_video_pipeline.py:2269`) | `openrouter_model` + `fallback_models` | `qwen/qwen3.5-flash-02-23` |
| Portrait / Still-Prompts (Flux 2) | `MusicVideoPrompter.generate_character_portrait_prompt` | `openrouter_model` + `fallback_models` | `qwen/qwen3.5-flash-02-23` |
| Lastframe-Continuity (Vision auf Vorgängersegment) | `MusicVideoPrompter.analyze_frames` (`music_video_pipeline.py:3007`) | `vision_model` + `vision_fallback_models` | `qwen/qwen-2.5-vl-7b-instruct` |
| Source-Image-Analyse (User-Endpoint `/enhance/prompt`) | `PromptEnhancer._analyze_source_image_with_openrouter` (`prompt_enhancer.py:383`) | `vision_model` + `vision_fallback_models` | `qwen/qwen-2.5-vl-7b-instruct` |
| Frame-Validierung | `frame_validator.py:64` | `vision_model` + `vision_fallback_models` | `qwen/qwen-2.5-vl-7b-instruct` |
| Genre-Validierung | `AudioEnhancer._validate_genre` | `genre_validation_model` | `qwen/qwen3.5-9b` |
| Prompt-Enhance (User-Submit Endpoint `/enhance/prompt`) | `prompt_enhancer.py:183` | `openrouter_model` + `fallback_models` | `qwen/qwen3.5-flash-02-23` |

## config.json Block

```jsonc
{
  "prompt_enhancer": {
    "openrouter_model": "qwen/qwen3.5-flash-02-23",
    "fallback_models": [
      "qwen/qwen3.5-9b",
      "nvidia/nemotron-3-super-120b-a12b"
    ],
    "vision_model": "qwen/qwen-2.5-vl-7b-instruct",
    "vision_fallback_models": [],
    "audio_llm_model": "google/gemma-4-31b-it",
    "genre_validation_model": "qwen/qwen3.5-9b",
    "openrouter_api_key": "sk-or-v1-...",
    "openrouter_base_url": "https://openrouter.ai/api/v1/chat/completions"
  }
}
```

## Fallback-Strategie

Auf 429 (Rate-Limit) oder 5xx-Fehlern werden `fallback_models` in der
gegebenen Reihenfolge versucht (`music_video_pipeline.py:2162`, Fix 23).
Identisches Verhalten für `vision_fallback_models`.

## Pro Task ein anderes Model setzen

Beispiel — Lyrics-Generation auf Claude-3.5-Sonnet via OpenRouter,
Video-Prompts auf Llama-3.1-70B, Vision auf Qwen3.5-Flash:

```jsonc
"prompt_enhancer": {
  "audio_llm_model": "anthropic/claude-3.5-sonnet",
  "openrouter_model": "meta-llama/llama-3.1-70b-instruct",
  "vision_model": "qwen/qwen3.5-flash-02-23",
  "genre_validation_model": "qwen/qwen3.5-9b",
  "openrouter_api_key": "sk-or-v1-..."
}
```

Hot-Reload: API (`hermes-ctb-api-1` Container) liest `config.json` beim
Lifespan-Startup. Nach Edit Container neustarten oder uvicorn-reload
triggern (z.B. Touch einer Python-Datei).

## Vision vs Text Endpoints

- **Vision-Model** wird ausschließlich genutzt wenn ein Image als Input
  übergeben wird (Lastframe-Continuity-Analyse, Source-Image-Analyse im
  User-Enhance-Endpoint, Frame-Validierung). Erfordert multi-modal-capable
  Modell (Qwen-VL, Claude-3 Opus/Sonnet, GPT-4o, etc).
- **Text-Model** wird für ALLE Text-Generierung benutzt — inklusive der
  MCA-Frame-Variant-Prompts (`frame_variant_prompt`). MCA bekommt einen
  Text-Prompt, kein Bild → text_model (`openrouter_model`), nicht vision.

Setze `vision_model` NUR auf ein vision-fähiges Model. Sonst schlagen
Frame-Variant-Calls fehl mit 400 (unsupported_modality).

## Quelle / Lade-Pfad

`config_loader.py` liest `config.json` beim Start. `CONFIG["prompt_enhancer"]`
wird an `MusicVideoPrompter` (api.py:165), `AudioEnhancer` und
`PromptEnhancer` weitergegeben. Diese Klassen nehmen die jeweiligen Keys
und initialisieren ihre eigenen OpenRouter-Clients.

## Kein lokales Ollama / kein anderer Provider

Aktuell ausschließlich OpenRouter. Wenn lokales Ollama oder anderer
Provider gewünscht: `openrouter_base_url` ist konfigurierbar, ABER alle
Clients erwarten OpenAI-kompatibles `/v1/chat/completions` Format.
Ollama unterstützt das via `/v1/chat/completions` Endpoint —
`base_url = "http://localhost:11434/v1"` setzen.
