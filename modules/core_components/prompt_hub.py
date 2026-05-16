"""Shared prompt utilities for cross-tab routing and LLM generation."""

import json
import re
import time
from pathlib import Path

import requests

PROMPTS_FILE = Path(__file__).parent.parent.parent / "prompts.json"

# Category keys used when storing prompts on disk, organised by prompt type.
# Voice Clone and Voice Presets share the same 'prompt' bucket so a saved
# prompt is reusable across both tools.
PROMPT_CATEGORIES = {
    "prompt":       "Prompt",
    "conversation": "Conversation",
    "voice_design":  "Voice Design",
    "sfx":          "SFX Design",
    "custom":       "Custom",
}

# Maps each tool's component prefix to the category it reads/writes.
TOOL_CATEGORY_MAP = {
    "vc":   "prompt",
    "conv": "conversation",
    "vd":   "voice_design",
    "sfx":  "sfx",
}

# Maps each PROMPT_TARGETS key to the prompt category it uses.
# '_custom' is a local sentinel used by the Prompt Manager for the Custom entry.
TARGET_CATEGORY_MAP = {
    "voice_clone.text":             "prompt",
    "conversation.script":          "conversation",
    "voice_design.reference":       "prompt",
    "voice_design.instructions":    "voice_design",
    "sfx.prompt":                   "sfx",
    "_custom":                      "custom",
}


def get_category_for_target(target_id):
    """Return the prompt category key for a given target ID."""
    return TARGET_CATEGORY_MAP.get(target_id, "custom")


DEFAULT_OPENAI_ENDPOINT = "https://api.openai.com/v1"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/v1"

# ============================================================================
# System Prompts
# ============================================================================

# These are the default system prompts for each preset option. They can be overridden by system_prompts.json
SYSTEM_PROMPTS = {
    "Voice Clone": (
        "You are a voice performance writer for DramaBox text-to-audio synthesis. "
        "Given a concept, character, or scene, write a compact audio prompt that describes: "
        "the speaker's voice (age, timbre, pitch, tone, accent, character archetype), "
        "the acoustic environment or scene if relevant, "
        "and the emotional delivery arc from start to finish. "
        "Place all spoken words inside double quotes exactly as they should be heard — "
        "performance direction goes around the quotes, never inside them. "
        "Use natural time-ordered language (then, a beat, as) to sequence emotional shifts. "
        "Aim for 3 to 6 sentences. End at the final closing quote — no trailing description after it. "
        "Output one compact paragraph with no headers, labels, or markdown."
    ),
    "Conversation": (
        "You are a voice performance writer for DramaBox text-to-audio synthesis. "
        "Given a topic, scenario, or conflict, write a multi-speaker exchange where each line is a self-contained DramaBox audio prompt. "
        "Each line must appear on its own line in this exact format: [n]: <audio prompt> — where n is the speaker number. "
        "Each audio prompt describes the speaker's voice (age, timbre, tone, accent), their emotional state and delivery, "
        "then places the spoken words inside double quotes. Performance direction goes around the quotes, never inside them. "
        "Give each speaker a distinct voice character that stays consistent across their lines. "
        "Vary emotional energy and pacing between speakers — not every line is the same intensity. "
        "End each line at its final closing quote — no trailing description after the spoken words. "
        "Example format:\n"
        "[1]: A woman in her early 30s, sharp and direct, barely looking up from her screen: \"Did you hear about the new project?\"\n"
        "[2]: A man in his mid 40s, measured and calm, sets down his coffee: \"Just got the email this morning.\"\n"
        "[1]: Her tone shifts — genuinely curious now, leaning forward slightly: \"What do you think about the timeline?\"\n"
        "Output only the lines in [n]: format — no narration, no headers, no extra formatting."
    ),
    "Voice Design (Simple)": (
        "You are a voice styling assistant for text-to-speech generation. "
        "Convert user intent into concise style instructions focused on delivery only: tone, pacing, energy, "
        "emotion, clarity, accent, and intensity. "
        "Do not write dialogue or script content. Do not explain. "
        "Output only the final style instruction text."
    ),
    "Voice Design (Json)": (
        "You are a voice design assistant for text-to-speech voice creation. "
        "Convert user intent into a structured voice design instruction. "
        "Do not write dialogue or script content. Do not explain. "
        "Output only one object using this exact key structure and key order:\n"
        "{\n"
        "  'label': '',\n"
        "  'role_type': '<e.g., Narrator, Podcast Host, Customer Agent, Actor, News Anchor, Teacher>',\n"
        "  'attributes': {\n"
        "    'age_range': '<e.g., young adult, middle-aged, elderly>',\n"
        "    'gender_presentation': '<male|female|nonbinary|androgynous|unspecified>',\n"
        "    'timbre': '',\n"
        "    'pitch': '<low|mid|high and numeric approximate in Hz if known, optional>',\n"
        "    'pitch_range': '<narrow|medium|wide>',\n"
        "    'speaking_rate': '<slow|medium|fast and approximate words/min if known, optional>',\n"
        "    'breathiness': '<none|light|moderate|heavy>',\n"
        "    'nasality': '<none|low|moderate|high>',\n"
        "    'articulation': '<clear|soft-mumbled|crisp|rounded>',\n"
        "    'warmth': '<cool|neutral|warm|very warm>',\n"
        "    'emotional_baseline': '<neutral|warm|authoritative|wry|cheerful|calm|serious|friendly etc.>',\n"
        "    'accent_locale': '<e.g., General American, RP British, Southern US, Australian, neutral European Spanish>',\n"
        "    'prosody_notes': '<short notes about intonation, stress, pausing>',\n"
        "    'phonetic_cues': ['<1-3 specific phonetic cues, e.g., elongated vowels, crisp plosives, soft sibilants>'],\n"
        "  }\n"
        "}"
    ),
    "SFX": (
        "You are a sound design prompt writer. The user will give you a short idea or concept. "
        "Your job is to expand it into a detailed, evocative description of a sound or soundscape. "
        "Focus on texture, layers, timing, spatial qualities, and acoustic characteristics. "
        "Describe what the listener should hear, not see. Output ONLY the final sound description."
    ),
}

SYSTEM_PROMPT_CHOICES = list(SYSTEM_PROMPTS.keys()) + ["Custom"]

# ============================================================================
# Prompt Targets - maps target IDs to tool tabs and component keys
# ============================================================================

PROMPT_TARGETS = {
    "voice_clone.text": {
        "label": "Voice Clone: Prompt",
        "tool": "Voice Clone",
        "tab_id": "tab_voice_clone",
        "component_key": "text_input",
        "default_system_preset": "Voice Clone",
        "template": (
            "Write final spoken text for voice generation. "
            "Return only the exact text to speak, no extra labels or notes.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_design.reference": {
        "label": "Voice Design: Prompt",
        "tool": "Voice Design",
        "tab_id": "tab_voice_design",
        "component_key": "design_text_input",
        "default_system_preset": "TTS / Voice",
        "template": (
            "Write final spoken reference text for a voice design sample. "
            "Return only the text to be spoken.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "voice_design.instructions": {
        "label": "Voice Design: Description",
        "tool": "Voice Design",
        "tab_id": "tab_voice_design",
        "component_key": "design_instruct_input",
        "default_system_preset": "Voice Design",
        "template": (
            "Write voice design instructions as one structured object. "
            "Fill all fields with specific values when possible. "
            "Return only the object, no markdown.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "conversation.script": {
        "label": "Conversation: Prompt",
        "tool": "Conversation",
        "tab_id": "tab_conversation",
        "component_key": "conversation_script",
        "default_system_preset": "Conversation",
        "template": (
            "Create a conversation script strictly in [n]: (emotion) format. "
            "No narration, no stage directions, no markdown.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
    "sfx.prompt": {
        "label": "Sound FX: Prompt",
        "tool": "Sound Effects",
        "tab_id": "tab_sound_effects",
        "component_key": "sfx_prompt",
        "default_system_preset": "SFX",
        "template": (
            "Write one high-quality sound design prompt for audio generation. "
            "Return only the final prompt text.\n\n"
            "User instruction:\n{instruction}"
        ),
    },
}


# ============================================================================
# Prompt Loader UI helpers
# ============================================================================

def create_prompt_loader(prefix, title="Saved Prompts"):
    """Create a compact 'Saved Prompts' accordion for any generation tab.

    Args:
        prefix: Unique key prefix for components (e.g. 'vc', 'vp', 'vd', etc.)
        title: Accordion label shown to the user

    Returns:
        dict with keys: {prefix}_pl_accordion, dropdown, preview,
                        load_replace_btn, load_append_btn
    """
    import gradio as gr

    components = {}
    with gr.Accordion(title, open=False) as acc:
        components[f'{prefix}_pl_accordion'] = acc
        category = TOOL_CATEGORY_MAP.get(prefix, "custom")
        components[f'{prefix}_pl_dropdown'] = gr.Dropdown(
            label="Saved Prompt",
            choices=get_prompt_names(category),
            value=None,
            interactive=True,
        )
        components[f'{prefix}_pl_preview'] = gr.Textbox(
            label="Preview",
            interactive=False,
            lines=3,
            max_lines=6,
        )
        with gr.Row():
            components[f'{prefix}_pl_load_replace_btn'] = gr.Button(
                "Load (Replace)",
                variant="primary",
                size="sm",
            )
            components[f'{prefix}_pl_load_append_btn'] = gr.Button(
                "Load (Append)",
                size="sm",
            )
    return components


def wire_prompt_loader(components, prefix, target_components):
    """Wire up events for a prompt loader created with create_prompt_loader.

    Args:
        components: The components dict returned by create_prompt_loader (and
                    merged into the tool's main components dict).
        prefix: The same prefix string passed to create_prompt_loader.
        target_components: Dict mapping target_id -> gr.Textbox component.
                           If there is only one target, use {'_default': component}.
    """
    import gradio as gr
    accordion = components[f'{prefix}_pl_accordion']
    dd = components[f'{prefix}_pl_dropdown']
    preview = components[f'{prefix}_pl_preview']
    load_replace_btn = components[f'{prefix}_pl_load_replace_btn']
    load_append_btn = components[f'{prefix}_pl_load_append_btn']

    # Flatten targets into a single ordered list for "load" operations.
    # When there is more than one target the first one is the primary target.
    target_list = list(target_components.values())
    primary_target = target_list[0]

    category = TOOL_CATEGORY_MAP.get(prefix, "custom")

    def _get_preview(name):
        return get_prompt_text(name, category) if name else ""

    def _refresh():
        names = get_prompt_names(category)
        return gr.update(choices=names, value=None), ""

    def _load_replace(name, *current_values):
        text = get_prompt_text(name, category) if name else ""
        updates = [gr.update(value=text)] + [gr.update() for _ in target_list[1:]]
        return updates if len(updates) > 1 else updates[0]

    def _load_append(name, current_val):
        text = get_prompt_text(name, category) if name else ""
        if not text:
            return gr.update()
        sep = "\n" if current_val and not current_val.endswith("\n") else ""
        return gr.update(value=current_val + sep + text)

    dd.change(_get_preview, inputs=[dd], outputs=[preview])
    accordion.expand(_refresh, outputs=[dd, preview])

    load_replace_btn.click(
        _load_replace,
        inputs=[dd] + [primary_target],
        outputs=[primary_target],
    )
    load_append_btn.click(
        _load_append,
        inputs=[dd, primary_target],
        outputs=[primary_target],
    )


# ============================================================================
# Prompt File Management
# ============================================================================

def _load_raw_prompts():
    """Load the raw prompts dict from disk, migrating legacy flat format if needed.

    New format  : {category_key: {name: text}}
    Legacy format: {name: text}  ->  automatically migrated to {"custom": {name: text}}
    """
    if PROMPTS_FILE.exists():
        try:
            with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            data = {}
    else:
        data = {}

    if not isinstance(data, dict) or not data:
        return {}

    # Detect legacy format: any top-level value is a plain string
    is_legacy = any(isinstance(v, str) for v in data.values())
    if is_legacy:
        migrated = {"custom": {str(k): str(v) for k, v in data.items() if isinstance(v, str)}}
        try:
            with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
                json.dump(migrated, f, indent=2, ensure_ascii=False)
            print("[Prompt Hub] Migrated legacy prompts.json -> category format (entries placed under 'custom').")
        except IOError:
            pass
        return migrated

    # New format: keep only valid {str: {str: str}} entries
    result = {}
    for cat, entries in data.items():
        if isinstance(entries, dict):
            result[str(cat)] = {str(k): str(v) for k, v in entries.items()}
    return result


def load_prompts(category=None):
    """Load prompts, optionally filtered to one category.

    Args:
        category: Category key such as 'prompt', 'custom', etc.
                  Pass None to get a merged flat dict across all categories.

    Returns:
        Dict of {name: text}
    """
    all_data = _load_raw_prompts()
    if category is not None:
        return all_data.get(str(category), {})
    merged = {}
    for entries in all_data.values():
        merged.update(entries)
    return merged


def save_prompts(prompts, category="custom"):
    """Save prompts for a category, sorted alphabetically.

    Args:
        prompts: Dict of {name: text} for this category.
        category: Category key to save under (default: 'custom').

    Returns:
        Sorted prompts dict that was written.
    """
    all_data = _load_raw_prompts()
    sorted_entries = dict(sorted(prompts.items(), key=lambda x: x[0].lower()))
    all_data[str(category)] = sorted_entries
    with open(PROMPTS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    return sorted_entries


def get_prompt_names(category=None):
    """Return sorted list of prompt names, optionally filtered by category."""
    return sorted(load_prompts(category).keys(), key=str.lower)


def get_prompt_text(name, category=None):
    """Get saved prompt text by name.

    Args:
        name: The prompt name.
        category: Category prefix to search in. None searches all categories.
    """
    if not name:
        return ""
    return load_prompts(category).get(name, "")


# ============================================================================
# Target Utilities
# ============================================================================

def get_target_config(target_id):
    """Return target config dict or None."""
    return PROMPT_TARGETS.get(target_id)


def get_target_default_preset(target_id):
    """Return default system preset name for a target."""
    cfg = get_target_config(target_id)
    if not cfg:
        return SYSTEM_PROMPT_CHOICES[0]
    return cfg.get("default_system_preset", SYSTEM_PROMPT_CHOICES[0])


def get_target_tab_id(target_id):
    """Return destination tab id for a target."""
    cfg = get_target_config(target_id)
    return cfg.get("tab_id", "") if cfg else ""


def get_enabled_target_choices(user_config, target_ids=None):
    """Return enabled prompt target choices as list of (label, id) tuples."""
    enabled_tools = user_config.get("enabled_tools", {}) if isinstance(user_config, dict) else {}
    choices = []

    ids = target_ids if target_ids is not None else list(PROMPT_TARGETS.keys())
    for target_id in ids:
        cfg = PROMPT_TARGETS.get(target_id)
        if not cfg:
            continue
        tool_name = cfg.get("tool", "")
        if enabled_tools.get(tool_name, True):
            choices.append((cfg.get("label", target_id), target_id))

    return choices


# ============================================================================
# Cross-Tab Routing
# ============================================================================

def build_apply_payload(target_id, mode, text, source="prompt_manager"):
    """Create serialized prompt-apply payload with unique nonce."""
    payload = {
        "target_id": target_id,
        "mode": mode,
        "text": text,
        "source": source,
        "nonce": time.time_ns(),
    }
    return json.dumps(payload, ensure_ascii=False)


def parse_apply_payload(raw_value):
    """Parse and validate serialized apply payload.

    Returns dict with target_id, mode, text, source keys or None if invalid.
    """
    if not raw_value or not str(raw_value).strip():
        return None

    try:
        payload = json.loads(raw_value)
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None

    target_id = str(payload.get("target_id", "")).strip()
    mode = str(payload.get("mode", "")).strip().lower()
    text = str(payload.get("text", ""))
    source = str(payload.get("source", "")).strip() or "unknown"

    if target_id not in PROMPT_TARGETS:
        return None
    if mode not in {"replace", "append"}:
        return None

    return {
        "target_id": target_id,
        "mode": mode,
        "text": text,
        "source": source,
    }


def merge_text(existing, new_text, mode):
    """Merge text using replace/append semantics."""
    old = existing or ""
    new = new_text or ""
    if mode == "append":
        if not old.strip():
            return new
        if not new.strip():
            return old
        return f"{old.rstrip()}\n{new.lstrip()}"
    return new


# ============================================================================
# HTTP / Endpoint Utilities
# ============================================================================

def normalize_v1_base_url(url, fallback=None):
    """Normalize endpoint base URL to include /v1 exactly once."""
    if fallback is None:
        fallback = DEFAULT_OPENAI_ENDPOINT
    base = (url or "").strip()
    if not base:
        base = fallback

    base = base.rstrip("/")
    for suffix in ("/chat/completions", "/completions"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break

    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def get_ollama_tags_url(ollama_v1_url):
    """Convert Ollama /v1 URL to /api/tags URL."""
    v1 = normalize_v1_base_url(ollama_v1_url, fallback=DEFAULT_OLLAMA_ENDPOINT)
    root = v1[:-3] if v1.endswith("/v1") else v1
    return f"{root}/api/tags"


def get_effective_base_url(use_local_ollama, endpoint_url, user_config):
    """Resolve active provider base URL from config and arguments."""
    if use_local_ollama:
        return normalize_v1_base_url(
            str(user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT)),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )
    return normalize_v1_base_url(
        endpoint_url or str(user_config.get("llm_endpoint_url", DEFAULT_OPENAI_ENDPOINT)),
        fallback=DEFAULT_OPENAI_ENDPOINT,
    )


def build_headers(user_config, use_local_ollama):
    """Build HTTP headers for endpoint calls."""
    headers = {"Content-Type": "application/json"}
    api_key = str(user_config.get("llm_api_key", "")).strip()
    if api_key and not use_local_ollama:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_error_message(response):
    """Extract error message text from an endpoint error response."""
    try:
        payload = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return text[:300] if text else f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("error")
            if msg:
                return str(msg)
        if isinstance(err, str):
            return err
        msg = payload.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()

    return f"HTTP {response.status_code}"


def format_http_error(response, base_url):
    """Map common HTTP statuses to actionable messages."""
    detail = _extract_error_message(response)
    if response.status_code in (401, 403):
        return (
            "Authentication failed (401/403). "
            "Check API key in Settings > LLM Endpoint. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 404:
        return (
            "Endpoint or model not found (404). "
            "Check endpoint URL and model name. "
            f"Endpoint: {base_url}\nDetails: {detail}"
        )
    if response.status_code == 400:
        return f"Invalid request (400): {detail}"
    return f"Request failed ({response.status_code}): {detail}"


# ============================================================================
# Model Discovery
# ============================================================================

def _parse_openai_model_ids(payload):
    """Extract model IDs from OpenAI-compatible /models response."""
    models = []
    if isinstance(payload, dict):
        for item in payload.get("data", []):
            if isinstance(item, dict):
                model_id = item.get("id")
                if isinstance(model_id, str) and model_id.strip():
                    models.append(model_id.strip())
    return sorted(set(models), key=lambda x: x.lower())


def discover_available_models(user_config, use_local_ollama, endpoint_url):
    """Discover models from provider.

    Returns (list_of_model_names, status_message).
    """
    timeout = 12

    if use_local_ollama:
        ollama_v1 = normalize_v1_base_url(
            str(user_config.get("llm_ollama_url", DEFAULT_OLLAMA_ENDPOINT)),
            fallback=DEFAULT_OLLAMA_ENDPOINT,
        )

        tags_url = get_ollama_tags_url(ollama_v1)
        try:
            response = requests.get(tags_url, timeout=timeout)
            if response.status_code == 200:
                payload = response.json()
                names = []
                if isinstance(payload, dict):
                    for item in payload.get("models", []):
                        if isinstance(item, dict):
                            name = item.get("model") or item.get("name")
                            if isinstance(name, str) and name.strip():
                                names.append(name.strip())
                if names:
                    unique = sorted(set(names), key=lambda x: x.lower())
                    return unique, f"Found {len(unique)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], f"Could not connect to local Ollama at {ollama_v1}. Start `ollama serve` and retry."
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception:
            pass

        try:
            response = requests.get(f"{ollama_v1}/models", timeout=timeout)
            if response.status_code != 200:
                return [], format_http_error(response, ollama_v1)
            models = _parse_openai_model_ids(response.json())
            if not models:
                return [], "Ollama responded but no models were found. Pull one first, e.g. `ollama pull llama3.1`."
            return models, f"Found {len(models)} local Ollama model(s)."
        except requests.exceptions.ConnectionError:
            return [], f"Could not connect to local Ollama at {ollama_v1}. Start `ollama serve` and retry."
        except requests.exceptions.Timeout:
            return [], f"Timed out while querying local Ollama at {ollama_v1}."
        except Exception as e:
            return [], f"Failed to query local Ollama models: {e}"

    base_url = get_effective_base_url(False, endpoint_url, user_config)
    headers = build_headers(user_config, use_local_ollama=False)
    try:
        response = requests.get(f"{base_url}/models", headers=headers, timeout=timeout)
        if response.status_code != 200:
            return [], format_http_error(response, base_url)
        models = _parse_openai_model_ids(response.json())
        if not models:
            return [], "No models returned by /models. You can still type the model name manually."
        return models, f"Found {len(models)} model(s) from endpoint."
    except requests.exceptions.ConnectionError:
        return [], f"Could not connect to endpoint: {base_url}."
    except requests.exceptions.Timeout:
        return [], f"Timed out while querying models from {base_url}."
    except Exception as e:
        return [], f"Failed to query models: {e}"


def resolve_system_prompt(preset_name, user_config, custom_text=None):
    """Resolve final system prompt text, substituting dynamic placeholders.
    """
    if preset_name == "Custom":
        return custom_text or ""

    return SYSTEM_PROMPTS.get(preset_name, "")
