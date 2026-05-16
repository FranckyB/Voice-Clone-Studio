"""
Prompt Manager Tool

Save, browse, and generate text prompts for TTS and sound effects.
Includes an LLM-powered prompt generator using llama.cpp.
Prompts are stored in a standalone prompts.json at the project root.
"""

import json
import os
import sys
import subprocess
import signal
import atexit
import time
import ctypes
import random
from datetime import datetime
import requests
import gradio as gr
from pathlib import Path

from modules.core_components.tool_base import Tool, ToolConfig
import modules.core_components.prompt_hub as prompt_hub

# ============================================================================
# Prompts file (standalone, same level as config.json)
# ============================================================================
PROMPTS_FILE = Path(__file__).parent.parent.parent.parent / "prompts.json"

# ============================================================================
# LLM Models available for download
# ============================================================================
LLM_MODELS = {
    "Qwen3-4B-Q8_0.gguf": {
        "repo": "Qwen/Qwen3-4B-GGUF",
    },
    "Qwen3.5-9B-UD-Q4_K_XL.gguf": {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
    },
    "Qwen3.5-9B-Q8_0.gguf": {
        "repo": "unsloth/Qwen3.5-9B-GGUF",
    },
}

# System prompt presets — loaded from system_prompts.json
SYSTEM_PROMPTS_FILE = Path(__file__).parent.parent / "system_prompts.json"


def _load_system_prompts():
    """Load system prompts from system_prompts.json.

    Supports two formats:
    - String values: used as-is
    - Array of strings: joined with spaces (empty strings become newlines)

    Returns:
        Dictionary of preset name -> system prompt text.
    """
    if SYSTEM_PROMPTS_FILE.exists():
        try:
            with open(SYSTEM_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                result = {}
                for key, value in data.items():
                    if isinstance(value, list):
                        # Join array lines: empty strings become newlines
                        parts = []
                        for line in value:
                            if line == "":
                                parts.append("\n")
                            else:
                                parts.append(line)
                        result[key] = " ".join(parts).replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
                    else:
                        result[key] = value
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"[Prompt Manager] Error loading system_prompts.json: {e}")
    return {}


def _get_merged_system_prompts():
    """Merge prompt_hub target-aware presets with user's system_prompts.json.

    Hub presets are the defaults; user's JSON adds/overrides on top.
    """
    merged = dict(prompt_hub.SYSTEM_PROMPTS)
    merged.update(_load_system_prompts())
    return merged


SYSTEM_PROMPTS = _get_merged_system_prompts()
SYSTEM_PROMPT_CHOICES = list(SYSTEM_PROMPTS.keys()) + ["Custom"]

# ============================================================================
# llama.cpp server management
# ============================================================================
_server_process = None
_current_model = None
_job_handle = None
SERVER_PORT = 8099  # Use a different port to avoid conflicts


def _setup_windows_job_object():
    """Create a Windows Job Object that kills child processes when parent exits."""
    global _job_handle
    if os.name != 'nt' or _job_handle:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
        JobObjectExtendedLimitInformation = 9

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_longlong),
                ("PerJobUserTimeLimit", ctypes.c_longlong),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_ulonglong),
                ("WriteOperationCount", ctypes.c_ulonglong),
                ("OtherOperationCount", ctypes.c_ulonglong),
                ("ReadTransferCount", ctypes.c_ulonglong),
                ("WriteTransferCount", ctypes.c_ulonglong),
                ("OtherTransferCount", ctypes.c_ulonglong),
            ]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ("IoInfo", IO_COUNTERS),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            raise ctypes.WinError(ctypes.get_last_error())

        extended_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        extended_info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

        if not kernel32.SetInformationJobObject(
            job, JobObjectExtendedLimitInformation,
            ctypes.byref(extended_info), ctypes.sizeof(extended_info)
        ):
            kernel32.CloseHandle(job)
            raise ctypes.WinError(ctypes.get_last_error())

        _job_handle = job
    except Exception:
        pass


def _assign_process_to_job(pid):
    """Assign subprocess pid to job object so it gets killed when parent exits."""
    global _job_handle
    if os.name != 'nt' or not _job_handle or not pid:
        return
    try:
        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        PROCESS_ALL_ACCESS = 0x1F0FFF
        proc_handle = kernel32.OpenProcess(PROCESS_ALL_ACCESS, False, int(pid))
        if not proc_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if not kernel32.AssignProcessToJobObject(_job_handle, proc_handle):
            kernel32.CloseHandle(proc_handle)
            raise ctypes.WinError(ctypes.get_last_error())
        kernel32.CloseHandle(proc_handle)
    except Exception:
        pass


def _cleanup_server():
    """Cleanup function to stop server on exit."""
    global _server_process, _job_handle
    if _server_process:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        except Exception:
            try:
                _server_process.kill()
            except Exception:
                pass
        _server_process = None

    if os.name == 'nt' and _job_handle:
        try:
            ctypes.WinDLL('kernel32').CloseHandle(_job_handle)
        except Exception:
            pass
        _job_handle = None


# Register cleanup
atexit.register(_cleanup_server)


def _is_server_alive():
    """Check if llama.cpp server is responding."""
    try:
        r = requests.get(f"http://localhost:{SERVER_PORT}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def _get_default_context_size():
    """Return a context size scaled to available GPU VRAM.

    Tiers:
        >= 24 GB  ->  8192
        >= 16 GB  ->  4096
         < 16 GB  ->  2048
        No GPU   ->  4096
    """
    try:
        import torch
        if torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_vram_gb >= 24:
                return 8192
            elif total_vram_gb >= 16:
                return 4096
            else:
                return 2048
    except Exception:
        pass
    return 4096


def _kill_llama_on_port():
    """Kill any process listening on our server port (safety net for orphans)."""
    try:
        if os.name == 'nt':
            # Parse netstat for PIDs on our port
            result = subprocess.run(
                ["netstat", "-aon"],
                capture_output=True, text=True, timeout=10
            )
            pids = set()
            for line in result.stdout.splitlines():
                if f":{SERVER_PORT}" in line and "LISTENING" in line:
                    parts = line.split()
                    if parts:
                        try:
                            pid = int(parts[-1])
                            if pid > 0:
                                pids.add(pid)
                        except ValueError:
                            pass
            for pid in pids:
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True, timeout=5
                    )
                except Exception:
                    pass
        else:
            subprocess.run(
                f"lsof -ti:{SERVER_PORT} | xargs -r kill -9",
                shell=True, capture_output=True, timeout=10
            )
    except Exception:
        pass


def _get_llama_models_dir(user_config):
    """Get the path to the default models/llama/ directory."""
    project_root = PROMPTS_FILE.parent
    models_base = project_root / user_config.get("models_folder", "models")
    llama_dir = models_base / "llama"
    llama_dir.mkdir(parents=True, exist_ok=True)
    return llama_dir


def _get_user_llama_models_dir(user_config):
    """Get the user-specified llama models directory, if configured.

    Returns:
        Path or None if not configured / empty.
    """
    user_path = user_config.get("llama_models_path", "").strip()
    if not user_path:
        return None
    p = Path(user_path)
    if p.exists() and p.is_dir():
        return p
    # Try creating it
    try:
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        return None


def _get_local_gguf_models(user_config):
    """Get list of local .gguf model filenames from all search directories."""
    seen = set()
    models = []

    # 1. Default models/llama/ folder
    llama_dir = _get_llama_models_dir(user_config)
    if llama_dir.exists():
        for f in sorted(llama_dir.glob("*.gguf")):
            if f.name not in seen:
                seen.add(f.name)
                models.append(f.name)

    # 2. User-specified models folder
    user_dir = _get_user_llama_models_dir(user_config)
    if user_dir and user_dir.exists():
        for f in sorted(user_dir.glob("*.gguf")):
            if f.name not in seen:
                seen.add(f.name)
                models.append(f.name)

    return models


def _get_all_model_choices(user_config):
    """Get combined list: local models first, then downloadable ones not yet local."""
    local = _get_local_gguf_models(user_config)
    all_choices = list(local)
    for name in LLM_MODELS:
        if name not in all_choices:
            all_choices.append(name)
    return all_choices


def _find_model_path(model_name, user_config):
    """Find the full path to a local model, searching all directories.

    Returns:
        Path string or None if not found.
    """
    # Check default models/llama/ first
    llama_dir = _get_llama_models_dir(user_config)
    default_path = llama_dir / model_name
    if default_path.exists():
        return str(default_path)

    # Check user-specified directory
    user_dir = _get_user_llama_models_dir(user_config)
    if user_dir:
        user_path = user_dir / model_name
        if user_path.exists():
            return str(user_path)

    return None


def _is_model_local(model_name, user_config):
    """Check if a model exists locally in any search directory."""
    return _find_model_path(model_name, user_config) is not None


def _download_model(model_name, user_config, progress=None):
    """Download a model from HuggingFace.

    Args:
        model_name: Filename of the model
        user_config: User config dict
        progress: Optional Gradio progress callback

    Returns:
        Path to downloaded model or None on error
    """
    if model_name not in LLM_MODELS:
        return None

    model_info = LLM_MODELS[model_name]
    repo_id = model_info["repo"]

    # Download to user-specified directory if configured, otherwise default
    user_dir = _get_user_llama_models_dir(user_config)
    download_dir = user_dir if user_dir else _get_llama_models_dir(user_config)
    local_path = download_dir / model_name

    if local_path.exists():
        return str(local_path)

    try:
        from huggingface_hub import HfApi
        print(f"[Prompt Manager] Downloading {model_name} from {repo_id}...")
        if progress:
            progress(0.0, desc=f"Downloading {model_name}...")

        api = HfApi()
        repo_info = api.repo_info(repo_id=repo_id, files_metadata=True)
        file_metadata = next((f for f in repo_info.siblings if f.rfilename == model_name), None)
        if not file_metadata or file_metadata.size is None:
            print(f"[Prompt Manager] Could not get file metadata for {model_name}")
            return None

        total_size = file_metadata.size
        total_mb = total_size / (1024 * 1024)
        print(f"[Prompt Manager] Model size: {total_mb:.0f} MB")
        download_url = f"https://huggingface.co/{repo_id}/resolve/main/{model_name}"

        downloaded = 0
        last_logged_pct = -10
        with requests.get(download_url, stream=True) as r, open(str(local_path), "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        pct = downloaded / total_size
                        dl_mb = downloaded / (1024 * 1024)
                        # Update UI progress
                        if progress:
                            progress(pct, desc=f"Downloading {model_name}... {dl_mb:.0f}MB / {total_mb:.0f}MB")
                        # Log to console every 10%
                        pct_int = int(pct * 100)
                        if pct_int >= last_logged_pct + 10:
                            last_logged_pct = pct_int
                            print(f"[Prompt Manager] Download: {pct_int}% ({dl_mb:.0f}MB / {total_mb:.0f}MB)")

        print(f"[Prompt Manager] Download complete: {local_path}")
        if progress:
            progress(1.0, desc="Download complete")
        return str(local_path)
    except Exception as e:
        # Clean up partial download
        if local_path.exists():
            try:
                local_path.unlink()
            except Exception:
                pass
        print(f"[Prompt Manager] Error downloading {model_name}: {e}")
        return None


def _start_server(model_name, user_config, progress=None):
    """Start llama.cpp server with specified model.

    Returns:
        Tuple: (success, error_message)
    """
    global _server_process, _current_model

    # If already running with same model, reuse
    if _server_process and _current_model == model_name and _is_server_alive():
        return True, None

    # Stop any existing server
    _stop_server()

    # Unload all AI models to free VRAM before starting llama.cpp
    try:
        from modules.core_components.ai_models.tts_manager import get_tts_manager
        from modules.core_components.ai_models.asr_manager import get_asr_manager
        from modules.core_components.ai_models.foley_manager import get_foley_manager

        tts = get_tts_manager()
        if tts:
            tts.unload_all()
        asr = get_asr_manager()
        if asr:
            asr.unload_all()
        foley = get_foley_manager()
        if foley:
            foley.unload_all()
    except Exception:
        pass

    # Download if not local
    needs_download = not _is_model_local(model_name, user_config)
    if needs_download:
        if model_name in LLM_MODELS:
            if progress:
                progress(0.05, desc=f"Downloading {model_name}...")
            print(f"[Prompt Manager] Model not found locally, downloading {model_name}...")
            model_path = _download_model(model_name, user_config, progress)
            if not model_path:
                return False, f"Failed to download model: {model_name}"
        else:
            return False, f"Model not found: {model_name}"
    else:
        model_path = _find_model_path(model_name, user_config)

    if not os.path.exists(model_path):
        return False, f"Model file not found: {model_path}"

    try:
        if progress:
            progress(0.3, desc="Starting llama.cpp server...")

        # Determine executable — use custom path from config if set
        custom_llama_path = user_config.get("llama_cpp_path", "").strip()
        if os.name == 'nt':
            if custom_llama_path:
                server_cmd = str(Path(custom_llama_path) / "llama-server.exe")
            else:
                server_cmd = "llama-server.exe"
            creation_flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            if custom_llama_path:
                server_cmd = str(Path(custom_llama_path) / "llama-server")
            else:
                server_cmd = "llama-server"
            creation_flags = 0

        context_size = _get_default_context_size()
        cmd_args = [
            server_cmd, "-m", model_path,
            "--port", str(SERVER_PORT),
            "--no-warmup",
            "-ngl", "100",
            "-c", str(context_size)
        ]

        # Assign specific GPU if configured
        llama_gpu = int(user_config.get("llama_gpu", 0))
        if llama_gpu > 0:
            cmd_args.extend(["--main-gpu", str(llama_gpu)])

        popen_kwargs = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        if os.name == 'nt':
            popen_kwargs["creationflags"] = creation_flags
        else:
            # On Unix, set PR_SET_PDEATHSIG so child gets SIGTERM when parent dies
            def _set_pdeathsig():
                try:
                    for libname in ("libc.so.6", "libc.dylib", "libc.so"):
                        try:
                            libc = ctypes.CDLL(libname)
                            break
                        except Exception:
                            libc = None
                    if not libc:
                        return
                    PR_SET_PDEATHSIG = 1
                    libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
                except Exception:
                    return
            popen_kwargs["preexec_fn"] = _set_pdeathsig

        _server_process = subprocess.Popen(cmd_args, **popen_kwargs)

        # On Windows, attach to job object
        if os.name == 'nt':
            _setup_windows_job_object()
            _assign_process_to_job(_server_process.pid)

        _current_model = model_name

        # Wait for server to be ready (up to 60 seconds for large models)
        for i in range(60):
            time.sleep(1)
            if progress:
                progress(0.3 + (i / 60) * 0.3, desc=f"Waiting for server... ({i + 1}s)")
            if _is_server_alive():
                return True, None
            # Check if process crashed
            if _server_process.poll() is not None:
                stderr = _server_process.stderr.read().decode(errors='replace') if _server_process.stderr else ""
                return False, f"Server exited unexpectedly.\n{stderr[:500]}"

        _stop_server()
        return False, "Server did not start in time (60s timeout)"

    except FileNotFoundError:
        custom_hint = ""
        if not user_config.get("llama_cpp_path", "").strip():
            custom_hint = "\nOr set the llama.cpp path in Settings > Folder Paths."
        return False, (
            "llama-server not found. Please install llama.cpp and add to PATH."
            f"{custom_hint}\n"
            "Installation: https://github.com/ggml-org/llama.cpp"
        )
    except Exception as e:
        return False, f"Error starting server: {e}"


def _stop_server():
    """Stop the llama.cpp server and ensure VRAM is freed."""
    global _server_process, _current_model
    was_running = _server_process is not None

    if _server_process:
        try:
            _server_process.terminate()
            _server_process.wait(timeout=5)
        except Exception:
            try:
                _server_process.kill()
                _server_process.wait(timeout=3)
            except Exception:
                pass
        _server_process = None
        _current_model = None

    _kill_llama_on_port()

    if was_running:
        # Give the GPU driver a moment to reclaim VRAM after process death
        time.sleep(0.5)


# Register llama.cpp shutdown as a pre-model-load hook so that loading
# any AI model (TTS, ASR, Foley) will stop the server first to free VRAM.
try:
    from modules.core_components.ai_models.model_utils import register_pre_load_hook
    register_pre_load_hook(_stop_server)
except Exception:
    pass


# ============================================================================
# Prompts file management
# ============================================================================

def _load_prompts(category=None):
    """Load prompts from prompts.json, optionally filtered by category.

    Args:
        category: Category key (e.g. 'prompt', 'custom'). None returns all merged.

    Returns:
        Dictionary of name -> text pairs
    """
    return prompt_hub.load_prompts(category)


def _save_prompts(prompts, category="custom"):
    """Save prompts for a category to prompts.json.

    Args:
        prompts: Dictionary of name -> text pairs
        category: Category prefix to save under

    Returns:
        Sorted prompts dictionary that was written
    """
    return prompt_hub.save_prompts(prompts, category)


def _get_prompt_names(category=None):
    """Get sorted list of prompt names, optionally filtered by category."""
    return prompt_hub.get_prompt_names(category)


# ============================================================================
# Temp prompts management
# ============================================================================
TEMP_PROMPTS_FILE = Path(__file__).parent.parent.parent.parent / "temp" / "temp_prompts.json"


def _clear_temp_prompts():
    """Delete the temp prompts file (called at launch)."""
    if TEMP_PROMPTS_FILE.exists():
        try:
            TEMP_PROMPTS_FILE.unlink()
        except Exception:
            pass


# Clear temp prompts on module load (i.e., every app launch)
_clear_temp_prompts()


def _load_temp_prompts():
    """Load temp prompts from temp_prompts.json.

    Returns:
        Dictionary of name -> {instruction, result, seed, model, temperature,
                               top_k, top_p, min_p, repeat_penalty, preset, timestamp}
    """
    if TEMP_PROMPTS_FILE.exists():
        try:
            with open(TEMP_PROMPTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def _save_temp_prompt(name, entry):
    """Save a single temp prompt entry.

    Args:
        name: Display name for the entry
        entry: Dict with instruction, result, seed, model, etc.
    """
    TEMP_PROMPTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    prompts = _load_temp_prompts()
    prompts[name] = entry
    with open(TEMP_PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)


def _get_temp_prompt_names():
    """Get list of temp prompt names for FileLister (newest first)."""
    prompts = _load_temp_prompts()
    # Sort by timestamp descending (newest first)
    items = sorted(prompts.items(), key=lambda x: x[1].get("timestamp", ""), reverse=True)
    return [name for name, _ in items]


def _make_temp_name(instruction, seed):
    """Generate a display name from the first 8 words + timestamp.

    Args:
        instruction: The user's LLM instruction text
        seed: The seed used for generation

    Returns:
        A short descriptive name like 'dramatic pirate monologue about - 14h32'
    """
    words = instruction.strip().split()[:8]
    short = " ".join(words)
    if len(instruction.strip().split()) > 8:
        short += "..."
    timestamp = datetime.now().strftime("%Hh%M")
    return f"{short} - {timestamp}"


def _format_temp_info(entry):
    """Format a temp prompt entry's metadata for display.

    Args:
        entry: Dict with generation metadata

    Returns:
        Formatted info string
    """
    lines = []
    if "seed" in entry:
        lines.append(f"Seed: {entry['seed']}")
    if "model" in entry:
        lines.append(f"Model: {entry['model']}")
    if "preset" in entry:
        lines.append(f"Preset: {entry['preset']}")
    if "temperature" in entry:
        lines.append(f"Temp: {entry['temperature']}  Top-K: {entry.get('top_k', '')}  Top-P: {entry.get('top_p', '')}")
    if "min_p" in entry:
        lines.append(f"Min-P: {entry['min_p']}  Repeat Penalty: {entry.get('repeat_penalty', '')}")
    if "timestamp" in entry:
        lines.append(f"Generated: {entry['timestamp']}")
    return "\n".join(lines)


# ============================================================================
# Prompt Manager Tool
# ============================================================================

class PromptManagerTool(Tool):
    """Prompt Manager - save, browse, and generate prompts with an LLM."""

    config = ToolConfig(
        name="Prompt Manager",
        module_name="prompt_generator",
        description="Save, browse, and generate text prompts using an LLM",
        enabled=True,
        category="utility"
    )

    @classmethod
    def create_tool(cls, shared_state):
        """Create the Prompt Manager UI."""
        from gradio_filelister import FileLister

        user_config = shared_state.get('_user_config', {})
        llm_backend = user_config.get("llm_backend", "llama.cpp")

        # Get model choices depending on the configured backend
        if llm_backend == "Ollama":
            model_choices, _disc_status = prompt_hub.discover_available_models(
                user_config, use_local_ollama=True, endpoint_url=""
            )
            if not model_choices:
                model_choices = []
        else:
            model_choices = _get_all_model_choices(user_config)

        # Restore saved LLM model selection
        saved_llm_model = user_config.get("llm_model", "")
        if saved_llm_model and saved_llm_model in model_choices:
            default_llm_model = saved_llm_model
        else:
            default_llm_model = model_choices[0] if model_choices else None

        # Cross-tab routing: target choices and availability
        target_choices = prompt_hub.get_enabled_target_choices(user_config)
        target_choices = target_choices + [("Custom", "_custom")]
        default_target = target_choices[0][1] if target_choices else "_custom"
        cross_tab_available = shared_state.get('prompt_apply_trigger') is not None

        components = {}

        with gr.TabItem("Prompt Manager") as prompt_generator_tab:
            components['prompt_generator_tab'] = prompt_generator_tab
            with gr.Row():
                # Left column: prompt editor and saved prompts
                with gr.Column(scale=2):
                    gr.Markdown("### Generated Prompt")

                    components['prompt_text'] = gr.Textbox(
                        label="Result",
                        lines=8,
                        max_lines=14,
                        placeholder="Your prompt will appear here, or select a saved one from the list...",
                        interactive=True
                    )

                    with gr.Row():
                        components['save_btn'] = gr.Button(
                            "Save Prompt",
                            variant="primary",
                            scale=1
                        )
                        components['delete_btn'] = gr.Button(
                            "Delete",
                            variant="stop",
                            scale=1
                        )
                        components['clear_btn'] = gr.Button(
                            "Clear",
                            scale=1
                        )

                    gr.Markdown("### Send To Tab")
                    components['pm_target'] = gr.Dropdown(
                        label="Target Field",
                        choices=target_choices,
                        value=default_target,
                        interactive=True,
                        info="Drives saved prompts shown below and where Send delivers the prompt.",
                    )
                    with gr.Row():
                        components['send_replace_btn'] = gr.Button(
                            "Send (Replace)",
                            variant="primary",
                            scale=1,
                            interactive=cross_tab_available and default_target != "_custom",
                        )
                        components['send_append_btn'] = gr.Button(
                            "Send (Append)",
                            scale=1,
                            interactive=cross_tab_available and default_target != "_custom",
                        )

                    gr.Markdown("### Saved Prompts")
                    _init_category = prompt_hub.get_category_for_target(default_target)
                    components['prompt_lister'] = FileLister(
                        value=_get_prompt_names(_init_category),
                        height=200,
                        show_footer=False,
                        interactive=True
                    )

                    gr.Markdown("### Recent Generations")
                    components['temp_lister'] = FileLister(
                        value=_get_temp_prompt_names(),
                        height=200,
                        show_footer=False,
                        interactive=True
                    )

                # Right column: LLM prompt generation
                with gr.Column(scale=2):
                    gr.Markdown("### LLM Prompt Generator")

                    components['llm_instruction'] = gr.Textbox(
                        label="Instructions for LLM",
                        lines=4,
                        max_lines=10,
                        placeholder="Describe what kind of prompt you want the LLM to generate...\ne.g., 'A dramatic monologue about a pirate finding treasure'",
                        interactive=True
                    )

                    components['llm_model'] = gr.Dropdown(
                        label="LLM Model",
                        choices=model_choices,
                        value=default_llm_model,
                        interactive=True,
                    )

                    with gr.Accordion("Advanced Settings", open=False) as llm_accordion:
                        components['llm_accordion'] = llm_accordion
                        with gr.Row():
                            components['llm_seed'] = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                minimum=-1,
                                info="-1 for random",
                                interactive=True
                            )
                            components['llm_temperature'] = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=2.0,
                                value=0.8,
                                step=0.05,
                                interactive=True
                            )
                        with gr.Row():
                            components['llm_top_k'] = gr.Slider(
                                label="Top-K",
                                minimum=0,
                                maximum=200,
                                value=40,
                                step=1,
                                info="0 = disabled",
                                interactive=True
                            )
                            components['llm_top_p'] = gr.Slider(
                                label="Top-P",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.95,
                                step=0.05,
                                interactive=True
                            )
                        with gr.Row():
                            components['llm_min_p'] = gr.Slider(
                                label="Min-P",
                                minimum=0.0,
                                maximum=1.0,
                                value=0.05,
                                step=0.01,
                                info="0 = disabled",
                                interactive=True
                            )
                            components['llm_repeat_penalty'] = gr.Slider(
                                label="Repeat Penalty",
                                minimum=1.0,
                                maximum=2.0,
                                value=1.1,
                                step=0.05,
                                interactive=True
                            )
                        with gr.Row():
                            components['llm_max_tokens'] = gr.Number(
                                label="Max Tokens",
                                value=0,
                                precision=0,
                                minimum=0,
                                info="0 = unlimited (model decides)",
                                interactive=True
                            )

                    components['system_prompt_preset'] = gr.Dropdown(
                        label="System Prompt Preset",
                        choices=SYSTEM_PROMPT_CHOICES,
                        value=SYSTEM_PROMPT_CHOICES[0],
                        interactive=True
                    )

                    components['system_prompt'] = gr.Textbox(
                        label="System Prompt",
                        lines=4,
                        max_lines=10,
                        value=SYSTEM_PROMPTS.get(SYSTEM_PROMPT_CHOICES[0], ""),
                        interactive=True
                    )

                    with gr.Row():
                        components['generate_btn'] = gr.Button(
                            "Generate Prompt",
                            variant="primary"
                        )
                        components['stop_server_btn'] = gr.Button(
                            "Stop LLM Server",
                            variant="stop",
                            visible=(llm_backend == "llama.cpp")
                        )

                    components['llm_status'] = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                    )

                    components['temp_info'] = gr.Textbox(
                        label="Generation Info",
                        interactive=False,
                        lines=4,
                    )

            # Hidden state
            components['pm_existing_files_json'] = gr.Textbox(visible=False)
            components['pm_suggested_name'] = gr.Textbox(visible=False)

        return components

    @classmethod
    def setup_events(cls, components, shared_state):
        """Wire up Prompt Manager events."""
        user_config = shared_state.get('_user_config', {})
        save_preference = shared_state.get('save_preference')
        show_input_modal_js = shared_state.get('show_input_modal_js')
        show_confirmation_modal_js = shared_state.get('show_confirmation_modal_js')
        input_trigger = shared_state.get('input_trigger')
        confirm_trigger = shared_state.get('confirm_trigger')
        cross_tab_available = shared_state.get('prompt_apply_trigger') is not None

        # Wire param persistence (auto-save on change)
        wire_param_persistence = shared_state['wire_param_persistence']
        param_map = {
            'llm': [
                ('llm_temperature', 'temperature'),
                ('llm_top_k', 'top_k'),
                ('llm_top_p', 'top_p'),
                ('llm_min_p', 'min_p'),
                ('llm_repeat_penalty', 'repeat_penalty'),
                ('system_prompt_preset', 'system_prompt_preset'),
            ],
        }
        wire_param_persistence(components, user_config, param_map)

        # Create restore handler for applying saved params on accordion expand
        create_param_restore_handler = shared_state['create_param_restore_handler']
        restore_fn, restore_outputs = create_param_restore_handler(components, user_config, param_map)

        # Restore saved params when accordion is opened
        components['llm_accordion'].expand(restore_fn, outputs=restore_outputs)

        # --- Target change: refresh lister + toggle Send buttons ---
        def on_target_change(target):
            """Refresh saved-prompts lister and enable/disable Send buttons based on target."""
            category = prompt_hub.get_category_for_target(target)
            can_send = cross_tab_available and target != "_custom"
            return (
                gr.update(value=_get_prompt_names(category)),
                gr.update(interactive=can_send),
                gr.update(interactive=can_send),
            )

        components['pm_target'].change(
            on_target_change,
            inputs=[components['pm_target']],
            outputs=[components['prompt_lister'], components['send_replace_btn'], components['send_append_btn']]
        )

        # --- Select prompt from lister ---
        def load_selected_prompt(lister_value, target):
            """Load a selected prompt into the text box."""
            if not lister_value:
                return gr.update()
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return gr.update()

            prompt_name = selected[0]
            category = prompt_hub.get_category_for_target(target)
            text = _load_prompts(category).get(prompt_name, "")
            return gr.update(value=text)

        components['prompt_lister'].change(
            load_selected_prompt,
            inputs=[components['prompt_lister'], components['pm_target']],
            outputs=[components['prompt_text']]
        )

        # --- Clear button ---
        components['clear_btn'].click(
            fn=lambda: (gr.update(value=""), ""),
            outputs=[components['prompt_text'], components['llm_status']]
        )

        # --- Save button: open input modal ---
        save_modal_js = show_input_modal_js(
            title="Save Prompt",
            message="Enter a name for this prompt:",
            placeholder="e.g., Dramatic Pirate, Thunder Storm, Calm Narrator",
            context="save_prompt_"
        )

        def get_existing_prompt_names(category):
            """Return JSON list of existing prompt names for overwrite detection in given category."""
            names = _get_prompt_names(category)
            return json.dumps(names)

        def get_selected_name(lister_value):
            """Get currently selected prompt name as suggested default."""
            if lister_value:
                selected = lister_value.get("selected", [])
                if len(selected) == 1:
                    return selected[0]
            return ""

        save_js = f"""
        (existingFilesJson, suggestedName) => {{
            try {{
                window.inputModalExistingFiles = JSON.parse(existingFilesJson || '[]');
            }} catch(e) {{
                window.inputModalExistingFiles = [];
            }}
            const openModal = {save_modal_js};
            openModal(suggestedName);
        }}
        """

        # Step 1: Get existing names (for the active category), then step 2: open modal
        components['save_btn'].click(
            fn=lambda lister_val, target: (get_existing_prompt_names(prompt_hub.get_category_for_target(target)), get_selected_name(lister_val)),
            inputs=[components['prompt_lister'], components['pm_target']],
            outputs=[components['pm_existing_files_json'], components['pm_suggested_name']],
        ).then(
            fn=None,
            inputs=[components['pm_existing_files_json'], components['pm_suggested_name']],
            js=save_js
        )

        # --- Input modal handler: save prompt ---
        def handle_save_prompt(input_value, prompt_text, target):
            """Process input modal result for saving prompts under the active category."""
            no_update = gr.update(), gr.update()

            if not input_value or not input_value.startswith("save_prompt_"):
                return no_update

            # Check for cancel
            parts = input_value.split("_", 3)
            if len(parts) >= 3 and parts[2] == "cancel":
                return no_update

            # Extract name: "save_prompt_<name>_<uuid>"
            raw_name = input_value[len("save_prompt_"):]
            name_parts = raw_name.rsplit("_", 1)
            chosen_name = name_parts[0] if len(name_parts) > 1 else raw_name

            if not chosen_name.strip():
                return "No name provided", gr.update()

            if not prompt_text or not prompt_text.strip():
                return "No prompt text to save", gr.update()

            cat = prompt_hub.get_category_for_target(target)
            try:
                prompts = _load_prompts(cat)
                prompts[chosen_name.strip()] = prompt_text.strip()
                _save_prompts(prompts, cat)

                new_names = _get_prompt_names(cat)
                cat_label = prompt_hub.PROMPT_CATEGORIES.get(cat, cat)
                return f"Saved '{chosen_name.strip()}' under {cat_label}", gr.update(value=new_names)
            except Exception as e:
                return f"Error saving: {e}", gr.update()

        input_trigger.change(
            handle_save_prompt,
            inputs=[input_trigger, components['prompt_text'], components['pm_target']],
            outputs=[components['llm_status'], components['prompt_lister']]
        )

        # --- Delete button: confirm and delete ---
        components['delete_btn'].click(
            fn=None,
            inputs=None,
            outputs=None,
            js=show_confirmation_modal_js(
                title="Delete Prompt?",
                message="This will permanently delete the selected prompt.",
                confirm_button_text="Delete",
                context="delete_prompt_"
            )
        )

        def handle_delete_prompt(confirm_value, lister_value, target):
            """Delete the selected prompt (from the active category) after confirmation."""
            no_update = gr.update(), gr.update(), gr.update()

            if not confirm_value or not confirm_value.startswith("delete_prompt_"):
                return no_update
            if "cancel" in confirm_value:
                return no_update

            if not lister_value:
                return "No prompt selected", gr.update(), gr.update()
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return "Select a single prompt to delete", gr.update(), gr.update()

            prompt_name = selected[0]
            cat = prompt_hub.get_category_for_target(target)
            try:
                prompts = _load_prompts(cat)
                if prompt_name not in prompts:
                    return f"Prompt not found: {prompt_name}", gr.update(), gr.update()

                del prompts[prompt_name]
                _save_prompts(prompts, cat)

                new_names = _get_prompt_names(cat)
                return f"Deleted: {prompt_name}", gr.update(value=new_names), gr.update(value="")
            except Exception as e:
                return f"Error deleting: {e}", gr.update(), gr.update()

        confirm_trigger.change(
            handle_delete_prompt,
            inputs=[confirm_trigger, components['prompt_lister'], components['pm_target']],
            outputs=[components['llm_status'], components['prompt_lister'], components['prompt_text']]
        )

        # --- Cross-tab routing ---
        prompt_apply_trigger = shared_state.get('prompt_apply_trigger')
        main_tabs_component = shared_state.get('main_tabs_component')

        def send_prompt_to_target(mode, prompt_text, target_id):
            """Build apply payload to send a prompt to a target tab field."""
            if not prompt_text or not str(prompt_text).strip():
                return "", "Prompt text is empty. Nothing sent."
            enabled_choices = prompt_hub.get_enabled_target_choices(user_config)
            enabled_ids = {value for _label, value in enabled_choices}
            if not target_id or target_id not in enabled_ids:
                return "", "Target is not available or disabled."

            payload = prompt_hub.build_apply_payload(target_id, mode, str(prompt_text), source="prompt_manager")
            target_label = prompt_hub.PROMPT_TARGETS.get(target_id, {}).get("label", target_id)
            return payload, f"Sent to {target_label} ({mode})."

        if prompt_apply_trigger is not None:
            if main_tabs_component is not None:
                def send_and_switch(mode, prompt_text, target_id):
                    payload, status = send_prompt_to_target(mode, prompt_text, target_id)
                    tab_id = prompt_hub.get_target_tab_id(target_id) if payload else ""
                    tab_update = gr.update(selected=tab_id) if tab_id else gr.update()
                    return payload, status, tab_update

                components['send_replace_btn'].click(
                    lambda text, target: send_and_switch("replace", text, target),
                    inputs=[components['prompt_text'], components['pm_target']],
                    outputs=[prompt_apply_trigger, components['llm_status'], main_tabs_component],
                )
                components['send_append_btn'].click(
                    lambda text, target: send_and_switch("append", text, target),
                    inputs=[components['prompt_text'], components['pm_target']],
                    outputs=[prompt_apply_trigger, components['llm_status'], main_tabs_component],
                )
            else:
                components['send_replace_btn'].click(
                    lambda text, target: send_prompt_to_target("replace", text, target),
                    inputs=[components['prompt_text'], components['pm_target']],
                    outputs=[prompt_apply_trigger, components['llm_status']],
                )
                components['send_append_btn'].click(
                    lambda text, target: send_prompt_to_target("append", text, target),
                    inputs=[components['prompt_text'], components['pm_target']],
                    outputs=[prompt_apply_trigger, components['llm_status']],
                )

        # --- Reload system prompts and refresh model list when tab is selected ---
        def on_tab_select(current_preset):
            """Reload system_prompts.json and rescan for models when tab is opened."""
            global SYSTEM_PROMPTS, SYSTEM_PROMPT_CHOICES
            SYSTEM_PROMPTS = _get_merged_system_prompts()
            SYSTEM_PROMPT_CHOICES = list(SYSTEM_PROMPTS.keys()) + ["Custom"]
            if current_preset not in SYSTEM_PROMPT_CHOICES:
                current_preset = SYSTEM_PROMPT_CHOICES[0] if SYSTEM_PROMPT_CHOICES else "Custom"
            prompt_text = SYSTEM_PROMPTS.get(current_preset, "")

            # Refresh model list
            backend = user_config.get("llm_backend", "llama.cpp")
            if backend == "Ollama":
                new_model_choices, _status = prompt_hub.discover_available_models(
                    user_config, use_local_ollama=True, endpoint_url=""
                )
                if not new_model_choices:
                    new_model_choices = []
            else:
                new_model_choices = _get_all_model_choices(user_config)
            saved = user_config.get("llm_model", "")
            new_model_value = saved if saved in new_model_choices else (new_model_choices[0] if new_model_choices else None)
            model_update = gr.update(choices=new_model_choices, value=new_model_value)

            return gr.update(choices=SYSTEM_PROMPT_CHOICES, value=current_preset), gr.update(value=prompt_text), model_update

        components['prompt_generator_tab'].select(
            on_tab_select,
            inputs=[components['system_prompt_preset']],
            outputs=[components['system_prompt_preset'], components['system_prompt'], components['llm_model']]
        )

        # --- System prompt preset selector ---
        def on_preset_change(preset_name):
            if preset_name in SYSTEM_PROMPTS:
                text = SYSTEM_PROMPTS[preset_name]
                # Resolve {available_emotions} placeholder if present
                if "{available_emotions}" in text:
                    text = text.replace("{available_emotions}", prompt_hub.get_available_emotions_text(user_config))
                return gr.update(value=text)
            return gr.update(value="")  # "Custom" — leave text as-is

        components['system_prompt_preset'].change(
            on_preset_change,
            inputs=[components['system_prompt_preset']],
            outputs=[components['system_prompt']]
        )

        # --- Generate prompt with LLM ---
        def generate_with_llm(instruction, system_prompt, model_name,
                              seed, temperature, top_k, top_p, min_p, repeat_penalty,
                              max_tokens, system_prompt_preset,
                              progress=gr.Progress()):
            """Send instruction to LLM and return generated prompt."""
            if not instruction or not instruction.strip():
                return gr.update(), "Please enter instructions for the LLM", gr.update(), gr.update()

            if not model_name:
                return gr.update(), "Please select a model", gr.update(), gr.update()

            try:
                # Build shared payload (llama.cpp ignores "model" field, Ollama requires it)
                effective_system = system_prompt.strip() if system_prompt and system_prompt.strip() else SYSTEM_PROMPTS[SYSTEM_PROMPT_CHOICES[0]]
                actual_seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
                payload = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": effective_system},
                        {"role": "user", "content": instruction.strip()}
                    ],
                    "stream": False,
                    "temperature": float(temperature),
                    "top_p": float(top_p),
                    "top_k": int(top_k),
                    "min_p": float(min_p),
                    "repeat_penalty": float(repeat_penalty),
                    "seed": actual_seed,
                }
                if int(max_tokens) > 0:
                    payload["max_tokens"] = int(max_tokens)

                backend = user_config.get("llm_backend", "llama.cpp")
                if backend == "Ollama":
                    progress(0.3, desc="Sending to Ollama...")
                    base_url = prompt_hub.get_effective_base_url(True, "", user_config)
                    headers = prompt_hub.build_headers(user_config, use_local_ollama=True)
                    response = requests.post(
                        f"{base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=120
                    )
                    progress(0.8, desc="Generating prompt...")
                else:
                    # Show appropriate message based on whether download is needed
                    if _is_model_local(model_name, user_config):
                        progress(0.1, desc="Starting LLM server...")
                    else:
                        progress(0.05, desc=f"Downloading {model_name}...")
                    success, error = _start_server(model_name, user_config, progress)
                    if not success:
                        return gr.update(), f"Server error: {error}", gr.update(), gr.update()

                    progress(0.6, desc="Generating prompt...")
                    response = requests.post(
                        f"http://localhost:{SERVER_PORT}/v1/chat/completions",
                        json=payload,
                        timeout=120
                    )

                    if response.status_code == 500:
                        # Server error, try restarting once
                        _stop_server()
                        success, error = _start_server(model_name, user_config, progress)
                        if not success:
                            return gr.update(), f"Server restart failed: {error}", gr.update(), gr.update()
                        response = requests.post(
                            f"http://localhost:{SERVER_PORT}/v1/chat/completions",
                            json=payload,
                            timeout=120
                        )

                response.raise_for_status()
                data = response.json()

                # Extract response text
                choices = data.get("choices", [])
                if not choices:
                    return gr.update(), "LLM returned no response", gr.update(), gr.update()

                generated_text = choices[0].get("message", {}).get("content", "").strip()
                if not generated_text:
                    return gr.update(), "LLM returned empty response", gr.update(), gr.update()

                # Strip thinking tags if present
                if "<think>" in generated_text:
                    import re
                    generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL).strip()

                progress(1.0, desc="Done")

                # Save to temp prompts history
                try:
                    preset_name = system_prompt_preset if system_prompt_preset else "Custom"
                    temp_name = _make_temp_name(instruction, actual_seed)
                    _save_temp_prompt(temp_name, {
                        "instruction": instruction.strip(),
                        "result": generated_text,
                        "seed": actual_seed,
                        "model": model_name,
                        "preset": preset_name,
                        "temperature": float(temperature),
                        "top_k": int(top_k),
                        "top_p": float(top_p),
                        "min_p": float(min_p),
                        "repeat_penalty": float(repeat_penalty),
                        "max_tokens": int(max_tokens),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except Exception:
                    pass

                return (
                    gr.update(value=generated_text),
                    "Prompt generated successfully",
                    gr.update(value=_get_temp_prompt_names()),
                    gr.update(value=""),
                )

            except requests.exceptions.ConnectionError:
                if user_config.get("llm_backend", "llama.cpp") == "Ollama":
                    return gr.update(), "Could not connect to Ollama. Start 'ollama serve' and retry.", gr.update(), gr.update()
                return gr.update(), "Could not connect to LLM server. Is llama-server installed?", gr.update(), gr.update()
            except requests.exceptions.Timeout:
                return gr.update(), "LLM request timed out (120s)", gr.update(), gr.update()
            except Exception as e:
                return gr.update(), f"Error: {e}", gr.update(), gr.update()

        def _disable_gen_btn():
            return gr.update(interactive=False)

        def _enable_gen_btn():
            return gr.update(interactive=True)

        components['generate_btn'].click(
            _disable_gen_btn, outputs=[components['generate_btn']]
        ).then(
            restore_fn, outputs=restore_outputs
        ).then(
            generate_with_llm,
            inputs=[
                components['llm_instruction'],
                components['system_prompt'],
                components['llm_model'],
                components['llm_seed'],
                components['llm_temperature'],
                components['llm_top_k'],
                components['llm_top_p'],
                components['llm_min_p'],
                components['llm_repeat_penalty'],
                components['llm_max_tokens'],
                components['system_prompt_preset'],
            ],
            outputs=[
                components['prompt_text'],
                components['llm_status'],
                components['temp_lister'],
                components['temp_info'],
            ]
        ).then(
            _enable_gen_btn, outputs=[components['generate_btn']]
        )

        # --- Stop server button ---
        def stop_llm_server():
            _stop_server()
            return "LLM server stopped"

        components['stop_server_btn'].click(
            stop_llm_server,
            outputs=[components['llm_status']]
        )

        # --- Select temp prompt from lister ---
        def load_temp_prompt(lister_value):
            """Load a selected temp prompt back into the UI fields."""
            no_update = gr.update(), gr.update(), gr.update()
            if not lister_value:
                return no_update
            selected = lister_value.get("selected", [])
            if len(selected) != 1:
                return no_update

            prompt_name = selected[0]
            temps = _load_temp_prompts()
            entry = temps.get(prompt_name)
            if not entry:
                return no_update

            result_text = entry.get("result", "")
            instruction_text = entry.get("instruction", "")
            info_text = _format_temp_info(entry)

            return (
                gr.update(value=result_text),
                gr.update(value=instruction_text),
                gr.update(value=info_text),
            )

        components['temp_lister'].change(
            load_temp_prompt,
            inputs=[components['temp_lister']],
            outputs=[
                components['prompt_text'],
                components['llm_instruction'],
                components['temp_info'],
            ]
        )

        # Save LLM model selection to config
        components['llm_model'].change(
            lambda x: save_preference("llm_model", x),
            inputs=[components['llm_model']],
            outputs=[]
        )


# Export for registry
get_tool_class = lambda: PromptManagerTool


# Standalone testing
if __name__ == "__main__":
    from modules.core_components.tools import run_tool_standalone
    run_tool_standalone(PromptManagerTool, port=7875, title="Prompt Manager - Standalone")
