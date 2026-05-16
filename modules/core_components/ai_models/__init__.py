"""
AI Models Management Package

Centralized management for TTS and ASR models.
"""

from .model_utils import (
    get_device,
    get_dtype,
    get_attention_implementation,
    check_model_available_locally,
    download_model_from_huggingface,
    empty_device_cache,
    empty_cuda_cache,
    log_gpu_memory,
    get_trained_models,
    get_trained_model_names,
    train_model,
    register_pre_load_hook,
    run_pre_load_hooks,
)

from .tts_manager import (
    TTSManager,
    get_tts_manager,
)

from .asr_manager import (
    ASRManager,
    get_asr_manager,
)

from .foley_manager import (
    FoleyManager,
    get_foley_manager,
)

__all__ = [
    # Utilities
    "get_device",
    "get_dtype",
    "get_attention_implementation",
    "check_model_available_locally",
    "download_model_from_huggingface",
    "empty_device_cache",
    "empty_cuda_cache",
    "log_gpu_memory",
    "get_trained_models",
    "get_trained_model_names",
    "train_model",
    "register_pre_load_hook",
    "run_pre_load_hooks",

    # TTS
    "TTSManager",
    "get_tts_manager",

    # ASR
    "ASRManager",
    "get_asr_manager",

    # Foley / Sound Effects
    "FoleyManager",
    "get_foley_manager",

]
