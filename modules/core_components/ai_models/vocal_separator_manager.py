"""
Vocal Separator Manager

Source separation via audio-separator (supports UVR MDX-Net, VR Arch, Demucs models).
Models are loaded per-call — no persistent state needed.
"""

import logging
from pathlib import Path

SUPPORTED_EXTENSIONS = {'.onnx', '.pth', '.th', '.yaml'}

# Show best vocal models first in the dropdown
_PREFERRED_ORDER = [
    'Kim_Vocal_2.onnx',
    'htdemucs_ft.yaml',
    'UVR-MDX-NET-Inst_HQ_4.onnx',
    '5_HP-Karaoke-UVR.pth',
]


def is_available() -> bool:
    """Check whether audio-separator is installed."""
    try:
        from audio_separator.separator import Separator  # noqa: F401
        return True
    except ImportError:
        return False


def scan_models(model_dir: str) -> list:
    """Return model filenames found in model_dir (recursive scan)."""
    if not model_dir:
        return []
    root = Path(model_dir)
    if not root.exists():
        return []

    found = set()
    for ext in SUPPORTED_EXTENSIONS:
        for p in root.rglob(f'*{ext}'):
            found.add(p.name)

    def _sort_key(name):
        try:
            return (0, _PREFERRED_ORDER.index(name), name)
        except ValueError:
            return (1, 0, name)

    return sorted(found, key=_sort_key)


def separate_vocals(audio_path: str, model_name: str, model_dir: str,
                    output_dir: str, progress=None):
    """
    Separate vocals and instrumental from *audio_path*.

    Returns:
        (vocals_path, instrumental_path, status_message)
        Either path may be None if the model only produces one stem.
    """
    if not audio_path:
        return None, None, "No audio file provided."
    if not model_name:
        return None, None, "No model selected."
    if not model_dir or not Path(model_dir).exists():
        return None, None, "Model directory not found. Set it in the tab settings."

    try:
        from audio_separator.separator import Separator
    except ImportError:
        return None, None, (
            "❌ audio-separator not installed.\n"
            "Run:  pip install audio-separator[gpu]"
        )

    try:
        if progress:
            progress(0.1, desc=f"Loading {model_name}…")

        sep = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(model_dir),
            output_dir=str(output_dir),
            output_format="WAV",
            normalization_threshold=0.9,
        )
        sep.load_model(model_filename=model_name)

        if progress:
            progress(0.35, desc="Separating audio…")

        output_files = sep.separate(str(audio_path))

        if not output_files:
            return None, None, "❌ Separation produced no output files."

        if progress:
            progress(0.9, desc="Identifying stems…")

        vocals_path = None
        instrumental_path = None

        for f in output_files:
            stem = Path(f).stem.lower()
            if any(t in stem for t in ('vocal', 'voice', 'singing')):
                if any(t in stem for t in ('no_', 'inst', 'music', 'back')):
                    instrumental_path = f
                else:
                    vocals_path = f
            elif any(t in stem for t in ('instrumental', 'no_vocal', 'music', 'backing', 'other')):
                instrumental_path = f

        # Fallback: first file = vocals, second = instrumental
        if vocals_path is None and instrumental_path is None:
            if len(output_files) >= 2:
                vocals_path, instrumental_path = output_files[0], output_files[1]
            elif len(output_files) == 1:
                vocals_path = output_files[0]

        if progress:
            progress(1.0, desc="Done!")

        return vocals_path, instrumental_path, f"Separated with {model_name}"

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return None, None, f"❌ {exc}"
