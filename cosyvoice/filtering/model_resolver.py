"""Resolve per-language Whisper ASR and CosyVoice3 TTS model directories."""

from pathlib import Path
from typing import Optional


ASR_MARKER_FILES = ("config.json",)
TTS_MARKER_FILES = ("cosyvoice3.yaml", "cosyvoice2.yaml", "cosyvoice.yaml", "llm.pt")


def _has_any_marker(directory: Path, marker_files: tuple[str, ...]) -> bool:
    """Return True if directory directly contains at least one marker file."""
    return any((directory / marker).is_file() for marker in marker_files)


def resolve_model_dir(
    models_dir: Path,
    marker_files: tuple[str, ...],
    modality: str,
    model_name: Optional[str] = None,
) -> Path:
    """Resolve a flat or nested model directory under `<root>/<lang>/models`.

    Supports both:
    - Flat layout: `models/` directly contains model artifact files.
    - Nested layout: `models/<subdir>/` contains model artifact files.
    """
    models_dir = Path(models_dir)
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"{modality} models directory does not exist: {models_dir}"
        )

    if model_name is not None:
        candidate = models_dir / model_name
        if candidate.is_dir() and _has_any_marker(candidate, marker_files):
            return candidate
        available = [
            p.name
            for p in sorted(models_dir.iterdir())
            if p.is_dir() and _has_any_marker(p, marker_files)
        ]
        raise FileNotFoundError(
            f"{modality} model '{model_name}' not found under {models_dir}. "
            f"Available candidates: {available or 'none'}"
        )

    if _has_any_marker(models_dir, marker_files):
        return models_dir

    candidates = [
        p
        for p in sorted(models_dir.iterdir())
        if p.is_dir() and not p.name.startswith(".") and _has_any_marker(p, marker_files)
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No valid {modality} model found in {models_dir} "
            f"(expected one of {marker_files} directly or inside a subdirectory)."
        )

    if len(candidates) > 1:
        names = [c.name for c in candidates]
        raise ValueError(
            f"Multiple {modality} models found in {models_dir}: {names}. "
            f"Please specify model_name explicitly."
        )

    return candidates[0]


def resolve_asr_model_dir(
    asr_root: Path, lang: str, model_name: Optional[str] = None
) -> Path:
    """Resolve Whisper ASR model directory for a language."""
    return resolve_model_dir(
        Path(asr_root) / lang / "models",
        ASR_MARKER_FILES,
        modality=f"ASR ({lang})",
        model_name=model_name,
    )


def resolve_tts_model_dir(
    tts_root: Path, lang: str, model_name: Optional[str] = None
) -> Path:
    """Resolve CosyVoice3 TTS model directory for a language."""
    return resolve_model_dir(
        Path(tts_root) / lang / "models",
        TTS_MARKER_FILES,
        modality=f"TTS ({lang})",
        model_name=model_name,
    )
