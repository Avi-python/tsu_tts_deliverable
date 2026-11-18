"""Closed-loop CosyVoice3 TTS + Whisper ASR CER filtering subpackage."""

from typing import TYPE_CHECKING

from cosyvoice.filtering.config import AttemptResult, FilterConfig, FilterResult
from cosyvoice.filtering.model_resolver import resolve_asr_model_dir, resolve_tts_model_dir

if TYPE_CHECKING:
    from cosyvoice.filtering.asr_engine import ASREngine, compute_cer, normalize_for_cer
    from cosyvoice.filtering.pipeline import TTSFilterPipeline
    from cosyvoice.filtering.tts_engine import TTSEngine

__all__ = [
    "ASREngine",
    "AttemptResult",
    "FilterConfig",
    "FilterResult",
    "TTSEngine",
    "TTSFilterPipeline",
    "compute_cer",
    "normalize_for_cer",
    "resolve_asr_model_dir",
    "resolve_tts_model_dir",
]


def __getattr__(name: str):
    if name in ("ASREngine", "compute_cer", "normalize_for_cer"):
        from cosyvoice.filtering import asr_engine

        return getattr(asr_engine, name)
    if name == "TTSEngine":
        from cosyvoice.filtering.tts_engine import TTSEngine

        return TTSEngine
    if name == "TTSFilterPipeline":
        from cosyvoice.filtering.pipeline import TTSFilterPipeline

        return TTSFilterPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
