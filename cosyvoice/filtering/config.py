"""Configuration and result data models for TTS + ASR CER filtering."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional


DEFAULT_ASR_ROOT = Path("/nfs/DGX2/Workspace/avi/fb_nchc/asr")
DEFAULT_TTS_ROOT = Path("/nfs/DGX2/Workspace/avi/fb_nchc/tts")


@dataclass
class FilterConfig:
    """Configuration for the TTS + ASR CER filtering pipeline."""

    lang: str = "tsu"
    asr_root: Path = DEFAULT_ASR_ROOT
    tts_root: Path = DEFAULT_TTS_ROOT
    asr_model_name: Optional[str] = None
    tts_model_name: Optional[str] = None
    asr_model_dir: Optional[Path] = None
    tts_model_dir: Optional[Path] = None
    cer_threshold: float = 0.05
    max_retries: int = 5
    base_seed: int = 42
    tts_mode: Literal["zero_shot", "instruct2"] = "zero_shot"
    default_prompt_wav: Optional[str] = None
    default_prompt_text: Optional[str] = None
    instruct_prefix: str = "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>"
    normalize_text: bool = True
    keep_best_on_fail: bool = True
    device: Optional[str] = None

    def __post_init__(self) -> None:
        self.asr_root = Path(self.asr_root)
        self.tts_root = Path(self.tts_root)
        if self.asr_model_dir is not None:
            self.asr_model_dir = Path(self.asr_model_dir)
        if self.tts_model_dir is not None:
            self.tts_model_dir = Path(self.tts_model_dir)
        if self.cer_threshold < 0:
            raise ValueError("cer_threshold must be non-negative")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")


@dataclass
class AttemptResult:
    """Result of a single TTS synthesis + ASR verification attempt."""

    attempt: int
    seed: int
    cer: float
    hypothesis: str
    normalized_hyp: str
    audio_path: Optional[str] = None
    duration_sec: float = 0.0


@dataclass
class FilterResult:
    """Aggregated filtering result across all attempts for one utterance."""

    text: str
    normalized_ref: str
    passed: bool
    best_attempt: AttemptResult
    tts_text: str = ""
    attempts: list[AttemptResult] = field(default_factory=list)
