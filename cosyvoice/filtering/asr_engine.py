"""Whisper ASR engine wrapper, Post-TN normalization, and CER computation."""

import json
from pathlib import Path
import re
from typing import Optional

import jiwer
import torch
import torchaudio
from transformers import WhisperForConditionalGeneration, WhisperProcessor


WHISPER_SAMPLE_RATE = 16000
_STRIP_PUNCT_RE = re.compile(r"[^\w\sʼ]", flags=re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_for_cer(text: str) -> str:
    """Apply Post-TN normalization: lowercase, strip punctuation (keeping U+02BC glottal), collapse spaces."""
    cleaned = _STRIP_PUNCT_RE.sub(" ", text.lower())
    return _MULTI_SPACE_RE.sub(" ", cleaned).strip()


def compute_cer(
    reference: str, hypothesis: str, normalize: bool = True
) -> tuple[float, str, str]:
    """Compute CER between reference and hypothesis, returning `(cer, ref_used, hyp_used)`."""
    ref_used = normalize_for_cer(reference) if normalize else reference.strip()
    hyp_used = normalize_for_cer(hypothesis) if normalize else hypothesis.strip()

    if not ref_used:
        return (0.0 if not hyp_used else 1.0), ref_used, hyp_used

    score = float(jiwer.cer(ref_used, hyp_used))
    return score, ref_used, hyp_used


def _detect_formosan_language_token(model_dir: Path) -> Optional[str]:
    """Return '<|formosan|>' if defined in generation_config.json, else None."""
    gen_cfg_path = model_dir / "generation_config.json"
    if not gen_cfg_path.is_file():
        return None
    try:
        data = json.loads(gen_cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    lang_to_id = data.get("lang_to_id") or {}
    if "<|formosan|>" in lang_to_id:
        return "<|formosan|>"
    return None


class ASREngine:
    """Wrapper around fine-tuned Whisper models for 16 kHz transcription."""

    def __init__(self, model_dir: Path, device: Optional[str] = None) -> None:
        self.model_dir = Path(model_dir)
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(str(self.model_dir))
        self.model = WhisperForConditionalGeneration.from_pretrained(
            str(self.model_dir),
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.language_token = _detect_formosan_language_token(self.model_dir)

    @torch.inference_mode()
    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """Resample `waveform` to 16 kHz in memory and return decoded transcription."""
        mono = waveform.detach().cpu().float()
        if mono.dim() == 2:
            if mono.size(0) > 1:
                mono = mono.mean(dim=0, keepdim=True)
            mono_1d = mono.squeeze(0)
        elif mono.dim() == 1:
            mono_1d = mono
        else:
            raise ValueError(f"Expected 1D or 2D waveform tensor, got shape {tuple(mono.shape)}")

        if sample_rate != WHISPER_SAMPLE_RATE:
            mono_1d = torchaudio.functional.resample(
                mono_1d.unsqueeze(0), orig_freq=sample_rate, new_freq=WHISPER_SAMPLE_RATE
            ).squeeze(0)

        inputs = self.processor(
            mono_1d.numpy(),
            sampling_rate=WHISPER_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device=self.device, dtype=self.dtype)

        generate_kwargs: dict = {}
        if self.language_token is not None:
            generate_kwargs["language"] = self.language_token
            generate_kwargs["task"] = "transcribe"

        predicted_ids = self.model.generate(input_features, **generate_kwargs)
        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()
