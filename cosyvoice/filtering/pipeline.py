"""Closed-loop TTS synthesis + ASR CER verification and seed-retry pipeline."""

from pathlib import Path
from typing import Optional, Protocol

import torch
import torchaudio

from cosyvoice.filtering.asr_engine import ASREngine, compute_cer, normalize_for_cer
from cosyvoice.filtering.config import AttemptResult, FilterConfig, FilterResult
from cosyvoice.filtering.model_resolver import resolve_asr_model_dir, resolve_tts_model_dir
from cosyvoice.filtering.tts_engine import TTSEngine


class TTSEngineProtocol(Protocol):
    def synthesize(
        self,
        text: str,
        seed: int,
        prompt_wav: str,
        prompt_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
    ) -> tuple[torch.Tensor, int]: ...


class ASREngineProtocol(Protocol):
    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str: ...


class TTSFilterPipeline:
    """Synthesize speech, verify CER via ASR, and retry with new seeds until passing."""

    def __init__(
        self,
        config: FilterConfig,
        tts_engine: TTSEngineProtocol,
        asr_engine: ASREngineProtocol,
    ) -> None:
        self.config = config
        self.tts_engine = tts_engine
        self.asr_engine = asr_engine

    @classmethod
    def from_config(cls, config: FilterConfig) -> "TTSFilterPipeline":
        """Resolve ASR/TTS model directories and initialize both engines."""
        asr_dir = (
            Path(config.asr_model_dir)
            if config.asr_model_dir is not None
            else resolve_asr_model_dir(
                config.asr_root, config.lang, model_name=config.asr_model_name
            )
        )
        tts_dir = (
            Path(config.tts_model_dir)
            if config.tts_model_dir is not None
            else resolve_tts_model_dir(
                config.tts_root, config.lang, model_name=config.tts_model_name
            )
        )
        tts_engine = TTSEngine(
            model_dir=tts_dir,
            mode=config.tts_mode,
            instruct_prefix=config.instruct_prefix,
        )
        asr_engine = ASREngine(model_dir=asr_dir, device=config.device)
        return cls(config=config, tts_engine=tts_engine, asr_engine=asr_engine)

    @staticmethod
    def _save_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tensor = waveform.detach().cpu()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(str(path), tensor, sample_rate)

    def filter_utterance(
        self,
        text: str,
        output_wav_path: Optional[Path] = None,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        instruct_text: Optional[str] = None,
        tts_text: Optional[str] = None,
    ) -> FilterResult:
        """Run up to `max_retries` seed attempts until CER <= `cer_threshold`."""
        eff_prompt_wav = prompt_wav or self.config.default_prompt_wav
        if not eff_prompt_wav:
            raise ValueError("No prompt_wav provided and default_prompt_wav is unset")
        eff_prompt_text = (
            prompt_text if prompt_text is not None else self.config.default_prompt_text
        )
        eff_tts_text = (
            tts_text.strip() if isinstance(tts_text, str) and tts_text.strip() else text
        )

        norm_ref = normalize_for_cer(text) if self.config.normalize_text else text.strip()
        attempts: list[AttemptResult] = []
        best_attempt: Optional[AttemptResult] = None
        best_audio: Optional[tuple[torch.Tensor, int]] = None

        for attempt_idx in range(self.config.max_retries):
            seed = self.config.base_seed + attempt_idx
            waveform, sample_rate = self.tts_engine.synthesize(
                text=eff_tts_text,
                seed=seed,
                prompt_wav=eff_prompt_wav,
                prompt_text=eff_prompt_text,
                instruct_text=instruct_text,
            )
            hypothesis = self.asr_engine.transcribe(waveform, sample_rate)
            cer, _, norm_hyp = compute_cer(
                text, hypothesis, normalize=self.config.normalize_text
            )
            duration_sec = float(waveform.shape[-1]) / float(sample_rate)

            record = AttemptResult(
                attempt=attempt_idx + 1,
                seed=seed,
                cer=cer,
                hypothesis=hypothesis,
                normalized_hyp=norm_hyp,
                audio_path=None,
                duration_sec=round(duration_sec, 4),
            )
            attempts.append(record)

            if best_attempt is None or cer < best_attempt.cer:
                best_attempt = record
                best_audio = (waveform, sample_rate)

            if cer <= self.config.cer_threshold:
                if output_wav_path is not None:
                    out_path = Path(output_wav_path)
                    self._save_wav(out_path, waveform, sample_rate)
                    record.audio_path = str(out_path)
                return FilterResult(
                    text=text,
                    normalized_ref=norm_ref,
                    passed=True,
                    best_attempt=record,
                    tts_text=eff_tts_text,
                    attempts=attempts,
                )

        assert best_attempt is not None and best_audio is not None
        if self.config.keep_best_on_fail and output_wav_path is not None:
            out_path = Path(output_wav_path)
            self._save_wav(out_path, best_audio[0], best_audio[1])
            best_attempt.audio_path = str(out_path)

        return FilterResult(
            text=text,
            normalized_ref=norm_ref,
            passed=False,
            best_attempt=best_attempt,
            tts_text=eff_tts_text,
            attempts=attempts,
        )
