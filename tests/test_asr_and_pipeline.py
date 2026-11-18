"""Unit tests for CER normalization, scoring, explicit model dirs, and TTSFilterPipeline retry logic."""

from pathlib import Path
import pytest
import torch

from cosyvoice.filtering import pipeline as pipeline_mod
from cosyvoice.filtering.asr_engine import compute_cer, normalize_for_cer
from cosyvoice.filtering.config import FilterConfig
from cosyvoice.filtering.pipeline import TTSFilterPipeline


class MockTTSEngine:
    def __init__(self, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.seeds_used: list[int] = []
        self.texts_used: list[str] = []

    def synthesize(
        self,
        text: str,
        seed: int,
        prompt_wav: str,
        prompt_text: str | None = None,
        instruct_text: str | None = None,
    ) -> tuple[torch.Tensor, int]:
        self.seeds_used.append(seed)
        self.texts_used.append(text)
        # Encode seed into amplitude so each attempt produces distinct audio
        waveform = torch.full((1, self.sample_rate // 2), float(seed) * 1e-4)
        return waveform, self.sample_rate


class MockASREngine:
    def __init__(self, hypotheses: list[str]) -> None:
        self.hypotheses = hypotheses
        self.calls = 0

    def transcribe(self, waveform: torch.Tensor, sample_rate: int) -> str:
        hyp = self.hypotheses[self.calls]
        self.calls += 1
        return hyp


def test_normalize_preserves_formosan_glottals() -> None:
    raw = "  ʼA ongko ta 'Pasuya', eʼe! "
    assert normalize_for_cer(raw) == "ʼa ongko ta pasuya eʼe"


def test_compute_cer_matches_expected() -> None:
    cer, ref, hyp = compute_cer("Mo tohiʼu!", "mo tohiʼu", normalize=True)
    assert ref == "mo tohiʼu"
    assert hyp == "mo tohiʼu"
    assert cer == 0.0

    cer_diff, _, _ = compute_cer("abcde", "abxde", normalize=False)
    assert cer_diff == pytest.approx(0.2)


def test_pipeline_passes_on_first_attempt(tmp_path: Path) -> None:
    cfg = FilterConfig(
        lang="tsu",
        cer_threshold=0.15,
        max_retries=5,
        base_seed=100,
        default_prompt_wav="dummy.wav",
        default_prompt_text="dummy",
    )
    tts = MockTTSEngine()
    asr = MockASREngine(["mo tohiʼu ta ecngi"])
    pipeline = TTSFilterPipeline(cfg, tts, asr)

    out_wav = tmp_path / "pass1.wav"
    res = pipeline.filter_utterance("mo tohiʼu ta ecngi", output_wav_path=out_wav)

    assert res.passed is True
    assert len(res.attempts) == 1
    assert res.best_attempt.seed == 100
    assert res.best_attempt.cer == 0.0
    assert out_wav.is_file()


def test_pipeline_retries_with_incrementing_seeds_until_pass(tmp_path: Path) -> None:
    cfg = FilterConfig(
        lang="tsu",
        cer_threshold=0.15,
        max_retries=5,
        base_seed=42,
        default_prompt_wav="dummy.wav",
        default_prompt_text="dummy",
    )
    tts = MockTTSEngine()
    asr = MockASREngine(["wrong text", "also bad", "mo tohiʼu"])
    pipeline = TTSFilterPipeline(cfg, tts, asr)

    out_wav = tmp_path / "retry_pass.wav"
    res = pipeline.filter_utterance("mo tohiʼu", output_wav_path=out_wav)

    assert res.passed is True
    assert tts.seeds_used == [42, 43, 44]
    assert len(res.attempts) == 3
    assert res.best_attempt.attempt == 3
    assert res.best_attempt.seed == 44
    assert out_wav.is_file()


def test_pipeline_exhausts_retries_keep_best_and_discard(tmp_path: Path) -> None:
    cfg_keep = FilterConfig(
        lang="tsu",
        cer_threshold=0.05,
        max_retries=3,
        base_seed=10,
        keep_best_on_fail=True,
        default_prompt_wav="dummy.wav",
        default_prompt_text="dummy",
    )
    tts = MockTTSEngine()
    # Target: "abcdef" (6 chars). Hyps have CER 0.50, 0.1667, 0.3333 -> best is seed 11
    asr = MockASREngine(["abcxxx", "abcdex", "abcdxx"])
    pipeline = TTSFilterPipeline(cfg_keep, tts, asr)

    keep_wav = tmp_path / "best_fail.wav"
    res_keep = pipeline.filter_utterance("abcdef", output_wav_path=keep_wav)
    assert res_keep.passed is False
    assert res_keep.best_attempt.seed == 11
    assert res_keep.best_attempt.audio_path == str(keep_wav)
    assert keep_wav.is_file()

    cfg_discard = FilterConfig(
        lang="tsu",
        cer_threshold=0.05,
        max_retries=2,
        base_seed=10,
        keep_best_on_fail=False,
        default_prompt_wav="dummy.wav",
        default_prompt_text="dummy",
    )
    discard_wav = tmp_path / "discard_fail.wav"
    res_discard = TTSFilterPipeline(
        cfg_discard, MockTTSEngine(), MockASREngine(["xxxxxx", "yyyyyy"])
    ).filter_utterance("abcdef", output_wav_path=discard_wav)
    assert res_discard.passed is False
    assert res_discard.best_attempt.audio_path is None
    assert not discard_wav.exists()


def test_pipeline_separate_tts_text_and_fallback(tmp_path: Path) -> None:
    cfg = FilterConfig(
        lang="tsu",
        cer_threshold=0.0,
        max_retries=2,
        base_seed=42,
        default_prompt_wav="dummy.wav",
        default_prompt_text="dummy",
    )
    ref = "ʼa ongko ta pasuya 我的 English 不是很好."
    hotfix = "ʼa ongko ta pasuya 我的 [IH1][NG][G][L][IH0][SH] 不是很好."

    tts = MockTTSEngine()
    asr = MockASREngine([ref, ref])
    pipeline = TTSFilterPipeline(cfg, tts, asr)

    res_inpaint = pipeline.filter_utterance(
        text=ref,
        tts_text=hotfix,
        output_wav_path=tmp_path / "inpaint.wav",
    )
    assert res_inpaint.passed is True
    assert res_inpaint.text == ref
    assert res_inpaint.tts_text == hotfix
    assert tts.texts_used == [hotfix]
    assert res_inpaint.best_attempt.cer == 0.0

    res_fallback = pipeline.filter_utterance(
        text=ref,
        tts_text="   ",
        output_wav_path=tmp_path / "fallback.wav",
    )
    assert res_fallback.passed is True
    assert res_fallback.tts_text == ref
    assert tts.texts_used == [hotfix, ref]


def test_filter_config_explicit_model_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    explicit_tts = tmp_path / "custom_tts"
    explicit_asr = tmp_path / "custom_asr"
    explicit_tts.mkdir()
    explicit_asr.mkdir()

    captured: dict[str, Path] = {}

    def fake_tts_engine(model_dir: Path, mode: str, instruct_prefix: str) -> MockTTSEngine:
        captured["tts_dir"] = model_dir
        return MockTTSEngine()

    def fake_asr_engine(model_dir: Path, device: str | None = None) -> MockASREngine:
        captured["asr_dir"] = model_dir
        return MockASREngine([])

    monkeypatch.setattr(pipeline_mod, "TTSEngine", fake_tts_engine)
    monkeypatch.setattr(pipeline_mod, "ASREngine", fake_asr_engine)

    cfg = FilterConfig(
        lang="tsu",
        tts_model_dir=str(explicit_tts),  # type: ignore[arg-type]
        asr_model_dir=str(explicit_asr),  # type: ignore[arg-type]
        tts_root=tmp_path / "nonexistent_tts",
        asr_root=tmp_path / "nonexistent_asr",
    )
    assert cfg.tts_model_dir == explicit_tts
    assert cfg.asr_model_dir == explicit_asr

    TTSFilterPipeline.from_config(cfg)
    assert captured["tts_dir"] == explicit_tts
    assert captured["asr_dir"] == explicit_asr
