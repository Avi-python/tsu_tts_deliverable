"""Unit tests for flat and nested model directory resolution."""

from pathlib import Path
import pytest

from cosyvoice.filtering.model_resolver import resolve_asr_model_dir, resolve_tts_model_dir


def test_resolve_flat_asr_layout(tmp_path: Path) -> None:
    models_dir = tmp_path / "tsu" / "models"
    models_dir.mkdir(parents=True)
    (models_dir / "config.json").write_text("{}", encoding="utf-8")

    resolved = resolve_asr_model_dir(tmp_path, "tsu")
    assert resolved == models_dir


def test_resolve_nested_asr_and_tts_layouts(tmp_path: Path) -> None:
    asr_sub = tmp_path / "asr" / "pyu" / "models" / "pyu_asr_20261115"
    asr_sub.mkdir(parents=True)
    (asr_sub / "config.json").write_text("{}", encoding="utf-8")

    tts_sub = tmp_path / "tts" / "tsu" / "models" / "tsu123"
    tts_sub.mkdir(parents=True)
    (tts_sub / "cosyvoice3.yaml").write_text("{}", encoding="utf-8")

    assert resolve_asr_model_dir(tmp_path / "asr", "pyu") == asr_sub
    assert resolve_tts_model_dir(tmp_path / "tts", "tsu") == tts_sub


def test_resolve_explicit_model_name_and_missing_override(tmp_path: Path) -> None:
    models_dir = tmp_path / "tsu" / "models"
    for name in ("v1", "v2"):
        sub = models_dir / name
        sub.mkdir(parents=True)
        (sub / "llm.pt").write_text("", encoding="utf-8")

    assert resolve_tts_model_dir(tmp_path, "tsu", model_name="v2") == models_dir / "v2"

    with pytest.raises(FileNotFoundError, match="Available candidates: \\['v1', 'v2'\\]"):
        resolve_tts_model_dir(tmp_path, "tsu", model_name="missing")


def test_resolve_empty_and_ambiguous_errors(tmp_path: Path) -> None:
    empty_dir = tmp_path / "pyu" / "models"
    empty_dir.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="No valid TTS \\(pyu\\) model found"):
        resolve_tts_model_dir(tmp_path, "pyu")

    ambig_dir = tmp_path / "trv" / "models"
    for name in ("ckpt_a", "ckpt_b"):
        sub = ambig_dir / name
        sub.mkdir(parents=True)
        (sub / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Please specify model_name explicitly"):
        resolve_asr_model_dir(tmp_path, "trv")
