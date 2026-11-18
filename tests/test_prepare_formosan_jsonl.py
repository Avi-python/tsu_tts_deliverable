"""Unit tests for self-contained train/dev/test splitting in tools/prepare_formosan_jsonl.py."""

import json
from pathlib import Path
import subprocess
import sys

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "tools" / "prepare_formosan_jsonl.py"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_kaldi_map(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            mapping[parts[0]] = parts[1]
    return mapping


def test_single_jsonl_split_tsu_zero_leakage(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "tsu_raw.jsonl"
    out_dir = tmp_path / "out_tsu"

    records = []
    # 40 records; first two share identical clean_text to test text-level leak prevention
    for i in range(40):
        clean_text = "shared sentence across speakers" if i < 2 else f"tsou sentence {i:02d}"
        records.append(
            {
                "id": f"tsu_{i:03d}",
                "lang": "tsu",
                "audio_path": f"/fake/audio/tsu_{i:03d}.wav",
                "clean_text": clean_text,
                "duration": 3.0,
                "speaker": f"spk_{i % 5}",
            }
        )
    input_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input_jsonl",
            str(input_jsonl),
            "--out_dir",
            str(out_dir),
            "--dev_ratio",
            "0.1",
            "--test_ratio",
            "0.1",
            "--seed",
            "42",
        ],
        check=True,
    )

    train_items = _read_jsonl(out_dir / "splits" / "train.jsonl")
    dev_items = _read_jsonl(out_dir / "splits" / "dev.jsonl")
    test_items = _read_jsonl(out_dir / "splits" / "test.jsonl")

    assert len(train_items) > 0
    assert len(dev_items) > 0
    assert len(test_items) > 0
    assert len(train_items) + len(dev_items) + len(test_items) == 40

    # Zero audio_path and clean_text overlap across train, dev, test
    split_audios = {
        name: {r["audio_path"] for r in items}
        for name, items in (("train", train_items), ("dev", dev_items), ("test", test_items))
    }
    split_texts = {
        name: {r["clean_text"] for r in items}
        for name, items in (("train", train_items), ("dev", dev_items), ("test", test_items))
    }
    assert split_audios["train"].isdisjoint(split_audios["dev"])
    assert split_audios["train"].isdisjoint(split_audios["test"])
    assert split_audios["dev"].isdisjoint(split_audios["test"])
    assert split_texts["train"].isdisjoint(split_texts["dev"])
    assert split_texts["train"].isdisjoint(split_texts["test"])
    assert split_texts["dev"].isdisjoint(split_texts["test"])

    # Verify Kaldi directories and unique_wavs (train + dev only, excluding test)
    for split_name in ("train", "dev", "test"):
        assert (out_dir / split_name / "wav.scp").is_file()
        assert (out_dir / split_name / "text").is_file()
        assert (out_dir / split_name / "utt2spk").is_file()
        assert (out_dir / split_name / "spk2utt").is_file()
        assert (out_dir / split_name / "instruct").is_file()

    unique_wavs = set(_read_kaldi_map(out_dir / "unique_wavs" / "wav.scp").values())
    assert unique_wavs == (split_audios["train"] | split_audios["dev"])
    assert unique_wavs.isdisjoint(split_audios["test"])


def test_dialect_stratified_split_auto_detected_for_pyu(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "pyu_raw.jsonl"
    out_dir = tmp_path / "out_pyu"

    records = []
    for dialect in ("Nanwang", "Jianhe"):
        for i in range(20):
            records.append(
                {
                    "id": f"pyu_{dialect}_{i:02d}",
                    "lang": "pyu",
                    "dialect": dialect,
                    "audio_path": f"/fake/audio/pyu_{dialect}_{i:02d}.wav",
                    "clean_text": f"{dialect} puyuma sentence {i:02d}",
                    "duration": 2.5,
                }
            )
    input_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input_jsonl",
            str(input_jsonl),
            "--out_dir",
            str(out_dir),
            "--lang",
            "pyu",
            "--dev_ratio",
            "0.05",
            "--test_ratio",
            "0.05",
            "--seed",
            "42",
        ],
        check=True,
    )

    for split_name in ("train", "dev", "test"):
        items = _read_jsonl(out_dir / "splits" / f"{split_name}.jsonl")
        dialects_present = {r["dialect"] for r in items}
        assert dialects_present == {"Nanwang", "Jianhe"}

    instructs = set(_read_kaldi_map(out_dir / "train" / "instruct").values())
    assert "You are a helpful assistant. 请用南王卑南語表达。<|endofprompt|>" in instructs
    assert "You are a helpful assistant. 请用建和卑南語表达。<|endofprompt|>" in instructs
