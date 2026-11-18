#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CosyVoice3 (SARC Voice) train/dev/test splits and Kaldi directories from a single Formosan JSONL file.

Key behaviors:
- Uses `clean_text` (no 4x augmentation variants; 1 unique `audio_path` -> 1 utterance).
- Self-contained by default: splits `--input_jsonl` directly into 90% train / 5% dev / 5% test
  (configurable via `--dev_ratio`, `--test_ratio`, `--seed`) without needing any external `test.jsonl`.
- Groups by `clean_text` so identical sentences never leak across train, dev, and test.
- Automatically detects `lang == "pyu"` (Puyuma) to stratify the split across dialects
  (Nanwang / Jianhe / Zhiben / Xiqun), while performing a direct split for `tsu` / `trv`.
- Exports `<out_dir>/splits/{train,dev,test}.jsonl` (so `test.jsonl` can be passed straight to
  `inference.py` or `inference_filter.py`), `<out_dir>/{train,dev,test}/` Kaldi directories,
  and `<out_dir>/unique_wavs/` (train + dev only) for Stage 1 & 2 feature extraction.
"""

import argparse
from collections import Counter, defaultdict
import json
import os
import random
import re
from typing import Optional

PYU_DIALECT_MAP = {
    "Nanwang": "南王",
    "Jianhe": "建和",
    "Zhiben": "知本",
    "Xiqun": "西群",
    "南王": "南王",
    "建和": "建和",
    "知本": "知本",
    "西群": "西群",
}

DEFAULT_LANG_INSTRUCT = {
    "tsu": "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>",
    "trv": "You are a helpful assistant. 请用太魯閣語表达。<|endofprompt|>",
    "pyu": "You are a helpful assistant. 请用{dialect}卑南語表达。<|endofprompt|>",
}


def build_record_instruct(lang: str, raw_dialect: Optional[str], instruct_arg: str) -> str:
    """Build language- and dialect-specific CosyVoice3 instruct prompt."""
    template = (
        instruct_arg
        if instruct_arg
        else DEFAULT_LANG_INSTRUCT.get(lang, "You are a helpful assistant.<|endofprompt|>")
    )
    if lang == "pyu" or "{dialect}" in template:
        zh_dialect = PYU_DIALECT_MAP.get(
            str(raw_dialect or "").strip(), str(raw_dialect or "").strip()
        )
        if "{dialect}" in template:
            return template.format(dialect=zh_dialect)
        if zh_dialect:
            return f"You are a helpful assistant. 请用{zh_dialect}卑南語表达。<|endofprompt|>"
    return template


def sanitize_id(val: str) -> str:
    s = re.sub(r"\s+", "_", str(val).strip())
    s = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]", "_", s)
    return s or "unknown"


def load_external_splits(splits_dir: str, lang: str) -> tuple[set[str], set[str]]:
    """Load optional external dev.jsonl and test.jsonl audio_path sets."""
    dev_audios: set[str] = set()
    test_audios: set[str] = set()
    if not splits_dir or not os.path.isdir(splits_dir):
        return dev_audios, test_audios

    for filename, target_set in (("dev.jsonl", dev_audios), ("test.jsonl", test_audios)):
        path = os.path.join(splits_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                rec_lang = d.get("lang")
                if (rec_lang is None or rec_lang == lang) and d.get("audio_path"):
                    target_set.add(str(d["audio_path"]).strip())

    return dev_audios, test_audios


def assign_splits_self_contained(
    records: list[tuple[str, str, dict]],
    lang: str,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, str]:
    """Assign each clean_text group to 'train', 'dev', or 'test' reproducibly.

    Automatically stratifies by dialect when `lang == 'pyu'`.
    Returns a mapping `{clean_text: split_name}`.
    """
    text_groups: dict[str, list[tuple[str, str, dict]]] = defaultdict(list)
    for rec in records:
        text_groups[rec[1]].append(rec)

    buckets: dict[str, list[str]] = defaultdict(list)
    if lang == "pyu":
        for text_key, grp in text_groups.items():
            raw_dialect = str(grp[0][2].get("dialect") or "").strip()
            dialect_key = PYU_DIALECT_MAP.get(raw_dialect, raw_dialect or "default")
            buckets[dialect_key].append(text_key)
    else:
        buckets["all"] = list(text_groups.keys())

    rng = random.Random(seed)
    text_to_split: dict[str, str] = {}

    for bucket_name in sorted(buckets.keys()):
        keys = sorted(buckets[bucket_name])
        rng.shuffle(keys)
        n = len(keys)
        if n == 0:
            continue

        n_test = round(n * test_ratio)
        if n >= 3 and test_ratio > 0 and n_test == 0:
            n_test = 1
        n_test = min(n_test, max(0, n - 1))

        n_dev = round(n * dev_ratio)
        if (n - n_test) >= 2 and dev_ratio > 0 and n_dev == 0:
            n_dev = 1
        n_dev = min(n_dev, max(0, n - n_test - 1))

        for k in keys[:n_test]:
            text_to_split[k] = "test"
        for k in keys[n_test : n_test + n_dev]:
            text_to_split[k] = "dev"
        for k in keys[n_test + n_dev :]:
            text_to_split[k] = "train"

    return text_to_split


def write_kaldi_dir(des_dir: str, records: list[dict], instruct_text: str) -> None:
    os.makedirs(des_dir, exist_ok=True)
    spk2utt: dict[str, list[str]] = defaultdict(list)

    with (
        open(os.path.join(des_dir, "wav.scp"), "w", encoding="utf-8") as f_wav,
        open(os.path.join(des_dir, "utt2wav_id"), "w", encoding="utf-8") as f_wid,
        open(os.path.join(des_dir, "text"), "w", encoding="utf-8") as f_txt,
        open(os.path.join(des_dir, "utt2spk"), "w", encoding="utf-8") as f_u2s,
        open(os.path.join(des_dir, "instruct"), "w", encoding="utf-8") as f_ins,
    ):
        for r in records:
            utt = r["utt_id"]
            wav_id = r["wav_id"]
            audio_path = r["audio_path"]
            text = r["text"]
            spk = r["spk_id"]

            f_wav.write(f"{utt} {audio_path}\n")
            f_wid.write(f"{utt} {wav_id}\n")
            f_txt.write(f"{utt} {text}\n")
            f_u2s.write(f"{utt} {spk}\n")
            rec_instruct = r.get("instruct") or instruct_text
            if rec_instruct:
                f_ins.write(f"{utt} {rec_instruct}\n")
            spk2utt[spk].append(utt)

    with open(os.path.join(des_dir, "spk2utt"), "w", encoding="utf-8") as f_s2u:
        for spk, utts in spk2utt.items():
            f_s2u.write(f"{spk} {' '.join(utts)}\n")


def write_split_jsonl(path: str, split_name: str, raw_items: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in raw_items:
            out_rec = dict(item)
            out_rec["split"] = split_name
            f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Formosan JSONL data (clean_text) into train/dev/test splits for CosyVoice3."
    )
    parser.add_argument("--input_jsonl", required=True, help="Path to input .jsonl file")
    parser.add_argument("--out_dir", required=True, help="Output base directory (e.g. data/tsu)")
    parser.add_argument(
        "--lang",
        default=None,
        help="Language code (tsu, pyu, trv). Auto-detected from JSONL 'lang' field if omitted; defaults to 'tsu'.",
    )
    parser.add_argument(
        "--splits_dir",
        default="",
        help="Optional directory containing external dev.jsonl and test.jsonl. If omitted, splits --input_jsonl directly into train/dev/test.",
    )
    parser.add_argument("--dev_ratio", type=float, default=0.05, help="Dev split ratio (default: 0.05)")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Test split ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splitting (default: 42)")
    parser.add_argument(
        "--instruct",
        default="",
        help="Optional CosyVoice3 instruct prompt override (supports {dialect} placeholder)",
    )
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration in seconds (<=30s for Whisper)")
    args = parser.parse_args()

    if not (0.0 <= args.dev_ratio < 1.0) or not (0.0 <= args.test_ratio < 1.0):
        raise ValueError("dev_ratio and test_ratio must be in [0.0, 1.0)")
    if args.dev_ratio + args.test_ratio >= 1.0:
        raise ValueError("dev_ratio + test_ratio must be strictly less than 1.0")

    # Pass 1: Read valid records, deduplicate by audio_path, and auto-detect language if needed
    seen_audios: set[str] = set()
    valid_records: list[tuple[str, str, dict]] = []
    detected_lang: Optional[str] = None
    skipped_duration = 0
    skipped_empty = 0
    skipped_duplicate_audio = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            audio_path = str(d.get("audio_path") or "").strip()
            text = re.sub(r"\s+", " ", str(d.get("clean_text") or d.get("text") or "").strip())

            if not audio_path or not text:
                skipped_empty += 1
                continue

            raw_duration = d.get("duration")
            if raw_duration is not None:
                duration = float(raw_duration)
                if duration < args.min_duration or duration > args.max_duration:
                    skipped_duration += 1
                    continue

            if audio_path in seen_audios:
                skipped_duplicate_audio += 1
                continue
            seen_audios.add(audio_path)

            if detected_lang is None and isinstance(d.get("lang"), str) and d["lang"].strip():
                detected_lang = d["lang"].strip()

            valid_records.append((audio_path, text, d))

    lang = args.lang.strip() if isinstance(args.lang, str) and args.lang.strip() else (detected_lang or "tsu")
    print(f"[{lang}] Loaded {len(valid_records):,} unique valid records from {args.input_jsonl}")

    dev_audios, test_audios = load_external_splits(args.splits_dir, lang)
    use_external_splits = bool(dev_audios or test_audios)

    if use_external_splits:
        print(
            f"[{lang}] Using external splits from {args.splits_dir}: "
            f"dev={len(dev_audios):,} wavs, test={len(test_audios):,} wavs"
        )
        test_texts = {txt for ap, txt, _ in valid_records if ap in test_audios}
        dev_texts = {txt for ap, txt, _ in valid_records if ap in dev_audios and txt not in test_texts}
        split_assignments: list[str] = []
        for ap, txt, _ in valid_records:
            if ap in test_audios or txt in test_texts:
                split_assignments.append("test")
            elif ap in dev_audios or txt in dev_texts:
                split_assignments.append("dev")
            else:
                split_assignments.append("train")
    else:
        split_mode = "dialect-stratified (pyu)" if lang == "pyu" else "grouped-random"
        print(
            f"[{lang}] Self-contained {split_mode} split: "
            f"train={1.0 - args.dev_ratio - args.test_ratio:.1%}, dev={args.dev_ratio:.1%}, test={args.test_ratio:.1%} (seed={args.seed})"
        )
        text_to_split = assign_splits_self_contained(
            valid_records,
            lang=lang,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        split_assignments = [text_to_split[txt] for _, txt, _ in valid_records]

    # Reuse existing unique_wavs/wav.scp mappings if present so extracted .pt features remain valid
    audio_to_wav_id: dict[str, str] = {}
    audio_to_spk_id: dict[str, str] = {}
    unique_dir = os.path.join(args.out_dir, "unique_wavs")
    existing_wav_scp = os.path.join(unique_dir, "wav.scp")
    existing_utt2spk = os.path.join(unique_dir, "utt2spk")
    if os.path.exists(existing_wav_scp) and os.path.exists(existing_utt2spk):
        wav_id_to_spk = {}
        with open(existing_utt2spk, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wav_id_to_spk[parts[0]] = parts[1]
        with open(existing_wav_scp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    wid, apath = parts[0], parts[1]
                    audio_to_wav_id[apath] = wid
                    if wid in wav_id_to_spk:
                        audio_to_spk_id[apath] = wav_id_to_spk[wid]
        print(f"[{lang}] Reusing {len(audio_to_wav_id):,} existing wav_id mappings from {existing_wav_scp}")

    kaldi_by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    jsonl_by_split: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}

    # Assign wav_id/spk_id for train/dev first, then test, so unique_wavs stays clean
    ordered_indices = [
        i for i, sp in enumerate(split_assignments) if sp in ("train", "dev")
    ] + [
        i for i, sp in enumerate(split_assignments) if sp == "test"
    ]

    for i in ordered_indices:
        audio_path, text, d = valid_records[i]
        split_name = split_assignments[i]

        if audio_path not in audio_to_wav_id:
            idx = len(audio_to_wav_id)
            audio_to_wav_id[audio_path] = f"{lang}_wav_{idx:06d}"

            raw_spk = d.get("speaker")
            if raw_spk and str(raw_spk).strip():
                spk_id = f"{lang}_spk_{sanitize_id(raw_spk)}"
            else:
                corpus = sanitize_id(d.get("corpus") or "corp")
                dialect = sanitize_id(d.get("dialect") or "default")
                spk_id = f"{lang}_{corpus}_{dialect}"
            audio_to_spk_id[audio_path] = spk_id

        wav_id = audio_to_wav_id[audio_path]
        spk_id = audio_to_spk_id[audio_path]
        rec_instruct = build_record_instruct(lang, d.get("dialect"), args.instruct)

        kaldi_by_split[split_name].append(
            {
                "utt_id": wav_id,
                "wav_id": wav_id,
                "audio_path": audio_path,
                "text": text,
                "spk_id": spk_id,
                "instruct": rec_instruct,
            }
        )
        jsonl_by_split[split_name].append(d)

    # Write unique_wavs (train + dev only, excluding test) for Stage 1 & 2 feature extraction
    os.makedirs(unique_dir, exist_ok=True)
    train_dev_recs = kaldi_by_split["train"] + kaldi_by_split["dev"]
    with (
        open(os.path.join(unique_dir, "wav.scp"), "w", encoding="utf-8") as f_wav,
        open(os.path.join(unique_dir, "utt2spk"), "w", encoding="utf-8") as f_u2s,
    ):
        for r in train_dev_recs:
            f_wav.write(f"{r['wav_id']} {r['audio_path']}\n")
            f_u2s.write(f"{r['wav_id']} {r['spk_id']}\n")

    splits_out_dir = os.path.join(args.out_dir, "splits")
    for split_name in ("train", "dev", "test"):
        write_kaldi_dir(os.path.join(args.out_dir, split_name), kaldi_by_split[split_name], args.instruct)
        write_split_jsonl(os.path.join(splits_out_dir, f"{split_name}.jsonl"), split_name, jsonl_by_split[split_name])

    print(f"[{lang}] Preparation complete -> {args.out_dir}")
    print(f"  Train records : {len(kaldi_by_split['train']):,} -> {os.path.join(splits_out_dir, 'train.jsonl')}")
    print(f"  Dev records   : {len(kaldi_by_split['dev']):,} -> {os.path.join(splits_out_dir, 'dev.jsonl')}")
    print(f"  Test records  : {len(kaldi_by_split['test']):,} -> {os.path.join(splits_out_dir, 'test.jsonl')}")
    print(f"  Unique train+dev wavs (Stage 1 & 2) : {len(train_dev_recs):,}")
    print(f"  Unique speaker IDs                  : {len(set(audio_to_spk_id.values())):,}")
    print(f"  Skipped: empty={skipped_empty:,}, duration={skipped_duration:,}, duplicate_audio={skipped_duplicate_audio:,}")
    instruct_counts = Counter(r["instruct"] for r in kaldi_by_split["train"])
    print("  Train instruct distribution:")
    for ins, cnt in instruct_counts.most_common():
        print(f"    {cnt:8,d} : {ins}")


if __name__ == "__main__":
    main()
