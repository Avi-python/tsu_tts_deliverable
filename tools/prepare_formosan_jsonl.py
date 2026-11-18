#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare CosyVoice3 (SARC Voice) training & validation directories from Formosan JSONL files.
- Groups strictly by `audio_path` (using official splits in /nfs/DGX2/Workspace/avi/fb_nchc/splits
  when available, or deterministic hash split on `audio_path`) to prevent any audio leakage
  between train, dev, and test across original + IPA-inpainted copies (copy_A, copy_B, copy_C).
- Creates `unique_wavs/` so CampPlus embeddings and SpeechTokenizerV3 discrete tokens are
  extracted only ONCE per unique audio file (~4x speedup).
"""

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

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


def build_record_instruct(lang: str, raw_dialect: str, instruct_arg: str) -> str:
    template = instruct_arg if instruct_arg else DEFAULT_LANG_INSTRUCT.get(lang, "You are a helpful assistant.<|endofprompt|>")
    if lang == "pyu" or "{dialect}" in template:
        zh_dialect = PYU_DIALECT_MAP.get(str(raw_dialect or "").strip(), str(raw_dialect or "").strip())
        if "{dialect}" in template:
            return template.format(dialect=zh_dialect)
        elif zh_dialect:
            return f"You are a helpful assistant. 请用{zh_dialect}卑南語表达。<|endofprompt|>"
    return template


def sanitize_id(val: str) -> str:
    s = re.sub(r"\s+", "_", str(val).strip())
    s = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]", "_", s)
    return s or "unknown"


def load_official_splits(splits_dir: str, lang: str):
    dev_audios = set()
    test_audios = set()
    if not splits_dir or not os.path.isdir(splits_dir):
        return dev_audios, test_audios

    dev_path = os.path.join(splits_dir, "dev.jsonl")
    test_path = os.path.join(splits_dir, "test.jsonl")

    if os.path.exists(dev_path):
        with open(dev_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("lang") == lang and d.get("audio_path"):
                    dev_audios.add(d["audio_path"])

    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if d.get("lang") == lang and d.get("audio_path"):
                    test_audios.add(d["audio_path"])

    return dev_audios, test_audios


def write_kaldi_dir(des_dir: str, records: list, instruct_text: str):
    os.makedirs(des_dir, exist_ok=True)
    spk2utt = defaultdict(list)

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


def main():
    parser = argparse.ArgumentParser(description="Prepare Formosan JSONL data for CosyVoice3 training.")
    parser.add_argument("--input_jsonl", required=True, help="Path to *_merged_all.jsonl")
    parser.add_argument("--out_dir", required=True, help="Output base directory for language (e.g. data/tsu)")
    parser.add_argument("--lang", required=True, help="Language code (pyu, trv, tsu)")
    parser.add_argument(
        "--splits_dir",
        default="/nfs/DGX2/Workspace/avi/fb_nchc/splits",
        help="Directory containing official dev.jsonl and test.jsonl splits",
    )
    parser.add_argument(
        "--instruct",
        default="",
        help="Optional CosyVoice3 instruct prompt override (supports {dialect} placeholder; defaults to language/dialect specific prompt)",
    )
    parser.add_argument("--min_duration", type=float, default=0.5, help="Minimum audio duration in seconds")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration in seconds (<=30s for Whisper)")
    parser.add_argument("--dev_keep_augmented", action="store_true", help="Keep copy_A/B/C in dev set (default: keep only 1st/orig record per wav in dev)")
    args = parser.parse_args()

    dev_audios, test_audios = load_official_splits(args.splits_dir, args.lang)
    use_official_splits = len(dev_audios) > 0

    print(f"[{args.lang}] Reading {args.input_jsonl} ...")
    print(f"[{args.lang}] Official splits loaded: dev={len(dev_audios)} wavs, test={len(test_audios)} wavs (excluded)")

    # Pass 1: Read all valid records and map each audio_path to its first (un-inpainted) clean_text.
    # Because clean_text.py already normalizes and strips punctuation, the first clean_text for each
    # audio_path is the canonical orthographic sentence before ARPAbet/IPA inpainting.
    raw_records = []
    audio_to_orig_text = {}

    skipped_duration = 0
    skipped_empty = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            audio_path = d.get("audio_path", "").strip()
            text = re.sub(r"\s+", " ", str(d.get("clean_text", "")).strip())
            duration = float(d.get("duration") or 0.0)

            if not audio_path or not text:
                skipped_empty += 1
                continue

            if duration < args.min_duration or duration > args.max_duration:
                skipped_duration += 1
                continue

            if audio_path not in audio_to_orig_text:
                audio_to_orig_text[audio_path] = text

            raw_records.append((audio_path, text, d))

    # Build text-level split sets (test > dev > train) so that if A, B, C share the same
    # original clean_text, ALL of A, B, C (and their inp1..3 copies) are assigned to the same split.
    test_texts = set()
    dev_texts = set()
    if use_official_splits:
        for audio_path, orig_text in audio_to_orig_text.items():
            if audio_path in test_audios:
                test_texts.add(orig_text)
        for audio_path, orig_text in audio_to_orig_text.items():
            if audio_path in dev_audios and orig_text not in test_texts:
                dev_texts.add(orig_text)
    print(
        f"[{args.lang}] Canonical text groups in splits: "
        f"test_texts={len(test_texts):,}, dev_texts={len(dev_texts):,}"
    )

    audio_to_wav_id = {}
    audio_to_spk_id = {}
    audio_variant_count = defaultdict(int)

    # Pre-load existing unique_wavs/wav.scp if present so wav_id keys remain 100% compatible
    # with any already-extracted utt2embedding.pt and utt2speech_token.pt in unique_wavs/
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
        print(f"[{args.lang}] Reusing {len(audio_to_wav_id):,} wav_id mappings from existing {existing_wav_scp}")

    train_records = []
    dev_records = []

    skipped_test_audio = 0
    skipped_test_text_leak = 0
    moved_dev_text_leak = 0

    for audio_path, text, d in raw_records:
        orig_text = audio_to_orig_text[audio_path]

        if use_official_splits:
            if audio_path in test_audios:
                skipped_test_audio += 1
                continue
            if orig_text in test_texts:
                skipped_test_text_leak += 1
                continue
            is_dev = (audio_path in dev_audios) or (orig_text in dev_texts)
            if is_dev and (audio_path not in dev_audios):
                moved_dev_text_leak += 1
        else:
            # Group by canonical orig_text hash so same-text recordings never split across train/dev
            h = int(hashlib.md5(orig_text.encode("utf-8")).hexdigest()[:8], 16)
            is_dev = (h % 100) < 2  # 2% dev by unique orig_text

        if audio_path not in audio_to_wav_id:
            idx = len(audio_to_wav_id)
            audio_to_wav_id[audio_path] = f"{args.lang}_wav_{idx:06d}"

            raw_spk = d.get("speaker")
            if raw_spk and str(raw_spk).strip():
                spk_id = f"{args.lang}_spk_{sanitize_id(raw_spk)}"
            else:
                corpus = sanitize_id(d.get("corpus") or "corp")
                dialect = sanitize_id(d.get("dialect") or "default")
                spk_id = f"{args.lang}_{corpus}_{dialect}"
            audio_to_spk_id[audio_path] = spk_id

        wav_id = audio_to_wav_id[audio_path]
        spk_id = audio_to_spk_id[audio_path]
        v_idx = audio_variant_count[audio_path]
        audio_variant_count[audio_path] += 1

        # For dev, default to keeping only the first (original orthographic) entry per unique wav
        # and only for wavs originally in dev_audios (preventing train-to-dev text leakage while keeping dev size clean)
        if is_dev:
            if use_official_splits and (audio_path not in dev_audios):
                continue
            if (not args.dev_keep_augmented) and v_idx > 0:
                continue

        utt_id = f"{wav_id}_v{v_idx}"
        rec_instruct = build_record_instruct(args.lang, d.get("dialect"), args.instruct)
        rec = {
            "utt_id": utt_id,
            "wav_id": wav_id,
            "audio_path": audio_path,
            "text": text,
            "spk_id": spk_id,
            "instruct": rec_instruct,
        }
        if is_dev:
            dev_records.append(rec)
        else:
            train_records.append(rec)

    # Write unique_wavs directory for 1x feature extraction
    os.makedirs(unique_dir, exist_ok=True)
    with (
        open(os.path.join(unique_dir, "wav.scp"), "w", encoding="utf-8") as f_wav,
        open(os.path.join(unique_dir, "utt2spk"), "w", encoding="utf-8") as f_u2s,
    ):
        for audio_path, wav_id in audio_to_wav_id.items():
            spk_id = audio_to_spk_id[audio_path]
            f_wav.write(f"{wav_id} {audio_path}\n")
            f_u2s.write(f"{wav_id} {spk_id}\n")

    write_kaldi_dir(os.path.join(args.out_dir, "train"), train_records, args.instruct)
    write_kaldi_dir(os.path.join(args.out_dir, "dev"), dev_records, args.instruct)

    print(f"[{args.lang}] Preparation complete -> {args.out_dir}")
    print(f"  Unique audio files (for 1x feature extraction): {len(audio_to_wav_id):,}")
    print(f"  Train records (orig + copy_A/B/C)             : {len(train_records):,}")
    print(f"  Dev records                                   : {len(dev_records):,}")
    print(f"  Unique speaker IDs                            : {len(set(audio_to_spk_id.values())):,}")
    print(
        f"  Filtered out: test_audio={skipped_test_audio:,}, test_text_leak={skipped_test_text_leak:,}, "
        f"dev_text_leak_removed_from_train={moved_dev_text_leak:,}, duration={skipped_duration:,}, empty={skipped_empty:,}"
    )
    instruct_counts = Counter(r["instruct"] for r in train_records)
    print(f"  Train instruct distribution:")
    for ins, cnt in instruct_counts.most_common():
        print(f"    {cnt:8,d} : {ins}")


if __name__ == "__main__":
    main()
