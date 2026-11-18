#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Adapted for SARC Voice Formosan deduplicated unique_wavs lookup and audio bytes caching.
import argparse
import json
import logging
import multiprocessing
import os
import time
import pandas as pd
import torch


def job(utt_list, parquet_file, utt2parquet_file, spk2parquet_file):
    start_time = time.time()
    # Filter out any utts missing valid embeddings or speech tokens
    valid_utts = []
    for utt in utt_list:
        wid = utt2wav_id.get(utt, utt)
        spk = utt2spk[utt]
        if utt2embedding is not None and wid not in utt2embedding:
            continue
        if spk2embedding is not None and spk not in spk2embedding:
            continue
        if utt2speech_token is not None and (wid not in utt2speech_token or len(utt2speech_token[wid]) == 0):
            continue
        if not os.path.exists(utt2wav[utt]):
            continue
        valid_utts.append(utt)

    # Cache last read audio_path bytes since v0, v1, v2, v3 of the same audio are adjacent
    data_list = []
    last_wav_path = None
    last_wav_bytes = None
    for utt in valid_utts:
        wpath = utt2wav[utt]
        if wpath != last_wav_path:
            with open(wpath, "rb") as f:
                last_wav_bytes = f.read()
            last_wav_path = wpath
        data_list.append(last_wav_bytes)

    wav_list = [utt2wav[utt] for utt in valid_utts]
    text_list = [utt2text[utt] for utt in valid_utts]
    spk_list = [utt2spk[utt] for utt in valid_utts]

    df = pd.DataFrame()
    df["utt"] = valid_utts
    df["wav"] = wav_list
    df["audio_data"] = data_list
    df["text"] = text_list
    df["spk"] = spk_list
    if utt2embedding is not None:
        df["utt_embedding"] = [utt2embedding[utt2wav_id.get(utt, utt)] for utt in valid_utts]
    if spk2embedding is not None:
        df["spk_embedding"] = [spk2embedding[utt2spk[utt]] for utt in valid_utts]
    if utt2speech_token is not None:
        df["speech_token"] = [utt2speech_token[utt2wav_id.get(utt, utt)] for utt in valid_utts]
    if utt2instruct is not None:
        df["instruct"] = [utt2instruct[utt] for utt in valid_utts]
    if args.dpo:
        df["reject_speech_token"] = [utt2reject_speech_token.get(utt, None) for utt in valid_utts]

    df.to_parquet(parquet_file)
    with open(utt2parquet_file, "w", encoding="utf-8") as f:
        json.dump({k: parquet_file for k in valid_utts}, f, ensure_ascii=False, indent=2)
    with open(spk2parquet_file, "w", encoding="utf-8") as f:
        json.dump({k: parquet_file for k in list(set(spk_list))}, f, ensure_ascii=False, indent=2)
    logging.info(f"Wrote {parquet_file} ({len(valid_utts)} utts) in {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_utts_per_parquet", type=int, default=1000, help="num utts per parquet")
    parser.add_argument("--num_processes", type=int, default=8, help="num processes for make parquets")
    parser.add_argument("--src_dir", type=str, required=True)
    parser.add_argument("--des_dir", type=str, required=True)
    parser.add_argument(
        "--unique_dir",
        type=str,
        default=None,
        help="Optional unique_wavs directory containing deduplicated utt2embedding.pt, spk2embedding.pt, utt2speech_token.pt",
    )
    parser.add_argument("--dpo", action="store_true", default=False, help="Use Direct Preference Optimization")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    os.makedirs(args.des_dir, exist_ok=True)
    utt2wav, utt2text, utt2spk, utt2wav_id = {}, {}, {}, {}
    with open(f"{args.src_dir}/wav.scp", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1]
    with open(f"{args.src_dir}/text", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2text[parts[0]] = parts[1]
    with open(f"{args.src_dir}/utt2spk", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2spk[parts[0]] = parts[1]

    utt2wav_id_path = f"{args.src_dir}/utt2wav_id"
    if os.path.exists(utt2wav_id_path):
        with open(utt2wav_id_path, "r", encoding="utf-8") as f:
            for l in f:
                parts = l.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt2wav_id[parts[0]] = parts[1]
    else:
        utt2wav_id = {u: u for u in utt2wav.keys()}

    if os.path.exists(f"{args.src_dir}/instruct"):
        utt2instruct = {}
        with open(f"{args.src_dir}/instruct", "r", encoding="utf-8") as f:
            for l in f:
                parts = l.strip().split(maxsplit=1)
                if len(parts) == 2:
                    utt2instruct[parts[0]] = parts[1]
    else:
        utt2instruct = None

    feat_dir = args.unique_dir if args.unique_dir else args.src_dir
    emb_path = f"{feat_dir}/utt2embedding.pt"
    spk_emb_path = f"{feat_dir}/spk2embedding.pt"
    tok_path = f"{feat_dir}/utt2speech_token.pt"

    print(f"[make_parquet_list] Loading features from {feat_dir} ...")
    utt2embedding = torch.load(emb_path) if os.path.exists(emb_path) else None
    spk2embedding = torch.load(spk_emb_path) if os.path.exists(spk_emb_path) else None
    utt2speech_token = torch.load(tok_path) if os.path.exists(tok_path) else None
    if args.dpo:
        utt2reject_speech_token = (
            torch.load(f"{args.src_dir}_reject/utt2speech_token.pt")
            if os.path.exists(f"{args.src_dir}_reject/utt2speech_token.pt")
            else {}
        )
    utts = list(utt2wav.keys())
    print(f"[make_parquet_list] Building Parquet shards for {len(utts):,} utterances into {args.des_dir} ...")

    pool = multiprocessing.Pool(processes=args.num_processes)
    parquet_list, utt2parquet_list, spk2parquet_list = [], [], []
    for i, j in enumerate(range(0, len(utts), args.num_utts_per_parquet)):
        parquet_file = os.path.join(args.des_dir, f"parquet_{i:09d}.tar")
        utt2parquet_file = os.path.join(args.des_dir, f"utt2parquet_{i:09d}.json")
        spk2parquet_file = os.path.join(args.des_dir, f"spk2parquet_{i:09d}.json")
        parquet_list.append(parquet_file)
        utt2parquet_list.append(utt2parquet_file)
        spk2parquet_list.append(spk2parquet_file)
        pool.apply_async(job, (utts[j : j + args.num_utts_per_parquet], parquet_file, utt2parquet_file, spk2parquet_file))
    pool.close()
    pool.join()

    with (
        open(f"{args.des_dir}/data.list", "w", encoding="utf8") as f1,
        open(f"{args.des_dir}/utt2data.list", "w", encoding="utf8") as f2,
        open(f"{args.des_dir}/spk2data.list", "w", encoding="utf8") as f3,
    ):
        for name in parquet_list:
            f1.write(name + "\n")
        for name in utt2parquet_list:
            f2.write(name + "\n")
        for name in spk2parquet_list:
            f3.write(name + "\n")
    print(f"[make_parquet_list] Done! Wrote {len(parquet_list)} shards to {args.des_dir}/data.list")
