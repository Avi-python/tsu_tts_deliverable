#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Adapted for SARC Voice Formosan multi-process CPU extraction on unique_wavs.
import argparse
import logging
import multiprocessing as mp
import os
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from tqdm import tqdm

_ORT_SESSION = None


def _init_worker(onnx_path: str):
    global _ORT_SESSION
    torch.set_num_threads(1)
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    option.inter_op_num_threads = 1
    option.log_severity_level = 3
    _ORT_SESSION = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=["CPUExecutionProvider"])


def _process_item(item):
    utt, wav_path = item
    try:
        audio, sample_rate = torchaudio.load(wav_path, backend="soundfile")
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        if audio.shape[1] > 10 * 16000:
            audio = audio[:, : 10 * 16000]
        feat = kaldi.fbank(audio, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        in_name = _ORT_SESSION.get_inputs()[0].name
        embedding = _ORT_SESSION.run(None, {in_name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        return utt, embedding
    except Exception as e:
        logging.warning(f"Failed to extract embedding for {utt} ({wav_path}): {e}")
        return utt, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--num_thread", type=int, default=32)
    parser.add_argument("--force", action="store_true", help="Overwrite existing .pt files")
    args = parser.parse_args()

    out_utt = os.path.join(args.dir, "utt2embedding.pt")
    out_spk = os.path.join(args.dir, "spk2embedding.pt")
    if os.path.exists(out_utt) and os.path.exists(out_spk) and not args.force:
        print(f"[extract_embedding] {out_utt} and {out_spk} already exist, skipping (use --force to overwrite).")
        return

    utt2wav, utt2spk = {}, {}
    with open(f"{args.dir}/wav.scp", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1]
    with open(f"{args.dir}/utt2spk", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2spk[parts[0]] = parts[1]

    items = list(utt2wav.items())
    utt2embedding, spk2embedding = {}, {}
    skipped = 0

    with mp.Pool(processes=args.num_thread, initializer=_init_worker, initargs=(args.onnx_path,)) as pool:
        iterator = tqdm(
            pool.imap_unordered(_process_item, items, chunksize=64),
            total=len(items),
            desc=f"Extracting CampPlus embeddings ({args.num_thread} procs)",
        )
        for utt, embedding in iterator:
            if embedding is None:
                skipped += 1
                continue
            utt2embedding[utt] = embedding
            spk = utt2spk[utt]
            if spk not in spk2embedding:
                spk2embedding[spk] = []
            spk2embedding[spk].append(embedding)

    for k, v in spk2embedding.items():
        spk2embedding[k] = torch.tensor(v).mean(dim=0).tolist()

    torch.save(utt2embedding, out_utt)
    torch.save(spk2embedding, out_spk)
    print(
        f"[extract_embedding] Saved {len(utt2embedding):,} utt embeddings and {len(spk2embedding):,} spk embeddings to {args.dir} (skipped={skipped})."
    )


if __name__ == "__main__":
    main()
