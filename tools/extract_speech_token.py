#!/usr/bin/env python3
# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu)
# Enhanced with multi-GPU sharded worker pool and cuDNN HEURISTIC conv search (~300x faster).
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import multiprocessing as mp
import os
import numpy as np
import onnxruntime
import torch
import torchaudio
from tqdm import tqdm
import whisper


def load_audio_16k(item):
    utt, wav_path = item
    try:
        audio, sample_rate = torchaudio.load(wav_path, backend="soundfile")
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio, sample_rate, 16000)
        if audio.shape[1] / 16000.0 > 30.0 or audio.shape[1] == 0:
            return utt, None
        return utt, audio
    except Exception as e:
        logging.warning(f"Error loading {utt} ({wav_path}): {e}")
        return utt, None


def gpu_worker(gpu_id: int, shard_items: list, onnx_path: str, num_threads: int, worker_idx: int):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 2
    option.log_severity_level = 3
    providers = [
        ("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
        "CPUExecutionProvider",
    ]
    ort_session = onnxruntime.InferenceSession(onnx_path, sess_options=option, providers=providers)
    in0 = ort_session.get_inputs()[0].name
    in1 = ort_session.get_inputs()[1].name

    res = {}
    with ThreadPoolExecutor(max_workers=num_threads) as io_pool:
        iterator = tqdm(
            io_pool.map(load_audio_16k, shard_items, chunksize=16),
            total=len(shard_items),
            desc=f"SpeechToken [GPU {gpu_id}]",
            position=worker_idx,
            leave=True,
        )
        for utt, audio in iterator:
            if audio is None:
                res[utt] = []
                continue
            try:
                feat = whisper.log_mel_spectrogram(audio.to(device), n_mels=128)
                speech_token = (
                    ort_session.run(
                        None,
                        {
                            in0: feat.detach().cpu().numpy(),
                            in1: np.array([feat.shape[2]], dtype=np.int32),
                        },
                    )[0]
                    .flatten()
                    .tolist()
                )
                res[utt] = speech_token
            except Exception as e:
                logging.warning(f"[GPU {gpu_id}] ORT error on {utt}: {e}")
                res[utt] = []
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--onnx_path", type=str, required=True)
    parser.add_argument("--num_thread", type=int, default=6, help="I/O prefetch threads per GPU worker")
    parser.add_argument("--gpus", type=str, default=None, help="Comma-separated GPU IDs, e.g. '0,1,2,3,4,5,6,7'")
    parser.add_argument("--force", action="store_true", help="Overwrite existing utt2speech_token.pt")
    args = parser.parse_args()

    out_path = os.path.join(args.dir, "utt2speech_token.pt")
    if os.path.exists(out_path) and not args.force:
        print(f"[extract_speech_token] {out_path} already exists, skipping (use --force to overwrite).")
        return

    utt2wav = {}
    with open(f"{args.dir}/wav.scp", "r", encoding="utf-8") as f:
        for l in f:
            parts = l.strip().split(maxsplit=1)
            if len(parts) == 2:
                utt2wav[parts[0]] = parts[1]

    items = list(utt2wav.items())
    if args.gpus:
        gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    elif os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu_list = [int(x.strip()) for x in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if x.strip() != ""]
    else:
        num_avail = torch.cuda.device_count()
        gpu_list = list(range(max(1, num_avail)))

    print(f"[extract_speech_token] Extracting {len(items):,} unique audio files across GPUs {gpu_list} ...")
    if len(gpu_list) == 1:
        utt2speech_token = gpu_worker(gpu_list[0], items, args.onnx_path, args.num_thread, 0)
    else:
        ctx = mp.get_context("spawn")
        shards = [items[i :: len(gpu_list)] for i in range(len(gpu_list))]
        with ctx.Pool(processes=len(gpu_list)) as pool:
            async_results = [
                pool.apply_async(gpu_worker, (gpu_list[i], shards[i], args.onnx_path, args.num_thread, i))
                for i in range(len(gpu_list))
            ]
            utt2speech_token = {}
            for ar in async_results:
                utt2speech_token.update(ar.get())

    non_empty = sum(1 for v in utt2speech_token.values() if len(v) > 0)
    torch.save(utt2speech_token, out_path)
    print(f"[extract_speech_token] Saved {len(utt2speech_token):,} entries ({non_empty:,} non-empty) to {out_path}.")


if __name__ == "__main__":
    main()
