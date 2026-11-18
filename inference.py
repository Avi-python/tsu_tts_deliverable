#!/usr/bin/env python3
"""Standard CosyVoice3 TTS inference CLI (single/multi-text, multi-seed, and batch JSONL)."""

import argparse
import functools
import logging
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent
MATCHA_DIR = ROOT_DIR / "third_party" / "Matcha-TTS"
for _path in (str(MATCHA_DIR), str(ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DEFAULT_TTS_ROOT = Path("/nfs/DGX2/Workspace/avi/fb_nchc/tts")
DEFAULT_PROMPT_WAV = "/nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav"
DEFAULT_PROMPT_TEXT = "a panto pangka to kiosicumia."
DEFAULT_INSTRUCT_PREFIX = "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>"


def _auto_select_gpu() -> None:
    """Select the GPU with the most free VRAM if CUDA_VISIBLE_DEVICES is unset."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[GPU] Using CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        candidates = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                candidates.append((int(parts[0]), int(parts[1])))
        if candidates:
            best_gpu = max(candidates, key=lambda x: x[1])[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
            print(f"[GPU] Auto-selected CUDA_VISIBLE_DEVICES={best_gpu}")
    except Exception as exc:
        print(f"[GPU] Notice: Could not auto-detect GPU via nvidia-smi: {exc}")


def _extract_text(record: dict, text_field: Optional[str]) -> str:
    """Extract target synthesis text from a JSONL record."""
    if text_field:
        val = record.get(text_field)
        if isinstance(val, str) and val.strip():
            return val.strip()
        raise KeyError(f"Field '{text_field}' missing or empty in JSONL record")
    for key in ("clean_text", "text", "target_text"):
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise KeyError("JSONL record has no non-empty 'clean_text', 'text', or 'target_text'")


def _apply_sampling_params(
    model, top_p: float, top_k: int, win_size: int, tau_r: float
) -> None:
    """Configure ras_sampling hyperparameters on the loaded CosyVoice3 LLM."""
    try:
        from cosyvoice.utils.common import ras_sampling

        model.model.llm.sampling = functools.partial(
            ras_sampling,
            top_p=top_p,
            top_k=top_k,
            win_size=win_size,
            tau_r=tau_r,
        )
    except Exception as exc:
        print(f"[Warning] Failed to apply custom ras_sampling parameters: {exc}")


def _synthesize_to_file(
    model,
    set_seed_fn,
    torchaudio_mod,
    torch_mod,
    *,
    mode: str,
    text: str,
    seed: int,
    prompt_wav: str,
    prompt_text: str,
    instruct_prefix: str,
    output_path: Path,
) -> Path:
    """Run one seeded synthesis pass and save waveform to `output_path`."""
    set_seed_fn(seed)
    if mode == "zero_shot":
        full_prompt = (
            prompt_text
            if "<|endofprompt|>" in prompt_text
            else f"{instruct_prefix}{prompt_text}"
        )
        outputs = list(
            model.inference_zero_shot(
                text,
                full_prompt,
                prompt_wav,
                stream=False,
                text_frontend=False,
            )
        )
    elif mode == "instruct2":
        outputs = list(
            model.inference_instruct2(
                text,
                instruct_prefix,
                prompt_wav,
                stream=False,
                text_frontend=False,
            )
        )
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if not outputs:
        raise RuntimeError(f"CosyVoice3 returned no audio chunks for text: {text!r}")

    waveform = torch_mod.cat([out["tts_speech"].cpu() for out in outputs], dim=-1)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torchaudio_mod.save(str(output_path), waveform, model.sample_rate)
    return output_path.resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standard CosyVoice3 TTS inference (single/multi-text, multi-seed, batch JSONL)."
    )
    parser.add_argument("--lang", type=str, default="tsu", help="Language code (default: tsu)")
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=None,
        help="Explicit CosyVoice3 model directory (overrides --tts_root / --lang)",
    )
    parser.add_argument("--tts_root", type=Path, default=DEFAULT_TTS_ROOT)
    parser.add_argument("--tts_model_name", type=str, default=None)

    parser.add_argument("--mode", choices=["zero_shot", "instruct2"], default="zero_shot")
    parser.add_argument("--prompt_wav", type=str, default=DEFAULT_PROMPT_WAV)
    parser.add_argument("--prompt_text", type=str, default=DEFAULT_PROMPT_TEXT)
    parser.add_argument("--instruct_prefix", type=str, default=DEFAULT_INSTRUCT_PREFIX)

    # Text / Batch input
    parser.add_argument(
        "--text",
        nargs="+",
        default=None,
        help="One or more target utterances to synthesize",
    )
    parser.add_argument(
        "--input_jsonl",
        type=Path,
        default=None,
        help="Path to input .jsonl file for batch inference",
    )
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument(
        "--filter_lang",
        type=str,
        default=None,
        help="Optional language filter when reading multi-language JSONL",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--use_record_audio_as_prompt",
        action="store_true",
        help="In batch mode, use each record's audio_path and text as prompt if available",
    )

    # Seeding & Output
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="One or more random seeds (e.g. --seeds 0 42 1234 2026)",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/inference"))

    # LLM sampling hyperparameters
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=25)
    parser.add_argument("--win_size", type=int, default=10)
    parser.add_argument("--tau_r", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input_jsonl:
        raise SystemExit("Provide either --text (CLI sentences) or --input_jsonl (batch mode).")

    _auto_select_gpu()

    import json
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel
    from cosyvoice.filtering.model_resolver import resolve_tts_model_dir
    from cosyvoice.utils.common import set_all_random_seed

    logging.root.handlers.clear()
    logging.basicConfig(level=logging.CRITICAL)

    model_dir = (
        Path(args.model_dir)
        if args.model_dir is not None
        else resolve_tts_model_dir(args.tts_root, args.lang, model_name=args.tts_model_name)
    )

    print(f"Loading CosyVoice3 model: {model_dir}")
    model = AutoModel(model_dir=str(model_dir))
    _apply_sampling_params(model, args.top_p, args.top_k, args.win_size, args.tau_r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str, str, str]] = []

    if args.text:
        for idx, utterance in enumerate(args.text):
            stem = f"{args.lang}_utt_{idx:03d}"
            items.append((stem, utterance, args.prompt_wav, args.prompt_text))
    else:
        assert args.input_jsonl is not None
        matched: list[tuple[int, dict]] = []
        with args.input_jsonl.open("r", encoding="utf-8") as fin:
            for line_num, raw_line in enumerate(fin, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if args.filter_lang and record.get("lang") != args.filter_lang:
                    continue
                matched.append((line_num, record))

        end_idx = None if args.limit is None else args.start_idx + args.limit
        for idx, (line_num, record) in enumerate(matched[args.start_idx : end_idx]):
            target_text = _extract_text(record, args.text_field)
            rec_audio = record.get("audio_path")
            if args.use_record_audio_as_prompt and rec_audio and Path(rec_audio).is_file():
                p_wav = rec_audio
                p_text = target_text
            else:
                p_wav = record.get("prompt_wav") or args.prompt_wav
                p_text = record.get("prompt_text") or args.prompt_text
            stem = str(record.get("id") or f"{args.lang}_line_{line_num:05d}_{idx:04d}")
            items.append((stem, target_text, p_wav, p_text))

    print(f"Synthesizing {len(items)} item(s) x {len(args.seeds)} seed(s) {args.seeds} -> {args.output_dir.resolve()}")
    for item_idx, (stem, target_text, p_wav, p_text) in enumerate(items, start=1):
        print(f"\n[{item_idx}/{len(items)}] Text: {target_text}")
        for seed in args.seeds:
            out_name = f"{stem}_seed{seed}.wav" if len(args.seeds) > 1 else f"{stem}.wav"
            saved = _synthesize_to_file(
                model,
                set_all_random_seed,
                torchaudio,
                torch,
                mode=args.mode,
                text=target_text,
                seed=seed,
                prompt_wav=p_wav,
                prompt_text=p_text,
                instruct_prefix=args.instruct_prefix,
                output_path=args.output_dir / out_name,
            )
            print(f"  [Seed {seed}] file://{saved}")


if __name__ == "__main__":
    main()
