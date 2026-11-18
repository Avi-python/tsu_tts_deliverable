#!/usr/bin/env python3
"""CLI for closed-loop CosyVoice3 TTS + Whisper ASR CER filtering inference."""

import argparse
from dataclasses import asdict
import json
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

from cosyvoice.filtering.config import (
    DEFAULT_ASR_ROOT,
    DEFAULT_TTS_ROOT,
    FilterConfig,
)

DEFAULT_INSTRUCT_PREFIX = "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>"


def _auto_select_gpu() -> None:
    """Select the GPU with the most free VRAM if CUDA_VISIBLE_DEVICES is unset."""
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
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
    except Exception:
        pass


def _extract_text(record: dict, text_field: Optional[str]) -> str:
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


def _extract_optional_text(record: dict, field: Optional[str]) -> Optional[str]:
    if not field:
        return None
    val = record.get(field)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter CosyVoice3 TTS outputs with Whisper ASR CER and seed retries."
    )
    parser.add_argument("--lang", type=str, default="tsu", help="Language code (default: tsu)")
    parser.add_argument("--asr_root", type=Path, default=DEFAULT_ASR_ROOT)
    parser.add_argument("--tts_root", type=Path, default=DEFAULT_TTS_ROOT)
    parser.add_argument("--asr_model_name", type=str, default=None)
    parser.add_argument("--tts_model_name", type=str, default=None)
    parser.add_argument(
        "--asr_model_dir",
        type=Path,
        default=None,
        help="Explicit Whisper ASR model directory (overrides --asr_root / --lang)",
    )
    parser.add_argument(
        "--tts_model_dir",
        type=Path,
        default=None,
        help="Explicit CosyVoice3 TTS model directory (overrides --tts_root / --lang)",
    )

    parser.add_argument("--cer_threshold", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=5)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--tts_mode", choices=["zero_shot", "instruct2"], default="zero_shot"
    )
    parser.add_argument("--prompt_wav", type=str, default=None)
    parser.add_argument("--prompt_text", type=str, default=None)
    parser.add_argument(
        "--instruct_prefix",
        type=str,
        default=DEFAULT_INSTRUCT_PREFIX,
    )
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--discard_on_fail", action="store_true")
    parser.add_argument("--device", type=str, default=None)

    # Single-text mode
    parser.add_argument("--text", type=str, default=None, help="Single utterance reference text")
    parser.add_argument(
        "--tts_text",
        type=str,
        default=None,
        help="Optional synthesis text (e.g. phoneme inpainting); defaults to --text",
    )
    parser.add_argument("--output_wav", type=Path, default=None)

    # Batch JSONL mode
    parser.add_argument("--input_jsonl", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--output_jsonl", type=Path, default=None)
    parser.add_argument("--text_field", type=str, default=None)
    parser.add_argument(
        "--tts_text_field",
        type=str,
        default=None,
        help="Optional JSONL field for TTS synthesis text; falls back to reference text",
    )
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text and not args.input_jsonl:
        raise SystemExit("Provide either --text (single mode) or --input_jsonl (batch mode).")

    _auto_select_gpu()

    from cosyvoice.filtering.pipeline import TTSFilterPipeline

    config = FilterConfig(
        lang=args.lang,
        asr_root=args.asr_root,
        tts_root=args.tts_root,
        asr_model_name=args.asr_model_name,
        tts_model_name=args.tts_model_name,
        asr_model_dir=args.asr_model_dir,
        tts_model_dir=args.tts_model_dir,
        cer_threshold=args.cer_threshold,
        max_retries=args.max_retries,
        base_seed=args.base_seed,
        tts_mode=args.tts_mode,
        default_prompt_wav=args.prompt_wav,
        default_prompt_text=args.prompt_text,
        instruct_prefix=args.instruct_prefix,
        normalize_text=not args.no_normalize,
        keep_best_on_fail=not args.discard_on_fail,
        device=args.device,
    )
    pipeline = TTSFilterPipeline.from_config(config)

    if args.text:
        out_wav = args.output_wav or (args.output_dir / f"{args.lang}_sample.wav")
        res = pipeline.filter_utterance(
            text=args.text,
            output_wav_path=out_wav,
            tts_text=args.tts_text,
        )
        summary = {
            "text": res.text,
            "tts_input_text": res.tts_text,
            "passed": res.passed,
            "best_cer": round(res.best_attempt.cer, 4),
            "best_seed": res.best_attempt.seed,
            "attempts": len(res.attempts),
            "hypothesis": res.best_attempt.hypothesis,
            "audio_path": res.best_attempt.audio_path,
            "attempt_history": [asdict(a) for a in res.attempts],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    assert args.input_jsonl is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_jsonl or (args.output_dir / "filtered_results.jsonl")

    total = 0
    passed_count = 0
    cer_sum = 0.0

    with (
        args.input_jsonl.open("r", encoding="utf-8") as fin,
        out_jsonl.open("w", encoding="utf-8") as fout,
    ):
        for idx, line in enumerate(fin):
            if args.limit is not None and total >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            target_text = _extract_text(item, args.text_field)
            item_tts_text = _extract_optional_text(item, args.tts_text_field)
            item_prompt_wav = item.get("prompt_wav") or args.prompt_wav
            item_prompt_text = item.get("prompt_text") or args.prompt_text
            stem = str(item.get("id") or f"{args.lang}_{idx:05d}")
            out_wav = args.output_dir / f"{stem}.wav"

            res = pipeline.filter_utterance(
                text=target_text,
                output_wav_path=out_wav,
                prompt_wav=item_prompt_wav,
                prompt_text=item_prompt_text,
                tts_text=item_tts_text,
            )

            total += 1
            if res.passed:
                passed_count += 1
            cer_sum += res.best_attempt.cer

            out_item = dict(item)
            out_item.update(
                {
                    "tts_audio": res.best_attempt.audio_path,
                    "tts_input_text": res.tts_text,
                    "asr_hypothesis": res.best_attempt.hypothesis,
                    "cer": round(res.best_attempt.cer, 4),
                    "seed": res.best_attempt.seed,
                    "attempts": len(res.attempts),
                    "passed": res.passed,
                    "attempt_history": [asdict(a) for a in res.attempts],
                }
            )
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")

    mean_cer = (cer_sum / total) if total else 0.0
    print(
        f"Done: {passed_count}/{total} passed "
        f"(mean best CER={mean_cer:.4f}). Output JSONL: {out_jsonl}"
    )


if __name__ == "__main__":
    main()
