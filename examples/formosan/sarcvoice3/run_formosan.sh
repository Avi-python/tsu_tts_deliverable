#!/bin/bash
# SARC Voice (CosyVoice 3) Formosan Training Pipeline
# Supports training three indigenous languages: tsu (Tsou), pyu (Puyuma), trv (Truku)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/Matcha-TTS:${PYTHONPATH:-}"

# User's own cloned conda environment
PYTHON_BIN="${PYTHON_BIN:-/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python}"

# Configurable parameters
stage="${stage:-0}"
stop_stage="${stop_stage:-6}"
langs="${langs:-tsu}"  # Default to tsu first; set langs="tsu pyu trv" for all three
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9}"

DATA_JSONL_DIR="${DATA_JSONL_DIR:-/nfs/DGX2/Workspace/avi/fb_nchc/tts/v2_arpa_jsonl}"
INPUT_JSONL_OVERRIDE="${INPUT_JSONL:-}"
SPLITS_DIR="${SPLITS_DIR:-}"  # Optional external splits dir; leave empty for self-contained 90/5/5 split
DEV_RATIO="${DEV_RATIO:-0.05}"
TEST_RATIO="${TEST_RATIO:-0.05}"
SPLIT_SEED="${SPLIT_SEED:-42}"
STORAGE_ROOT="${STORAGE_ROOT:-/nfs/DGX2/Workspace/avi/fb_nchc/tts/sarcvoice_exp}"
if [ -z "${PRETRAINED_DIR:-}" ]; then
  if [ -d "${REPO_ROOT}/pretrained_models/台語CV3模型" ]; then
    PRETRAINED_DIR="${REPO_ROOT}/pretrained_models/台語CV3模型"
  else
    PRETRAINED_DIR="${REPO_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B-2512"
  fi
fi
CONFIG_FILE="${REPO_ROOT}/examples/formosan/sarcvoice3/conf/cosyvoice3_formosan.yaml"
DS_CONFIG="${REPO_ROOT}/examples/formosan/sarcvoice3/conf/ds_stage2.json"

train_engine="${train_engine:-torch_ddp}"
num_workers="${num_workers:-4}"
prefetch="${prefetch:-100}"
average_num="${average_num:-5}"

mkdir -p "${STORAGE_ROOT}/data" "${STORAGE_ROOT}/exp" "${STORAGE_ROOT}/tensorboard" "${STORAGE_ROOT}/exported_models"

for lang in ${langs}; do
  if [ -n "${INPUT_JSONL_OVERRIDE}" ]; then
    INPUT_JSONL="${INPUT_JSONL_OVERRIDE}"
  elif [ -f "${DATA_JSONL_DIR}/fb_nchc_clean_${lang}_filtered_merged_all.jsonl" ]; then
    INPUT_JSONL="${DATA_JSONL_DIR}/fb_nchc_clean_${lang}_filtered_merged_all.jsonl"
  elif [ -f "${DATA_JSONL_DIR}/fb_nchc_clean_filtered_${lang}_merged_all.jsonl" ]; then
    INPUT_JSONL="${DATA_JSONL_DIR}/fb_nchc_clean_filtered_${lang}_merged_all.jsonl"
  else
    INPUT_JSONL="${DATA_JSONL_DIR}/${lang}.jsonl"
  fi
  LANG_DATA_DIR="${STORAGE_ROOT}/data/${lang}"
  UNIQUE_WAV_DIR="${LANG_DATA_DIR}/unique_wavs"

  echo "================================================================================"
  echo " [SARC Voice 3] Processing language: ${lang}"
  echo "   Input JSONL : ${INPUT_JSONL}"
  echo "   Storage Dir : ${STORAGE_ROOT} (data/${lang}, exp/${lang})"
  echo "   GPUs        : ${CUDA_VISIBLE_DEVICES}"
  echo "================================================================================"

  # Stage 0: Prepare train/dev/test splits and Kaldi directories from single JSONL
  if [ "${stage}" -le 0 ] && [ "${stop_stage}" -ge 0 ]; then
    if [ "${lang}" = "tsu" ]; then
      DEFAULT_INSTRUCT="You are a helpful assistant. 请用鄒語表达。<|endofprompt|>"
    elif [ "${lang}" = "trv" ]; then
      DEFAULT_INSTRUCT="You are a helpful assistant. 请用太魯閣語表达。<|endofprompt|>"
    else
      DEFAULT_INSTRUCT="You are a helpful assistant. 请用{dialect}卑南語表达。<|endofprompt|>"
    fi
    INSTRUCT_PROMPT="${INSTRUCT_PROMPT:-${DEFAULT_INSTRUCT}}"
    echo "[Stage 0][${lang}] Preparing train/dev/test splits from ${INPUT_JSONL} (dev=${DEV_RATIO}, test=${TEST_RATIO}, seed=${SPLIT_SEED}) ..."
    PREP_ARGS=(
      --input_jsonl "${INPUT_JSONL}"
      --out_dir "${LANG_DATA_DIR}"
      --lang "${lang}"
      --dev_ratio "${DEV_RATIO}"
      --test_ratio "${TEST_RATIO}"
      --seed "${SPLIT_SEED}"
      --instruct "${INSTRUCT_PROMPT}"
    )
    if [ -n "${SPLITS_DIR}" ]; then
      PREP_ARGS+=(--splits_dir "${SPLITS_DIR}")
    fi
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/prepare_formosan_jsonl.py" "${PREP_ARGS[@]}"
  fi

  # Stage 1: Extract CampPlus speaker/utterance embeddings ONCE per unique audio file
  if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "[Stage 1][${lang}] Extracting CampPlus embeddings on ${UNIQUE_WAV_DIR} ..."
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/extract_embedding.py" \
      --dir "${UNIQUE_WAV_DIR}" \
      --onnx_path "${PRETRAINED_DIR}/campplus.onnx" \
      --num_thread 48
  fi

  # Stage 2: Extract SpeechTokenizerV3 discrete speech tokens ONCE per unique audio file across GPUs
  if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "[Stage 2][${lang}] Extracting SpeechTokenizerV3 tokens on ${UNIQUE_WAV_DIR} across GPUs ${CUDA_VISIBLE_DEVICES} ..."
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/extract_speech_token.py" \
      --dir "${UNIQUE_WAV_DIR}" \
      --onnx_path "${PRETRAINED_DIR}/speech_tokenizer_v3.onnx" \
      --gpus "${CUDA_VISIBLE_DEVICES}" \
      --num_thread 6
  fi

  # Stage 3: Build Parquet shards for train and dev (referencing unique_wavs features)
  if [ "${stage}" -le 3 ] && [ "${stop_stage}" -ge 3 ]; then
    for part in dev train; do
      echo "[Stage 3][${lang}] Building Parquet shards for ${part} ..."
      mkdir -p "${LANG_DATA_DIR}/${part}/parquet"
      "${PYTHON_BIN}" "${REPO_ROOT}/tools/make_parquet_list.py" \
        --num_utts_per_parquet 1000 \
        --num_processes 24 \
        --src_dir "${LANG_DATA_DIR}/${part}" \
        --unique_dir "${UNIQUE_WAV_DIR}" \
        --des_dir "${LANG_DATA_DIR}/${part}/parquet"
      cp "${LANG_DATA_DIR}/${part}/parquet/data.list" "${LANG_DATA_DIR}/${part}.data.list"
    done
  fi

  # Stage 5: Train LLM starting from Fun-CosyVoice3-0.5B RL checkpoint (or auto-resume latest checkpoint)
  if [ "${stage}" -le 5 ] && [ "${stop_stage}" -ge 5 ]; then
    num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F "," '{print NF}')
    MODEL_EXP_DIR="${STORAGE_ROOT}/exp/${lang}/llm/${train_engine}"
    TB_DIR="${STORAGE_ROOT}/tensorboard/${lang}/llm/${train_engine}"
    mkdir -p "${MODEL_EXP_DIR}" "${TB_DIR}"

    # Auto-detect the most recent epoch/step checkpoint if CHECKPOINT is not explicitly set
    if [ -n "${CHECKPOINT:-}" ]; then
      RESUME_CKPT="${CHECKPOINT}"
    else
      LATEST_CKPT=$(ls -t "${MODEL_EXP_DIR}"/epoch_*.pt 2>/dev/null | head -n 1 || true)
      if [ -n "${LATEST_CKPT}" ]; then
        RESUME_CKPT="${LATEST_CKPT}"
      else
        RESUME_CKPT="${PRETRAINED_DIR}/llm.rl.pt"
      fi
    fi

    echo "[Stage 5][${lang}] Starting LLM training on ${num_gpus} GPUs (checkpoint: ${RESUME_CKPT}) ..."
    "${PYTHON_BIN}" -m torch.distributed.run --nnodes=1 --nproc_per_node="${num_gpus}" \
      --rdzv_id="1986_${lang}" --rdzv_backend="c10d" --rdzv_endpoint="localhost:29501" \
      "${REPO_ROOT}/cosyvoice/bin/train.py" \
      --train_engine "${train_engine}" \
      --config "${CONFIG_FILE}" \
      --train_data "${LANG_DATA_DIR}/train.data.list" \
      --cv_data "${LANG_DATA_DIR}/dev.data.list" \
      --qwen_pretrain_path "${PRETRAINED_DIR}/CosyVoice-BlankEN" \
      --model llm \
      --checkpoint "${RESUME_CKPT}" \
      --model_dir "${MODEL_EXP_DIR}" \
      --tensorboard_dir "${TB_DIR}" \
      --ddp.dist_backend nccl \
      --num_workers "${num_workers}" \
      --prefetch "${prefetch}" \
      --pin_memory \
      --use_amp \
      --wandb_disable \
      --deepspeed_config "${DS_CONFIG}" \
      --deepspeed.save_states model+optimizer
  fi

  # Stage 6: Average best checkpoints and assemble inference-ready model directory
  if [ "${stage}" -le 6 ] && [ "${stop_stage}" -ge 6 ]; then
    MODEL_EXP_DIR="${STORAGE_ROOT}/exp/${lang}/llm/${train_engine}"
    DECODE_CHECKPOINT="${MODEL_EXP_DIR}/llm_avg${average_num}.pt"
    EXPORT_DIR="${STORAGE_ROOT}/exported_models/SarcVoice3-${lang}"

    echo "[Stage 6][${lang}] Averaging best ${average_num} checkpoints -> ${DECODE_CHECKPOINT} ..."
    "${PYTHON_BIN}" "${REPO_ROOT}/cosyvoice/bin/average_model.py" \
      --dst_model "${DECODE_CHECKPOINT}" \
      --src_path "${MODEL_EXP_DIR}" \
      --num "${average_num}" \
      --val_best

    mkdir -p "${EXPORT_DIR}"
    cp "${DECODE_CHECKPOINT}" "${EXPORT_DIR}/llm.pt"
    cp "${PRETRAINED_DIR}/cosyvoice3.yaml" "${EXPORT_DIR}/cosyvoice3.yaml"
    ln -sf "${PRETRAINED_DIR}/flow.pt" "${EXPORT_DIR}/flow.pt"
    ln -sf "${PRETRAINED_DIR}/hift.pt" "${EXPORT_DIR}/hift.pt"
    ln -sf "${PRETRAINED_DIR}/campplus.onnx" "${EXPORT_DIR}/campplus.onnx"
    ln -sf "${PRETRAINED_DIR}/speech_tokenizer_v3.onnx" "${EXPORT_DIR}/speech_tokenizer_v3.onnx"
    ln -sf "${PRETRAINED_DIR}/CosyVoice-BlankEN" "${EXPORT_DIR}/CosyVoice-BlankEN"
    echo "[Stage 6][${lang}] Inference-ready model assembled at: ${EXPORT_DIR}"
  fi
done
