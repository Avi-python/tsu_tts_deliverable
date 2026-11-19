# Formosan SARC Voice (CosyVoice 3) 語音合成與 ASR 品質過濾系統（鄒語 `tsu` 交付版）

本專案整合 **CosyVoice 3 原住民族語 TTS 訓練管線**、**標準語音合成推論** 與 **Whisper ASR 閉迴路 CER 品質過濾推論**，並以**鄒語（`tsu`）**作為主要示範設定，整理為可直接交付與複現的單一程式碼庫。

---

## 1. 系統架構與功能概述

本專案包含三個核心工作流程：

1. **模型訓練（Training, Stage 0 ～ Stage 6）**
   - 基於 **CosyVoice 3 (`Fun-CosyVoice3-0.5B`, 24 kHz)** 微調 **LLM (`CosyVoice3LM` / Qwen2-0.5B)**，保留預訓練的 **Flow Matching (`flow.pt`)** 與 **HiFi-GAN (`hift.pt`)** 進行 Zero-shot 語者音色克隆。
   - 支援從單一 `.jsonl`（使用 `clean_text`）自動執行防資料洩漏的 **90% Train / 5% Dev / 5% Test** 切分（偵測到 `pyu` 時自動依方言分層抽樣）、離線特徵提取（CampPlus + SpeechTokenizerV3）、Parquet 分片打包、多卡 DDP/DeepSpeed 訓練及最佳 Checkpoint 權重平均匯出。
2. **標準語音合成推論（`inference.py`）**
   - 單純執行 CosyVoice 3 TTS 推論，**不需要載入 Whisper ASR 模型**。
   - 支援單句／多句輸入、批次 `.jsonl` 合成、多隨機種子（Multi-seed）比較、以及 `[IPA]` / CMU 音標混合 Inpainting 合成。
3. **ASR 閉迴路品質過濾推論（`inference_filter.py` / `cosyvoice.filtering`）**
   - 結合 **CosyVoice3 TTS** 與微調後的 **Whisper ASR**，對每筆目標語句進行閉迴路字元錯誤率（CER）檢核與隨機種子重試：
     1. 使用 `seed = base_seed + attempt` 合成語音。
     2. 在記憶體中將 24 kHz 波形重採樣至 16 kHz，送入 Whisper ASR 辨識並套用 Post-TN 正規化（保留喉塞音 `ʼ`）計算 CER。
     3. 若 `CER <= cer_threshold`，立即通過並儲存 `.wav`；否則更換下一個 seed 重新生成，直到通過或達到 `max_retries`。

---

## 2. 目錄結構

```text
tsu_tts_deliverable/
├── README.md                              # 本使用手冊
├── LICENSE                                # Apache 2.0 授權條款
├── requirements.txt                       # Python 相依套件清單（含 jiwer, pytest）
├── .gitignore                             # Git 忽略規則（完整保留 tools/ 與 examples/）
├── inference.py                           # [推論入口 1] 標準 TTS 推論 CLI（不載入 ASR）
├── inference_filter.py                    # [推論入口 2] TTS + Whisper ASR 閉迴路過濾推論 CLI
├── cosyvoice/                             # 核心 Python 套件
│   ├── __init__.py                        # 自動掛載 third_party/Matcha-TTS
│   ├── bin/                               # 訓練與匯出程式 (train.py, average_model.py 等)
│   ├── cli/                               # AutoModel 與前端文本處理 (cosyvoice.py, frontend.py)
│   ├── dataset/ , flow/ , hifigan/ , llm/ , tokenizer/ , transformer/ , utils/ , vllm/
│   └── filtering/                         # ASR 輔助品質過濾子模組
│       ├── __init__.py                    # 匯出 FilterConfig, TTSFilterPipeline 等核心介面
│       ├── config.py                      # FilterConfig / AttemptResult / FilterResult
│       ├── model_resolver.py              # 自動解析扁平或巢狀 ASR / TTS 模型目錄
│       ├── tts_engine.py                  # CosyVoice3 封裝（支援確定性 per-attempt seeding）
│       ├── asr_engine.py                  # Whisper 16 kHz 辨識、Post-TN 正規化與 CER 計算
│       └── pipeline.py                    # TTSFilterPipeline 閉迴路重試主流程
├── tools/                                 # 訓練資料前處理工具（Stage 0 ～ Stage 3）
│   ├── prepare_formosan_jsonl.py          # Stage 0: 單一 JSONL 自動 90/5/5 切分 train/dev/test 與 splits/*.jsonl
│   ├── extract_embedding.py               # Stage 1: CampPlus 192 維語者向量提取
│   ├── extract_speech_token.py            # Stage 2: 多 GPU SpeechTokenizerV3 離散 Token 提取
│   └── make_parquet_list.py               # Stage 3: 打包 Parquet 訓練分片
├── examples/formosan/sarcvoice3/          # 原住民族語訓練配置與主控腳本
│   ├── run_formosan.sh                    # Stage 0 ～ Stage 6 一鍵訓練、續訓與模型匯出腳本
│   ├── TRAINING_GUIDE.md                  # 完整訓練原理、參數調整與實戰指令手冊
│   └── conf/
│       ├── cosyvoice3_formosan.yaml       # 模型架構、資料管線與優化器設定
│       └── ds_stage2.json                 # DeepSpeed ZeRO-2 設定
├── docs/
│   └── FILTER_CYCLE_STUDY.md              # 鄒語（tsu）Filter Cycle 重試機制實例研究報告
├── tests/                                 # 單元測試（可離線驗證資料切分、模型解析、CER 與重試邏輯）
│   ├── conftest.py
│   ├── test_asr_and_pipeline.py
│   ├── test_model_resolver.py
│   └── test_prepare_formosan_jsonl.py
├── third_party/Matcha-TTS/                # Flow Matching ODE Solver 依賴
├── runtime/                               # FastAPI / gRPC / Triton 部署範例
└── pretrained_models/                     # 預訓練基底模型目錄（如 Fun-CosyVoice3-0.5B-2512）
```

---

## 3. 環境建置

本機已配置好包含 `numpy<2`（`1.26.4`）、`torch==2.3.1`、`transformers==4.51.3` 與 `jiwer` 的 `sarcvoice` Conda 環境：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python
```

若需於新機器重建環境（Python 3.10）：

```bash
conda create -n sarcvoice -y python=3.10
conda activate sarcvoice
pip install -r requirements.txt
```

> [!TIP]
> `inference.py` 與 `inference_filter.py` 皆會在未設定 `CUDA_VISIBLE_DEVICES` 時自動偵測並選用剩餘 VRAM 最多的 GPU；如需手動指定 GPU，可在指令前加上 `CUDA_VISIBLE_DEVICES=0`。

---

## 4. 模型訓練流程（Stage 0 ～ Stage 6）

完整的資料準備、去重加速技巧、中斷續訓（Resume）與模型平均匯出細節請參閱 [TRAINING_GUIDE.md](file:///mnt/usr1/avi/Projects/formosan/tsu_tts_deliverable/examples/formosan/sarcvoice3/TRAINING_GUIDE.md)。

透過 [`examples/formosan/sarcvoice3/run_formosan.sh`](file:///mnt/usr1/avi/Projects/formosan/tsu_tts_deliverable/examples/formosan/sarcvoice3/run_formosan.sh) 可一鍵執行全部或指定階段（支援 `tsu`、`pyu`、`trv`）。在 Stage 0 只需提供單一 `.jsonl`（讀取 `clean_text`），腳本便會自動以防洩漏方式切出 **90% Train / 5% Dev / 5% Test**（若 `lang=pyu` 會自動依方言分層），並於 `<STORAGE_ROOT>/data/<lang>/splits/` 輸出 `train.jsonl`、`dev.jsonl`、`test.jsonl`：

```bash
# 直接指定單一 JSONL，從頭執行鄒語（tsu）切分（90/5/5）、訓練與模型匯出
INPUT_JSONL=/path/to/tsu.jsonl stage=0 stop_stage=6 langs=tsu \
  bash examples/formosan/sarcvoice3/run_formosan.sh

# 自訂切分比例與隨機種子（預設 DEV_RATIO=0.05 TEST_RATIO=0.05 SPLIT_SEED=42）
INPUT_JSONL=/path/to/tsu.jsonl DEV_RATIO=0.05 TEST_RATIO=0.05 SPLIT_SEED=42 \
  stage=0 stop_stage=3 langs=tsu \
  bash examples/formosan/sarcvoice3/run_formosan.sh

# 若訓練中斷，直接執行 Stage 5~6 會自動搜尋最新的 epoch_*.pt 接續訓練並匯出
stage=5 stop_stage=6 langs=tsu \
  bash examples/formosan/sarcvoice3/run_formosan.sh
```

> [!NOTE]
> Stage 0 自動產出的 `<STORAGE_ROOT>/data/<lang>/splits/test.jsonl` 可直接作為 `inference.py` 或 `inference_filter.py --input_jsonl` 的輸入，無須依賴任何外部事先切分好的 `test.jsonl`。

---

## 5. 標準語音合成推論（`inference.py`）

當只需要進行 TTS 合成、比較多個隨機種子，或測試混合音標（Inpainting）而**不需載入 Whisper ASR** 時，請使用 `inference.py`：

### 範例 A：鄒語單句推論（自動解析 `/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/models`）

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference.py \
  --lang tsu \
  --text "mo pio na avʼu ta oko?" \
  --output_dir outputs/tsu_simple
```

### 範例 B：多種子比較與混合音標（Inpainting）測試

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference.py \
  --model_dir /nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/models/tsu_tts_20261115 \
  --text "ʼa ongko ta pasuya 我的 [IH1][NG][G][L][IH0][SH] 不是很好." \
  --seeds 0 42 1234 2026 \
  --output_dir outputs/tsu_multiseed
```

### 範例 C：批次 JSONL 推論

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference.py \
  --lang tsu \
  --input_jsonl /nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/sim_10/sim_10.jsonl \
  --limit 5 \
  --output_dir outputs/tsu_batch_raw
```

---

## 6. ASR 閉迴路品質過濾推論（`inference_filter.py`）

若需自動透過 Whisper ASR 驗證合成語音的發音正確率（CER）並自動重試不同 Seed，請使用 `inference_filter.py`。針對鄒語實際重試案例的分析，可參考 [FILTER_CYCLE_STUDY.md](file:///mnt/usr1/avi/Projects/formosan/tsu_tts_deliverable/docs/FILTER_CYCLE_STUDY.md)。

### 模型目錄自動解析規則
依 `--lang` 自動搜尋 `<asr_root>/<lang>/models` 與 `<tts_root>/<lang>/models`，同時相容：
- **扁平結構**：`models/` 底下直接放置模型檔（如 `config.json`）。
- **巢狀子目錄結構**：`models/<model_name>/` 底下放置模型檔（同一語言有多個模型時可加 `--asr_model_name` / `--tts_model_name`，或直接使用 `--asr_model_dir` / `--tts_model_dir` 指定完整路徑）。

### 範例 A：單句模式（Single-text Mode）

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference_filter.py \
  --lang tsu \
  --text "mo pio na avʼu ta oko?" \
  --prompt_wav /nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav \
  --prompt_text "a panto pangka to kiosicumia." \
  --instruct_prefix "You are a helpful assistant. 请用鄒語表达。<|endofprompt|>" \
  --cer_threshold 0.05 \
  --max_retries 5 \
  --base_seed 42 \
  --output_wav outputs/tsu_single.wav
```

### 範例 B：批次 JSONL 模式與嚴格過濾

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference_filter.py \
  --lang tsu \
  --input_jsonl /nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/sim_10/sim_10.jsonl \
  --prompt_wav /nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav \
  --prompt_text "a panto pangka to kiosicumia." \
  --cer_threshold 0.10 \
  --max_retries 5 \
  --discard_on_fail \
  --output_dir outputs/tsu_filtered \
  --output_jsonl outputs/tsu_filtered/results.jsonl
```

> [!NOTE]
> 未加上 `--discard_on_fail` 時，若耗盡 `max_retries` 次數仍未達標，系統預設保留 CER 最低的候選音檔並標記 `"passed": false`；加上 `--discard_on_fail` 則不會輸出未通過的 `.wav`。

### 範例 C：Inpainting（Hotfix）分離答案與合成文字

當特定詞彙需使用 CMU / IPA 音標修正發音，但 Whisper ASR CER 仍需以原始文字當標準答案時：

- **單句模式**：使用 `--tts_text` 指定合成用文字
```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/python inference_filter.py \
  --lang tsu \
  --text "ʼa ongko ta pasuya 我的 English 不是很好." \
  --tts_text "ʼa ongko ta pasuya 我的 [IH1][NG][G][L][IH0][SH] 不是很好." \
  --prompt_wav /nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav \
  --prompt_text "a panto pangka to kiosicumia." \
  --output_wav outputs/tsu_inpaint.wav
```

- **批次模式**：使用 `--tts_text_field` 指定合成用欄位（若該筆資料該欄位為空，自動 fallback 回參考答案欄位）。

### 常用參數一覽（`inference_filter.py`）

| 參數 | 預設值 | 說明 |
| :--- | :--- | :--- |
| `--lang` | `tsu` | 語言代碼（如 `tsu`、`pyu`、`trv`） |
| `--tts_model_dir` / `--asr_model_dir` | `None` | 直接指定 TTS / ASR 模型完整資料夾路徑（覆寫自動搜尋） |
| `--tts_model_name` / `--asr_model_name` | `None` | 指定 `<root>/<lang>/models/` 下的子目錄名稱 |
| `--cer_threshold` | `0.05` | 通過門檻（`CER <= cer_threshold` 即通過） |
| `--max_retries` | `5` | 單句最多嘗試的 seed 次數 |
| `--base_seed` | `42` | 起始 seed（第 `i` 次嘗試使用 `base_seed + i`） |
| `--tts_mode` | `zero_shot` | TTS 推論模式（`zero_shot` 或 `instruct2`） |
| `--prompt_wav` / `--prompt_text` | `None` | 預設參考音檔與文本（若 JSONL 內含對應欄位則優先採用） |
| `--instruct_prefix` | `You are a helpful assistant. 请用鄒語表达。<|endofprompt|>` | 語言控制指令前綴 |
| `--no_normalize` | `False` | 停用 Post-TN 文本正規化（預設小寫並去除標點，保留 `ʼ` 喉塞音） |
| `--discard_on_fail` | `False` | 超過重試上限仍未通過時捨棄 `.wav` |
| `--tts_text` / `--tts_text_field` | `None` | 指定送入 TTS 合成的音標修補文字或 JSONL 欄位 |

### 輸出 JSONL 欄位說明
批次模式會保留原始輸入欄位，並新增：
- `tts_audio`：最終儲存的 `.wav` 路徑（若 `--discard_on_fail` 且未通過則為 `null`）
- `tts_input_text`：實際送入 CosyVoice 3 合成的文字
- `asr_hypothesis`：最終選定嘗試的 Whisper ASR 辨識結果
- `cer`、`seed`、`attempts`、`passed`、`attempt_history`

### Python API 呼叫範例

```python
from pathlib import Path
from cosyvoice.filtering import FilterConfig, TTSFilterPipeline

config = FilterConfig(
    lang="tsu",
    cer_threshold=0.05,
    max_retries=5,
    base_seed=42,
    default_prompt_wav="/nfs/DGX2/Workspace/avi/nchc-2026-cou/協作平台/wav/c19/cou_e_04842_c19.wav",
    default_prompt_text="a panto pangka to kiosicumia.",
)

pipeline = TTSFilterPipeline.from_config(config)
result = pipeline.filter_utterance(
    text="mo pio na avʼu ta oko?",
    output_wav_path=Path("outputs/api_sample.wav"),
)
print(f"passed={result.passed}, cer={result.best_attempt.cer:.4f}, seed={result.best_attempt.seed}")
```

---

## 7. 執行單元測試

無須 GPU 或載入模型權重即可執行單元測試，驗證目錄解析、Post-TN 正規化、CER 計算與 Seed 重試流程：

```bash
/mnt/usr1/avi/miniconda3/envs/sarcvoice/bin/pytest tests -v
```
