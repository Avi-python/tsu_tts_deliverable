# Filter Cycle 實例研究報告（`tsu` / `nat_acc_20.jsonl`）

本報告整理自 `/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/nat_acc_20/nat_acc_20.jsonl`（共 20 筆鄒語測試句）實際執行 CosyVoice3 TTS + Whisper ASR 過濾迴圈後，**真正進入重試流程（`attempts > 1`）** 的真實案例，供您分析不同 `seed` 對發音穩定度與 ASR 辨識結果的影響。

---

## 1. 實驗設定

- **語言 (`--lang`)**：`tsu`
- **TTS 模型**：`/nfs/DGX2/Workspace/avi/fb_nchc/tts/tsu/models/tsu_tts_20261115`
- **ASR 模型**：`/nfs/DGX2/Workspace/avi/fb_nchc/asr/tsu/models`
- **Prompt 音檔 / 文本**：`cou_e_04842_c19.wav` / `"a panto pangka to kiosicumia."`
- **參數**：`base_seed=42`、`max_retries=5`
- **測試門檻**：
  - **Run A（`--cer_threshold 0.05`）**：20/20 通過，平均最佳 CER = `0.0049`，其中 **1 筆**進入重試迴圈（第 12 筆 `cou_e_04554_c25`）。
  - **Run B（`--cer_threshold 0.0` 嚴格模式）**：19/20 通過，平均最佳 CER = `0.0008`，共有 **4 筆**進入重試迴圈（3 筆經換 seed 後修正為 `CER = 0.0`，1 筆耗盡 5 次重試）。

---

## 2. 進入 Filter Cycle 的案例總覽

| Index | `source_id` | 目標句子 (`clean_text`) | Run A (`<=0.05`) | Run B (`<=0.0`) 嘗試次數 | 起始 CER (seed 42) | 最終 CER / Seed | 現象分類 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **11** | `cou_e_04554_c25` | `oʼa, miʼo seolʉa no oʼte tusiʼngi.` | **2 次通過** | **2 次通過** | `0.0625` | **`0.0000`** (`43`) | **真實發音/音素錯誤**（`tusiʼngi` → `tusiʼmi`） |
| **9** | `cou_e_06208_c09` | `moso uhne patungkuonʉ ho eafou.` | 1 次通過 | **3 次通過** | `0.0333` | **`0.0000`** (`44`) | **連寫/斷詞空白差異**（`eafou` → `ea fou`） |
| **13** | `cou_e_00155_c10` | `ʼiaʼa! te monʼi yuovei?` | 1 次通過 | **2 次通過** | `0.0476` | **`0.0000`** (`43`) | **句首喉塞音 `ʼ` 漏失**（`ʼiaʼa` → `iaʼa`） |
| **10** | `cou_p_00454_c33` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to niamaameoi.` | 1 次通過 | **5 次未過** (`passed=false`) | `0.0161` | `0.0161` (`42`) | **系統性斷詞差異 / 子音混淆**（`niamaameoi` ↔ `nia maameoi` / `miamaameoi`） |

---

## 3. 各案例詳細重試紀錄

### 案例 1：`cou_e_04554_c25`（Index 11）— 唯一在 `0.05` 門檻就觸發重試且成功修正

- **中文翻譯**：不，我沒有正在掃地。
- **Target (`clean_text`)**：`oʼa, miʼo seolʉa no oʼte tusiʼngi.`
- **結果**：`passed=True`，第 2 次嘗試成功（`seed=43`）
- **輸出音檔**：[tsu_00011.wav](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_strict/tsu_00011.wav)

| Attempt | Seed | CER | 時長 (s) | Whisper ASR 辨識結果 (`hypothesis`) | 差異分析 |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `42` | `0.0625` | `7.32` | `oʼa, miʼo seolʉa no oʼte tusiʼmi.` | 句尾 `tusiʼngi` 被合成／辨識成 `tusiʼmi`（`/ŋ/` → `/m/`） |
| **2** | **`43`** | **`0.0000`** | `7.72` | `oʼa, miʼo seolʉa no oʼte tusiʼngi.` | **完全吻合，通過門檻** |

---

### 案例 2：`cou_e_06208_c09`（Index 9）— 嘗試 3 個 Seed 後修正斷詞空白

- **中文翻譯**：他去玉山打獵，
- **Target (`clean_text`)**：`moso uhne patungkuonʉ ho eafou.`
- **結果**：`passed=True`，第 3 次嘗試成功（`seed=44`）
- **輸出音檔**：[tsu_00009.wav](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_strict/tsu_00009.wav)

| Attempt | Seed | CER | 時長 (s) | Whisper ASR 辨識結果 (`hypothesis`) | 差異分析 |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `42` | `0.0333` | `5.40` | `moso uhne patungkuonʉ ho ea fou.` | `eafou` 中間多出空白（`ea fou`） |
| 2 | `43` | `0.0333` | `5.80` | `moso uhne patungkuonʉ ho ea fou.` | 仍為 `ea fou`（但語速/時長不同） |
| **3** | **`44`** | **`0.0000`** | `5.64` | `moso uhne patungkuonʉ ho eafou.` | **連寫正確，完全吻合** |

---

### 案例 3：`cou_e_00155_c10`（Index 13）— 句首喉塞音 `ʼ` 漏失後經重試補回

- **中文翻譯**：喔，會很快回來嗎？
- **Target (`clean_text`)**：`ʼiaʼa! te monʼi yuovei?`
- **結果**：`passed=True`，第 2 次嘗試成功（`seed=43`）
- **輸出音檔**：[tsu_00013.wav](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_strict/tsu_00013.wav)

| Attempt | Seed | CER | 時長 (s) | Whisper ASR 辨識結果 (`hypothesis`) | 差異分析 |
| :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | `42` | `0.0476` | `4.16` | `iaʼa! te monʼi yuovei?` | 句首漏掉喉塞音 `ʼ`，且整體語長明顯偏短（`4.16s`） |
| **2** | **`43`** | **`0.0000`** | `5.16` | `ʼiaʼa! te monʼi yuovei?` | **句首 `ʼiaʼa` 發音清晰，通過** |

---

### 案例 4：`cou_p_00454_c33`（Index 10）— 耗盡 5 次重試仍未達 `0.0`（`passed=false`）

- **中文翻譯**：是為守住祖先踏過的獵場。
- **Target (`clean_text`)**：`ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to niamaameoi.`
- **結果**：`passed=False`，保留最低 CER 嘗試（`seed=42`, `cer=0.0161`）
- **輸出音檔**：[tsu_00010.wav](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_strict/tsu_00010.wav)

| Attempt | Seed | CER | 時長 (s) | Whisper ASR 辨識結果 (`hypothesis`) | 差異分析 |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **1** | **`42`** | **`0.0161`** | `13.68` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to nia maameoi.` | `niamaameoi` 被拆成 `nia maameoi`（僅差 1 個空白） |
| 2 | `43` | `0.0161` | `13.96` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to nia maameoi.` | 同上 |
| 3 | `44` | `0.0161` | `11.88` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to nia maameoi.` | 同上 |
| 4 | `45` | `0.0161` | `12.00` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to nia maameoi.` | 同上 |
| 5 | `46` | `0.0161` | `13.32` | `ʼaisieno ʼucia atukca ʼo ohecula pʼecihi ci hupa to miamaameoi.` | 變成單字但字首誤聽/誤發為 `miamaameoi`（`n` → `m`） |

---

## 4. 研究觀察與建議

1. **Seed 重試確實有效**：
   - 在**音素混淆**（`tusiʼngi` vs `tusiʼmi`）與**句首喉塞音漏失**（`ʼiaʼa` vs `iaʼa`）這類隨機取樣造成的品質波動上，切換 seed 後都在 1–2 次重試內完全修復（`CER -> 0.0`）。
2. **「空白斷詞差異」對 CER 門檻的影響**：
   - `eafou` ↔ `ea fou` 與 `niamaameoi` ↔ `nia maameoi` 在語音上可能發音正確，只是詞素邊界停頓導致 Whisper 多輸出一個空白。
   - 因此實務上建議把 `--cer_threshold` 設在 **`0.05`～`0.10`**（而非 `0.0`），既能攔截真實發音錯誤（如案例 1 的 `0.0625`），又不會因為單純的斷詞空白差異（`0.0161`～`0.0333`）而浪費重試時間。
3. **原始結果檔位置**：
   - `cer_threshold=0.05`：[outputs/nat_acc_20_study/results.jsonl](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_study/results.jsonl)
   - `cer_threshold=0.0`：[outputs/nat_acc_20_strict/results.jsonl](file:///mnt/usr1/avi/Projects/formosan/Filtering/outputs/nat_acc_20_strict/results.jsonl)
