## 檔案說明
- **download.py**: 下載多語言多選題測試資料集
- **model.py**: 實現模型接口，包含 Gemini 和 Mistral AI 的連接和回答處理
- **prompt_format.py**: 定義各種提示格式（純文字、JSON、XML）和輸入輸出組合
- **shuffle.py**: 打亂題目選項順序，保持答案正確性
- **experiment.py**: 主要實驗程式，使用不同格式讓模型回答問題並記錄結果

## 執行順序

1. 先執行 download.py 獲取測試資料：
   ```bash
   python download.py
   ```
   這會下載 MMLU 資料集並產生 `mmlu_17subjects_2langs_100samples.json`

2. 執行 shuffle.py 打亂選項順序：
   ```bash
   python shuffle.py
   ```
   這會讀取 `mmlu_17subjects_2langs_100samples.json` 並產生 `shuffle_mmlu_17subjects_2langs_100samples.json`

3. 執行實驗（使用原始數據或打亂後的數據）：
   ```bash
   # 使用原始數據
   python experiment.py --input mmlu_17subjects_2langs_100samples.json
   
   # 或使用打亂選項後的數據
   python experiment.py --input shuffle_mmlu_17subjects_2langs_100samples.json
   ```

## 重要說明

- 執行 experiment.py 時必須手動指定使用哪個輸入檔案，可以是原始的 `mmlu_17subjects_2langs_100samples.json` 或打亂後的 `shuffle_mmlu_17subjects_2langs_100samples.json`
- 實驗結果會保存回輸入的 JSON 檔案中，並生成兩個摘要報告：
  - `{model}_main_formats_summary.csv`: 比較基本格式、JSON和XML格式的效果
  - `{model}_json_variants_summary.csv`: 比較不同JSON輸入輸出組合的效果
- 默認情況下，程式會依序使用 Gemini 和 Mistral 兩個模型進行測試
- 使用前需要在環境變數或 model.py 中設定 API 金鑰：
  - Gemini：設置 `GOOGLE_API_KEY` 環境變數
  - Mistral：設置 `MISTRAL_API_KEY` 環境變數

## 格式測試說明

實驗測試了以下格式：

1. 基本格式比較：
   - 純文字輸入，純文字輸出（基準點）
   - JSON輸入，JSON輸出
   - XML輸入，XML輸出

2. JSON變體比較：
   - 純文字輸入，純文字輸出（基準點）
   - JSON輸入，純文字輸出
   - 純文字輸入，JSON輸出
   - JSON輸入，JSON輸出
