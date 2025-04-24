## 檔案說明
- **download.py**: 下載 2 語言 17 subtasks 資料集
- **model.py**: model setting，包含 Gemini 和 Mistral 和回答處理
- **prompt_format.py**: 定義各種提示格式（純文字、JSON、XML）和輸入輸出組合
- **shuffle.py**: 打亂題目選項順序，保持答案正確性
- **experiment.py**: 主要實驗程式，使用不同格式讓模型回答問題並記錄結果
- **eval.py**: 評估模型回答的 RSD、CKLD 等指標
- **visualize.py**: 生成各種視覺化圖表，包括準確率比較、波動率分析、語言和學科敏感性分析等 
   > 這個是暫時 GPT 自己想的圖

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
3. 執行 eval.py 評估模型表現：
   ```bash
   python eval.py
   ```
   這會生成評估報告，包含準確率、波動率等指標

4. 執行 visualize.py 生成視覺化圖表：
   > 這個是暫時 GPT 自己想的圖
   ```bash
   python visualize.py
   ```
   這會生成各種分析圖表，包括：
   - 準確率比較圖
   - 波動率分析圖
   - 語言敏感性分析圖
   - 學科敏感性熱圖
   - 信心與順序敏感性關係圖
   - 偏差指標雷達圖


5. 或你可以直接執行 
```bash
bash run.sh
```
## 重要說明

- 執行 experiment.py 時必須手動指定使用哪個輸入檔案，可以是原始的 `mmlu_17subjects_2langs_100samples.json` 或打亂後的 `shuffle_mmlu_17subjects_2langs_100samples.json`
- 默認情況下，程式會依序使用 Gemini 和 Mistral 兩個模型進行測試
- 使用前需要在 model.py 中設定 API token：
  - Gemini：設置 `GOOGLE_API_KEY` 
  - Mistral：設置 `MISTRAL_API_KEY`

## 格式測試說明

實驗測試了以下格式：

1. 基本格式比較：
   - 純文字輸入，純文字輸出（base）
   - JSON輸入，JSON輸出
   - XML輸入，XML輸出

2. JSON變體比較：
   - 純文字輸入，純文字輸出（base）
   - JSON輸入，純文字輸出
   - 純文字輸入，JSON輸出
   - JSON輸入，JSON輸出
