from datasets import load_dataset
import pandas as pd
import requests
from io import StringIO

# 支援的語言
languages = ['ZH_CN']  # 中文從 HuggingFace 下載

# 要抓取的子任務
target_subjects = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics", "econometrics",
    "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts"
]

# 每個子任務取前 N 筆資料
N = 100

all_data = []

# 處理 HuggingFace 中文數據
for lang in languages:
    print(f"從 HuggingFace 下載 {lang} 語言資料集")
    dataset = load_dataset("openai/MMMLU", name=lang, split="test")
    df = dataset.to_pandas()

    # 過濾指定科目並取前 N 筆
    for subject in target_subjects:
        filtered = df[df["Subject"] == subject].head(N).copy()
        filtered["language"] = lang
        all_data.append(filtered)

# 處理英文數據集（從 URL 下載）
english_url = "https://openaipublic.blob.core.windows.net/simple-evals/mmlu.csv"
print(f"從 URL 下載英文資料集: {english_url}")

response = requests.get(english_url)

# 將 CSV 內容讀入 DataFrame
en_df = pd.read_csv(StringIO(response.text))

# 過濾指定科目並取前 N 筆
for subject in target_subjects:
    filtered = en_df[en_df["Subject"] == subject].head(N).copy()
    filtered["language"] = "EN_US"  # 添加英文標記
    all_data.append(filtered)


# 合併所有資料
final_df = pd.concat(all_data, ignore_index=True)

print(f"共 {len(final_df)} 筆資料")
print(final_df.head())

final_df.to_json("mmlu_17subjects_2langs_100samples.json", orient="records", indent=2)
print("資料已儲存至 mmlu_17subjects_2langs_100samples.json")

