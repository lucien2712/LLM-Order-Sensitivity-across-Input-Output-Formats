from datasets import load_dataset
import pandas as pd

# 支援的語言（default 是英文，ZH_CN 是中文）
languages = ['default', 'ZH_CN']

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

for lang in languages:
    print(f"Loading dataset for language: {lang}")
    dataset = load_dataset("openai/MMMLU", name=lang, split="test")
    df = dataset.to_pandas()

    # 過濾指定科目並取前 N 筆
    for subject in target_subjects:
        filtered = df[df["Subject"] == subject].head(N).copy()
        filtered["language"] = lang
        all_data.append(filtered)


final_df = pd.concat(all_data, ignore_index=True)

print(final_df.head())

final_df.to_json("mmlu_17subjects_2langs_100samples.json", orient="records", indent=2)
