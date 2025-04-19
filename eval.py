import json
import numpy as np
import pandas as pd
from scipy.stats import entropy

'''
1. 讀取 mmlu_17subjects_2langs_100samples.json & shuffle_mmlu_17subjects_2langs_100samples.json
2. 計算 shuffle 前後的 metric
3. 計算 不同語言在不同 format 的 metric (shuffle 前、後)
4. 計算 不同 subwork 在不同 format 的 metric (shuffle 前、後)

metric 包含
RSD (Relative Standard Deviation)
Ref: Beyond Performance: Quantifying and Mitigating Label Bias in LLMs (NAACL, 2024)

RStd (Standard Deviation of Recalls)
Ref: Large Language Models Are Not Robust Multiple Choice Selectors (ICLR, 2024)

Fluctuation Rate
Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models (ACL, 2024)

CKLD (Choice Kullback-Leibler Divergence)
Ref: Mitigating Selection Bias with Node Pruning and Auxiliary Options
'''


# 1. RSD (相對標準差) - Relative Standard Deviation
"""
RSD = (σ / μ) × 100%

其中：
σ 是選項分佈的標準差
μ 是選項分佈的平均值

RSD 用於測量模型選擇各個選項的均勻性。較高的 RSD 表示選項分佈不均衡。
參考文獻：Beyond Performance: Quantifying and Mitigating Label Bias in LLMs (NAACL, 2024)
"""

# 2. RStd (召回率標準差) - Standard Deviation of Recalls
"""
RStd = σ_r

其中：
σ_r 是各選項召回率的標準差
召回率 = 正確選擇某選項的次數 / 該選項為正確答案的總次數

RStd 衡量模型對不同選項的偏好程度。較高的 RStd 表示模型對某些選項的處理能力不均衡。
參考文獻：Large Language Models Are Not Robust Multiple Choice Selectors (ICLR, 2024)
"""

# 3. 波動率 - Fluctuation Rate
"""
Fluctuation Rate = |A_orig ≠ A_shuf| / N

其中：
|A_orig ≠ A_shuf| 是原始順序和打亂順序後答案不一致的問題數量
N 是問題總數

波動率反映了模型對選項順序變化的敏感程度。較高的波動率表示模型易受選項順序影響。
參考文獻：Unveiling Selection Biases: Exploring Order and Token Sensitivity in Large Language Models (ACL, 2024)
"""

# 4. CKLD (選項 KL 散度) - Choice Kullback-Leibler Divergence
"""
CKLD = D_KL(P || Q) = Σ P(i) × log(P(i)/Q(i))

其中：
P 是模型的選項分佈 (如 {A:0.4, B:0.3, C:0.2, D:0.1})
Q 是均勻分佈 (如 {A:0.25, B:0.25, C:0.25, D:0.25})
D_KL 是 KL 散度

CKLD 測量模型選項分佈與理想均勻分佈的差異。較高的 CKLD 表示模型存在選項偏好。
參考文獻：Mitigating Selection Bias with Node Pruning and Auxiliary Options
"""

def load_data(original_file, shuffled_file):
    """載入原始和打亂後的問題集"""
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(shuffled_file, 'r', encoding='utf-8') as f:
        shuffled_data = json.load(f)
    
    return original_data, shuffled_data

def calculate_accuracy(questions, model_name, format_name):
    """計算準確率"""
    correct = 0
    total = 0
    
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"]
        correct_answer = question["Answer"]
        
        if model_answer.upper() == correct_answer.upper():
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def calculate_answer_distribution(questions, model_name, format_name):
    """計算答案分佈"""
    distribution = {"A": 0, "B": 0, "C": 0, "D": 0}
    total = len(questions)
    
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"].upper()
        if model_answer in distribution:
            distribution[model_answer] += 1
    
    # 將計數轉換為比例
    for key in distribution:
        distribution[key] /= total
    
    return distribution

def calculate_rsd(questions, model_name, format_name):
    """計算 Relative Standard Deviation (RSD)"""
    distribution = calculate_answer_distribution(questions, model_name, format_name)
    values = list(distribution.values())
    
    mean = np.mean(values)
    std = np.std(values)
    return (std / mean) * 100 if mean > 0 else 0

def calculate_rstd(questions, model_name, format_name):
    """計算 Standard Deviation of Recalls (RStd)"""
    recalls = {"A": 0, "B": 0, "C": 0, "D": 0}
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"].upper()
        correct_answer = question["Answer"].upper()
        
        counts[correct_answer] += 1
        if model_answer == correct_answer:
            recalls[correct_answer] += 1
    
    # 計算每個選項的召回率
    recall_values = []
    for key in recalls:
        if counts[key] > 0:
            recall_values.append(recalls[key] / counts[key])
    
    return np.std(recall_values) if recall_values else 0

def calculate_fluctuation_rate(original_questions, shuffled_questions, model_name, format_name):
    """計算 Fluctuation Rate"""
    fluctuations = 0
    total_pairs = len(original_questions)
    
    for orig_q, shuf_q in zip(original_questions, shuffled_questions):
        orig_answer = orig_q["results"][f"{model_name}_{format_name}"]["answer"].upper()
        shuf_answer = shuf_q["results"][f"{model_name}_{format_name}"]["answer"].upper()
        
        # 檢查答案是否在打亂前後發生變化
        if orig_answer != shuf_answer:
            fluctuations += 1
    
    return fluctuations / total_pairs

def calculate_ckld(questions, model_name, format_name):
    """計算 Choice Kullback-Leibler Divergence (CKLD)"""
    distribution = calculate_answer_distribution(questions, model_name, format_name)
    uniform_distribution = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    
    p = np.array(list(distribution.values()))
    q = np.array(list(uniform_distribution.values()))
    
    # 替換為小值以避免除零錯誤
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    
    return entropy(p, q)

def get_language(question):
    """判斷問題的語言"""
    return question.get("Language", "default")

def get_subject(question):
    """獲取問題的學科領域"""
    return question.get("Subject", "unknown")

def evaluate_and_save_metrics(original_data, shuffled_data, model_names, format_names):
    """評估指標並儲存在DataFrame中"""
    # 初始化三個DataFrame的資料
    overall_metrics = []
    language_metrics = []
    subject_metrics = []
    
    # 按模型和格式進行評估
    for model_name in model_names:
        for format_name in format_names:
            format_key = f"{format_name}"
            
            # 總體指標
            orig_accuracy = calculate_accuracy(original_data, model_name, format_key)
            shuf_accuracy = calculate_accuracy(shuffled_data, model_name, format_key)
            orig_rsd = calculate_rsd(original_data, model_name, format_key)
            shuf_rsd = calculate_rsd(shuffled_data, model_name, format_key)
            orig_rstd = calculate_rstd(original_data, model_name, format_key)
            shuf_rstd = calculate_rstd(shuffled_data, model_name, format_key)
            fluc_rate = calculate_fluctuation_rate(original_data, shuffled_data, model_name, format_key)
            orig_ckld = calculate_ckld(original_data, model_name, format_key)
            shuf_ckld = calculate_ckld(shuffled_data, model_name, format_key)
            
            # 添加到總體指標列表
            overall_metrics.append({
                "Model": model_name,
                "Format": format_name,
                "Original_Accuracy": orig_accuracy,
                "Shuffled_Accuracy": shuf_accuracy,
                "Original_RSD": orig_rsd,
                "Shuffled_RSD": shuf_rsd,
                "Original_RStd": orig_rstd,
                "Shuffled_RStd": shuf_rstd,
                "Fluctuation_Rate": fluc_rate,
                "Original_CKLD": orig_ckld,
                "Shuffled_CKLD": shuf_ckld
            })
            
            # 按語言分組
            languages = set(get_language(q) for q in original_data)
            for lang in languages:
                orig_lang_questions = [q for q in original_data if get_language(q) == lang]
                shuf_lang_questions = [q for q in shuffled_data if get_language(q) == lang]
                
                lang_orig_accuracy = calculate_accuracy(orig_lang_questions, model_name, format_key)
                lang_shuf_accuracy = calculate_accuracy(shuf_lang_questions, model_name, format_key)
                lang_orig_rsd = calculate_rsd(orig_lang_questions, model_name, format_key)
                lang_shuf_rsd = calculate_rsd(shuf_lang_questions, model_name, format_key)
                lang_orig_rstd = calculate_rstd(orig_lang_questions, model_name, format_key)
                lang_shuf_rstd = calculate_rstd(shuf_lang_questions, model_name, format_key)
                lang_fluc_rate = calculate_fluctuation_rate(orig_lang_questions, shuf_lang_questions, model_name, format_key)
                lang_orig_ckld = calculate_ckld(orig_lang_questions, model_name, format_key)
                lang_shuf_ckld = calculate_ckld(shuf_lang_questions, model_name, format_key)
                
                language_metrics.append({
                    "Model": model_name,
                    "Format": format_name,
                    "Language": lang,
                    "Original_Accuracy": lang_orig_accuracy,
                    "Shuffled_Accuracy": lang_shuf_accuracy,
                    "Original_RSD": lang_orig_rsd,
                    "Shuffled_RSD": lang_shuf_rsd,
                    "Original_RStd": lang_orig_rstd,
                    "Shuffled_RStd": lang_shuf_rstd,
                    "Fluctuation_Rate": lang_fluc_rate,
                    "Original_CKLD": lang_orig_ckld,
                    "Shuffled_CKLD": lang_shuf_ckld
                })
            
            # 按學科分組
            subjects = set(get_subject(q) for q in original_data if get_subject(q) != "unknown")
            for subj in subjects:
                orig_subj_questions = [q for q in original_data if get_subject(q) == subj]
                shuf_subj_questions = [q for q in shuffled_data if get_subject(q) == subj]
                
                subj_orig_accuracy = calculate_accuracy(orig_subj_questions, model_name, format_key)
                subj_shuf_accuracy = calculate_accuracy(shuf_subj_questions, model_name, format_key)
                subj_orig_rsd = calculate_rsd(orig_subj_questions, model_name, format_key)
                subj_shuf_rsd = calculate_rsd(shuf_subj_questions, model_name, format_key)
                subj_orig_rstd = calculate_rstd(orig_subj_questions, model_name, format_key)
                subj_shuf_rstd = calculate_rstd(shuf_subj_questions, model_name, format_key)
                subj_fluc_rate = calculate_fluctuation_rate(orig_subj_questions, shuf_subj_questions, model_name, format_key)
                subj_orig_ckld = calculate_ckld(orig_subj_questions, model_name, format_key)
                subj_shuf_ckld = calculate_ckld(shuf_subj_questions, model_name, format_key)
                
                subject_metrics.append({
                    "Model": model_name,
                    "Format": format_name,
                    "Subject": subj,
                    "Original_Accuracy": subj_orig_accuracy,
                    "Shuffled_Accuracy": subj_shuf_accuracy,
                    "Original_RSD": subj_orig_rsd,
                    "Shuffled_RSD": subj_shuf_rsd,
                    "Original_RStd": subj_orig_rstd,
                    "Shuffled_RStd": subj_shuf_rstd,
                    "Fluctuation_Rate": subj_fluc_rate,
                    "Original_CKLD": subj_orig_ckld,
                    "Shuffled_CKLD": subj_shuf_ckld
                })
    
    # 創建 DataFrame
    overall_df = pd.DataFrame(overall_metrics)
    language_df = pd.DataFrame(language_metrics)
    subject_df = pd.DataFrame(subject_metrics)
    
    # 儲存到CSV
    overall_df.to_csv("result/overall_metrics.csv", index=False, encoding="utf-8")
    language_df.to_csv("result/language_metrics.csv", index=False, encoding="utf-8")
    subject_df.to_csv("result/subject_metrics.csv", index=False, encoding="utf-8")
    
    # 印出概要資訊
    print("評估完成！")
    
    return overall_df, language_df, subject_df

def main():
    # 檔案路徑
    original_file = "mmlu_17subjects_2langs_100samples.json"
    shuffled_file = "shuffle_mmlu_17subjects_2langs_100samples.json"
    
    # 載入資料
    print(f"載入資料: {original_file} 和 {shuffled_file}")
    original_data, shuffled_data = load_data(original_file, shuffled_file)
    
    # 定義模型和格式
    model_names = ["gemini", "mistral"]
    format_names = ["base_format", "json_format", "xml_format", 
                   "json_input_text_output", "text_input_json_output"]
    
    # 評估和儲存指標
    overall_df, language_df, subject_df = evaluate_and_save_metrics(
        original_data, shuffled_data, model_names, format_names
    )
    
    return overall_df, language_df, subject_df

if __name__ == "__main__":
    overall_df, language_df, subject_df = main()