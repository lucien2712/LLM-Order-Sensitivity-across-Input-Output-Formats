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

def load_data(original_file, shuffled_file):
    with open(original_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    
    with open(shuffled_file, 'r', encoding='utf-8') as f:
        shuffled_data = json.load(f)
    
    return original_data, shuffled_data

def calculate_accuracy(questions, model_name, format_name):
    correct = 0
    total = 0
    
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"]
        correct_answer = question["Answer"]
        
        if model_answer.upper() == correct_answer.upper():
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0


def calculate_rsd(questions, model_name, format_name):
    '''
    RSD = √[(1/|Y|) * Σ(acc_i - acc)²] / acc
    
    其中：
    - acc_i 是第i個選項(A/B/C/D)作為正確答案時的精確度
    - acc 是整體平均精確度 = (Σacc_i) / |Y|
    - |Y| 是選項數量(此處為4)
    - Σ 表示從i=1到|Y|的總和
    '''
    
    accuracies = {"A": 0, "B": 0, "C": 0, "D": 0}
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # 統計每個選項作為正確答案的次數及被正確選中的次數
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"].upper()
        correct_answer = question["Answer"].upper()
        
        counts[correct_answer] += 1  # 此選項作為正確答案的總次數
        if model_answer == correct_answer:
            accuracies[correct_answer] += 1  # 模型正確選擇此選項的次數
    
    # 計算每個選項的精確度 (acc_i)
    class_accuracies = []
    for key in accuracies:
        if counts[key] > 0:
            class_accuracies.append(accuracies[key] / counts[key])
    
    # 計算平均精確度 (acc)
    mean_acc = np.mean(class_accuracies)
    # 計算標準差 (√(1/|Y|) * Σ(acc_i - acc)²)
    std_acc = np.std(class_accuracies)
    
    # 計算RSD = (標準差/平均值)
    return (std_acc / mean_acc) if mean_acc > 0 else 0


def calculate_rstd(questions, model_name, format_name):
    '''
    RStd = σ_R = [(Σ(R_i - R̄)²) / |Y|] ^ (1/2)
    
    - R_i = 當正確答案為選項i時的召回率 = 正確選擇i的次數 / i作為正確答案的總次數
    - R̄ = (ΣR_i) / |Y| 是平均召回率
    - |Y| 是選項數量(此處為4)
    '''
    recalls = {"A": 0, "B": 0, "C": 0, "D": 0}  # 各選項回答正確的次數
    counts = {"A": 0, "B": 0, "C": 0, "D": 0}   # 各選項作為正確答案的總次數
    
    for question in questions:
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"].upper()
        correct_answer = question["Answer"].upper()
        
        counts[correct_answer] += 1  # 增加此選項作為正確答案的計數
        if model_answer == correct_answer:
            recalls[correct_answer] += 1  # 模型答對時增加對應選項的正確計數
    
    # 計算每個選項的召回率
    recall_values = []
    for key in recalls:
        if counts[key] > 0:  # 避免除以零
            recall_values.append(recalls[key] / counts[key])
    
    # 返回召回率的標準差(若無數據則返回0)
    return np.std(recall_values)


def calculate_fluctuation_rate(original_questions, shuffled_questions, model_name, format_name):
    '''
    FR = (1/N) * Σ(r_orig(i) ≠ r_shuf(i))
    '''
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
    '''
    CKLD = Σ p_i * log(p_i/q_i)
    
    - p_i 是真實標籤在選項i上的分佈比例
    - q_i 是模型預測在選項i上的分佈比例
    - i∈{A,B,C,D} 表示選項
    '''
    # 計算真實答案分佈 (p)
    ground_truth_distribution = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    # 計算模型預測分佈 (q)
    model_distribution = {"A": 0, "B": 0, "C": 0, "D": 0}
    
    total = len(questions)
    
    for question in questions:
        # 獲取真實答案和模型答案
        correct_answer = question["Answer"].upper()
        model_answer = question["results"][f"{model_name}_{format_name}"]["answer"].upper()
        
        # 更新真實答案計數
        if correct_answer in ground_truth_distribution:
            ground_truth_distribution[correct_answer] += 1
        
        # 更新模型預測計數
        if model_answer in model_distribution:
            model_distribution[model_answer] += 1
    
    # 將計數轉換為比例
    for key in ground_truth_distribution:
        ground_truth_distribution[key] /= total
        model_distribution[key] /= total
    
    # 依照論文定義，p是真實標籤分佈，q是模型預測分佈
    p = np.array(list(ground_truth_distribution.values()))
    q = np.array(list(model_distribution.values()))
    
    # 替換為小值以避免除零錯誤
    p = np.where(p == 0, 1e-10, p)
    q = np.where(q == 0, 1e-10, q)
    
    return entropy(p, q)

def get_language(question):
    """判斷問題的語言"""
    return question.get("Language")

def get_subject(question):
    """獲取問題的學科領域"""
    return question.get("Subject")

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
    

    overall_df = pd.DataFrame(overall_metrics)
    language_df = pd.DataFrame(language_metrics)
    subject_df = pd.DataFrame(subject_metrics)
    
    
    overall_df.to_csv("result/overall_metrics.csv", index=False, encoding="utf-8")
    language_df.to_csv("result/language_metrics.csv", index=False, encoding="utf-8")
    subject_df.to_csv("result/subject_metrics.csv", index=False, encoding="utf-8")
 
    print("done!")
    
    return overall_df, language_df, subject_df

def main():
    
    original_file = "mmlu_17subjects_2langs_100samples.json"
    shuffled_file = "shuffle_mmlu_17subjects_2langs_100samples.json"

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