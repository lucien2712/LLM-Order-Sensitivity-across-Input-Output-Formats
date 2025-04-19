import json
import random
import copy
import os

def shuffle_options(question):
    """打亂問題選項順序，並更新正確答案的選項標記"""
    # 創建一個問題的深度拷貝，避免修改原問題
    new_question = copy.deepcopy(question)
    
    # 獲取原始正確答案的索引 (A=0, B=1, C=2, D=3)
    original_answer = new_question.get("Answer", "")
    if not original_answer or original_answer not in "ABCD":
        print(f"警告: 問題缺少有效答案或答案格式不正確: {original_answer}")
        return new_question
    
    answer_index = ord(original_answer) - ord('A')
    
    # 獲取原始選項內容
    options = [
        new_question.get("A", ""),
        new_question.get("B", ""),
        new_question.get("C", ""),
        new_question.get("D", "")
    ]
    
    # 記錄正確答案的內容
    correct_option_content = options[answer_index]
    
    # 創建選項的索引列表，然後打亂
    indices = list(range(4))
    random.shuffle(indices)
    
    # 根據打亂後的索引重新分配選項
    new_options = [options[i] for i in indices]
    new_question["A"] = new_options[0]
    new_question["B"] = new_options[1]
    new_question["C"] = new_options[2]
    new_question["D"] = new_options[3]
    
    # 找出正確答案的新位置
    for i, content in enumerate(new_options):
        if content == correct_option_content:
            new_answer = chr(ord('A') + i)
            new_question["Answer"] = new_answer
            break
    
    return new_question

def shuffle_questions_file():
    """打亂 JSON 檔案中每個問題的選項順序"""
    # 固定輸入檔案
    input_file = "mmlu_17subjects_2langs_100samples.json"
    
    # 設置輸出檔案名稱
    output_file = f"shuffle_{input_file}"
    
    print(f"讀取問題檔案: {input_file}")
    
    try:
        # 讀取問題集
        with open(input_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
    except Exception as e:
        print(f"讀取檔案時發生錯誤: {e}")
        return
    
    print(f"共讀取 {len(questions)} 個問題")
    
    # 處理每個問題
    shuffled_questions = []
    for i, question in enumerate(questions):
        # 打亂選項順序
        shuffled_question = shuffle_options(question)
        shuffled_questions.append(shuffled_question)
        
        # 顯示進度
        if (i + 1) % 10 == 0 or i == 0 or i == len(questions) - 1:
            print(f"已處理 {i+1}/{len(questions)} 個問題")
    
    # 儲存結果
    print(f"儲存打亂後的問題到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(shuffled_questions, f, ensure_ascii=False, indent=2)
    
    print("處理完成!")

if __name__ == "__main__":
    shuffle_questions_file()