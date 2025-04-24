import json
from tqdm import tqdm
import argparse

from model import GeminiModel, MistralModel
from prompt_format import (
    base_format,
    json_format,
    xml_format,
    json_input_text_output,
    text_input_json_output
)

def run_experiment_with_model(model, model_name, questions, input_file):
 
    print(f"\n使用 {model_name} 模型進行實驗")
    
    # 定義實驗設置 - 主要格式比較
    main_experiments = [
        {
            "name": f"{model_name}_base_format",
            "prompt_generator": base_format,
            "format_type": "text",
            "description": "Base format: 純文字輸入，純文字輸出"
        },
        {
            "name": f"{model_name}_json_format",
            "prompt_generator": json_format,
            "format_type": "json",
            "description": "JSON format: JSON輸入，JSON輸出"
        },
        {
            "name": f"{model_name}_xml_format",
            "prompt_generator": xml_format,
            "format_type": "xml",
            "description": "XML format: XML輸入，XML輸出"
        }
    ]
    
    # 定義實驗設置
    json_variants = [
        {
            "name": f"{model_name}_base_format",
            "prompt_generator": base_format,
            "format_type": "text",
            "description": "Base: 純文字輸入和純文字輸出"
        },
        {
            "name": f"{model_name}_json_input_text_output",
            "prompt_generator": json_input_text_output,
            "format_type": "text",
            "description": "JSON輸入，純文字輸出"
        },
        {
            "name": f"{model_name}_text_input_json_output",
            "prompt_generator": text_input_json_output,
            "format_type": "json",
            "description": "純文字輸入，JSON輸出"
        },
        {
            "name": f"{model_name}_json_format",
            "prompt_generator": json_format,
            "format_type": "json",
            "description": "JSON輸入和JSON輸出"
        }
    ]
    
    # 合併所有實驗，共 3400 題
    all_experiments = []
    seen_names = set()
    
    for exp in main_experiments:
        if exp["name"] not in seen_names:
            all_experiments.append(exp)
            seen_names.add(exp["name"])
    
    for exp in json_variants:
        if exp["name"] not in seen_names:
            all_experiments.append(exp)
            seen_names.add(exp["name"])
    
    # 進行實驗
    for i, question in enumerate(tqdm(questions, desc=f"使用 {model_name} 處理問題")):
        print(f"\n處理問題 {i+1}/{len(questions)} 使用 {model_name}")
        
        # 為每個問題添加實驗結果
        for exp in all_experiments:
            exp_name = exp["name"]
            print(f"  執行 {exp['description']}")
            
            # 檢查是否已經有結果，避免重複處理
            # if "results" in question and exp_name in question["results"]:
            #    print(f"  已存在結果，跳過")
            #    continue
            
         
            # 生成提示並獲取回答
            prompt = exp["prompt_generator"](question)
            response = model.answer_question(prompt)
            answer = model.extract_answer(response, exp["format_type"])
            
            # 將結果加到問題中
            if "results" not in question:
                question["results"] = {}
            
            # 儲存結果但不儲存完整回應，只保存提取出的答案
            question["results"][exp_name] = {
                "answer": answer
            }
            
            # 列印提取出的答案
            print(f"  回答: {answer}")
                
            
            # 每處理完一個實驗就保存一次檔案，以防萬一程式中途停止
            with open(input_file, 'w', encoding='utf-8') as f:
                json.dump(questions, f, ensure_ascii=False, indent=2)

def run_experiment(input_file):
  
    print(f"讀取問題集: {input_file}")
    
    # 讀取問題
    with open(input_file, 'r', encoding='utf-8') as f:
        questions = json.load(f)
    print(f"共讀取 {len(questions)} 個問題")

   
    print("初始化 Gemini 模型")
    gemini_model = GeminiModel()
    
    # 使用 Gemini 模型進行實驗
    run_experiment_with_model(gemini_model, "gemini", questions, input_file)

    
    print("初始化 Mistral 模型")
    mistral_model = MistralModel()
    
    # 使用 Mistral 模型進行實驗
    run_experiment_with_model(mistral_model, "mistral", questions, input_file)

    
    print(f"\n更新最終結果到: {input_file}")
    with open(input_file, 'w', encoding='utf-8') as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用不同格式提示的問答 prompt 實驗")
    parser.add_argument("--input", "-i", type=str, default="mmlu_17subjects_2langs_100samples.json",
                       help="輸入 JSON 檔案路徑，也作為輸出檔案 (預設: mmlu_17subjects_2langs_100samples.json)")
    
    args = parser.parse_args()
    
    run_experiment(args.input)