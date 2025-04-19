import google.generativeai as genai
import os
import json
from mistralai import Mistral
import re

class BaseModel:
    
    def extract_answer(self, response, format_type="text"):
        """從回應中提取答案，支援英文和中文"""
        
        # 標準化回應，移除格式
        normalized_response = response.replace("**", "").replace("$", "").replace("\\", "")
        
        # 如果回應 free txt
        if format_type == "text":
            
            # 英文
            en_match = re.search(r"(?i)Answer\s*:\s*([A-D])", normalized_response)
            if en_match:
                return en_match.group(1).upper()
                
            # 中文
            cn_patterns = [
                r"答案\s*[:：]\s*([A-D])",
                r"答\s*[:：]\s*([A-D])",
                r"選項\s*[:：]\s*([A-D])",
                r"选项\s*[:：]\s*([A-D])"
            ]
            
            for pattern in cn_patterns:
                cn_match = re.search(pattern, normalized_response)
                if cn_match:
                    return cn_match.group(1).upper()
            
            # 如果沒找到明確的格式，檢查是否有單獨的 A/B/C/D
            single_letter = re.search(r"(?<!\w)([A-D])(?!\w)", normalized_response)
            if single_letter:
                return single_letter.group(1).upper()
            
            # 都沒找到，嘗試返回第一個字母
            # for line in normalized_response.strip().split('\n'):
            #    if line.strip() and line.strip()[0].upper() in "ABCD":
            #        return line.strip()[0].upper()
            
            return response
        
        # 如果回應是 json
        elif format_type == "json":
            try:
                # 尋找 JSON 格式的答案
                for line in normalized_response.strip().split('\n'):
                    if "{" in line and "}" in line and "answer" in line.lower():
                        json_str = line.strip()
                        # 提取 JSON 部分
                        start = json_str.find("{")
                        end = json_str.rfind("}") + 1
                        if start >= 0 and end > start:
                            parsed = json.loads(json_str[start:end])
                            if "answer" in parsed:
                                return parsed["answer"].upper()
                
                # 嘗試解析整個回應
                parsed_json = json.loads(normalized_response)
                if isinstance(parsed_json, dict) and "answer" in parsed_json:
                    return parsed_json["answer"].upper()
                return parsed_json
            except json.JSONDecodeError:
                # JSON解析失敗時，使用文本方式再次提取
                return self.extract_answer(normalized_response, format_type="text")
            
        # 如果回應是 xml
        elif format_type == "xml":
            
            # 尋找 <answer>X</answer> 格式
            match = re.search(r"<answer>([A-D])</answer>", normalized_response)
            if match:
                return match.group(1).upper()
            
            # 如果沒找到，嘗試文本方式提取
            return self.extract_answer(normalized_response, format_type="text")
        
    
        return response


class GeminiModel(BaseModel):
    
    def __init__(self, api_key = "AIzaSyAYay20N3vkRCMrpbg65ZSfJ-k_Y0uxes8", model_name = "gemini-2.0-flash"):

        self.api_key = api_key 
        
        # 設置 API 金鑰並初始化模型
        genai.configure(api_key=self.api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
    
    def answer_question(self, prompt, temperature = 0.7, max_tokens = 1024) :
        
    
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        )
        return response.text

class MistralModel(BaseModel):
    
    
    def __init__(self, api_key = "klJYBumsimymwd5pADaw04ZvTMetcXYO", model_name = "mistral-large-latest"):

        self.api_key = api_key

        self.client = Mistral(api_key=self.api_key)
        self.model_name = model_name
    
    def answer_question(self, prompt, temperature = 0.7, max_tokens = 1024) :
        
        
           
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        
        chat_response = self.client.chat.complete(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # 提取回應內容
        return chat_response.choices[0].message.content
            
     
        
if __name__ == "__main__":
    question = "請回答以下選擇題：\n法國的首都是？\nA. 柏林\nB. 倫敦\nC. 巴黎\nD. 羅馬\n\n請只回答字母，格式為「Answer: X」，其中 X 是選項的字母。"
    print(question)
    print()

    print("===== 測試 Gemini 模型 =====")
    gemini_model = GeminiModel()
    print(gemini_model.answer_question(question))
    
    print("===== 測試 Mistral 模型 =====")
    mistral_model = MistralModel()
    print(mistral_model.answer_question(question))
