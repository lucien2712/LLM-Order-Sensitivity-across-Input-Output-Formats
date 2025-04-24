import json

# Base format: 純文字輸入，純文字輸出
def base_format(row):
    QUERY_TEMPLATE = """
    Answer the following multiple choice question. The last line of your response
    should be of the following format: 'Answer: $LETTER' (without quotes) where
    LETTER is one of ABCD. Think step by step before answering.

    {Question}

    A) {A}
    B) {B}
    C) {C}
    D) {D}
    """.strip()
    
    return QUERY_TEMPLATE.format(Question=row["Question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"])

# JSON format: JSON輸入，JSON輸出
def json_format(row):
    
    question_obj = {
        "question": row["Question"],
        "options": {
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"]
        }
    }
    json_output = {
        "answer": "LETTER"
        }

    # 使用 indent=2 格式化 JSON
    formatted_json = json.dumps(question_obj, ensure_ascii=False, indent=2)
    json_output = json.dumps(json_output, indent=2)
    
    QUERY_TEMPLATE = """
    Answer the following multiple choice question. The last line of your response
    should be of the following JSON format: 
    {json_output} 
    where LETTER is one of ABCD. Think step by step before answering.

    {formatted_json}
    """.strip()
    
    return QUERY_TEMPLATE.format(formatted_json=formatted_json)

# XML format: XML輸入，XML輸出
def xml_format(row):
    # 創建縮排式 XML
    xml_content = f"""
                    <question>{row["Question"]}</question>
                    <options>
                        <option id="A">{row["A"]}</option>
                        <option id="B">{row["B"]}</option>
                        <option id="C">{row["C"]}</option>
                        <option id="D">{row["D"]}</option>
                    </options>
                    """
    
    QUERY_TEMPLATE = """
    Answer the following multiple choice question. The last line of your response
    should be of the following XML format:
    <answer>LETTER</answer>
    where LETTER is one of ABCD. Think step by step before answering.

    {xml_content}
    """.strip()
    
    return QUERY_TEMPLATE.format(xml_content=xml_content)

# JSON輸入，純文字輸出
def json_input_text_output(row):
    
    question_obj = {
        "question": row["Question"],
        "options": {
            "A": row["A"],
            "B": row["B"],
            "C": row["C"],
            "D": row["D"]
        }
    }
    
    # 使用 indent=2 格式化 JSON
    formatted_json = json.dumps(question_obj, ensure_ascii=False, indent=2)
    
    QUERY_TEMPLATE = """
    Answer the following multiple choice question. The last line of your response
    should be of the following format: 'Answer: $LETTER' (without quotes) where
    LETTER is one of ABCD. Think step by step before answering.

    {formatted_json}
    """.strip()
    
    return QUERY_TEMPLATE.format(formatted_json=formatted_json)

# 純文字輸入，JSON輸出
def text_input_json_output(row):

    json_output= {
        "answer": "LETTER"
        }
    json_output = json.dumps(json_output, indent=2)
    
    QUERY_TEMPLATE = """
    Answer the following multiple choice question. The last line of your response
    should be of the following JSON format:
    {json_output} 
    where LETTER is one of ABCD. Think step by step before answering.

    {Question}

    A) {A}
    B) {B}
    C) {C}
    D) {D}
    """.strip()
    
    return QUERY_TEMPLATE.format(Question=row["Question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"])