#%%
import os
import json
from pathlib import Path

def merge_json_files_in_folder(folder_path, output_filename="merged_data.json"):
    """
    지정된 폴더 내의 모든 JSON 파일을 하나의 JSON 배열로 합칩니다.

    Args:
        folder_path (str): JSON 파일이 있는 폴더 경로.
        output_filename (str): 합쳐진 JSON 파일의 이름.
    """
    merged_data = []
    
    if not os.path.isdir(folder_path):
        print(f"오류: '{folder_path}' 폴더를 찾을 수 없습니다.")
        return

    file_list = os.listdir(folder_path)
    
    json_files = [f for f in file_list if f.endswith('.json')]
    
    if not json_files:
        print(f"'{folder_path}' 폴더에 JSON 파일이 없습니다.")
        return
        
    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
            print(f"'{json_file}' 파일을 성공적으로 로드했습니다.")
        except json.JSONDecodeError:
            print(f"경고: '{json_file}' 파일이 유효한 JSON 형식이 아닙니다. 이 파일을 건너뜁니다.")
        except Exception as e:
            print(f"오류: '{json_file}' 파일을 처리하는 중 예외가 발생했습니다: {e}")

    output_path = os.path.join(folder_path, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
        
    print(f"\n총 {len(json_files)}개의 JSON 파일을 합쳐 '{output_path}'에 저장했습니다.")

import json
import csv
import re

def clean_text(text):
    """텍스트 내의 불필요한 공백과 줄바꿈 문자를 제거합니다."""
    # 유니코드 공백 문자 포함
    text = re.sub(r'[\s\ufeff\u200b]+', ' ', text)
    return text.strip()

def extract_qa_from_text(text):
    """'Q :'와 'A :'를 기준으로 질문과 답변을 추출합니다."""
    # "A :"를 기준으로 텍스트를 분할
    parts = text.split("A :", 1)
    
    if len(parts) < 2:
        return {"Q": clean_text(parts[0]), "A": None}
    
    question = clean_text(parts[0].replace("Q :", "", 1))
    answer = clean_text(parts[1])
    
    return {"Q": question, "A": answer}

def process_and_save_to_csv(input_json_file, output_csv_file):
    """
    JSON 파일에서 QA 데이터를 추출하여 CSV 파일로 저장합니다.
    각 필드는 큰따옴표로 묶어 내부 쉼표 오류를 방지합니다.
    """
    try:
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: '{input_json_file}' 파일을 찾을 수 없습니다.")
        return
    except json.JSONDecodeError:
        print(f"오류: '{input_json_file}' 파일이 유효한 JSON 형식이 아닙니다.")
        return

    # CSV 파일에 쓰기 (quoting=csv.QUOTE_ALL 옵션 추가)
    with open(output_csv_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
        fieldnames = ['Q', 'A']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)

        writer.writeheader()
        
        for item in data:
            if 'consulting_content' in item:
                content = item['consulting_content']
                qa_pair = extract_qa_from_text(content)
                
                if qa_pair['Q'] and qa_pair['A']:
                    writer.writerow({'Q': qa_pair['Q'], 'A': qa_pair['A']})
    
    print(f"데이터가 '{output_csv_file}'에 성공적으로 저장되었습니다.")

# 사용 예시
# 가정: 'merged_data.json' 파일에 모든 데이터가 합쳐져 있습니다.
if __name__ == '__main__':
    project_dir = Path(
        os.getenv("HACKATHON_ROOT", "/opt/hackathon")
    ).resolve()
    module_dir = project_dir / "RAG" / "Preprocessing"
    process_and_save_to_csv(
        module_dir / "merged_data.json",
        module_dir / "qa_data.csv",
    )
