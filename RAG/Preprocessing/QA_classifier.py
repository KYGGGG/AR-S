#%%
import os
import sys
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
MODULE_DIR = PROJECT_DIR / "RAG" / "Preprocessing"
sys.path.append(str(PROJECT_DIR))

# kobert_data_utils.py 파일에서 필요한 함수들을 임포트합니다.
from Kobert.kobert_data_utils import load_inference_assets, classify_text

def process_qa_data_and_classify():
    """
    qa_data.csv 파일을 읽어와 각 텍스트를 분류하고 결과를 출력합니다.
    """
    # 1. 모델과 필요한 자산 로드
    # 'Kobert/Data/' 폴더 경로가 맞는지 확인하세요.
    print("모델 및 추론 자산 로드 중...")
    try:
        load_inference_assets(
            checkpoint_path=PROJECT_DIR / 'Kobert' / 'Data' / 'checkpoint.pt',
            source_encoder_path=PROJECT_DIR / 'Kobert' / 'Data' / 'source_encoder.pkl',
            category_encoder_path=PROJECT_DIR / 'Kobert' / 'Data' / 'category_encoder.pkl',
            map_path=PROJECT_DIR / 'Kobert' / 'Data' / 'main_to_sub_category_map.json'
        )
        print("로드 완료.\n")
    except RuntimeError as e:
        print(f"로드 실패: {e}")
        return

    # 2. qa_data.csv 파일 로드
    qa_file_path = MODULE_DIR / 'qa_data.csv'
    if not qa_file_path.exists():
        print(f"오류: '{qa_file_path}' 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        return

    print(f"'{qa_file_path}' 파일 로드 중...")
    df = pd.read_csv(qa_file_path)
    print("로드 완료.\n")

    # 3. 각 행의 Q 텍스트를 분류하고 결과 출력
    results = []
    print("각 질문(Q)에 대한 분류를 시작합니다...")
    for index, row in df.iterrows():
        question_text = row['Q']
        answer_text = row['A']
        if pd.isna(question_text):
            continue

        try:
            # classify_text 함수 호출
            classification_result = classify_text(question_text)
            
            # 결과 저장
            results.append({
                'Q': question_text,
                'A': answer_text,
                'Predicted_Source': classification_result['predicted_source'],
                'Predicted_Category': classification_result['predicted_category']
            })
            print(f"질문 {index+1} 분류 완료.")
        except Exception as e:
            print(f"질문 {index+1} 분류 중 오류 발생: {e}")
            continue

    # 4. 분류 결과를 새 CSV 파일로 저장
    output_df = pd.DataFrame(results)
    df_sorted = output_df.sort_values(by=['Predicted_Source', 'Predicted_Category'], ascending=True)
    df_sorted.to_csv(MODULE_DIR / 'classified_qa_data.csv', index=False, encoding='utf-8-sig')
    print(f"\n모든 질문의 분류가 완료되었고, 결과는 'classified_qa_data.csv'에 저장되었습니다.")

# 스크립트 실행
if __name__ == '__main__':
    process_qa_data_and_classify()
# %%
