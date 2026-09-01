import warnings

warnings.filterwarnings("ignore")

import torch
import os
import argparse
import json
from pathlib import Path

# Pipeline components
from Kobert.kobert_data_utils import load_inference_assets, classify_text, reclassify_text
from STT.STT_utils import setup_stt_pipeline, run_stt
from Summarization.Summarization import setup_summarization_pipeline, summarize_text
from RAG.Retrieval_utils import set_retrieval, get_most_similar_answer

BASE_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MP3 파일에서 텍스트를 추출하고, 이를 Qwen으로 요약한 뒤 KoBERT로 분류하는 통합 CLI")
    parser.add_argument('audio_file', type=str, help='분류 및 요약할 MP3 파일 경로를 입력하세요.')
    
    args = parser.parse_args()
    
    try:
        # Initialize classification and speech-recognition resources.
        setup_stt_pipeline()
        load_inference_assets(
            checkpoint_path=str(BASE_DIR / 'Kobert' / 'Data' / 'checkpoint.pt'),
            source_encoder_path=str(BASE_DIR / 'Kobert' / 'Data' / 'source_encoder.pkl'),
            category_encoder_path=str(BASE_DIR / 'Kobert' / 'Data' / 'category_encoder.pkl'),
            map_path=str(BASE_DIR / 'Kobert' / 'Data' / 'main_to_sub_category_map.json')
        )
        # Initialize the summarization model.
        tokenizer, summary_model, generation_config = setup_summarization_pipeline()
        
        # Initialize the ChromaDB retrieval collection.
        db_path = str(BASE_DIR / "RAG" / "db_path")
        client, retrieval_collection = set_retrieval(db_path)
        
        print("\n모든 모델 로드 완료. 이제 요약 및 분류를 시작할 수 있습니다.")
        
        stt_result = run_stt(args.audio_file)
        original_text = stt_result
        
        summary = summarize_text(original_text, tokenizer, summary_model, generation_config)
        print(f"원본 텍스트: {original_text}")
        print(f"요약된 내용: {summary}")
        
        final_result = classify_text(original_text)
        
        excluded_categories = []
        classification_history = []
        
        while True:
            record = {
                "step": len(classification_history) + 1,
                "original_text": original_text,
                "summary": summary,
                "predicted_source": final_result['predicted_source'],
                "predicted_category": final_result['predicted_category']
            }
            classification_history.append(record)
            
            print("\n--- 현재 분류 결과 ---")
            print(f"요약 텍스트: {summary}")
            print(f"예측된 기관: {final_result['predicted_source']}")
            print(f"예측된 부서: {final_result['predicted_category']}")
            
            user_feedback = input("\n위 분류 결과가 정확합니까? (Yes/No): ")
            
            if user_feedback.strip().lower() == "yes":
                print("\n✅ 분류 결과가 확정되었습니다.")
                record["feedback"] = "Yes"
                predicted_source = final_result['predicted_source']
                predicted_category = final_result['predicted_category']
                retrieved_answer = get_most_similar_answer(retrieval_collection, original_text, predicted_source, predicted_category)
                print(f"참고 답변: {retrieved_answer}")
                break 
            
            elif user_feedback.strip().lower() == "no":
                print("\n--- 재분류 프로세스 시작 ---")
                record["feedback"] = "No"
                excluded_categories.append(final_result['predicted_category'])
                
                reclassified_result = reclassify_text(
                    text=original_text,
                    incorrect_category_labels=excluded_categories
                )
                final_result = reclassified_result
            else:
                print("유효하지 않은 입력입니다. 'Yes' 또는 'No'를 입력해주세요.")
                
        output_path = BASE_DIR / "result.json"
        
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(classification_history, f, ensure_ascii=False, indent=4)
            print(f"\n모든 요약 및 분류 과정이 '{output_path}'에 저장되었습니다.")
            
    except RuntimeError as e:
        print(f"\n초기화 오류 발생: {e}")
        print("필요한 파일들이 올바른 경로에 있는지 확인하고 다시 시도하세요.")
