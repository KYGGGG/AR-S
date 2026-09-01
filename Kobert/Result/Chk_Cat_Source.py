#%%
import pandas as pd
from pathlib import Path
import os

PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
KOBERT_DIR = PROJECT_DIR / "Kobert"
MODULE_DIR = KOBERT_DIR / "Result"

# Pandas DataFrame 출력 설정 (텍스트가 잘리지 않도록)
pd.set_option('display.max_rows', None)        # 모든 행 출력
pd.set_option('display.max_columns', None)     # 모든 컬럼 출력
pd.set_option('display.width', None)           # 출력 너비 제한 없앰
pd.set_option('display.max_colwidth', None)    # 컬럼 내용 잘리지 않도록

df = pd.read_csv(KOBERT_DIR / 'Data' / 'merged_all.csv')
results_df = pd.read_csv(MODULE_DIR / 'Chk_Category_Source.csv')

# 원본 데이터프레임 `df`에서 유효한 Source-Category 짝을 추출
valid_pairs_df = df[['source', 'consulting_category']].drop_duplicates()
valid_source_category_map = {}
for source, category in valid_pairs_df.itertuples(index=False):
    if source not in valid_source_category_map:
        valid_source_category_map[source] = set()
    valid_source_category_map[source].add(category)

# 예측된 Source-Category 짝이 유효하지 않은 경우를 판별하는 함수
def is_invalid_pair(row, valid_map):
    predicted_source = str(row['Predicted Source']) # str로 명시적 변환
    predicted_category = str(row['Predicted Category']) # str로 명시적 변환
    
    # 예측된 Source가 유효한 Source 목록에 없거나,
    # 예측된 Source에 대해 예측된 Category가 유효한 Category 목록에 없는 경우
    if predicted_source not in valid_map or \
       predicted_category not in valid_map[predicted_source]:
        return True
    return False

# 잘못된 짝인 경우만 필터링
invalid_pair_predictions = results_df[
    results_df.apply(lambda row: is_invalid_pair(row, valid_source_category_map), axis=1)
]

print(f"\n총 {len(invalid_pair_predictions)}개의 예측에서 Source-Category 짝이 잘못되었습니다.")
if not invalid_pair_predictions.empty:
    print("\n[잘못된 Source-Category 짝의 모든 예시 (Text, 실제 Source, 예측 Source, 실제 Category, 예측 Category)]:")
    print(invalid_pair_predictions[['Text', 'True Source', 'Predicted Source', 'True Category', 'Predicted Category']].to_string(index=False))
else:
    print("잘못된 Source-Category 짝 예측이 없습니다.")


# True Source는 맞았지만, 예측된 Category가 해당 True Source의 유효한 Category가 아닌 경우를 판별하는 함수
def is_invalid_category_for_true_source(row, valid_map):
    true_source = str(row['True Source']) # str로 명시적 변환
    predicted_category = str(row['Predicted Category']) # str로 명시적 변환
    
    # True Source가 맵에 있고, 예측된 Category가 해당 True Source의 유효한 Category 목록에 없는 경우
    if true_source in valid_map and \
       predicted_category not in valid_map[true_source]:
        return True
    return False

invalid_category_for_true_source_predictions = results_df[
    (results_df['Source Correct'] == True) & # Source 예측은 맞았는데
    results_df.apply(lambda row: is_invalid_category_for_true_source(row, valid_source_category_map), axis=1)
]

invalid_category_for_true_source_predictions
# %%
