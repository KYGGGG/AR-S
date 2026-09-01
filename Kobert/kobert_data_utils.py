import warnings

warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel
from torch import nn
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import joblib
import json
import pandas as pd
from pathlib import Path
import os
from kobert_tokenizer import KoBERTTokenizer

PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
DATA_DIR = PROJECT_DIR / "Kobert" / "Data"

class KoBERTMultiTaskDataset(Dataset):
    def __init__(self, texts, sources, categories, tokenizer):
        """
        인스턴스 초기화 생성자

        Args:
            texts (list): 처리할 텍스트가 담긴 리스트
            sources (list): 각 텍스트에 해당하는 출처 레이블이 담긴 리스트
            categories (list): 각 텍스트에 해당하는 카테고리 레이블이 담긴 리스트
            tokenizer (transformers.PreTrainedTokenizer): 텍스트를 토큰화하는 데 사용할 토크나이저
        """
        self.texts = texts
        self.sources = sources
        self.categories = categories
        self.tokenizer = tokenizer
        self.model_max_len = 512 

    def __len__(self):
        """
        데이터셋의 전체 샘플 수를 반환

        Returns:
            int: 데이터셋의 총 길이.
        """
        return len(self.texts)

    def __getitem__(self, item):
        """
        주어진 인덱스에 해당하는 하나의 샘플 반환

        텍스트 토큰화, 해당 레이블과 함께 KoBERT 모델 입력에 맞는 형태로 변환하여 딕셔너리로 반환

        Args:
            item (int): 데이터셋에서 가져올 샘플의 인덱스

        Returns:
            dict: 다음 키를 포함하는 딕셔너리
                - 'text' (str): 원본 텍스트
                - 'input_ids' (torch.Tensor): 토큰화된 텍스트의 입력 ID
                - 'attention_mask' (torch.Tensor): 어텐션 마스크
                - 'token_type_ids' (torch.Tensor): 토큰 타입 ID
                - 'source_label' (torch.Tensor): 출처 레이블
                - 'category_label' (torch.Tensor): 카테고리 레이블
        """
        text = str(self.texts[item]) 
        source = self.sources[item]
        category = self.categories[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.model_max_len,
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'source_label': torch.tensor(source, dtype=torch.long),
            'category_label': torch.tensor(category, dtype=torch.long)
        }

def collate_fn(batch, tokenizer):
    """
    데이터로더에서 배치를 구성할 때, 배치 내 모든 시퀀스를 가장 긴 시퀀스에 맞춰 패딩하고 텐서들을 스택하는 함수

    Args:
        batch (list):
            `KoBERTMultiTaskDataset`의 `__getitem__` 메소드에서 반환된 딕셔너리들의 리스트
            각 딕셔너리는 'input_ids', 'attention_mask', 'token_type_ids' 등 포함
        tokenizer (transformers.PreTrainedTokenizer):
            패딩 토큰 ID를 가져오기 위해 사용되는 토크나이저

    Returns:
        dict:
            패딩 및 스택이 완료된 텐서들을 포함하는 딕셔너리
            - 'input_ids' (torch.Tensor): 패딩된 입력 ID 텐서
            - 'attention_mask' (torch.Tensor): 패딩된 어텐션 마스크 텐서
            - 'token_type_ids' (torch.Tensor): 패딩된 토큰 타입 ID 텐서
            - 'source_label' (torch.Tensor): 출처 레이블 텐서
            - 'category_label' (torch.Tensor): 카테고리 레이블 텐서
            - 'text' (list): 원본 텍스트 리스트
    """
    max_len = max([len(item['input_ids']) for item in batch])

    input_ids = []
    attention_mask = []
    token_type_ids = []
    source_labels = []
    category_labels = []
    texts = []

    for item in batch:
        padding_length = max_len - len(item['input_ids'])
        
        input_ids.append(torch.cat([item['input_ids'], torch.tensor([tokenizer.pad_token_id] * padding_length, dtype=torch.long)]))
        attention_mask.append(torch.cat([item['attention_mask'], torch.tensor([0] * padding_length, dtype=torch.long)]))
        token_type_ids.append(torch.cat([item['token_type_ids'], torch.tensor([0] * padding_length, dtype=torch.long)]))
        
        source_labels.append(item['source_label'])
        category_labels.append(item['category_label'])
        texts.append(item['text'])

    return {
        'input_ids': torch.stack(input_ids),
        'attention_mask': torch.stack(attention_mask),
        'token_type_ids': torch.stack(token_type_ids),
        'source_label': torch.stack(source_labels),
        'category_label': torch.stack(category_labels),
        'text': texts 
    }

class KoBERTMultiTaskClassifier(nn.Module):
    """
    KoBERT를 기반으로 하는 멀티태스크 분류 모델
    """
    def __init__(self, num_sources, num_categories):
        """
        인스턴스 초기화 생성자
        
        Args:
            num_sources (int): 출처(source) 클래스의 총 개수.
            num_categories (int): 카테고리(category) 클래스의 총 개수.
        
        """
        super(KoBERTMultiTaskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.source_classifier = nn.Linear(self.bert.config.hidden_size, num_sources)
        self.category_classifier = nn.Linear(self.bert.config.hidden_size, num_categories)

    def forward(self, input_ids, attention_mask, token_type_ids):
        """
        forward

        Args:
            input_ids (torch.Tensor): 토큰화된 입력 ID
            attention_mask (torch.Tensor): 어텐션 마스크
            token_type_ids (torch.Tensor): 토큰 타입 ID

        Returns:
            tuple:
                - source_logits (torch.Tensor)
                - category_logits (torch.Tensor)
        """
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = output.pooler_output

        pooled_output = self.dropout(pooled_output)

        source_logits = self.source_classifier(pooled_output)
        category_logits = self.category_classifier(pooled_output)

        return source_logits, category_logits
    
def apply_hierarchical_constraint(category_logits, predicted_main_categories, main_to_sub_category_map, device):
    """
    hierarchical_constraint 조건을 적용하여 예측된 중분류에 속하지 않는 모든 소분류의 로짓을 낮은 값으로 조정

    Args:
        category_logits (torch.Tensor):
            모델이 출력한 로짓 값. 형태는 (batch_size, total_sub_categories).
        predicted_main_categories (torch.Tensor):
            각 샘플에 대해 예측된 메인 중분류 인덱스. 형태는 (batch_size,).
        main_to_sub_category_map (dict):
            메인 카테고리 인덱스를 해당 소분류 인덱스 리스트로 매핑하는 딕셔너리.
            예: {0: [0, 1, 2], 1: [3, 4]}
        device (torch.device):
            tensor가 위치할 장치(CPU 또는 GPU).

    Returns:
        torch.Tensor:
            계층적 제약 조건이 적용된 새로운 로짓 텐서.
            허용되지 않는 서브 카테고리의 로짓은 -100으로 설정됩니다.
            형태는 category_logits와 동일합니다.
    """
    constrained_category_logits = category_logits.clone().to(device) 

    all_sub_indices_tensor = torch.arange(category_logits.size(1), device=device)

    for i in range(category_logits.size(0)):
        predicted_main_idx = predicted_main_categories[i].item()
        
        allowed_sub_indices = main_to_sub_category_map.get(predicted_main_idx, [])
        
        if not allowed_sub_indices:
            continue 

        not_allowed_indices = [
            idx.item() for idx in all_sub_indices_tensor 
            if idx.item() not in allowed_sub_indices
        ]

        if not not_allowed_indices:
            continue
        
        constrained_category_logits[i, not_allowed_indices] =  -100

    return constrained_category_logits

def preprocess_and_encode_data(df: pd.DataFrame):
    """
    제공된 DataFrame을 전처리하고 인코딩을 수행한 후,
    인코더와 매핑 정보를 파일로 저장합니다.

    Args:
        df (pd.DataFrame): 전처리할 데이터가 포함된 DataFrame.

    Returns:
        tuple: 전처리된 데이터프레임, 소스 인코더, 카테고리 인코더, 중분류-소분류 맵.
    """
    # Select the required columns and remove incomplete rows.
    df = df[['processed_consulting_content_combined_for_kobert', 'source', 'consulting_category']].dropna()
    
    # Encode source and category labels.
    source_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    
    df['source_encoded'] = source_encoder.fit_transform(df['source'])
    df['category_encoded'] = category_encoder.fit_transform(df['consulting_category'])

    joblib.dump(source_encoder, DATA_DIR / 'source_encoder.pkl')
    joblib.dump(category_encoder, DATA_DIR / 'category_encoder.pkl')
    
    # Build the hierarchical source-to-category mapping.
    main_to_sub_category_map = {}
    for _, row in df.iterrows():
        main_encoded = row['source_encoded']
        sub_encoded = row['category_encoded']
        
        if main_encoded not in main_to_sub_category_map:
            main_to_sub_category_map[main_encoded] = set()
        main_to_sub_category_map[main_encoded].add(sub_encoded)

    for main_id, sub_set in main_to_sub_category_map.items():
        main_to_sub_category_map[main_id] = sorted(list(sub_set))

    with (DATA_DIR / 'main_to_sub_category_map.json').open('w', encoding='utf-8') as f:
        serializable_map = {str(k): list(v) for k, v in main_to_sub_category_map.items()}
        json.dump(serializable_map, f, ensure_ascii=False, indent=4)
        
    return df, source_encoder, category_encoder, main_to_sub_category_map

def load_inference_assets(
    checkpoint_path: str | Path = DATA_DIR / 'checkpoint.pt',
    source_encoder_path: str | Path = DATA_DIR / 'source_encoder.pkl',
    category_encoder_path: str | Path = DATA_DIR / 'category_encoder.pkl',
    map_path: str | Path = DATA_DIR / 'main_to_sub_category_map.json'
):
    warnings.filterwarnings("ignore")
    
    global tokenizer, model, source_encoder, category_encoder, main_to_sub_category_map_loaded, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')

        source_encoder = joblib.load(source_encoder_path)
        category_encoder = joblib.load(category_encoder_path)

        with open(map_path, 'r', encoding='utf-8') as f:
            temp_map = json.load(f)
            main_to_sub_category_map_loaded = {int(k): v for k, v in temp_map.items()}
        
        num_sources = len(source_encoder.classes_)
        num_categories = len(category_encoder.classes_)

        model = KoBERTMultiTaskClassifier(num_sources, num_categories).to(device)
        model_state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(model_state_dict)
        model.eval()

        return tokenizer, model, source_encoder, category_encoder, main_to_sub_category_map_loaded

    except FileNotFoundError as e:
        print(f"ERROR: 필수 파일 로드 실패: {e}. 경로를 확인하세요.")
        raise RuntimeError(f"초기화 실패: {e}")
    except Exception as e:
        print(f"ERROR: 모델 로드 또는 초기화 중 오류 발생: {e}")
        raise RuntimeError(f"초기화 실패: {e}")
    
    
def classify_text(text: str) -> dict:
    """
    텍스트를 입력받아 출처와 카테고리를 계층적 제약 조건에 따라 분류하는 함수

    Args:
        text (str): 분류할 입력 텍스트.

    Returns:
        dict:
            분류 결과를 담은 딕셔너리.
            - 'text' (str): 입력 텍스트.
            - 'predicted_source_label' (str): 예측된 출처 레이블.
            - 'predicted_category_label' (str): 예측된 카테고리 레이블.
    """
    if model is None or tokenizer is None or source_encoder is None or category_encoder is None:
        raise RuntimeError("모델 또는 인코더가 로드되지 않았습니다. load_inference_assets를 먼저 호출하세요.")

    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512, 
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length' 
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        source_logits, category_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
    
    source_probs = torch.softmax(source_logits, dim=-1)
    predicted_source_id = torch.argmax(source_probs, dim=-1).item()
    predicted_source_label = source_encoder.inverse_transform([predicted_source_id])[0]

    predicted_main_categories_tensor = torch.tensor([predicted_source_id], dtype=torch.long).to(device)
    
    constrained_category_logits = apply_hierarchical_constraint(
        category_logits, 
        predicted_main_categories_tensor, 
        main_to_sub_category_map_loaded, 
        device
    )

    category_probs = torch.softmax(constrained_category_logits, dim=-1)
    predicted_category_id = torch.argmax(category_probs, dim=-1).item()
    predicted_category_label = category_encoder.inverse_transform([predicted_category_id])[0]

    result = [str(text), '\n', str(predicted_source_label), '\n', str(predicted_category_label)]
    
    return {
        "original_text": text,
        "predicted_source": predicted_source_label,
        "predicted_category": predicted_category_label,
        "source_probs": source_probs,
        "category_probs": category_probs,
    }
    
def reclassify_text(text: str, incorrect_category_labels: list) -> dict:
    """
    민원인이 지목한 잘못된 카테고리(부서) 목록을 제외하고 텍스트를 재분류하는 함수.

    Args:
        text (str): 재분류할 입력 텍스트.
        incorrect_category_labels (list): 민원인이 불일치한다고 지정한 카테고리 레이블 목록.

    Returns:
        dict: 재분류 결과를 담은 딕셔너리.
    """
    if model is None or tokenizer is None or source_encoder is None or category_encoder is None:
        raise RuntimeError("모델 또는 인코더가 로드되지 않았습니다. load_inference_assets를 먼저 호출하세요.")

    inputs = tokenizer(
        text,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
        padding='max_length'
    )

    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    with torch.no_grad():
        source_logits, category_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

    source_probs = torch.softmax(source_logits, dim=-1)
    predicted_source_id = torch.argmax(source_probs, dim=-1).item()
    predicted_source_label = source_encoder.inverse_transform([predicted_source_id])[0]

    predicted_main_categories_tensor = torch.tensor([predicted_source_id], dtype=torch.long).to(device)

    constrained_category_logits = apply_hierarchical_constraint(
        category_logits,
        predicted_main_categories_tensor,
        main_to_sub_category_map_loaded,
        device
    )

    for label in incorrect_category_labels:
        if label in category_encoder.classes_:
            incorrect_category_id = category_encoder.transform([label])[0]
            constrained_category_logits[:, incorrect_category_id] = -100

    category_probs = torch.softmax(constrained_category_logits, dim=-1)
    predicted_category_id = torch.argmax(category_probs, dim=-1).item()
    predicted_category_label = category_encoder.inverse_transform([predicted_category_id])[0]

    return {
        "original_text": text,
        "predicted_source": predicted_source_label,
        "predicted_category": predicted_category_label,
        "source_probs": source_probs,
        "category_probs": category_probs,
    }
