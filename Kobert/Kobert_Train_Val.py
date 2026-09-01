import torch
from torch import nn
from torch.optim import AdamW 
from tqdm.auto import tqdm

from kobert_data_utils import apply_hierarchical_constraint 

def train_epoch(model, data_loader, optimizer, source_criterion, category_criterion, device, main_to_sub_category_map):
    """
    epoch 동안 모델 학습

    멀티태스크 학습을 위해 source와 category 두 가지 손실을 계산,
    이 손실을 합산하여 역전파를 수행, 카테고리 예측에는 계층적 제약 조건을 적용

    Args:
        model (nn.Module): 학습시킬 모델.
        data_loader (DataLoader): 학습 데이터 로더.
        optimizer (AdamW): 모델 파라미터를 업데이트하는 옵티마이저.
        source_criterion (nn.Module): 출처 예측을 위한 손실 함수.
        category_criterion (nn.Module): 카테고리 예측을 위한 손실 함수.
        device (torch.device): 연산에 사용될 장치.
        main_to_sub_category_map (dict): 메인 카테고리를 서브 카테고리 인덱스에 매핑하는 딕셔너리.

    Returns:
        tuple:
            - avg_loss (float): 에포크 동안의 평균 손실.
            - source_accuracy (torch.Tensor): 출처 예측 정확도.
            - category_accuracy (torch.Tensor): 카테고리 예측 정확도.
    """
    model.train()
    total_loss = 0
    source_correct_predictions = 0
    category_correct_predictions = 0
    num_samples = 0

    weight_source = 0.5
    weight_category = 0.5 

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        source_labels = batch['source_label'].to(device)
        category_labels = batch['category_label'].to(device)

        optimizer.zero_grad()

        source_logits, category_logits = model(input_ids, attention_mask, token_type_ids)

        _, predicted_main_categories = torch.max(source_logits, dim=1)

        constrained_category_logits = apply_hierarchical_constraint(
            category_logits, 
            predicted_main_categories, 
            main_to_sub_category_map, 
            device
        )

        source_loss = source_criterion(source_logits, source_labels)
        category_loss = category_criterion(constrained_category_logits, category_labels)

        loss = (weight_source * source_loss) + (weight_category * category_loss)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, source_preds = torch.max(source_logits, dim=1)
        _, category_preds = torch.max(constrained_category_logits, dim=1) 

        source_correct_predictions += torch.sum(source_preds == source_labels)
        category_correct_predictions += torch.sum(category_preds == category_labels)
        num_samples += source_labels.size(0)

    avg_loss = total_loss / len(data_loader)
    source_accuracy = source_correct_predictions.double() / num_samples
    category_accuracy = category_correct_predictions.double() / num_samples

    return avg_loss, source_accuracy, category_accuracy

def eval_model(model, data_loader, source_criterion, category_criterion, device, main_to_sub_category_map):
    """
    학습된 모델로 데이터셋의 손실과 정확도를 계산하고, 예측 결과와 정답 레이블을 수집
    
    Args:
        model (nn.Module): 평가할 모델.
        data_loader (DataLoader): 평가 데이터 로더.
        source_criterion (nn.Module): 출처 예측을 위한 손실 함수.
        category_criterion (nn.Module): 카테고리 예측을 위한 손실 함수.
        device (torch.device): 연산에 사용될 장치.
        main_to_sub_category_map (dict): 메인 카테고리를 서브 카테고리 인덱스에 매핑하는 딕셔너리.

    Returns:
        tuple:
            - avg_loss (float): 평가 데이터셋의 평균 손실.
            - source_accuracy (torch.Tensor): 출처 예측 정확도.
            - category_accuracy (torch.Tensor): 카테고리 예측 정확도.
            - all_source_preds (list): 모든 출처 예측값.
            - all_source_labels (list): 모든 출처 정답 레이블.
            - all_category_preds (list): 모든 카테고리 예측값.
            - all_category_labels (list): 모든 카테고리 정답 레이블.
            - all_texts (list): 모든 원본 텍스트.
    """
    model.eval()
    total_loss = 0
    source_correct_predictions = 0
    category_correct_predictions = 0
    num_samples = 0

    all_source_preds = []
    all_source_labels = []
    all_category_preds = []
    all_category_labels = []
    all_texts = [] 

    weight_source = 0.5
    weight_category = 0.5

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            source_labels = batch['source_label'].to(device)
            category_labels = batch['category_label'].to(device)
            texts = batch['text']

            source_logits, category_logits = model(input_ids, attention_mask, token_type_ids)

            _, predicted_main_categories = torch.max(source_logits, dim=1)

            constrained_category_logits = apply_hierarchical_constraint(
                category_logits, 
                predicted_main_categories, 
                main_to_sub_category_map, 
                device
            )

            source_loss = source_criterion(source_logits, source_labels)
            category_loss = category_criterion(constrained_category_logits, category_labels) 

            loss = (weight_source * source_loss) + (weight_category * category_loss)
            total_loss += loss.item()

            _, source_preds = torch.max(source_logits, dim=1)
            _, category_preds = torch.max(constrained_category_logits, dim=1) 

            source_correct_predictions += torch.sum(source_preds == source_labels)
            category_correct_predictions += torch.sum(category_preds == category_labels)
            num_samples += source_labels.size(0)

            all_source_preds.extend(source_preds.cpu().numpy())
            all_source_labels.extend(source_labels.cpu().numpy())
            all_category_preds.extend(category_preds.cpu().numpy())
            all_category_labels.extend(category_labels.cpu().numpy())
            all_texts.extend(texts)

    avg_loss = total_loss / len(data_loader)
    source_accuracy = source_correct_predictions.double() / num_samples
    category_accuracy = category_correct_predictions.double() / num_samples

    return avg_loss, source_accuracy, category_accuracy, \
           all_source_preds, all_source_labels, \
           all_category_preds, all_category_labels, \
           all_texts