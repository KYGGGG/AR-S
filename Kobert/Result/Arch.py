import torch
from torch import nn
from transformers import BertModel # KoBERTMultiTaskClassifier 정의에 필요
from torchviz import make_dot
import os

# Graphviz 설치 경로를 명시적으로 설정 (Windows 등에서 필요할 수 있음)
# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/' # Graphviz 설치 경로에 맞게 수정

# KoBERTMultiTaskClassifier 클래스 (기존 모델 코드)
class KoBERTMultiTaskClassifier(nn.Module):
    def __init__(self, num_sources, num_categories):
        super(KoBERTMultiTaskClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('skt/kobert-base-v1')
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)

        self.source_classifier = nn.Linear(self.bert.config.hidden_size, num_sources)
        self.category_classifier = nn.Linear(self.bert.config.hidden_size, num_categories)

    def forward(self, input_ids, attention_mask, token_type_ids):
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

def visualize_model_architecture(model: nn.Module, 
                                 input_shape=(1, 128), # 예시 입력 시퀀스 길이
                                 num_sources: int = 2, 
                                 num_categories: int = 5,
                                 filename: str = "kobert_multitask_classifier", 
                                 directory: str = "./model_architecture_diagrams",
                                 format: str = "png"):

    batch_size, seq_len = input_shape
    dummy_input_ids = torch.randint(0, 30000, input_shape) # 예시 vocab size 30000
    dummy_attention_mask = torch.ones(input_shape)
    dummy_token_type_ids = torch.zeros(input_shape)

    if next(model.parameters()).is_cuda:
        device = torch.device("cuda")
        dummy_input_ids = dummy_input_ids.to(device)
        dummy_attention_mask = dummy_attention_mask.to(device)
        dummy_token_type_ids = dummy_token_type_ids.to(device)
    else:
        device = torch.device("cpu")

    model.eval()

    try:
        source_logits, category_logits = model(dummy_input_ids, dummy_attention_mask, dummy_token_type_ids)
        
        # make_dot은 스칼라 출력을 선호하지만, 다중 출력이면 튜플로 전달 가능
        # 각 출력을 개별적으로 시각화하거나, 손실처럼 합쳐서 하나의 스칼라로 만들어 전달할 수도 있습니다.
        # 여기서는 두 출력을 모두 포함하는 계산 그래프를 만듭니다.
        # make_dot에 전달할 `params`는 선택적이며, 노드에 이름을 붙일 때 유용합니다.
        # 여기서는 모델의 이름을 명시적으로 부여합니다.
        
        # (source_logits, category_logits) 튜플의 그라디언트를 추적
        # make_dot은 보통 최종 스칼라 손실로부터 역추적하지만, 
        # 직접적으로 특정 텐서들로부터 그래프를 그릴 수도 있습니다.
        # 여기서는 각 로짓을 이름과 함께 make_dot에 전달하여 시각화합니다.
        
        # make_dot에 이름과 함께 텐서 딕셔너리로 전달
        output_dict = {'source_logits': source_logits, 'category_logits': category_logits}
        
        # 모델 파라미터에 대한 노드를 추가하여 아키텍처를 더 잘 표현
        graph = make_dot(output_dict, params=dict(model.named_parameters()))
        
        # 저장 디렉토리 생성
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filepath = os.path.join(directory, filename)
        graph.render(filepath, format=format, view=False) # view=True 시 자동 팝업
        print(f"모델 아키텍처 다이어그램이 '{filepath}.{format}'으로 저장되었습니다.")

    except Exception as e:
        print(f"모델 시각화 중 오류 발생: {e}")
        print("Graphviz가 제대로 설치되어 있고 PATH에 추가되었는지 확인하세요.")
        print("또한, 모델의 forward 메서드가 make_dot이 추적할 수 있도록 구성되어야 합니다.")

# 사용 예시:
if __name__ == "__main__":
    # 모델 인스턴스 생성 (임의의 클래스 수)
    num_sources = 5 # 예시
    num_categories = 20 # 예시
    model = KoBERTMultiTaskClassifier(num_sources, num_categories)
    
    # 모델을 CPU로 이동 (시각화는 보통 CPU에서 진행)
    model.to("cpu")

    # 시각화 함수 호출
    visualize_model_architecture(model, 
                                 input_shape=(1, 128), # 배치 1, 시퀀스 길이 128
                                 num_sources=num_sources, 
                                 num_categories=num_categories,
                                 filename="kobert_multitask_classifier_diagram", 
                                 directory="./model_diagrams",
                                 format="png")