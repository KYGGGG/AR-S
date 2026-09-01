import torch
import joblib
import json
import numpy as np
import os
import sys
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from Summarization.qwen_generation_utils import make_context, decode_tokens
from peft import PeftModel, PeftConfig

PROJECT_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
MODULE_DIR = PROJECT_DIR / "Summarization"


def anonymize_and_finalize(summary_text: str) -> str:
    """Remove personal information and refine a response for consultation.

    Returns the original text if the optional OpenAI dependency, credentials,
    or remote request is unavailable, keeping the main pipeline operational.
    """
    try:
        from openai import OpenAI

        client = OpenAI()
        prompt = (
            "다음 텍스트를 익명화한 뒤 상담 답변 형태로 다듬어주세요.\n"
            "- 이름, 전화번호, 이메일, 주소, 주민번호 등 개인정보는 제거하거나 일반적인 표현으로 대체\n"
            "- 마지막에 '이 내용은 참고 사례이며, 최신 정책은 상담사 확인이 필요합니다.' 문구 포함\n"
            "- 존중하는 상담사 어조 사용\n\n"
            f"텍스트:\n{summary_text}\n\n최종 답변:"
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        print(f"답변 익명화 후처리 오류: {exc}")
        return summary_text

def setup_summarization_pipeline():
    try:
        adapter_path = MODULE_DIR / "output_qwen"
        config = PeftConfig.from_pretrained(str(adapter_path))
        
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '<|extra_0|>'})
        if tokenizer.eos_token is None:
            tokenizer.add_special_tokens({'eos_token': '<|endoftext|>'})
            
        model = AutoModelForCausalLM.from_pretrained(
            str(adapter_path),
            device_map="auto",
            trust_remote_code=True
        ).eval()
        
        generation_config = GenerationConfig(
            max_new_tokens=512,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3,
            temperature=1.0
        )
        
        return tokenizer, model, generation_config
    
    except Exception as e:
        print(f"텍스트 요약 모델 로드 중 오류 발생: {e}")
        return None, None, None

def summarize_text(text, tokenizer, model, generation_config):
    qa_prompt = (
        f"다음 질의를 읽고, 핵심 문맥이 드러나는 요약문을 한국어 및 숫자만 사용하여 200자 이내로 요약하시오.\n\n"
        f"Q: {text}\n"
        "요약 (200자 이내):"
    )
    
    inputs = tokenizer(qa_prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    outputs = model.generate(
        **inputs,
        generation_config=generation_config
    )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = decoded.split("요약 (200자 이내):")[-1].strip().split("\n")[0].split("<")[0].strip()
    return summary
