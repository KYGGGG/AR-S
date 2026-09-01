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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_beams=3,
            early_stopping=True,
            repetition_penalty=1.0,
            no_repeat_ngram_size=3
        )
        
        return tokenizer, model, generation_config
    except Exception as e:
        print(f"텍스트 요약 모델 로드 중 오류 발생: {e}")
        return None, None, None

def summarize_text(text, tokenizer, model, generation_config):
    qa_prompt = (
        f"다음 질의를 읽고, 핵심 문맥이 드러나게 무조건 **한국어**로 200자 이내로 요약하시오.\n\n"
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
