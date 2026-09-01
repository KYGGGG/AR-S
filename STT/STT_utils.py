import warnings

warnings.filterwarnings("ignore")

import torch
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from Kobert.kobert_data_utils import classify_text

stt_pipeline = None
device_stt = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype_stt = torch.float16 if torch.cuda.is_available() else torch.float32
model_id_stt = "openai/whisper-large-v3"

def setup_stt_pipeline():
    """Initialize and cache the speech-to-text pipeline."""
    global stt_pipeline
    
    warnings.filterwarnings("ignore")

    model_stt = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id_stt, torch_dtype=torch_dtype_stt, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model_stt.to(device_stt)
    processor_stt = AutoProcessor.from_pretrained(model_id_stt)

    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_stt,
        return_timestamps=True,
        tokenizer=processor_stt.tokenizer,
        feature_extractor=processor_stt.feature_extractor,
        torch_dtype=torch_dtype_stt,
        device=device_stt,
    )

def run_stt(audio_path: str):
    """Transcribe an audio file with the initialized STT pipeline.

    Args:
        audio_path: Path to the input audio file.
    """
    if stt_pipeline is None:
        raise RuntimeError("STT 파이프라인이 로드되지 않았습니다. setup_stt_pipeline()을 먼저 호출하세요.")
    
    stt_result = stt_pipeline(audio_path)
    extracted_text = stt_result["text"]
      
    return extracted_text
