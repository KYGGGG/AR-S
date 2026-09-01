"""FastAPI application for the integrated AR-S pipeline.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from threading import Lock
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool


BASE_DIR = Path(os.getenv("HACKATHON_ROOT", "/opt/hackathon")).resolve()
RESULTS_FILE = BASE_DIR / "results.json"
TEMP_AUDIO_DIR = BASE_DIR / "temp_audio"

load_dotenv(BASE_DIR / ".env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AR-S System API",
    description="Integrated STT, KoBERT classification, Qwen summarization, and ChromaDB retrieval API",
    version="1.0.0",
)


class ARSRequest(BaseModel):
    text: str = Field(min_length=1)


class ClassificationResponse(BaseModel):
    original_text: str
    summary: str
    predicted_source: str
    predicted_category: str
    source_prob: float
    category_prob: float
    status_message: str


class FeedbackRequest(BaseModel):
    original_text: str
    predicted_source: str
    predicted_category: str
    user_feedback: bool
    reclassify_labels: list[str] = Field(default_factory=list)
    additional_text: str | None = None


class ARSResponse(BaseModel):
    original_text: str
    predicted_source: str
    predicted_category: str
    retrieved_answer: str
    status_message: str
    summary: str | None = None


class SummaryResponse(BaseModel):
    original_text: str
    summary: str


_resources: dict[str, Any] = {}
_resource_lock = Lock()


def _load_resources() -> dict[str, Any]:
    """Load model resources and the vector database once, on first use."""
    if _resources:
        return _resources

    with _resource_lock:
        if _resources:
            return _resources

        from Kobert.kobert_data_utils import load_inference_assets
        from RAG.Retrieval_utils import set_retrieval
        from STT.STT_utils import setup_stt_pipeline
        from Summarization.Summarization import setup_summarization_pipeline

        logger.info("모델 및 검색 DB를 로드합니다.")
        classifier_assets = load_inference_assets(
            checkpoint_path=str(BASE_DIR / "Kobert" / "Data" / "checkpoint.pt"),
            source_encoder_path=str(BASE_DIR / "Kobert" / "Data" / "source_encoder.pkl"),
            category_encoder_path=str(BASE_DIR / "Kobert" / "Data" / "category_encoder.pkl"),
            map_path=str(BASE_DIR / "Kobert" / "Data" / "main_to_sub_category_map.json"),
        )
        retrieval_client, retrieval_collection = set_retrieval(
            db_path=str(BASE_DIR / "RAG" / "db_path")
        )
        summary_tokenizer, summary_model, generation_config = setup_summarization_pipeline()
        setup_stt_pipeline()

        _resources.update(
            classifier_assets=classifier_assets,
            retrieval_client=retrieval_client,
            retrieval_collection=retrieval_collection,
            summary_tokenizer=summary_tokenizer,
            summary_model=summary_model,
            generation_config=generation_config,
        )
        logger.info("모델 및 검색 DB 로드가 완료됐습니다.")
        return _resources


async def _resources_or_503() -> dict[str, Any]:
    try:
        return await run_in_threadpool(_load_resources)
    except Exception as exc:
        logger.exception("리소스 로드 실패")
        raise HTTPException(status_code=503, detail=f"모델 로드 실패: {exc}") from exc


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "models_loaded": bool(_resources)}


static_dir = BASE_DIR / "static"
if static_dir.is_dir():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    index_file = static_dir / "index.html"
    if index_file.is_file():
        return HTMLResponse(index_file.read_text(encoding="utf-8"))
    return HTMLResponse(
        "<h1>AR-S System API</h1>"
        "<p>웹 UI는 아직 없으며 <a href='/docs'>/docs</a>에서 API를 사용할 수 있습니다.</p>"
    )


@app.post("/ars/classify", response_model=ClassificationResponse)
async def classify_ars_request(request: ARSRequest) -> ClassificationResponse:
    resources = await _resources_or_503()
    from Kobert.kobert_data_utils import classify_text
    from Summarization.Summarization import summarize_text

    try:
        classification = await run_in_threadpool(classify_text, request.text)
        summary = await run_in_threadpool(
            summarize_text,
            request.text,
            resources["summary_tokenizer"],
            resources["summary_model"],
            resources["generation_config"],
        )
        source_prob = classification["source_probs"].max().item()
        category_prob = classification["category_probs"].max().item()

        predicted_source = classification["predicted_source"]
        predicted_category = classification["predicted_category"]
        if predicted_source == "음성 민원":
            predicted_category = "음성 민원"
            message = "음성 민원으로 분류되어 자동 처리를 종료합니다."
        elif source_prob < 0.8:
            predicted_source = "처리불가"
            predicted_category = "처리불가"
            message = "분류 신뢰도가 낮아 상담사 확인이 필요합니다."
        else:
            message = "분류가 완료됐습니다. 사용자 피드백을 기다립니다."

        return ClassificationResponse(
            original_text=request.text,
            summary=summary,
            predicted_source=predicted_source,
            predicted_category=predicted_category,
            source_prob=source_prob,
            category_prob=category_prob,
            status_message=message,
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("분류 처리 실패")
        raise HTTPException(status_code=500, detail=f"분류 처리 실패: {exc}") from exc


@app.post("/ars/feedback", response_model=ARSResponse)
async def process_feedback(request: FeedbackRequest) -> ARSResponse:
    resources = await _resources_or_503()
    from Kobert.kobert_data_utils import reclassify_text
    from RAG.Retrieval_utils import get_most_similar_answer
    from Summarization.Summarization import anonymize_and_finalize, summarize_text

    try:
        if request.user_feedback:
            answer = await run_in_threadpool(
                get_most_similar_answer,
                resources["retrieval_collection"],
                request.original_text,
                request.predicted_source,
                request.predicted_category,
            )
            final_answer = await run_in_threadpool(anonymize_and_finalize, answer)
            return ARSResponse(
                original_text=request.original_text,
                predicted_source=request.predicted_source,
                predicted_category=request.predicted_category,
                retrieved_answer=final_answer,
                status_message="피드백을 반영해 유사 답변을 검색했습니다.",
            )

        combined_text = " ".join(
            part for part in (request.original_text, request.additional_text) if part
        )
        excluded = [*request.reclassify_labels, request.predicted_category]
        result = await run_in_threadpool(reclassify_text, combined_text, excluded)
        summary = await run_in_threadpool(
            summarize_text,
            combined_text,
            resources["summary_tokenizer"],
            resources["summary_model"],
            resources["generation_config"],
        )
        return ARSResponse(
            original_text=request.original_text,
            predicted_source=result["predicted_source"],
            predicted_category=result["predicted_category"],
            retrieved_answer=(
                f"{result['predicted_source']} - {result['predicted_category']} 관련 문의가 맞나요?"
            ),
            status_message="피드백에 따라 재분류했습니다.",
            summary=summary,
        )
    except Exception as exc:
        logger.exception("피드백 처리 실패")
        raise HTTPException(status_code=500, detail=f"피드백 처리 실패: {exc}") from exc


@app.post("/ars/summarize", response_model=SummaryResponse)
async def summarize_ars_request(request: ARSRequest) -> SummaryResponse:
    resources = await _resources_or_503()
    from Summarization.Summarization import summarize_text

    try:
        summary = await run_in_threadpool(
            summarize_text,
            request.text,
            resources["summary_tokenizer"],
            resources["summary_model"],
            resources["generation_config"],
        )
        return SummaryResponse(original_text=request.text, summary=summary)
    except Exception as exc:
        logger.exception("요약 처리 실패")
        raise HTTPException(status_code=500, detail=f"요약 처리 실패: {exc}") from exc


@app.post("/ars/audio")
async def process_audio(file: UploadFile = File(...)) -> dict[str, str]:
    await _resources_or_503()
    from STT.STT_utils import run_stt

    TEMP_AUDIO_DIR.mkdir(exist_ok=True)
    safe_name = Path(file.filename or "audio.bin").name
    temp_path = TEMP_AUDIO_DIR / f"{time.time_ns()}_{safe_name}"
    try:
        with temp_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                output.write(chunk)
        stt_result = await run_in_threadpool(run_stt, str(temp_path))
        return {"stt_result": stt_result}
    except Exception as exc:
        logger.exception("음성 처리 실패")
        raise HTTPException(status_code=500, detail=f"음성 처리 실패: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def append_to_results(log_data: dict[str, Any]) -> int:
    results: list[Any] = []
    if RESULTS_FILE.is_file():
        try:
            loaded = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                results = loaded
        except (json.JSONDecodeError, OSError):
            logger.warning("기존 results.json을 읽지 못해 새 목록을 생성합니다.")
    results.append(log_data)
    RESULTS_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(results)


@app.post("/ars/save_results")
async def save_results(request: Request) -> dict[str, Any]:
    try:
        saved_count = await run_in_threadpool(append_to_results, await request.json())
        return {"status": "ok", "saved_count": saved_count}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"결과 저장 실패: {exc}") from exc
