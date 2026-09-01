# AR-S: Intelligent Civil Complaint Response System

AR-S is an end-to-end AI pipeline for processing voice-based civil complaints. It converts incoming audio into text, summarizes the request, classifies the responsible organization and complaint category, and retrieves a relevant response from a vector database.

The project combines speech recognition, KoBERT-based hierarchical classification, Qwen-based summarization, and ChromaDB retrieval behind both a command-line interface and a FastAPI server.

## Pipeline

```text
Audio Input
    -> Speech-to-Text
    -> Complaint Summarization
    -> Organization and Category Classification
    -> User Feedback and Optional Reclassification
    -> Similar-Case Retrieval
    -> Anonymized Final Response
```

## Project Structure

```text
hackathon/
|-- main.py                  # FastAPI application
|-- classifier.py            # End-to-end CLI application
|-- Kobert/                  # Hierarchical complaint classifier
|   `-- Data/                # Encoders, label map, and model checkpoint
|-- STT/                     # Speech recognition pipeline
|-- Summarization/           # Qwen summarization and response post-processing
|-- RAG/                     # ChromaDB retrieval and preprocessing
|-- requirements-api.txt     # Web API dependencies
`-- README.md
```

## Key Features

- Speech-to-text conversion for uploaded complaint recordings
- KoBERT-based multi-task classification of organizations and categories
- Hierarchical constraints for valid organization-category predictions
- Qwen-based complaint summarization
- Confidence-aware routing for low-confidence predictions
- User-feedback-driven reclassification
- ChromaDB similarity search for relevant complaint responses
- Optional OpenAI-based anonymization and response refinement
- Lazy model loading to keep API startup lightweight

## Getting Started

The default Linux deployment path is:

```text
/opt/hackathon
```

Run all commands from this directory:

```bash
cd /opt/hackathon
```

### Web API

Install the API dependencies:

```bash
python3 -m pip install -r /opt/hackathon/requirements-api.txt
```

Start the server:

```bash
cd /opt/hackathon
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Open the interactive API documentation at:

```text
http://localhost:8000/docs
```

The health endpoint is available at `GET /health`. Large models and the vector database are loaded only when an inference endpoint is called for the first time.

### Command-Line Interface

Process an audio file through the complete pipeline:

```bash
python3 /opt/hackathon/classifier.py \
  /opt/hackathon/Test.mp3
```

## API Overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check server status and model-loading state |
| `POST` | `/ars/audio` | Transcribe an uploaded audio file |
| `POST` | `/ars/classify` | Summarize and classify complaint text |
| `POST` | `/ars/feedback` | Retrieve an answer or reclassify from feedback |
| `POST` | `/ars/summarize` | Summarize complaint text |
| `POST` | `/ars/save_results` | Store a processing result locally |

## Required Runtime Assets

Model artifacts and generated databases are excluded from Git because of their size and potential data sensitivity. Obtain them separately and place them at the following paths before running inference:

| Asset | Expected Path |
| --- | --- |
| KoBERT classification checkpoint | `/opt/hackathon/Kobert/Data/checkpoint.pt` |
| Source and category encoders | `/opt/hackathon/Kobert/Data/source_encoder.pkl`, `/opt/hackathon/Kobert/Data/category_encoder.pkl` |
| Hierarchical label mapping | `/opt/hackathon/Kobert/Data/main_to_sub_category_map.json` |
| Qwen adapter and training output | `/opt/hackathon/Summarization/output_qwen/` |
| ChromaDB vector database | `/opt/hackathon/RAG/db_path/` |
| Private complaint datasets | `/opt/hackathon/Kobert/Data/merged_all.csv`, `/opt/hackathon/RAG/Preprocessing/` |

All application paths resolve from `/opt/hackathon` by default. To deploy the project elsewhere, set `HACKATHON_ROOT` to another absolute Linux path.

## Acknowledgement

This repository was developed with support from the 서울시립대학교 데이터 사이언스 플러스 차세대 융합인재 양성사업단 - http://dsplus.uos.ac.kr/
