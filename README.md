# Zepto Data & AI Platform — Capstone

A single repository containing three internally connected capabilities required by the Certificate Program in Artificial Intelligence and Machine Learning:

- `/data_pipeline` — web scraping, cleaning, currency conversion, normalized SQLite storage and SQL/pandas analysis.
- `/analytics` — Titanic profiling, EDA, preprocessing, classification, imbalance analysis, hyperparameter tuning, regression and model persistence.
- `/support_assistant` — local embeddings, ChromaDB retrieval, LangGraph routing, deterministic mock generation, Pydantic output validation and FastAPI.

## Setup

This repository uses **one consolidated `requirements.txt` at the root**.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Run each module

### 1. Data pipeline
```bash
python data_pipeline/scrape_and_load.py
python data_pipeline/queries.py
```
The required conversion is the fixed project-defined **1 GBP = 105.50 INR**. No currency API is required.

### 2. Analytics
```bash
python analytics/01_eda.py
python analytics/02_modeling.py
```
The first script calls `sns.load_dataset('titanic')` only when the committed offline fallback `analytics/titanic.csv` is absent, immediately writes that CSV, and then performs EDA. The modeling script reads the CSV/cleaned CSV and does not reload the raw dataset from Seaborn.

### 3. Support assistant
```bash
python -m support_assistant.ingest
MOCK_LLM=1 uvicorn support_assistant.main:app --port 7860
```
The default mock path is the graded baseline and makes no LLM-provider call. See `/support_assistant/README.md` for curl examples and Docker commands.

## Design decisions

**Data pipeline:** scrape at least 60 books from the public practice site, parse fields defensively, median-impute numeric parsing failures, drop rows missing essential identity/category data, use the fixed GBP/INR rate, and normalize categories into a parent table referenced by books with a foreign key.

**Analytics:** use the requested missingness thresholds, separate EDA cleaning from modeling preprocessing, stratify the classification split before preprocessing, and enforce train-only fitting with scikit-learn `Pipeline`/`ColumnTransformer`. SMOTE is placed inside an imbalanced-learn pipeline so oversampling occurs only on the training fold. The saved model artifact includes preprocessing plus estimator.

**Support assistant:** use one chunk per policy document because the documents are short, embed locally with `all-MiniLM-L6-v2`, store vectors in ChromaDB using cosine similarity, route with a three-node LangGraph, and use deterministic mock responses by default. Only generation/classification behavior changes when `MOCK_LLM=0`; retrieval remains real in both modes.

## Required Git workflow

Create a feature branch, make at least two commits, then merge it into `main`:

```bash
git checkout -b feature/zepto-capstone
git add . && git commit -m "feat: add data pipeline"
git add . && git commit -m "feat: add analytics and support assistant"
git checkout main
git merge --no-ff feature/zepto-capstone -m "merge: zepto capstone feature"
git log --graph --oneline --all
```

## Submission checklist

Before publishing, run all three modules and commit the generated required artifacts: `data_pipeline/zepto_books.db` (or regeneration script), `data_pipeline/books_clean.csv`, SQL outputs, `analytics/titanic.csv`, EDA/model results, charts, `analytics/best_pipeline.joblib`, the ChromaDB data if desired, and the support-assistant example JSON transcripts. Do not add screenshots, PDFs, slides, audio or video.
