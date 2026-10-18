# SupportSense AI

[![Run Tests](https://github.com/zarnab123/supportsense-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/zarnab123/supportsense-ai/actions)

An end-to-end banking customer-support intelligence copilot. It classifies incoming support tickets, suggests urgency, and retrieves relevant support guidance for a human agent to review.

> Portfolio project by Zarnab Jawadani. This tool provides decision support only; it does not make final customer-service or fraud decisions.

## Problem

Banking support teams receive large volumes of tickets about card payments, cash withdrawals, transfers, card delivery, and potential fraud. Manual routing can be slow and inconsistent.

SupportSense AI helps an agent quickly understand:

- What the ticket is about
- Whether it may need urgent attention
- Which support-policy guidance is relevant
- Why human review is still required

## Features

- **77-class intent classification** using the Banking77 dataset
- **84% held-out test accuracy** using TF-IDF features and a LinearSVC classifier
- **Priority suggestion** using transparent safety-focused rules
- **Retrieval-augmented guidance** using TF-IDF cosine similarity over a curated demo policy knowledge base
- **FastAPI REST API** with interactive Swagger documentation
- **Automated tests** for priority logic, API validation, and RAG guidance
- **GitHub Actions CI** runs tests automatically after every push

## How It Works

1. A customer support ticket is sent to the `/triage` API endpoint.
2. The ML classifier predicts one of 77 banking support categories.
3. A transparent rule layer suggests `HIGH` or `NORMAL` priority.
4. The retriever finds the most relevant policy chunk from the knowledge base.
5. The API returns the prediction, priority, confidence indicator, guidance, and a human-review flag.

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Machine learning | scikit-learn, TF-IDF, LinearSVC |
| API | FastAPI, Uvicorn, Pydantic |
| Retrieval | TF-IDF and cosine similarity |
| Testing | pytest, FastAPI TestClient |
| CI | GitHub Actions |
| Dataset | Banking77 |

## Project Structure

```text
supportsense-ai/
├── app.py
├── data/
│   └── raw/
├── knowledge_base/
│   └── banking_support_policies.md
├── models/
│   └── ticket_classifier.joblib
├── src/
│   ├── eda.py
│   ├── predict.py
│   ├── retriever.py
│   └── train.py
├── tests/
│   ├── test_api.py
│   └── test_predict.py
├── .github/workflows/ci.yml
├── requirements.txt
└── README.md
## Run the Streamlit dashboard

After training the model, start the interactive dashboard:

```bash
streamlit run ui.py