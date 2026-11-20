# Customer Feedback Insight Analyzer
Author: Kannvi (UID 24BAI70705) - Industrial Internship: Azure AI Fundamentals

Turns customer feedback (text, CSV, or scanned image) into sentiment, key phrases and a short summary
using Azure AI Vision (OCR), Azure AI Language and Azure OpenAI.

## Run
    pip install -r requirements.txt
    python run.py                 # CLI on data/sample_feedback.csv
    streamlit run app.py          # web UI
    pytest tests                  # offline tests

With no Azure variables set it runs in **offline demo mode** (simple word-list rules, only for demo/tests).
To use real Azure services: create Language, Vision and Azure OpenAI resources (free tier F0 where
available), copy `.env.example` values into environment variables, and run again.
Delete the resource group when done to avoid cost.
