---
title: Stock Analysis
emoji: 📈
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.16.0
app_file: app.py
pinned: true
license: mit
short_description: AI trading terminal using Finviz & RAG strategy auditing.
---

📈 Trading Brain AI: Strategic Stock Auditor
A professional-grade stock market terminal built with Streamlit and LangChain. This app automates the "Triple Screen" audit process by combining real-time market data from Alpha Vantage, fundamental screening via Finviz, and custom strategy knowledge using a RAG (Retrieval-Augmented Generation) architecture powered by IBM WatsonX.

✨ Key Features
Automated Screener: Filters stocks based on high-growth criteria (EPS > 25%, Sales > 25%, Price > SMA200).

RAG Strategy Engine: Upload your trading books (PDFs) to "train" the AI on your specific rules.

Triple Screen Audit: 1.  Technical: Relative Strength (RS) scoring against the S&P 500 (SPY).
2.  Fundamental: Q/Q growth metrics via Finviz.
3.  Strategic: AI-driven audit based on your uploaded knowledge base.

Cloud Persistence: Automatically syncs your FAISS vector database to Hugging Face Datasets for persistent memory on ephemeral cloud hosting.

🚀 Deployment Guide (Streamlit Cloud)
1. Requirements
Ensure your repository contains a requirements.txt with:
streamlit, pandas, alpha_vantage, finvizfinance, huggingface_hub, langchain-ibm, langchain-huggingface, faiss-cpu, and pypdf.

2. Configure Secrets
In the Streamlit Cloud dashboard, navigate to Settings > Secrets and paste the following (replacing with your actual keys):

Ini, TOML
HF_TOKEN = "your_huggingface_write_token"
ALPHA_VANTAGE_KEY = "your_alpha_vantage_api_key"
WATSONX_APIKEY = "your_ibm_watsonx_apikey"
WATSONX_PROJECT_ID = "your_watsonx_project_id"
DATASET_REPO_ID = "your-username/trading-brain-db"
3. Usage
Training: Go to the Strategy tab and upload PDF books containing your trading rules. Click Train AI.

Scanning: Go to the Terminal tab and click Run Strategic Scan.

Auditing: Review the AI-generated "Audit Verdicts" which compare current market data against your uploaded strategies.

🛠️ Tech Stack
UI Framework: Streamlit

LLM: Meta-Llama-3.3-70B-Instruct (via IBM WatsonX)

Embeddings: Sentence-Transformers (via Hugging Face Inference API)

Vector Store: FAISS

Data APIs: Alpha Vantage & Finviz

⚠️ Important Notes
API Limits: The Alpha Vantage free tier is limited to 25 requests per day in this configuration.

Memory Management: This app is optimized for the Streamlit Free Tier (1GB RAM) by using API-based embeddings rather than local model loading.
