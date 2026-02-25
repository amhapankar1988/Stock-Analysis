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
short_description: An AI-powered trading terminal for scanning growth stocks using RAG.
---

# 🚀 AI Strategic Trader Pro

This AI-powered terminal automates the "Triple Screen" trading workflow: 
1. **Fundamentals**: Quarterly EPS and Sales growth > 25%.
2. **Technicals**: Relative Strength (RS) scoring and 200-SMA trend alignment.
3. **Catalyst**: Real-time news sentiment analysis via Alpha Vantage.

## 🧠 Features

- **RAG-Powered Strategy**: Upload your favorite trading books (PDFs) to create a custom "Brain" that the AI uses to audit stock setups.
- **Smart Filters**: Automatically screens for stocks with high relative volume, price > $20, and positive growth metrics.
- **Visual Gut-Check**: Integrated Finviz chart previews to spot VCP and Pivot patterns instantly.
- **Cloud Persistence**: Automatically syncs your indexed strategy books to a private Hugging Face dataset for use across sessions.

## 🛠️ Setup Instructions

### 1. Environment Secrets
To run this Space, you must add the following **Secrets** in your Space Settings:
* `WATSONX_APIKEY`: Your IBM Watsonx.ai API key.
* `WATSONX_PROJECT_ID`: Your IBM Watsonx Project ID.
* `ALPHA_VANTAGE_KEY`: Your API key from Alpha Vantage.
* `HF_TOKEN`: A Hugging Face Write Token for cloud backups.

### 2. Knowledge Base
Go to the **Strategy** tab and upload your PDF strategy guides. Click **Train AI**. This will build your vector database and save it to your private dataset repo.

## 📈 Usage
1. Open the **Terminal** tab.
2. Click **Run Triple Screen**.
3. View the AI verdict, technical stats, and chart setups for the top 3 momentum leaders in the market.

## ⚠️ Disclaimer
*This tool is for educational and research purposes only. Trading involves significant risk. Always consult with a financial advisor before making investment decisions.*
