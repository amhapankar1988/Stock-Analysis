import os
import gradio as gr
import pandas as pd
import yfinance as yf
from finvizfinance.screener.overview import Overview
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION & SECRETS ---
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = "https://ca-tor.ml.cloud.ibm.com"

# Initialize ChatWatsonx (The modern LangChain-IBM standard)
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-1-70b-instruct",
    url=WATSONX_URL,
    project_id=PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params={
        "decoding_method": "sample",
        "max_new_tokens": 800,
        "temperature": 0.2,
    }
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- STRATEGIC CALCULATIONS ---

def calculate_rs(ticker, hist, spy_hist):
    """Calculates weighted Relative Strength vs SPY."""
    stock_perf = (hist['Close'].iloc[-1] / hist['Close'].iloc[0]) - 1
    spy_perf = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]) - 1
    # RS Score: % Outperformance
    return round((stock_perf - spy_perf) * 100, 2)

def check_pivot_status(hist):
    """Checks if price is within 5% of 52-week high (Livermore Pivot)."""
    high_52w = hist['High'].max()
    curr_price = hist['Close'].iloc[-1]
    is_at_pivot = curr_price >= (high_52w * 0.95)
    return is_at_pivot, curr_price, high_52w

# --- CORE APP LOGIC ---

def ingest_strategy_books(files):
    global vector_db
    if not files: return "Please upload at least one PDF."
    
    documents = []
    for file in files:
        loader = PyPDFLoader(file.name)
        documents.extend(loader.load())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    splits = text_splitter.split_documents(documents)
    vector_db = FAISS.from_documents(splits, embeddings)
    return "Knowledge base ready. AI trained on strategy books!"

def run_strategic_scan():
    if vector_db is None:
        return None, "### ⚠️ Error: Please upload and 'Train' on your books first!"

    # 1. CANSLIM Screen (Finviz)
    foverview = Overview()
    filters = {
        'EPS growth qtr over qtr': 'Over 25%',
        'Sales growth qtr over qtr': 'Over 25%',
        'Price': 'Above SMA200'
    }
    foverview.set_filter(filters_dict=filters)
    df_screener = foverview.screener_view()
    
    if df_screener.empty:
        return None, "No stocks currently meet the basic CANSLIM fundamental criteria."

    spy_hist = yf.Ticker("SPY").history(period="1y")
    candidates = df_screener['Ticker'].tolist()[:12] # Filter top 12 for speed
    
    table_data = []
    ai_verdict = ""

    for ticker in candidates:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if len(hist) < 200: continue
        
        rs_score = calculate_rs(ticker, hist, spy_hist)
        is_pivot, curr, high = check_pivot_status(hist)
        
        # Only analyze if it's outperforming the market and near a pivot
        if rs_score > 0 and is_pivot:
            table_data.append([ticker, f"${curr:.2f}", f"{rs_score}%", "PIVOT WATCH"])
            
            # RAG: Retrieve context from Livermore/O'Neal books
            docs = vector_db.similarity_search(f"How to trade a breakout at a pivotal point for {ticker}", k=3)
            context = "\n".join([d.page_content for d in docs])
            
            prompt = f"Context: {context}\n\nData: {ticker} is at ${curr:.2f} with a 52w high of ${high:.2f}. RS is {rs_score}. Analyze this based on the strategy books."
            response = llm.invoke(prompt)
            ai_verdict += f"## {ticker} Analysis\n{response.content}\n\n---\n"

    final_df = pd.DataFrame(table_data, columns=["Ticker", "Current Price", "RS vs SPY", "Status"])
    return final_df, ai_verdict or "Market scan complete. No high-conviction pivots detected."

# --- GRADIO 6 DASHBOARD ---

with gr.Blocks(theme=gr.themes.Soft(), title="Livermore-CANSLIM AI", fill_height=True) as demo:
    gr.Markdown("# 📈 Strategic Trader AI Dashboard")
    
    with gr.Sidebar(label="Knowledge Management", open=True):
        gr.Markdown("### 1. Training")
        book_upload = gr.File(label="Upload Strategy Books (PDF)", file_count="multiple")
        train_btn = gr.Button("🧠 Train AI on Books", variant="secondary")
        status_label = gr.Textbox(label="System Status", value="Idle", interactive=False)
        gr.Markdown("---")
        gr.Markdown("### 2. Execution")
        scan_btn = gr.Button("🚀 Run Real-Time Scan", variant="primary")

    with gr.Row(equal_height=True):
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Live Candidates")
            output_table = gr.DataFrame(
                headers=["Ticker", "Current Price", "RS vs SPY", "Status"],
                label="Filtered Momentum Tickers"
            )
            
        with gr.Column(scale=3):
            gr.Markdown("### 🤖 Strategy Analysis")
            output_text = gr.Markdown("Waiting for market data analysis...")

    # Event Handlers
    train_btn.click(ingest_strategy_books, inputs=[book_upload], outputs=[status_label])
    scan_btn.click(run_strategic_scan, outputs=[output_table, output_text])

demo.launch()
