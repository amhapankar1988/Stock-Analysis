import os
import time
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

# Initialize ChatWatsonx
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
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
    return round((stock_perf - spy_perf) * 100, 2)

def check_pivot_status(hist):
    """Checks if price is within 5% of 52-week high (Livermore Pivot)."""
    high_52w = hist['High'].max()
    curr_price = hist['Close'].iloc[-1]
    is_at_pivot = curr_price >= (high_52w * 0.95)
    return is_at_pivot, curr_price, high_52w

# --- CORE APP LOGIC ---

def ingest_strategy_books(file_objs, progress=gr.Progress()):
    global vector_db
    if not file_objs: 
        return "❌ Please upload at least one PDF."
    
    try:
        documents = []
        # Phase 1: Text Extraction
        progress(0, desc="📄 Starting text extraction...")
        
        # Ensure file_objs is a list even if single file
        if not isinstance(file_objs, list):
            file_objs = [file_objs]

        for i, file in enumerate(file_objs):
            # Correcting the path handling for Gradio 5
            file_path = file if isinstance(file, str) else file.get("path", file.name)
            
            progress((i/len(file_objs)) * 0.3, desc=f"Reading: {os.path.basename(file_path)}")
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        
        # Phase 2: Chunking
        progress(0.4, desc="✂️ Splitting text into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)
        
        # Phase 3: Embedding (This is the slow part for 90MB)
        progress(0.5, desc="🧠 Embedding text (This may take a few minutes)...")
        vector_db = FAISS.from_documents(splits, embeddings)
        
        # Optional: Save locally so it persists during the current session
        vector_db.save_local("faiss_index")
        
        progress(1.0, desc="✅ Success!")
        return f"Knowledge base ready! Trained on {len(splits)} strategy chunks."
    
    except Exception as e:
        return f"❌ Error: {str(e)}"

def run_strategic_scan(progress=gr.Progress()):
    global vector_db
    if vector_db is None:
        return None, "### ⚠️ Error: Please upload and 'Train' on your books first!"

    try:
        progress(0.1, desc="🔍 Screening market for CANSLIM fundamentals...")
        foverview = Overview()
        
        # FIXED: Specific spacing required by finvizfinance library
        filters = {
            'EPS growthqtr over qtr': 'Over 25%',
            'Sales growthqtr over qtr': 'Over 25%',
            '200-Day Simple Moving Average': 'Price above SMA200'
        }
        
        foverview.set_filter(filters_dict=filters)
        df_screener = foverview.screener_view()
        
        if df_screener is None or df_screener.empty:
            return None, "No stocks currently meet the basic CANSLIM fundamental criteria."

        # SPEED WIN: Batch download all tickers in one go
        candidates = df_screener['Ticker'].tolist()[:15]
        progress(0.3, desc=f"📈 Downloading data for {len(candidates)} tickers...")
        
        all_data = yf.download(candidates + ["SPY"], period="1y", threads=True, progress=False)
        spy_hist = all_data['Close']['SPY']
        
        table_data = []
        ai_verdict = ""

        # Pre-fetch context once to save AI processing time
        docs = vector_db.similarity_search("How to identify a high-volume pocket pivot breakout", k=4)
        context = "\n".join([d.page_content for d in docs])

        for ticker in candidates:
            if ticker == "SPY": continue
            
            # Extract individual stock data from the batch
            hist_close = all_data['Close'][ticker].dropna()
            hist_high = all_data['High'][ticker].dropna()
            
            if len(hist_close) < 200: continue
            
            # Simple manual RS and Pivot checks
            stock_perf = (hist_close.iloc[-1] / hist_close.iloc[0]) - 1
            spy_perf = (spy_hist.iloc[-1] / spy_hist.iloc[0]) - 1
            rs_score = round((stock_perf - spy_perf) * 100, 2)
            
            high_52w = hist_high.max()
            curr_price = hist_close.iloc[-1]
            
            if rs_score > 0 and curr_price >= (high_52w * 0.95):
                table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", "PIVOT WATCH"])
                
                # AI Analysis
                prompt = (f"Context: {context}\n\n"
                         f"Analysis: {ticker} is at ${curr_price:.2f} (52w High: ${high_52w:.2f}). "
                         f"Market outperformance is {rs_score}%. Is this a valid breakout setup?")
                
                response = llm.invoke(prompt)
                ai_verdict += f"## {ticker} Analysis\n{response.content}\n\n---\n"

        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS vs SPY", "Status"])
        progress(1.0, desc="Scan Complete!")
        return final_df, ai_verdict or "Market scan complete. No high-conviction pivots detected."

    except Exception as e:
        return None, f"### ❌ Error during scan: {str(e)}"

# --- GRADIO DASHBOARD ---

with gr.Blocks(theme=gr.themes.Soft(), title="Livermore-CANSLIM AI") as demo:
    gr.Markdown("# 📈 Strategic Trader AI Dashboard")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 📚 Knowledge Management")
            book_upload = gr.File(label="Upload Strategy Books (PDF)", file_count="multiple")
            train_btn = gr.Button("🧠 Train AI on Books", variant="primary")
            status_label = gr.Textbox(label="System Status", value="Idle")
            gr.Markdown("---")
            scan_btn = gr.Button("🚀 Run Real-Time Scan", variant="primary")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Market Scan"):
                    output_table = gr.DataFrame(label="Filtered Candidates")
                with gr.TabItem("Strategy Analysis"):
                    output_text = gr.Markdown("Waiting for data analysis...")

    # Event Handlers
    train_btn.click(
        fn=ingest_strategy_books, 
        inputs=[book_upload], 
        outputs=[status_label]
    )
    scan_btn.click(
        fn=run_strategic_scan, 
        outputs=[output_table, output_text]
    )

demo.launch()
