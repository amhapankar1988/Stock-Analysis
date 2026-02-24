import os
import time
import gradio as gr
import pandas as pd
from finvizfinance.screener.overview import Overview
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from alpha_vantage.timeseries import TimeSeries

# --- CONFIGURATION ---
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
AV_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
WATSONX_URL = "https://ca-tor.ml.cloud.ibm.com"

# Initialize Models
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=WATSONX_URL,
    project_id=PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params={"decoding_method": "sample", "max_new_tokens": 800, "temperature": 0.2}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- CORE LOGIC ---

def ingest_strategy_books(file_objs, progress=gr.Progress()):
    global vector_db
    if not file_objs: return "❌ No files uploaded."
    try:
        documents = []
        progress(0.1, desc="📄 Extracting text...")
        if not isinstance(file_objs, list): file_objs = [file_objs]
        for file in file_objs:
            path = file if isinstance(file, str) else file.get("path", file.name)
            documents.extend(PyPDFLoader(path).load())
        
        progress(0.4, desc="✂️ Chunking...")
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(documents)
        
        progress(0.6, desc="🧠 Embedding (This takes time)...")
        vector_db = FAISS.from_documents(splits, embeddings)
        return f"✅ Success! Trained on {len(splits)} chunks."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def run_strategic_scan(progress=gr.Progress()):
    global vector_db
    if vector_db is None: return None, None, "### ⚠️ Train on books first!"
    if not AV_API_KEY: return None, None, "### ⚠️ Missing ALPHA_VANTAGE_KEY Secret!"

    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    
    try:
        progress(0.1, desc="🔍 Fundamental Screen & Momentum Sort...")
        foverview = Overview()
        
        # We add 'Relative Volume' to ensure we only look at stocks traders are active in today
        filters = {
            'EPS growthqtr over qtr': 'Over 25%',
            'Sales growthqtr over qtr': 'Over 25%',
            '200-Day Simple Moving Average': 'Price above SMA200',
            'Relative Volume': 'Over 1.5' 
        }
        foverview.set_filter(filters_dict=filters)
        
        # Sort by Weekly Performance DESCENDING to find the strongest momentum
        df_screener = foverview.screener_view(order='Performance (Week)', ascend=False)
        
        if df_screener is None or df_screener.empty:
            return None, None, "No candidates found meeting the momentum criteria."

        # Take only the Top 4 to stay safe with Alpha Vantage Free Tier (5/min limit)
        candidates = df_screener['Ticker'].tolist()[:4]
        table_data = []
        ai_verdict = ""

        progress(0.3, desc="📊 Fetching SPY Benchmark...")
        spy_data, _ = ts.get_daily_adjusted(symbol='SPY', outputsize='full')
        spy_close = spy_data['5. adjusted close'].iloc[-252:]

        for ticker in candidates:
            progress(0.5, desc=f"📈 API Analysis: {ticker}...")
            
            # 12-second sleep is MANDATORY for Alpha Vantage Free Tier
            time.sleep(12) 
            
            data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize='full')
            close = data['5. adjusted close'].iloc[-252:]
            
            # Technical Calculations
            stock_perf = (close.iloc[-1] / close.iloc[0]) - 1
            spy_perf = (spy_close.iloc[-1] / spy_close.iloc[0]) - 1
            rs_score = round((stock_perf - spy_perf) * 100, 2)
            
            high_52w = close.max()
            curr = close.iloc[-1]
            
            # Pivot Check (Livermore Style)
            if rs_score > 0 and curr >= (high_52w * 0.95):
                table_data.append([ticker, f"${curr:.2f}", f"{rs_score}%", "TOP MOMENTUM"])
                
                # RAG Analysis
                docs = vector_db.similarity_search(f"High volume breakout strategy for {ticker}", k=2)
                context = "\n".join([d.page_content for d in docs])
                res = llm.invoke(f"Strategy: {context}\n\nAnalyze {ticker} at ${curr:.2f} (RS: {rs_score}%).")
                ai_verdict += f"## {ticker} Analysis\n{res.content}\n\n---\n"

        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS vs SPY", "Status"])
        
        # Save a local copy for the user to download
        csv_path = "latest_scan_results.csv"
        final_df.to_csv(csv_path, index=False)
        
        return final_df, csv_path, ai_verdict or "Scan complete. No high-conviction pivots found."

    except Exception as e:
        return None, None, f"### ❌ Error: {str(e)}"

# --- UPDATED GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Strategic Trader AI") as demo:
    gr.Markdown("# 📈 Alpha-Vantage Powered Strategic AI")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            book_upload = gr.File(label="1. Upload PDFs", file_count="multiple")
            train_btn = gr.Button("🧠 Train AI", variant="primary")
            status = gr.Textbox(label="Status", value="Idle")
            gr.Markdown("---")
            scan_btn = gr.Button("🚀 Run Quality Scan", variant="primary")
            # Added a File component for downloading the CSV
            csv_download = gr.File(label="2. Download Scan Report")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Market Scan"):
                    output_table = gr.DataFrame(label="Top Momentum Candidates")
                with gr.TabItem("AI Strategy Analysis"):
                    output_text = gr.Markdown("Analysis will appear here...")

    train_btn.click(ingest_strategy_books, [book_upload], [status])
    # Note: Added csv_download to the outputs
    scan_btn.click(run_strategic_scan, None, [output_table, csv_download, output_text])

demo.launch()