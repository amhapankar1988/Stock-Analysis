import os
import time
import requests
import pandas as pd
import gradio as gr
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from finvizfinance.screener.overview import Overview
from huggingface_hub import HfApi, snapshot_download, repo_info
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
DATASET_REPO_ID = "amhapankar/my-trading-brain" 
HF_TOKEN = os.getenv("HF_TOKEN")
AV_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# --- CUSTOM CSS ---
css = """
.gradio-container { background-color: #0b0f19 !important; color: #e5e7eb !important; }
button.primary { background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%) !important; border: none !important; }
#terminal-df { border: 1px solid #374151 !important; border-radius: 8px !important; background-color: #111827 !important; }
"""

# --- INITIALIZE MODELS ---
if not all([WATSONX_APIKEY, PROJECT_ID]):
    raise ValueError("❌ Missing WATSONX Secrets!")

llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://ca-tor.ml.cloud.ibm.com",
    project_id=PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params={"decoding_method": "sample", "max_new_tokens": 1000, "temperature": 0.2}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- CORE FUNCTIONS ---

def ingest_strategy_books(file_objs, progress=gr.Progress()):
    global vector_db
    if not file_objs: return "❌ Please upload strategy PDFs."
    try:
        documents = []
        progress(0.2, desc="📄 Reading PDFs...")
        files = file_objs if isinstance(file_objs, list) else [file_objs]
        for file in files:
            path = file.name if hasattr(file, 'name') else file
            documents.extend(PyPDFLoader(path).load())
        
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
        progress(0.7, desc="🧠 Indexing...")
        vector_db = FAISS.from_documents(splits, embeddings)
        save_brain_to_hub()
        return f"✅ Strategy Ready: {len(splits)} chunks indexed."
    except Exception as e:
        return f"❌ Ingestion Error: {str(e)}"

def save_brain_to_hub():
    if vector_db is None: return "❌ No brain to save."
    try:
        vector_db.save_local("faiss_index")
        api = HfApi()
        api.upload_folder(folder_path="faiss_index", repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return "☁️ Sync Success"
    except: return "❌ Sync Failed"

def load_brain_from_hub():
    global vector_db
    try:
        local_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        vector_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return "✅ Brain Loaded"
    except: return "ℹ️ New Session"

def run_strategic_scan(usage_count, progress=gr.Progress()):
    global vector_db
    if vector_db is None: return None, None, "### ⚠️ Train on books first!", usage_count, None
    
    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    usage = usage_count
    
    try:
        progress(0.1, desc="🔍 Screening Finviz...")
        foverview = Overview()
        foverview.set_filter(filters_dict={
            'EPS growthqtr over qtr': 'Over 25%',
            'Relative Volume': 'Over 1.5',
            'Price': 'Over $20',
            'Sales growthqtr over qtr': 'Over 25%',
            '200-Day Simple Moving Average': 'Price above SMA200'
        })
        df_screener = foverview.screener_view(order='Relative Volume', ascend=False)
        candidates = df_screener['Ticker'].tolist()[:3] if df_screener is not None else []
        
        table_data, chart_previews, ai_verdict = [], [], ""
        
        # Benchmark (SPY)
        spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
        usage += 1
        spy_close = spy_data['4. close']

        for ticker in candidates:
            if usage >= 25: break
            time.sleep(12) # Rate limit protection
            
            # Fetch Price & News
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            usage += 1
            curr_price = data['4. close'].iloc[0]
            
            # Relative Strength Calculation
            stock_perf = (curr_price / data['4. close'].iloc[-1]) - 1
            spy_perf = (spy_close.iloc[0] / spy_close.iloc[-1]) - 1
            rs_score = round((stock_perf - spy_perf) * 100, 2)
            
            # Sentiment Analysis
            news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={AV_API_KEY}"
            sentiment_resp = requests.get(news_url).json()
            sentiment = sentiment_resp.get("feed", [{}])[0].get("overall_sentiment_label", "Neutral")
            usage += 1

            # RAG Context
            docs = vector_db.similarity_search(f"strategy for {ticker}", k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # --- FIXED PROMPT (Prevents Tool-Call Hallucinations) ---
            structured_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are a professional stock market auditor. Your goal is to provide a text-based analysis. 
            Do NOT output tool calls or function requests. Use the provided strategy context to audit the stock data.
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            
            STRATEGY CONTEXT:
            {context}

            MARKET DATA:
            Ticker: {ticker}
            Price: ${curr_price:.2f}
            Relative Strength: {rs_score}% vs SPY
            Sentiment: {sentiment}

            Provide a concise 'Triple Screen' audit. End with a Recommendation (BUY/WATCH/AVOID).<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
            
            res = llm.invoke(structured_prompt)
            
            ai_verdict += f"## {ticker} Audit\n{res.content}\n\n---\n"
            table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", sentiment])
            chart_previews.append((f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l", f"{ticker} Daily"))

        # --- SAVE TO DISK ---
        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS Score", "Sentiment"])
        final_df.to_csv("report.csv", index=False) 
        
        return final_df, "report.csv", ai_verdict, usage, chart_previews

    except Exception as e:
        return None, None, f"### ❌ Error: {str(e)}", usage, None

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    usage_state = gr.State(value=0)
    
    with gr.Tabs():
        with gr.TabItem("📈 Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    usage_slider = gr.Slider(0, 25, label="API Usage", interactive=False)
                    scan_btn = gr.Button("🚀 RUN TRIPLE SCREEN", variant="primary")
                    csv_out = gr.File(label="Report")
                with gr.Column(scale=3):
                    picks_table = gr.DataFrame(elem_id="terminal-df")
                    charts_gallery = gr.Gallery(label="Visual Analysis", columns=3)
                    verdict_md = gr.Markdown()

        with gr.TabItem("📚 Strategy"):
            book_upload = gr.File(label="Upload PDFs", file_count="multiple")
            train_btn = gr.Button("🧠 Train AI")
            train_status = gr.Textbox(label="Status")

        with gr.TabItem("⚙️ Admin"):
            sync_btn = gr.Button("🔄 Cloud Sync")
            admin_logs = gr.Textbox(label="Logs")

    train_btn.click(ingest_strategy_books, [book_upload], [train_status])
    scan_btn.click(run_strategic_scan, [usage_state], [picks_table, csv_out, verdict_md, usage_state, charts_gallery])
    sync_btn.click(load_brain_from_hub, None, admin_logs)

load_brain_from_hub()
demo.launch()
