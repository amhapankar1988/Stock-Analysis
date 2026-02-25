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
DATASET_REPO_ID = "amhapankar/my-trading-brain"  # <--- UPDATE THIS
HF_TOKEN = os.getenv("HF_TOKEN")
AV_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# --- INITIALIZE MODELS ---
# Added a check to prevent crash if secrets are missing
if not all([WATSONX_APIKEY, PROJECT_ID]):
    raise ValueError("❌ Missing WATSONX_APIKEY or WATSONX_PROJECT_ID in Secrets!")

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
        for file in (file_objs if isinstance(file_objs, list) else [file_objs]):
            path = file.name if hasattr(file, 'name') else file
            documents.extend(PyPDFLoader(path).load())
        
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
        progress(0.7, desc="🧠 Creating Vector Index...")
        vector_db = FAISS.from_documents(splits, embeddings)
        
        # Auto-backup to cloud after training
        save_brain_to_hub()
        return f"✅ Knowledge Base Ready: {len(splits)} chunks indexed and synced to cloud."
    except Exception as e:
        return f"❌ Ingestion Error: {str(e)}"

def save_brain_to_hub():
    if vector_db is None: return "❌ No brain to save."
    try:
        vector_db.save_local("faiss_index")
        api = HfApi()
        api.upload_folder(
            folder_path="faiss_index",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
        return f"✅ Backup Success: {datetime.now().strftime('%H:%M:%S')}"
    except Exception as e:
        return f"❌ Backup Failed: {str(e)}"

def load_brain_from_hub():
    global vector_db
    try:
        local_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        vector_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return "✅ Brain Loaded from Cloud"
    except Exception:
        return "ℹ️ New Session: No Cloud Brain found."

def get_last_backup_time():
    try:
        info = repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return f"Last Cloud Update: {info.last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
    except:
        return "No Backup History Found."

def run_strategic_scan(usage_count, progress=gr.Progress()):
    global vector_db
    if vector_db is None: return None, None, "### ⚠️ Train on books first!", usage_count
    if usage_count >= 25: return None, None, "### ❌ Daily Limit Reached.", usage_count

    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    usage = usage_count
    
    try:
        progress(0.1, desc="🔍 Screening Finviz...")
        foverview = Overview()
        foverview.set_filter(filters_dict={
            'EPS growthqtr over qtr': 'Over 25%',
            'Relative Volume': 'Over 1.5',
            'Price': 'Over $20',  # Stocks greater than $20
            'Sales growthqtr over qtr': 'Over 25%',  # Quarterly Sales growth
            '200-Day Simple Moving Average': 'Price above SMA200' # Confirming long-term trend
        })
        
        df_screener = foverview.screener_view(order='Performance (Week)', ascend=False)
        
        candidates = df_screener['Ticker'].tolist()[:4] if df_screener is not None else []
        table_data = []
        ai_verdict = ""

        # Fetch SPY
        spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
        usage += 1
        spy_close = spy_data['4. close']

        for ticker in candidates:
            if usage >= 25: break
            time.sleep(12)

            # Technical Data
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            usage += 1
            close = data['4. close']
            curr_price = close.iloc[0]
            high_100d = close.max()
            dist_from_high = round((1 - curr_price/high_100d)*100, 2)
            rs_score = round(((close.iloc[0]/close.iloc[-1]) - (spy_close.iloc[0]/spy_close.iloc[-1])) * 100, 2)

            # News Sentiment Call
            time.sleep(12)
            news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={AV_API_KEY}"
            news_resp = requests.get(news_url).json()
            usage += 1
            sentiment = news_resp.get("feed", [{}])[0].get("overall_sentiment_label", "Neutral")

            # Strategy Retrieval
            docs = vector_db.similarity_search(f"strategy for {ticker}", k=3)
            context = "\n".join([d.page_content for d in docs])
            
           # DEFINE THE PROMPT 
            prompt = f"""
            ### STRATEGY CONTEXT: {context}
            ### DATA: {ticker} | Price: ${curr_price:.2f} | RS: {rs_score}% | Sentiment: {sentiment}
            Audit this stock based on the strategy: 
            - Identify the Pivot.
            - Catalyst check.
            - Final Verdict (BUY/WATCH/AVOID).
            """
            
            # INVOKE LLM
            res = llm.invoke(prompt) 
            ai_verdict += f"## {ticker}\n{res.content}\n\n---\n"
        
            table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", sentiment])

        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS Score", "Sentiment"])
        return final_df, "report.csv", ai_verdict or "Scan finished.", usage

    except Exception as e:
        return None, None, f"### ❌ Error: {str(e)}", usage

# --- UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    usage_state = gr.State(value=0)
    
    gr.Markdown("# 🚀 AI Strategic Trader Pro")
    
    with gr.Tabs():
        with gr.TabItem("📈 Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    usage_slider = gr.Slider(0, 25, label="API Credits Used", interactive=False)
                    scan_btn = gr.Button("🔥 Run Scan", variant="primary")
                    csv_out = gr.File(label="Download Report")
                with gr.Column(scale=3):
                    picks_table = gr.DataFrame(label="Top Market Momentum")
                    verdict_md = gr.Markdown("Waiting for scan...")

        with gr.TabItem("📚 Knowledge"):
            book_upload = gr.File(label="Upload PDFs", file_count="multiple")
            train_btn = gr.Button("🧠 Train Strategy")
            train_status = gr.Textbox(label="Status", value="Ready")

        with gr.TabItem("⚙️ Admin"):
            backup_info = gr.Markdown(get_last_backup_time())
            manual_backup_btn = gr.Button("☁️ Save Brain to Cloud")
            force_load_btn = gr.Button("🔄 Sync from Cloud")
            admin_status = gr.Textbox(label="Logs")

    train_btn.click(ingest_strategy_books, [book_upload], [train_status])
    scan_btn.click(run_strategic_scan, [usage_state], [picks_table, csv_out, verdict_md, usage_state])
    usage_state.change(lambda x: x, usage_state, usage_slider)
    manual_backup_btn.click(save_brain_to_hub, None, admin_status)
    force_load_btn.click(load_brain_from_hub, None, admin_status)

# On Startup
load_brain_from_hub()
demo.launch()
