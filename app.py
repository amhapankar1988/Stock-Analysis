import os
import time
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
PROJECT_ID = os.getenv("PROJECT_ID")

# --- INITIALIZE MODELS ---
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url="https://ca-tor.ml.cloud.ibm.com",
    project_id=PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params={"decoding_method": "sample", "max_new_tokens": 1000, "temperature": 0.2}
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- CLOUD BRAIN LOGIC ---

def save_brain_to_hub():
    if vector_db is None: return "❌ No brain to save. Train first."
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
        return "ℹ️ First Run: No Cloud Brain found."

def get_last_backup_time():
    try:
        info = repo_info(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return f"Last Cloud Update: {info.last_modified.strftime('%Y-%m-%d %H:%M:%S')}"
    except:
        return "No Backup History Found."

# --- STRATEGIC SCANNING ---

def run_strategic_scan(usage_count, progress=gr.Progress()):
    global vector_db
    if vector_db is None: return None, None, "### ⚠️ Train on books first!", usage_count
    if usage_count >= 25: return None, None, "### ❌ Daily Limit Reached.", usage_count

    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    usage = usage_count
    
    try:
        progress(0.1, desc="🔍 Screening Finviz...")
        foverview = Overview()
        foverview.set_filter(filters_dict={'EPS growthqtr over qtr': 'Over 25%', 'Relative Volume': 'Over 1.5'})
        df_screener = foverview.screener_view(order='Performance (Week)', ascend=False)
        
        candidates = df_screener['Ticker'].tolist()[:4] if df_screener is not None else []
        table_data = []
        ai_verdict = ""

        # Fetch SPY (1 Credit)
        spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
        usage += 1
        spy_close = spy_data['4. close']

        for ticker in candidates:
            if usage >= 25: break
            time.sleep(12) # Rate limit delay
            
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            usage += 1
            close = data['4. close']
            
            curr_price = close.iloc[0]
            high_100d = close.max()
            
            # Distance from High
            dist_from_high = round((1 - curr_price/high_100d)*100, 2)
            # RS Score (Simplified 100-day)
            rs_score = round(((close.iloc[0]/close.iloc[-1]) - (spy_close.iloc[0]/spy_close.iloc[-1])) * 100, 2)
            
            # WIZARD PROMPT LOGIC
            status_tag = "🚀 PIVOT" if dist_from_high < 5 else "Consolidating"
            
            docs = vector_db.similarity_search(f"trading strategy for {ticker}", k=3)
            context = "\n".join([d.page_content for d in docs])
            
            wizard_prompt = f"""
            ### STRATEGY CONTEXT:
            {context}

            ### REAL-TIME DATA:
            - Ticker: {ticker} | Price: ${curr_price:.2f}
            - RS Score: {rs_score}% | 100d High: ${high_100d:.2f}
            - Distance from High: {dist_from_high}%

            ### INSTRUCTIONS:
            Perform a Technical Audit of {ticker} using the provided strategy:
            1. **Pivotal Point**: Is this breaking resistance at the 100-day high?
            2. **VCP Check**: Identify any Volatility Contraction Pattern.
            3. **Cheat Level**: Suggest a tight stop-loss based on the patterns in the text.
            4. **Verdict**: BUY, WATCH, or AVOID.
            """
            
            res = llm.invoke(wizard_prompt)
            ai_verdict += f"## {ticker} Analysis\n{res.content}\n\n---\n"
            table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", status_tag])

        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS Score", "Status"])
        return final_df, "report.csv", ai_verdict or "No high-conviction pivots found.", usage
    except Exception as e:
        return None, None, f"### ❌ Error: {str(e)}", usage

# --- UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    usage_state = gr.State(value=0)
    
    gr.Markdown("# 🚀 AI Strategic Trader Pro")
    
    with gr.Tabs():
        with gr.TabItem("📈 Trading Terminal"):
            with gr.Row():
                with gr.Column(scale=1):
                    usage_slider = gr.Slider(0, 25, label="API Credits (25 Max)", interactive=False)
                    scan_btn = gr.Button("🔥 Run Strategy Scan", variant="primary")
                    csv_out = gr.File(label="Download Report")
                with gr.Column(scale=3):
                    picks_table = gr.DataFrame(label="Market Momentum (100-Day Window)")
                    verdict_md = gr.Markdown("Waiting for scan...")

        with gr.TabItem("📚 Strategy Ingestion"):
            gr.Markdown("### Upload PDF Books (Minervini, Livermore, O'Neil)")
            book_upload = gr.File(label="Upload Books", file_count="multiple")
            train_btn = gr.Button("🧠 Update Strategy Brain")
            train_status = gr.Textbox(label="Status", value="Ready")

        with gr.TabItem("⚙️ Admin Panel"):
            gr.Markdown("### Cloud Knowledge Storage")
            backup_info = gr.Markdown(get_last_backup_time())
            with gr.Row():
                manual_backup_btn = gr.Button("☁️ Save Brain to Cloud")
                force_load_btn = gr.Button("🔄 Sync from Cloud")
            admin_status = gr.Textbox(label="Admin Logs")

    # Wire up Events
    train_btn.click(ingest_strategy_books, [book_upload], [train_status])
    scan_btn.click(run_strategic_scan, [usage_state], [picks_table, csv_out, verdict_md, usage_state])
    usage_state.change(lambda x: x, usage_state, usage_slider)
    
    manual_backup_btn.click(save_brain_to_hub, None, admin_status)
    force_load_btn.click(load_brain_from_hub, None, admin_status)

# On Startup
load_brain_from_hub()
demo.launch()