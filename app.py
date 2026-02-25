import os
import time
import pandas as pd
import gradio as gr
from alpha_vantage.timeseries import TimeSeries
from finvizfinance.screener.overview import Overview
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- INITIALIZATION ---
AV_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_URL = "https://ca-tor.ml.cloud.ibm.com"

# Models
llm = ChatWatsonx(
    model_id="meta-llama/llama-3-3-70b-instruct",
    url=WATSONX_URL,
    project_id=PROJECT_ID,
    apikey=WATSONX_APIKEY,
    params={"decoding_method": "sample", "max_new_tokens": 800, "temperature": 0.2}
)
# Using a high-performance open-source embedding model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = None

# --- FUNCTIONS ---

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
        return f"✅ Knowledge Base Ready: {len(splits)} strategy chunks indexed."
    except Exception as e:
        return f"❌ Ingestion Error: {str(e)}"

def run_strategic_scan(usage_count, progress=gr.Progress()):
    global vector_db
    if vector_db is None: return None, None, "### ⚠️ Train the AI on your books first!", usage_count
    if usage_count >= 25: return None, None, "### ❌ Daily Alpha Vantage Limit Reached (25/25).", usage_count

    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    usage = usage_count
    
    try:
        progress(0.1, desc="🔍 Screening Finviz for Momentum...")
        foverview = Overview()
        foverview.set_filter(filters_dict={
            'EPS growthqtr over qtr': 'Over 25%',
            'Relative Volume': 'Over 1.5',
            '200-Day Simple Moving Average': 'Price above SMA200'
        })
        df_screener = foverview.screener_view(order='Performance (Week)', ascend=False)
        
        if df_screener is None or df_screener.empty:
            return None, None, "No stocks currently match the momentum criteria.", usage

        candidates = df_screener['Ticker'].tolist()[:4]
        table_data = []
        ai_verdict = ""

        # Fetch SPY Benchmark (1 API Call) - Using 'compact' for FREE tier
        progress(0.2, desc="📊 Fetching SPY (Free API Call)...")
        spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact') 
        spy_close = spy_data['4. close']
        usage += 1

        for ticker in candidates:
            if usage >= 25: break
            
            progress(0.3 + (usage/25), desc=f"📈 Analyzing {ticker} (Usage: {usage}/25)...")
            time.sleep(12) 
            
            try:
                # Switching to 'compact' to stay on Free Tier
                data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                usage += 1
                close = data['4. close']
                
                # Calculations (Using 100-day window)
                # Performance is now calculated over the available 100 days
                stock_perf = (close.iloc[0] / close.iloc[-1]) - 1 # Note: AV data is desc (newest first)
                spy_perf = (spy_close.iloc[0] / spy_close.iloc[-1]) - 1
                rs_score = round((stock_perf - spy_perf) * 100, 2)
                
                curr_price = close.iloc[0]
                high_period = close.max() # This is now the 100-day high
                
                # Check for "Power Pivot": Within 3% of 100-day High
                if curr_price >= (high_period * 0.97):
                    status_tag = "🚀 BREAKOUT WATCH"
                    docs = vector_db.similarity_search(f"trading strategy for {ticker}", k=2)
                    context = "\n".join([d.page_content for d in docs])
                    prompt = f"Strategy: {context}\n\nAnalyze {ticker} at ${curr_price:.2f}. It is at a 100-day high with RS score {rs_score}%."
                    res = llm.invoke(prompt)
                    ai_verdict += f"## {ticker} Verdict\n{res.content}\n\n---\n"
                else:
                    status_tag = "Consolidating"

                table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", status_tag])

            except Exception as e: 
                print(f"Error on {ticker}: {e}")
                continue

        final_df = pd.DataFrame(table_data, columns=["Ticker", "Price", "RS (100d)", "Status"])
        csv_path = "scan_results.csv"
        final_df.to_csv(csv_path, index=False)
        
        return final_df, csv_path, ai_verdict or "Scan complete. No 100-day breakouts detected.", usage

    except Exception as e:
        return None, None, f"### ❌ API Error: {str(e)}", usage

# --- GRADIO UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="AI Strategic Scanner") as demo:
    usage_state = gr.State(value=0) # Persists usage during session
    
    gr.Markdown("# 📈 Strategic Trader AI")
    
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Strategy Setup")
            book_upload = gr.File(label="Upload Trading Books (PDF)", file_count="multiple")
            train_btn = gr.Button("🧠 Train AI", variant="primary")
            status = gr.Textbox(label="System Status", value="Waiting for books...")
            
            gr.Markdown("### 2. Market Scan")
            usage_bar = gr.Slider(0, 25, label="API Credits Used (Daily Limit: 25)", interactive=False)
            scan_btn = gr.Button("🚀 Run Quality Scan", variant="primary")
            csv_download = gr.File(label="Download Report")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Live Market Picks"):
                    output_table = gr.DataFrame(interactive=False)
                with gr.TabItem("AI Strategy Verdicts"):
                    output_text = gr.Markdown("Analysis will appear here after the scan...")

    # Event Mapping
    train_btn.click(ingest_strategy_books, [book_upload], [status])
    
    scan_btn.click(
        run_strategic_scan, 
        inputs=[usage_state], 
        outputs=[output_table, csv_download, output_text, usage_state]
    )
    
    # Update usage bar whenever state changes
    usage_state.change(lambda x: x, inputs=[usage_state], outputs=[usage_bar])

demo.launch()
