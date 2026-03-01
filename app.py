import os
import time
import requests
import pandas as pd
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from finvizfinance.screener.overview import Overview
from huggingface_hub import HfApi, snapshot_download
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# --- 1. CLOUD CONFIGURATION ---
st.set_page_config(page_title="Trading Brain AI", layout="wide", page_icon="📈")

# Accessing Streamlit Secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")
AV_API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = st.secrets.get("WATSONX_APIKEY")
PROJECT_ID = st.secrets.get("WATSONX_PROJECT_ID")
DATASET_REPO_ID = st.secrets.get("DATASET_REPO_ID", "amhapankar/my-trading-brain")

# --- 2. INITIALIZE MODELS (Cloud Optimized) ---
@st.cache_resource
def init_models():
    if not all([WATSONX_APIKEY, PROJECT_ID, HF_TOKEN]):
        st.error("🔑 Secrets missing! Add HF_TOKEN, ALPHA_VANTAGE_KEY, WATSONX_APIKEY, and WATSONX_PROJECT_ID to Streamlit Secrets.")
        st.stop()

    # WatsonX Setup
    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://ca-tor.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
        apikey=WATSONX_APIKEY,
        params={"decoding_method": "sample", "max_new_tokens": 1000, "temperature": 0.2}
    )

    # API-Based Embeddings (Saves ~500MB RAM compared to local models)
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    
    return llm, embeddings

llm, embeddings = init_models()

# --- 3. SESSION STATE MANAGEMENT ---
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- 4. CORE FUNCTIONS ---

def save_brain_to_hub():
    if st.session_state.vector_db is None: return
    try:
        st.session_state.vector_db.save_local("faiss_index")
        api = HfApi()
        api.upload_folder(
            folder_path="faiss_index", 
            repo_id=DATASET_REPO_ID, 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        st.toast("☁️ Sync Success: Brain saved to Hugging Face")
    except Exception as e:
        st.error(f"Sync Failed: {e}")

def load_brain_from_hub():
    try:
        local_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        st.session_state.vector_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return True
    except:
        return False

def ingest_strategy_books(uploaded_files):
    try:
        documents = []
        for uploaded_file in uploaded_files:
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            documents.extend(PyPDFLoader(temp_path).load())
            os.remove(temp_path)
        
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
        st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
        save_brain_to_hub()
        st.success(f"✅ Strategy Ready: {len(splits)} chunks indexed.")
    except Exception as e:
        st.error(f"❌ Ingestion Error: {str(e)}")

def run_strategic_scan():
    if st.session_state.vector_db is None:
        st.warning("⚠️ Train on books first or sync from cloud!")
        return
    
    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    
    try:
        with st.status("🔍 Screening Finviz Candidates...", expanded=True) as status:
            foverview = Overview()
            foverview.set_filter(filters_dict={
                'EPS growthqtr over qtr': 'Over 25%',
                'Relative Volume': 'Over 1.5',
                'Price': 'Over $20',
                'Sales growthqtr over qtr': 'Over 25%',
                '200-Day Simple Moving Average': 'Price above SMA200'
            })
            df_screener = foverview.screener_view(order='Relative Volume', ascend=False)
            
            if df_screener is None or df_screener.empty:
                st.error("No stocks met the criteria today.")
                return

            candidates = df_screener['Ticker'].tolist()[:3]
            table_data, chart_urls, ai_verdict = [], [], ""
            
            # Benchmark (SPY)
            spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
            st.session_state.usage_count += 1
            spy_close = spy_data['4. close']

            for ticker in candidates:
                if st.session_state.usage_count >= 25: 
                    st.warning("API Limit Reached (25 calls).")
                    break

                status.write(f"Auditing {ticker}...")
                ticker_row = df_screener[df_screener['Ticker'] == ticker].iloc[0]
                
                time.sleep(12) # Alpha Vantage free tier rate limit
                
                # Fetch Data
                data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                st.session_state.usage_count += 1
                curr_price = data['4. close'].iloc[0]
                
                # Performance & Sentiment
                stock_perf = (curr_price / data['4. close'].iloc[-1]) - 1
                spy_perf = (spy_close.iloc[0] / spy_close.iloc[-1]) - 1
                rs_score = round((stock_perf - spy_perf) * 100, 2)
                
                news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={AV_API_KEY}"
                sentiment_resp = requests.get(news_url).json()
                sentiment = sentiment_resp.get("feed", [{}])[0].get("overall_sentiment_label", "Neutral")
                st.session_state.usage_count += 1

                # RAG Audit
                docs = st.session_state.vector_db.similarity_search(f"strategy for {ticker}", k=3)
                context = "\n".join([d.page_content for d in docs])
                
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Professional Stock Auditor. Evaluate if the setup follows the provided strategy context.<|eot_id|><|start_header_id|>user<|end_header_id|>
                STRATEGY: {context}
                STOCK: {ticker} (Price: ${curr_price:.2f}, RS: {rs_score}%, Sentiment: {sentiment})
                GROWTH: EPS Q/Q {ticker_row.get('EPS Q/Q')}, Sales Q/Q {ticker_row.get('Sales Q/Q')}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                res = llm.invoke(prompt)
                ai_verdict += f"## {ticker} Audit\n{res.content}\n\n---\n"
                table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", sentiment])
                chart_urls.append(f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l")

            status.update(label="✅ Analysis Complete", state="complete")

            st.session_state.scan_results = {
                "df": pd.DataFrame(table_data, columns=["Ticker", "Price", "RS Score", "Sentiment"]),
                "verdict": ai_verdict,
                "charts": chart_urls
            }
    except Exception as e:
        st.error(f"Scan Error: {e}")

# --- 5. UI LAYOUT ---
st.title("📊 My Trading Brain AI")

# Sidebar for Cloud Connection Status
with st.sidebar:
    st.header("Admin")
    if st.button("🔄 Sync with Cloud"):
        if load_brain_from_hub():
            st.success("Brain Loaded!")
        else:
            st.info("No brain found in cloud repository.")
    
    st.divider()
    st.metric("API Usage", f"{st.session_state.usage_count} / 25")
    if st.button("🗑️ Reset Usage"):
        st.session_state.usage_count = 0
        st.rerun()

# Main Tabs
tab1, tab2 = st.tabs(["📈 Terminal", "📚 Strategy"])

with tab1:
    col_run, col_dl = st.columns([1, 1])
    with col_run:
        if st.button("🚀 RUN STRATEGIC SCAN", use_container_width=True):
            run_strategic_scan()
    
    if st.session_state.scan_results:
        with col_dl:
            csv = st.session_state.scan_results["df"].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV Report", data=csv, file_name="report.csv", use_container_width=True)

        st.dataframe(st.session_state.scan_results["df"], use_container_width=True)
        
        # Display Charts
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        for i, url in enumerate(st.session_state.scan_results["charts"]):
            cols[i % 3].image(url, caption=f"Chart {i+1}")
        
        st.markdown(st.session_state.scan_results["verdict"])
    else:
        st.info("No scan data found. Run a scan or upload strategy books.")

with tab2:
    st.subheader("Knowledge Ingestion")
    files = st.file_uploader("Upload Strategy PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🧠 Train AI Knowledge Base"):
        if files:
            ingest_strategy_books(files)
        else:
            st.warning("Please upload at least one PDF.")

# Initial Cloud Load on First Start
if st.session_state.vector_db is None:
    load_brain_from_hub()
