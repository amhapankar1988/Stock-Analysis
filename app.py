import os
import time
import requests
import pandas as pd
import streamlit as st
from datetime import datetime
from alpha_vantage.timeseries import TimeSeries
from finvizfinance.screener.overview import Overview
from huggingface_hub import HfApi, snapshot_download
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIG ---
st.set_page_config(page_title="Trading Brain AI", layout="wide", page_icon="📈")

# --- STYLE ---
st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #e5e7eb; }
    div.stButton > button:first-child {
        background: linear-gradient(180deg, #2563eb 0%, #1d4ed8 100%);
        color: white; border: none; width: 100%;
    }
    .reportview-container .main .block-container { padding-top: 2rem; }
    </style>
    """, unsafe_allow_html=True)

# --- SECRETS & CONFIG ---
DATASET_REPO_ID = os.getenv("DATASET_REPO_ID", "amhapankar/my-trading-brain")
HF_TOKEN = os.getenv("HF_TOKEN")
AV_API_KEY = os.getenv("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")

# --- INITIALIZE STATE ---
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- MODELS ---
@st.cache_resource
def init_models():
    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://ca-tor.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
        apikey=WATSONX_APIKEY,
        params={"decoding_method": "sample", "max_new_tokens": 1000, "temperature": 0.2}
    )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = init_models()

# --- CORE LOGIC ---

def ingest_strategy_books(uploaded_files):
    try:
        documents = []
        for uploaded_file in uploaded_files:
            # Streamlit provides a file-like object; we save temporarily to load via PyPDF
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            documents.extend(PyPDFLoader(temp_path).load())
            os.remove(temp_path)
        
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents)
        st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
        
        # Sync to Hub
        st.session_state.vector_db.save_local("faiss_index")
        api = HfApi()
        api.upload_folder(folder_path="faiss_index", repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        return f"✅ Strategy Ready: {len(splits)} chunks indexed & synced."
    except Exception as e:
        return f"❌ Ingestion Error: {str(e)}"

def load_brain_from_hub():
    try:
        local_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        st.session_state.vector_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return "✅ Brain Loaded from Cloud"
    except Exception as e:
        return f"ℹ️ Session started (No cloud index found: {e})"

def run_strategic_scan():
    if st.session_state.vector_db is None:
        st.error("### ⚠️ Train on books first!")
        return

    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    
    try:
        with st.status("🔍 Scanning Markets...", expanded=True) as status:
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
            
            table_data, chart_urls, ai_verdicts = [], [], []
            
            # Benchmark (SPY)
            spy_data, _ = ts.get_daily(symbol='SPY', outputsize='compact')
            st.session_state.usage_count += 1
            spy_close = spy_data['4. close']

            for ticker in candidates:
                if st.session_state.usage_count >= 25: break
                
                status.write(f"Analyzing {ticker}...")
                ticker_row = df_screener[df_screener['Ticker'] == ticker].iloc[0]
                
                time.sleep(12) # Rate limit
                
                data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
                st.session_state.usage_count += 1
                curr_price = data['4. close'].iloc[0]
                
                stock_perf = (curr_price / data['4. close'].iloc[-1]) - 1
                spy_perf = (spy_close.iloc[0] / spy_close.iloc[-1]) - 1
                rs_score = round((stock_perf - spy_perf) * 100, 2)
                
                # Sentiment
                news_url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&limit=3&apikey={AV_API_KEY}"
                sentiment_resp = requests.get(news_url).json()
                sentiment = sentiment_resp.get("feed", [{}])[0].get("overall_sentiment_label", "Neutral")
                st.session_state.usage_count += 2 # Accounting for news + daily

                # RAG
                docs = st.session_state.vector_db.similarity_search(f"strategy for {ticker}", k=3)
                context = "\n".join([d.page_content for d in docs])
                
                prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                Professional stock auditor. Audit based on Strategy Context.<|eot_id|><|start_header_id|>user<|end_header_id|>
                CONTEXT: {context}
                DATA: {ticker}, Price: {curr_price}, RS: {rs_score}%, Sentiment: {sentiment}
                Growth: EPS {ticker_row.get('EPS Q/Q')}, Sales {ticker_row.get('Sales Q/Q')}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
                
                res = llm.invoke(prompt)
                
                ai_verdicts.append(f"## {ticker} Audit\n{res.content}")
                table_data.append([ticker, f"${curr_price:.2f}", f"{rs_score}%", sentiment])
                chart_urls.append(f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l")

            status.update(label="✅ Scan Complete!", state="complete")

        # Save to session state
        st.session_state.scan_results = {
            "df": pd.DataFrame(table_data, columns=["Ticker", "Price", "RS Score", "Sentiment"]),
            "verdicts": "\n\n---\n".join(ai_verdicts),
            "charts": chart_urls
        }
    except Exception as e:
        st.error(f"Scan Error: {e}")

# --- UI LAYOUT ---
st.title("🚀 My Trading Brain")

tabs = st.tabs(["📈 Terminal", "📚 Strategy", "⚙️ Admin"])

with tabs[0]:
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("API Usage", f"{st.session_state.usage_count}/25")
        if st.button("🚀 RUN TRIPLE SCREEN"):
            run_strategic_scan()
        
        if st.session_state.scan_results:
            csv = st.session_state.scan_results["df"].to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Report", data=csv, file_name="report.csv", mime="text/csv")

    with col2:
        if st.session_state.scan_results:
            st.dataframe(st.session_state.scan_results["df"], use_container_width=True)
            
            # Display Charts
            cols = st.columns(3)
            for idx, url in enumerate(st.session_state.scan_results["charts"]):
                cols[idx % 3].image(url, use_column_width=True)
            
            st.markdown(st.session_state.scan_results["verdicts"])
        else:
            st.info("Start a scan to see results here.")

with tabs[1]:
    st.subheader("Knowledge Ingestion")
    files = st.file_uploader("Upload Strategy PDFs", type="pdf", accept_multiple_files=True)
    if st.button("🧠 Train AI"):
        if files:
            msg = ingest_strategy_books(files)
            st.success(msg)
        else:
            st.warning("Please upload files first.")

with tabs[2]:
    st.subheader("System Administration")
    if st.button("🔄 Cloud Sync (Load Brain)"):
        msg = load_brain_from_hub()
        st.info(msg)

# Initial Load
if st.session_state.vector_db is None:
    with st.spinner("Connecting to brain..."):
        load_brain_from_hub()
