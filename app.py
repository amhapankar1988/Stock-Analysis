import os
import time
import json
import requests
import pandas as pd
import streamlit as st
from alpha_vantage.timeseries import TimeSeries
from finvizfinance.screener.overview import Overview
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
from langchain_ibm import ChatWatsonx
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Trading Brain AI", layout="wide", page_icon="📈")

# Secrets Management (Streamlit Cloud Dashboard)
HF_TOKEN = st.secrets.get("HF_TOKEN")
AV_API_KEY = st.secrets.get("ALPHA_VANTAGE_KEY")
WATSONX_APIKEY = st.secrets.get("WATSONX_APIKEY")
PROJECT_ID = st.secrets.get("WATSONX_PROJECT_ID")
DATASET_REPO_ID = st.secrets.get("DATASET_REPO_ID", "amhapankar/my-trading-brain")

# --- 2. INITIALIZE MODELS ---
@st.cache_resource
def init_models():
    if not all([WATSONX_APIKEY, PROJECT_ID, HF_TOKEN]):
        st.error("🔑 Missing Secrets in Streamlit Dashboard!")
        st.stop()

    llm = ChatWatsonx(
        model_id="meta-llama/llama-3-3-70b-instruct",
        url="https://ca-tor.ml.cloud.ibm.com",
        project_id=PROJECT_ID,
        apikey=WATSONX_APIKEY,
        params={"decoding_method": "sample", "max_new_tokens": 1000, "temperature": 0.2}
    )

    # RAM-Efficient Embeddings via HF API
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=HF_TOKEN
    )
    return llm, embeddings

llm, embeddings = init_models()

# --- 3. PERSISTENCE HELPERS ---

def save_portfolio_to_hub():
    """Saves the current portfolio list to Hugging Face as a JSON file."""
    try:
        with open("portfolio.json", "w") as f:
            json.dump(st.session_state.portfolio, f)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj="portfolio.json",
            path_in_repo="portfolio.json",
            repo_id=DATASET_REPO_ID,
            repo_type="dataset",
            token=HF_TOKEN
        )
    except Exception as e:
        st.sidebar.error(f"Sync Error: {e}")

def load_portfolio_from_hub():
    """Loads the portfolio list from Hugging Face."""
    try:
        file_path = hf_hub_download(
            repo_id=DATASET_REPO_ID, 
            filename="portfolio.json", 
            repo_type="dataset", 
            token=HF_TOKEN
        )
        with open(file_path, "r") as f:
            st.session_state.portfolio = json.load(f)
    except:
        st.session_state.portfolio = []

def load_brain_from_hub():
    """Loads the FAISS vector database from Hugging Face."""
    try:
        local_dir = snapshot_download(repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        st.session_state.vector_db = FAISS.load_local(local_dir, embeddings, allow_dangerous_deserialization=True)
        return True
    except:
        st.session_state.vector_db = None
        return False

# --- 4. SESSION STATE INITIALIZATION ---
if 'vector_db' not in st.session_state:
    load_brain_from_hub()
if 'portfolio' not in st.session_state:
    load_portfolio_from_hub()
if 'usage_count' not in st.session_state:
    st.session_state.usage_count = 0
if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# --- 5. CORE LOGIC FUNCTIONS ---

def ingest_strategy(files):
    """Processes PDF files and updates the knowledge base with a progress bar."""
    try:
        docs = []
        progress = st.progress(0, text="📄 Reading PDFs...")
        for i, f in enumerate(files):
            temp_path = f"temp_{f.name}"
            with open(temp_path, "wb") as tmp:
                tmp.write(f.getbuffer())
            docs.extend(PyPDFLoader(temp_path).load())
            os.remove(temp_path)
            progress.progress(int((i+1)/len(files) * 30))

        progress.progress(50, text="✂️ Chunking text...")
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
        
        progress.progress(80, text="🧠 Indexing Brain...")
        st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
        
        # Save brain back to cloud
        st.session_state.vector_db.save_local("faiss_index")
        HfApi().upload_folder(folder_path="faiss_index", repo_id=DATASET_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        
        progress.progress(100)
        st.success(f"✅ Strategy Loaded: {len(splits)} chunks indexed.")
    except Exception as e:
        st.error(f"Ingestion Error: {e}")

def run_triple_screen():
    """Main market scanner logic."""
    if not st.session_state.vector_db:
        st.warning("Please train the AI on strategy books first!")
        return
    
    ts = TimeSeries(key=AV_API_KEY, output_format='pandas')
    
    with st.status("🔍 Scanning Markets...", expanded=True) as status:
        foverview = Overview()
        foverview.set_filter(filters_dict={'Price': 'Over $20', 'Relative Volume': 'Over 1.5'})
        df = foverview.screener_view()
        candidates = df['Ticker'].tolist()[:3] if df is not None else []
        
        results, charts, audit = [], [], ""
        for ticker in candidates:
            status.write(f"Analyzing {ticker}...")
            time.sleep(12) # Rate limiting
            
            data, _ = ts.get_daily(symbol=ticker, outputsize='compact')
            curr_price = data['4. close'].iloc[0]
            
            # RAG Audit
            context = "\n".join([d.page_content for d in st.session_state.vector_db.similarity_search(ticker, k=2)])
            prompt = f"Audit {ticker} (Price: {curr_price}) based on Strategy: {context}"
            res = llm.invoke(prompt)
            
            audit += f"## {ticker} Audit\n{res.content}\n\n---\n"
            results.append([ticker, f"${curr_price}", "Neutral"])
            charts.append(f"https://charts2.finviz.com/chart.ashx?t={ticker}&ty=c&ta=1&p=d&s=l")
            st.session_state.usage_count += 2

        st.session_state.scan_results = {"df": pd.DataFrame(results, columns=["Ticker", "Price", "Sent"]), "charts": charts, "verdict": audit}

# --- 6. UI TABS ---
st.title("🚀 My Trading Brain AI")

tabs = st.tabs(["📈 Terminal", "💼 Portfolio", "📚 Strategy"])

with tabs[0]: # TERMINAL
    if st.button("🚀 RUN TRIPLE SCREEN"):
        run_triple_screen()
    if st.session_state.scan_results:
        st.dataframe(st.session_state.scan_results["df"], use_container_width=True)
        cols = st.columns(3)
        for i, url in enumerate(st.session_state.scan_results["charts"]):
            cols[i%3].image(url)
        st.markdown(st.session_state.scan_results["verdict"])

with tabs[1]: # PORTFOLIO AUDIT
    st.subheader("Holdings Risk Manager")
    col_in, col_btn = st.columns([3, 1])
    new_t = col_in.text_input("Add Ticker").upper()
    if col_btn.button("➕ Add"):
        if new_t and new_t not in st.session_state.portfolio:
            st.session_state.portfolio.append(new_t)
            save_portfolio_to_hub()
            st.rerun()

    if st.button("🔍 AUDIT MY PORTFOLIO", type="primary"):
        # Audit logic for each stock in session_state.portfolio
        for stock in st.session_state.portfolio:
            with st.expander(f"Audit for {stock}"):
                st.write(f"Evaluating {stock} health against knowledge base...")
                # Simplified audit for this example
                st.info(f"AI Audit for {stock}: Data suggests holding based on strategy context.")
    
    st.write("Current Holdings:")
    for t in st.session_state.portfolio:
        if st.button(f"❌ Remove {t}"):
            st.session_state.portfolio.remove(t)
            save_portfolio_to_hub()
            st.rerun()

with tabs[2]: # STRATEGY
    uploaded = st.file_uploader("Upload Strategy Books (PDF)", accept_multiple_files=True)
    if st.button("🧠 Train AI Knowledge"):
        if uploaded:
            ingest_strategy(uploaded)

