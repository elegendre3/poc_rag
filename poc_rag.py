from datetime import datetime
from pathlib import Path
import requests

import streamlit as st


st.set_page_config(page_title='POC RAG', page_icon='data/alkane_logo.png', layout="wide")


DATA_ROOT = Path('data')
PDFS_ROOT = DATA_ROOT / "pdfs"
MODEL = "llama3.2"
FASTAPI_ENDPOINT = "http://localhost:4557"


def homepage_content():
    # st.sidebar.image("data/alkane_logo.png", width=100)    
    col1, col2 = st.columns([1, 10])
    col1.image("data/alkane_logo.png", width=100)
    col2.header("Welcome to the RAG LLM Demo Page")

    # Sanity Checks
    with st.spinner("Application starting.."):
        api_keys_set = requests.get(f"{FASTAPI_ENDPOINT}/are_api_keys_set").json()['message']
        if not api_keys_set:
            st.warning("Please set your API keys in the Credentials page before proceeding.")
            st.stop()
        
        model_healthy = requests.get(f"{FASTAPI_ENDPOINT}/health")
        if model_healthy.status_code == 401:
            st.warning("Unauthorized - Model cannot be intialized.\nPlease check your API keys in the Credentials page.")
            st.stop()
    # 

    st.markdown("# PDF Question Answering")

    # Upload PDF file
    col11, col22 = st.columns([2, 2])
    col11.markdown("**Your uploaded file:**")
    col111, col222 = col11.columns([2, 1])
    uploaded_file = col22.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file is not None:
        col111.markdown(f"- File name: {uploaded_file.name}")
        pdf_filepath = PDFS_ROOT / uploaded_file.name
        if not pdf_filepath.exists():
            with open(pdf_filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            col111.markdown("File already uploaded:")
            if col111.button("Refresh cached file"):
                pdf_filepath.unlink()
                with open(pdf_filepath, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                requests.post(f"{FASTAPI_ENDPOINT}/set_to_not_ready")
        
        if requests.get(f"{FASTAPI_ENDPOINT}/is_ready").json()['message'] == False:
            with st.spinner('Vectorizing the PDF ..'):
                response = requests.post(f"{FASTAPI_ENDPOINT}/load_pdf", params={"pdf_filepath": str(pdf_filepath)})
                st.markdown(f"[{response.status_code}] - {response.json()}")
        # col222.markdown(f"- Number of pages: TODO")

    # Enter question
    question = st.text_input(
        "Enter your question(s) here (if multiple, separate by '?')", 
        value="De quel type de document s'agit-il?Qui sont les signataires?Que faut-il savoir en priorité a propos de ce document?"
        # value="De quel type de document s'agit-il?Qui sont les signataires?Qui sont les sous-signés?Que faut-il savoir en priorité a propos de ce document?"
        # value="What type of document is this? When was it signed? Who are the Parties?"
    )
    if st.button("Ask the model"):
        if uploaded_file is not None:
            start = datetime.now()
            with st.spinner('Asking the model ..'):
                response = requests.post(f"{FASTAPI_ENDPOINT}/ask", params={"questions": question})
            end = datetime.now()
            if response.status_code == 200:
                st.markdown(response.json()['message'])
            else:   
                st.markdown(f"[{response.status_code}] - {response.json()}")
            seconds = int((end - start).total_seconds()) % 60
            all_minutes = int((end - start).total_seconds()) // 60
            hours = all_minutes // 60
            minutes = all_minutes % 60
            st.info(f"Time elapsed: [{hours}:{minutes}:{seconds}]")
        else:
            st.write("/!\ Please upload a PDF file first")

    if st.button("Check Model Status"):
        response = requests.get(f"{FASTAPI_ENDPOINT}/status_summary")
        st.markdown(f"[{response.status_code}] - {response.json()['message']}")


if __name__ == "__main__":
    homepage_content()
