from datetime import datetime
from pathlib import Path

import streamlit as st

from source.rag import (
    load_pdf,
    load_model_and_embeddings,
    pdf_question_answering_pipeline,
)

st.set_page_config(page_title='POC RAG', page_icon='data/alkane_logo.png', layout="wide")


DATA_ROOT = Path('data')
PDFS_ROOT = DATA_ROOT / "pdfs"
MODEL = "llama3.2"
CHAIN = None

@st.cache_data
def load_model_and_embeddings_cached(model_name):
    return load_model_and_embeddings(model_name)

def homepage_content():
    # st.sidebar.image("data/alkane_logo.png", width=100)    
    col1, col2 = st.columns([1, 10])
    col1.image("data/alkane_logo.png", width=100)
    col2.header("Welcome to the RAG LLM Demo Page")
    
    st.markdown("# PDF Question Answering")

    # Load Model (cached)
    model, embeddings = load_model_and_embeddings_cached(MODEL)

    # Upload PDF file
    col11, col22 = st.columns([2, 2])
    col11.markdown("**Your file:**")
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
        
        # load file with PyPDFLoader
        pages = load_pdf(pdf_filepath)
        col222.markdown(f"- Number of pages: {len(pages)}")
        
        try:
            assert CHAIN is None
        except UnboundLocalError:
            with st.spinner('Vectorizing your PDF ..'):
                CHAIN = pdf_question_answering_pipeline(pages, embeddings, model)


    # Enter question
    question = st.text_input(
        "Enter your question(s) here (if multiple, separate by '?')", 
        value="What type of document is this? When was it signed? Who are the Parties?"
    )
    if st.button("Ask the model"):
        if uploaded_file is not None:
            if CHAIN is None:
                with st.spinner('Vectorizing your PDF ..'):
                    CHAIN = pdf_question_answering_pipeline(pages, embeddings, model)

            output = ""
            start = datetime.now()
            with st.spinner('Asking the model ..'):
                for question in question.split("?"):
                    if len(question) > 2:
                        question = question.strip(" ") + "?"
                        output += f"- **Question**: {question}\n"
                        output += f"- **Answer**: {CHAIN.invoke({'question': question})}\n\n"
            end = datetime.now()
            st.markdown(output)
            seconds = int((end - start).total_seconds()) % 60
            all_minutes = int((end - start).total_seconds()) // 60
            hours = all_minutes // 60
            minutes = all_minutes % 60
            st.info(f"Time elapsed: [{hours}:{minutes}:{seconds}]")
        else:
            st.write("/!\ Please upload a PDF file first")


if __name__ == "__main__":
    homepage_content()
