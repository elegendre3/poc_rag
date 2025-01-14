from pathlib import Path
import requests

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Bank Statement Analyzer', page_icon='data/alkane_logo.png', layout="wide")


DATA_ROOT = Path('data')
PDFS_ROOT = DATA_ROOT / "pdfs"
MODEL = "llama3.2"
FASTAPI_ENDPOINT = "http://localhost:4557"

import streamlit as st

def display_operation_legend():
    """Display color legend for operation types"""
    st.markdown("### Legend")
    
    # Define colors and operations
    legend_items = {
        'paiement carte': '#FFE6E6',
        'prelevement': '#E6FFE6',
        'virement': '#E6E6FF',
        'remise cheque': '#FFFDE6',
        'frais carte': '#FFE6FF',
        'autre': '#F2F2F2'
    }
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        for op, color in list(legend_items.items())[:3]:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                ">
                    <div style="
                        width: 20px;
                        height: 20px;
                        background-color: {color};
                        margin-right: 10px;
                        border: 1px solid #ccc;
                    "></div>
                    <span>{op}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        for op, color in list(legend_items.items())[3:]:
            st.markdown(
                f"""
                <div style="
                    display: flex;
                    align-items: center;
                    margin: 5px 0;
                ">
                    <div style="
                        width: 20px;
                        height: 20px;
                        background-color: {color};
                        margin-right: 10px;
                        border: 1px solid #ccc;
                    "></div>
                    <span>{op}</span>
                </div>
                """,
                unsafe_allow_html=True
            )

def color_rows_by_operation(df):
    """
    Color rows based on operation type
    """
    def get_color(operation):
        colors = {
            'paiement carte': '#FFE6E6',  # light red
            'prelevement': '#E6FFE6',     # light green
            'virement': '#E6E6FF',        # light blue
            'remise cheque': '#FFFDE6',   # light yellow
            'frais carte': '#FFE6FF',     # light purple
            'autre': '#F2F2F2'            # light gray
        }
        return colors.get(operation, '#FFFFFF')
    
    return [f'background-color: {get_color(df["Operation"])}'] * len(df)


def main():
    st.title("Bank Statement Analyzer")

    col1, col2 = st.columns([1, 10])

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

    # Upload PDF file
    col11, col22 = st.columns([2, 2])
    col11.markdown("**Your uploaded file:**")
    col111, col222 = col11.columns([2, 1])
    uploaded_file = col22.file_uploader("Upload your bank statement file", type="pdf")

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
        
        with st.spinner('Chunking the PDF ..'):
            st.markdown(str(pdf_filepath))
            response = requests.post(f"{FASTAPI_ENDPOINT}/chunk_bank_statement", params={"pdf_filepath": str(pdf_filepath)})
        
        st.write('---')
        chunks = response.json()['data']
        st.write(f'Solde Precedent [{chunks["solde_precedent"]}]')
        st.write(f'Nouveau Solde [{chunks["nouveau_solde"]}]')
        st.write(f'Debits: [{chunks["debit"]}] - Credits: [{chunks["credit"]}]')
    
        
        # context = "  \n".join(chunks['lines'])
        # question = st.text_input(label='Ask your question', value="What is the most likely type of document?")
        credits, debits = [], []
        for c in chunks['lines']:
            if len(c) == 0:
                continue
            credit_debit = c['debit_credit']
            if credit_debit == 'credit':
                credits.append([c["date"], c["operation"], c["amount"], c["operation_txt"], c["extras"]])
            else:
                debits.append([c["date"], c["operation"], c["amount"], c["operation_txt"], c["extras"]])
        
        credits_df = pd.DataFrame(credits, columns=["Date", "Operation", "Amount", "Operation Text", "Extras"])
        credits_df = credits_df.sort_values(by=["Amount", "Date"], ascending=True)
        total_credit = credits_df["Amount"].sum()
        credits_df.loc["Total"] = credits_df.sum(numeric_only=True)

        debits_df = pd.DataFrame(debits, columns=["Date", "Operation", "Amount", "Operation Text", "Extras"])
        debits_df = debits_df.sort_values(by=["Amount", "Date"], ascending=False)
        total_debit = debits_df["Amount"].sum()
        debits_df.loc["Total"] = debits_df.sum(numeric_only=True)

        # Apply styling to DataFrames
        credits_styled = credits_df.style.apply(color_rows_by_operation, axis=1).format({'Amount': '{:.2f}'})
        debits_styled = debits_df.style.apply(color_rows_by_operation, axis=1).format({'Amount': '{:.2f}'})

        display_operation_legend()
        # Display styled DataFrames
        st.markdown("**Credits**")
        st.dataframe(credits_styled, use_container_width=True)
        st.markdown("**Debits**")
        st.dataframe(debits_styled, use_container_width=True)


        if not np.abs((total_credit + chunks["credit"])) < 1.0:     
                st.markdown("**Credits**")
                st.dataframe(credits_df.style.apply(color_rows_by_operation, axis=1), use_container_width=True)
                st.markdown("**Debits**")
                st.dataframe(debits_df.style.apply(color_rows_by_operation, axis=1), use_container_width=True)

                if not np.abs((total_credit + chunks["credit"])) < 1.0:
                    st.error(f"Total credit mismatch: [{total_credit}] vs [{chunks['credit']}]")
                if not np.abs((total_debit - chunks["debit"])) < 1.0:
                    st.error(f"Total debit mismatch: [{total_debit}] vs [{chunks['debit']}]")

            # context = c
            # question = "Réponds par 'Credit' ou 'Debit', Cette ligne de mon relevé de banque représente-t-elle un débit ou un crédit? Si c'est une dépense, de quelle type s'agit-il? Choisis une seule catégorie parmi: 'Alimentation', 'Transport', 'Logement', 'Loisirs', 'Santé', 'Education', 'Autre'."
            # response = requests.post(
            #     f"{FASTAPI_ENDPOINT}/ask_with_context", 
            #     params={"context": context, "question": question},
            # )
            # if response.status_code == 200:
            #     st.markdown(f"Context:")
            #     st.markdown(context)
            #     # st.markdown(f"**Question: {question}:**")
            #     st.write(response.json()['message'])
            #     st.write('---')


if __name__ == "__main__":
    main()