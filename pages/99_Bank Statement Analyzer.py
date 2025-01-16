from pathlib import Path
import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title='Bank Statement Analyzer', page_icon='data/alkane_logo.png', layout="wide")


DATA_ROOT = Path('data')
PDFS_ROOT = DATA_ROOT / "pdfs"
MODEL = "llama3.2"
FASTAPI_ENDPOINT = "http://localhost:4557"

CATEGORY_COLORS = {
        'paiement carte': '#FFE6E6',        # light red
        'prelevement': '#E6FFE6',           # light green
        'virement': '#E6E6FF',              # light blue
        'remise cheque': '#FFFDE6',         # light yellow
        'frais carte': '#FFE6FF',           # light purple
        'emprunt': '#E6FFFE',               # light cyan
        'autre': '#F2F2F2',
        'alimentation': '#FFE0B3',          # light orange
        'restaurant': '#B3FFB3',            # mint green
        'loisirs': '#E6B3FF',               # lavender
        'santé': '#E6FFFE',                 # light cyan
        'voyage': '#B3E6FF',                # sky blue
        'logement': '#CCFFCC',              # pale green
        'edf': '#FFB3B3',                   # salmon
        'internet': '#CCE6FF',              # baby blue
        'assurance habitation': '#FFCCFF',  # pink
        'utilities': '#E6FFCC',             # lime
        'xx': '#FFE6CC',                    # peach
    }

EXPENSE_CATEGORIES = [
    "Alimentation",
    "Restaurant",
    "Voyage",
    "Loisirs",
    "Santé",
    "Autre",
]

def create_operation_pie_chart(df, title="Operations Distribution", column='Operation'):
    """
    Create a pie chart of operations using the same color scheme
    """

    # Aggregate by Operation
    operation_totals = df.groupby(column)['Amount'].sum().reset_index()
    
    # Create pie chart
    fig = go.Figure(data=[go.Pie(
        labels=operation_totals[column],
        values=operation_totals['Amount'],
        marker_colors=[CATEGORY_COLORS.get(op.lower(), '#FFFFFF') for op in operation_totals[column]],
        hovertemplate="<b>%{label}</b><br>" +
                      "Amount: %{value:.2f}€<br>" +
                      "<extra></extra>"
    )])
    
    fig.update_layout(
        title=title,
        showlegend=True,
        width=800,
        height=500
    )
    return fig

def display_operation_legend():
    """Display color legend for operation types"""
    st.markdown("### Legend")

    # Create two columns for layout
    col1, col2 = st.columns(2)
    half_point = (len(CATEGORY_COLORS) // 2)
    with col1:
        for op, color in list(CATEGORY_COLORS.items())[:half_point]:
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
        for op, color in list(CATEGORY_COLORS.items())[half_point:]:
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
        return CATEGORY_COLORS.get(operation, '#FFFFFF')
    
    return [f'background-color: {get_color(df["Operation"])}'] * len(df)

def categorize_transactions(df, color, categories):
    """
    Interactive categorization of transactions
    Returns list of updated categories
    """
    # Initialize session state
    if 'categorized' not in st.session_state:
        st.session_state.categorized = {}

    st.markdown("### Transaction Categorization:")
            
    # Container for all transactions
    with st.container():
        for idx, row in df.iterrows():
            # Unique key for each radio
            radio_key = f"transaction_{idx}"
            st.markdown(f"**Transaction {idx}**")
            
            # Display transaction details
            text = f"Date: {row['Date']} | Operation: {row['Operation']} | Amount: {row['Amount']} | Operation Text: {row['Operation Text']} |  Extras: {row['Extras']}"
            st.markdown(f"<span style='background-color: {color}; padding: 5px 10px; border-radius: 5px;'>{text}</span>", unsafe_allow_html=True)

            # Radio buttons for categorization
            selected = st.radio(
                "Category:",
                options=categories,
                key=radio_key,
                horizontal=True,
                index=categories.index(row['Operation']) if row['Operation'] in categories else 0
            )
            
            # Store selection in session state
            st.session_state.categorized[idx] = selected
            st.markdown("---")

    # Export button
    if st.button("Save Categories"):
        updated_categories = [st.session_state.categorized[i] for i in range(len(df))]
        return updated_categories
    
    return None

def export_to_excel(credits_df, debits_df, statement_date):
    """Export transaction data to styled Excel file"""
    # Create Excel writer
    filename = f"bank_statement_analysis_{statement_date.strftime('%Y%m%d')}.xlsx"
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    
    # Get workbook and add formats
    workbook = writer.book
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'bg_color': '#E6E6E6',
        'border': 1
    })
    
    # Write debits sheet
    debits_df.to_excel(writer, sheet_name='Debits', startrow=2, index=False)
    debits_worksheet = writer.sheets['Debits']
    debits_worksheet.write('A1', f'Debits Analysis - {statement_date.strftime("%B %Y")}', header_format)

    # Write credits sheet
    credits_df.to_excel(writer, sheet_name='Credits', startrow=2, index=False)
    credits_worksheet = writer.sheets['Credits']
    credits_worksheet.write('A1', f'Credits Analysis - {statement_date.strftime("%B %Y")}', header_format)
    
    # Format columns
    for worksheet in [credits_worksheet, debits_worksheet]:
        worksheet.set_column('A:A', 12)  # Date
        worksheet.set_column('B:B', 15)  # Operation
        worksheet.set_column('C:C', 12)  # Amount
        worksheet.set_column('D:D', 40)  # Operation Text
        worksheet.set_column('E:E', 20)  # Extras
    
    writer.close()
    return filename


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
        # st.write(f'Debits: [{chunks["debit"]}] - Credits: [{chunks["credit"]}]')
    
        
        credits, debits = [], []
        for c in chunks['lines']:
            if len(c) == 0:
                continue
            credit_debit = c['debit_credit']
            if credit_debit == 'credit':
                credits.append([c["date"], c["operation"], c["category"], c["amount"], c["operation_txt"], c["extras"]])
            else:
                debits.append([c["date"], c["operation"], c["category"], c["amount"], c["operation_txt"], c["extras"]])
        
        credits_df = pd.DataFrame(credits, columns=["Date", "Operation", "Category", "Amount", "Operation Text", "Extras"])
        credits_df = credits_df.sort_values(by=["Amount", "Date"], ascending=True)
        total_credit = credits_df["Amount"].sum()
        # credits_df.loc["Total"] = credits_df.sum(numeric_only=True)

        debits_df = pd.DataFrame(debits, columns=["Date", "Operation", "Category", "Amount", "Operation Text", "Extras"])
        debits_df = debits_df.sort_values(by=["Amount", "Date"], ascending=False)
        total_debit = debits_df["Amount"].sum()
        # debits_df.loc["Total"] = debits_df.sum(numeric_only=True)

        # Apply styling to DataFrames
        credits_styled = credits_df.style.apply(color_rows_by_operation, axis=1).format({'Amount': '{:.2f}'})
        debits_styled = debits_df.style.apply(color_rows_by_operation, axis=1).format({'Amount': '{:.2f}'})

        display_operation_legend()
        # Display styled DataFrames
        st.markdown("**Credits**")
        st.dataframe(credits_styled, use_container_width=True)
        st.markdown(f"**Total Credits [{chunks['credit']}]**")
        st.markdown("**Debits**")
        st.dataframe(debits_styled, use_container_width=True)
        st.markdown(f"**Total Debits [{chunks['debit']}]**")

        # Pie Chart
        st.markdown("### Expenses Distribution")
        st.plotly_chart(create_operation_pie_chart(debits_df, f"Total expenses [{chunks['debit']}]"), use_container_width=True)


        if not np.abs((total_credit + chunks["credit"])) < 1.0:     
                st.markdown("**Credits**")
                st.dataframe(credits_df.style.apply(color_rows_by_operation, axis=1), use_container_width=True)
                st.markdown("**Debits**")
                st.dataframe(debits_df.style.apply(color_rows_by_operation, axis=1), use_container_width=True)

                if not np.abs((total_credit + chunks["credit"])) < 1.0:
                    st.error(f"Total credit mismatch: [{total_credit}] vs [{chunks['credit']}]")
                if not np.abs((total_debit - chunks["debit"])) < 1.0:
                    st.error(f"Total debit mismatch: [{total_debit}] vs [{chunks['debit']}]")



        category, color = "paiement carte", CATEGORY_COLORS["paiement carte"]
        cats_df = debits_df[debits_df["Operation"] == category].reset_index(drop=True)
        categories = categorize_transactions(cats_df, color, EXPENSE_CATEGORIES)

        if categories:
            st.success("Categories updated!")
            cats_df['Category'] = categories
            concat_expenses = pd.concat([debits_df[debits_df['Operation'] != category], cats_df])

            # Pie Chart
            st.markdown("### Expenses Distribution")
            c1, c2 = st.columns([1, 1])
            c1.plotly_chart(create_operation_pie_chart(cats_df, f"Card expenses [{int(cats_df['Amount'].sum())}]", column="Category"))
            c2.plotly_chart(create_operation_pie_chart(concat_expenses, f"Total expenses [{int(concat_expenses['Amount'].sum())}]", column="Category"))

            # statement_date = pd.to_datetime(credits_df['Date'].iloc[0])
            statement_date = pd.to_datetime("today")
            excel_file = export_to_excel(
                credits_df, 
                concat_expenses, 
                statement_date,
            )
            st.success(f"Excel report exported: {excel_file}")


            # context = "  \n".join(chunks['lines'])
            # question = st.text_input(label='Ask your question', value="What is the most likely type of document?")

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