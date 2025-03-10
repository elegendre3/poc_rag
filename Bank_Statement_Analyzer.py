from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from source.text import chunk_bank_statement

st.set_page_config(page_title='Bank Statement Analyzer', page_icon='data/alkane_logo.png', layout="wide")


DATA_ROOT = Path('data')
PDFS_ROOT = DATA_ROOT / "pdfs"

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
        'vacances': '#B3E6FF',                # sky blue
        'esthetique': '#CCFFCC',              # pale green
        'edf': '#FFB3B3',                   # salmon
        'internet': '#CCE6FF',              # baby blue
        'assurance habitation': '#FFCCFF',  # pink
        'fringues': '#E6FFCC',             # lime
        'xx': '#FFE6CC',                    # peach
    }

EXPENSE_CATEGORIES = [
    "Alimentation",
    "Restaurant",
    "Transport",
    "Vacances",
    "Loisirs",
    "Santé",
    "Esthetique",
    "Fringues",
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
    
    # Write Analysis sheet
    grouped_expenses = debits_df.groupby('Category')['Amount'].sum().reset_index().sort_values(by='Amount', ascending=False).reset_index(drop=True)
    grouped_expenses.to_excel(writer, sheet_name='Analysis', startrow=2, index=False)
    analyis_worksheet = writer.sheets['Analysis']
    analyis_worksheet.write('A1', f'Expenses Grouped Analysis - {statement_date.strftime("%B %Y")}', header_format)

    # Format columns
    for worksheet in [credits_worksheet, debits_worksheet]:
        worksheet.set_column('A:A', 12)  # Date
        worksheet.set_column('B:B', 15)  # Operation
        worksheet.set_column('C:C', 12)  # Amount
        worksheet.set_column('D:D', 40)  # Operation Text
        worksheet.set_column('E:E', 20)  # Extras

    # add expenses pie chart
    expenses_chart = workbook.add_chart({'type': 'pie'})
    expenses_chart.add_series({
        'categories': f'=Analysis!$A$3:$A${len(grouped_expenses) + 2}',
        'values': f'=Analysis!$B$3:$B${len(grouped_expenses) + 2}',
        'data_labels': {'percentage': True}
    })
    expenses_chart.set_title({'name': 'Expenses Distribution'})
    expenses_chart.set_size({'width': 500, 'height': 400})
    expenses_chart.set_legend({'position': 'bottom'})
    expenses_chart.set_style(10)
    expenses_chart.set_table({
        'show_keys': True,
        'show_series_key': True,
        'show_category_key': True,
        'show_value': True,
        'show_percentage': True,
        'font': {'bold': True}
    })
    expenses_chart.set_title({
        'name': 'Expenses Distribution',
        'name_font': {'bold': True, 'italic': True}
    })
    analyis_worksheet.insert_chart('F2', expenses_chart, {'x_offset': 25, 'y_offset': 10}) 

    writer.close()
    return filename


def main():
    st.title("Bank Statement Analyzer")

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
        
        with st.spinner('Chunking the PDF ..'):
            st.markdown(str(pdf_filepath))
            chunks = chunk_bank_statement(pdf_filepath)
        
        st.write('---')
        st.write(f'Solde Precedent [{chunks["solde_precedent"]}]')
        st.write(f'Nouveau Solde [{chunks["nouveau_solde"]}]')    
        
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

        debits_df = pd.DataFrame(debits, columns=["Date", "Operation", "Category", "Amount", "Operation Text", "Extras"])
        debits_df = debits_df.sort_values(by=["Amount", "Date"], ascending=False)
        total_debit = debits_df["Amount"].sum()

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
            st.success(f"Excel report ready: {excel_file}")

            with open(excel_file, 'rb') as f:
                file_bytes = f.read()
            
            st.download_button(
                label="📥 Download Excel Report",
                data=file_bytes,
                file_name=f"bank_statement_analysis_{statement_date.strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )


if __name__ == "__main__":
    main()
