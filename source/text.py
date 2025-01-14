import logging
from pathlib import Path
import re
from typing import (List, Tuple)

from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: Path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of specified size with optional overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The size of each chunk.
        overlap (int): The number of overlapping tokens between chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def chunk_pdf(file_path: Path, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    pages = load_pdf(file_path)
    if len(pages) == 0:
        logging.info('Warning: PDF has no text layer')
        logging.info('OCRing..')
        if ".pdf" == file_path.as_posix()[-4:]:
            import fitz
            doc = fitz.open(file_path)
            zoom = 4
            mat = fitz.Matrix(zoom, zoom)
            count = 0
            # Count variable is to get the number of pages in the pdf
            for p in doc:
                count += 1
            page_files = []
            for i in range(count):
                val = file_path.parent / f"{file_path.stem}_page_{str(i+1)}.jpg"
                page_files.append(val.as_posix())
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=mat)
                pix.save(val)
            doc.close()
        else:
             page_files = [file_path.as_posix()]
        import easyocr
        reader = easyocr.Reader(['fr','en']) # this needs to run only once to load the model into memory
        ocred_texts = [reader.readtext(filepath) for filepath in page_files]
        flattened_list = [item[1] for sublist in ocred_texts for item in sublist]
        # full_text = " ".join(flattened_list)
        # chunks = chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        return flattened_list
    text = "\n ".join([page.page_content for page in pages])
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return chunks

def extract_amount_cents(text: str) -> int:
    """
    Extract amount from text and convert to cents integer
    Examples:
        '17.839,45' -> 1783945
        'SOLDE PRÉCÉDENT AU 06/12/2024 17.839,45' -> 17839
    """
    pattern = r'(\d+[.,\d]*\d+),(\d{2})$'
    match = re.search(pattern, text)
    if match:
        whole_part = match.group(1).replace('.', '')  # Remove thousand separators
        # cents_part = match.group(2)
        return int(whole_part)
    return -1

def extract_two_amounts_cents(text: str) -> tuple[int, int]:
    """
    Extract two amounts from text and convert to cents integers
    Example:
        'TOTAUX DES MOUVEMENTS 2.887,76 4.100,62' -> (288776, 410062)
    """
    pattern = r'(\d+[.,\d]*\d+),(\d{2})\s+(\d+[.,\d]*\d+),(\d{2})'
    match = re.search(pattern, text)
    if match:
        amount1 = int(match.group(1).replace('.', ''))
        amount2 = int(match.group(3).replace('.', ''))
        return amount1, amount2
    return -1, -1

def extract_two_dates_operation_amount(text: str) -> Tuple[int, int]:
    """
    Extract two dates, operation text and optional amount from a line
    Example:
        '16/12/2024 16/12/2024 PRELEVEMENT EUROPEEN 7409851689 1.234,56' -> 
        (True, '16/12/2024', '16/12/2024', 'PRELEVEMENT EUROPEEN 7409851689', '1.234,56')
        '16/12/2024 16/12/2024 PRELEVEMENT EUROPEEN' -> 
        (True, '16/12/2024', '16/12/2024', 'PRELEVEMENT EUROPEEN', '')
        'SOLDE PRÉCÉDENT AU 06/12/2024' -> (False, '', '', '', '')
    """
    pattern = r'^(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(.+?)(?:\s+(\d{1,3}(?:\.\d{3})*,\d{2}))?$'
    match = re.match(pattern, text)
    if match:
        date1 = match.group(1)
        date2 = match.group(2)
        operation_text = match.group(3)
        amount = match.group(4) or ''
        return True, date1, date2, operation_text, amount
    return False, '', '', '', ''

def is_line_beginning_of_item(text: str) -> bool:
   """
    Check if line starts with two dates in DD/MM/YYYY format
    Example:
        '16/12/2024 16/12/2024 PRELEVEMENT EUROPEEN 7409851689' -> True
        'SOLDE PRÉCÉDENT AU 06/12/2024' -> False
    """
   pattern = r'^(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})'
   match = re.match(pattern, text)
   return bool(match)


def bank_operation_classifier(text: str) -> Tuple[str, str]:
    """
    Classify bank operation type
    - SOCIETE GENERALE
    """
    lower_text = text.lower()
    if "carte" in lower_text:
        return ("paiement carte", "debit")
    elif "prelevement" in lower_text:
        return ("prelevement", "debit")
    elif "vir perm" in lower_text:
        return ("virement", "debit")
    elif "vir recu" in lower_text:
        return ("virement", "credit")
    elif "remise cheque" in lower_text:
        return ("remise cheque", "credit")
    elif ("cotisation jazz" in lower_text) or ("option tranquillite" in lower_text):
        return ("frais carte", "debit")
    else:
        return ("autre", "debit")
def append_line_item(line_item: List[str], lines: List) -> List:
    """
    Append line item to list of lines
    """
    if len(line_item) == 0:
        return lines
    date = line_item[0]
    operation_txt = line_item[2]
    amount = int(line_item[-1].split(',')[0].replace('.', '')) + float(f"0.{line_item[-1].split(',')[1]}")
    
    operation = bank_operation_classifier(operation_txt)
    if operation[1] == "credit":
        amount = -amount

    if len(line_item) > 4:
        extras = line_item[3:-1]
    else:
        extras = []

    lines.append(
        {
            "date": date,
            "operation": operation[0],
            "debit_credit": operation[1],
            "amount": amount,
            "operation_txt": operation_txt,
            "extras": extras
        }
    )
    return lines

def chunk_bank_statement_soge(file_path: Path):
    pages = load_pdf(file_path)

    lines = []
    skip = True
    solde_precedent = -1
    nouveau_solde = -1
    debit = -1
    credit = -1
    sublines = []
    for page in pages:
        for line in page.page_content.split("\n"):
            if len(line) > 0:
                is_beginning, date1, date2, operation_text, amount = extract_two_dates_operation_amount(line)
                if is_beginning:
                    if skip:
                        # header of new page, restart collection
                        skip=False
                    else:
                        # end of item
                        append_line_item(line_item=sublines, lines=lines)
                        # split out first line of item into date1, date2, operation_text, amount
                    sublines = [x for x in [date1, date2, operation_text, amount] if len(x) > 0]
                elif "SOLDE PRÉCÉDENT" in line:
                    # absolute start
                    skip = False
                    solde_precedent = extract_amount_cents(line)
                    continue
                else:
                    if skip:
                        if "NOUVEAU SOLDE" in line:
                            nouveau_solde = extract_amount_cents(line)
                            break
                        else:
                            continue
                    else:
                        if "suite >>>" in line:
                            # new page
                            append_line_item(line_item=sublines, lines=lines)
                            sublines = []
                            skip=True
                            continue
                        if "***" in line:
                            # end of month statement
                            continue
                        if "TOTAUX" in line:
                            # absolute end
                            append_line_item(line_item=sublines, lines=lines)
                            debit, credit = extract_two_amounts_cents(line)
                            skip = True
                            continue
                    sublines.append(line)
    return {
        "solde_precedent": solde_precedent,
        "nouveau_solde": nouveau_solde,
        "debit": debit,
        "credit": credit,
        "lines": lines
    }

if __name__ == "__main__":
    # chunks = chunk_pdf("data/CDI Eliott LEGENDRE.pdf")
    # chunks = chunk_pdf("/Users/eliottlegendre/Documents/prive/evolution_naturejournal_leeCronin_2023.pdf")
    # chunks = chunk_pdf(Path("/Users/eliottlegendre/Library/CloudStorage/Box-Box/PRO/OTTILE/2024/URSSAF courrier 1.pdf"))
    
    chunks = chunk_bank_statement_soge("data/bank_statement.pdf")
    solde_precedent = chunks["solde_precedent"]
    nouveau_solde = chunks["nouveau_solde"]
    lines = chunks["lines"]
    print('w')

    # chunks = chunk_pdf(Path("data/bank_statement.pdf"))
    # for i, chunk in enumerate(chunks):
    #     print(f"Chunk [{i+1}]:\n{chunk}\n")

    # import easyocr
    # reader = easyocr.Reader(['fr','en']) # this needs to run only once to load the model into memory
    # result = reader.readtext('data/screen.png')