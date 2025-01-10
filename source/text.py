import logging
from pathlib import Path
from typing import List

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


if __name__ == "__main__":
    # chunks = chunk_pdf("data/CDI Eliott LEGENDRE.pdf")
    # chunks = chunk_pdf("/Users/eliottlegendre/Documents/prive/evolution_naturejournal_leeCronin_2023.pdf")
    chunks = chunk_pdf(Path("/Users/eliottlegendre/Library/CloudStorage/Box-Box/PRO/OTTILE/2024/URSSAF courrier 1.pdf"))
    for i, chunk in enumerate(chunks):
        print(f"Chunk [{i+1}]:\n{chunk}\n")

    # import easyocr
    # reader = easyocr.Reader(['fr','en']) # this needs to run only once to load the model into memory
    # result = reader.readtext('data/screen.png')