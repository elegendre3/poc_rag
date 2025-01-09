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
    text = "\n ".join([page.page_content for page in pages])
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return chunks


if __name__ == "__main__":
    chunks = chunk_pdf("data/CDI Eliott LEGENDRE.pdf")
    for i, chunk in enumerate(chunks):
        print(f"Chunk [{i+1}]:\n{chunk}\n")
