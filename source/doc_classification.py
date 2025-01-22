import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from source.text import (
    chunk_docx,
    chunk_pdf,
    chunk_pptx,
    load_pdf,
)
from source.model import load_model_and_embeddings

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MODEL = "llama3.2"



def doc_classification_prompt():
    """
    usage: print(prompt.format(context="Here is some context", question="Here is a question"))
    """
    template = """
        Based on these excerpts of the document,
        Document excerpts: {context}
        
        Answer the question: What type of document is this? 
        You can add a few arguments that make you think
        this is the correct answer.
        If you're unsure, or cannot answer the question, simply reply "I don't know".
        Please format your output as follows:
        "This document is a [???]. The information that leads to this conclusion is: [???]"
    """
    prompt = PromptTemplate.from_template(template)
    return prompt


def doc_classification_pipeline(file_name: str, model, embeddings) -> str:
    
    seed = 22
    num_clusters = 4
    num_samples_per_cluster = 1
    chunk_size = 256
    overlap = 50

    filepath = Path(file_name)
    if filepath.suffix == ".pdf":
        chunks = chunk_pdf(filepath, chunk_size=chunk_size, overlap=overlap)
    elif filepath.suffix == ".docx":
        chunks = chunk_docx(filepath, chunk_size=chunk_size, overlap=overlap)
    elif filepath.suffix == ".pptx":
        chunks = chunk_pptx(filepath, chunk_size=chunk_size, overlap=overlap)
    else:
        return "Only file type accepted for now: [PDF, DOCX, PPTX]"


    if len(chunks) < num_clusters:
        context = " ".join(chunks)
    else:
        # Method 1 - sample k elements from chunks at random
        # context_chunks = np.random.choice(chunks, 4)
        
        # Method 2 - apply k-means clustering to the embeddings of the chunks
        doc_embeddings = embeddings.embed_documents(chunks)
        # apply clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=seed).fit(doc_embeddings)

        # randomly sample k from each cluster
        context_chunks_idx = []
        for cluster in range(num_clusters):
            cluster_indices = np.where(kmeans.labels_ == cluster)[0]
            if len(cluster_indices) <= num_samples_per_cluster:
                context_chunks_idx.extend(cluster_indices)
            else:
                context_chunks_idx.extend(np.random.choice(cluster_indices, num_samples_per_cluster))
        context_chunks = [chunks[i] for i in context_chunks_idx]
        context = " [...] ".join(context_chunks)
    
    chain = doc_classification_prompt() | model | StrOutputParser()
    response = chain.invoke({'context': context})
    return response


def main():
    # file_name = "data/pdfs/CDI Eliott LEGENDRE.pdf"
    # file_name = "data/pdfs/Eliott Legendre CV.pdf"
    file_name = "data/pdfs/lettre urssaf.pdf"
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)
    
    response = doc_classification_pipeline(file_name, model, embeddings)
    print(f"Answer: {response}")


if __name__ == "__main__":
   main()