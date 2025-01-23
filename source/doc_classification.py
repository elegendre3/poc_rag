import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from sklearn.cluster import KMeans

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from source.text import (
    chunk_doc,
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
    template_eng = """
        Based on these excerpts of the document,
        Document excerpts: {context}
        
        Answer the question: What type of document is this? 
        You can add a few arguments that make you think
        this is the correct answer.
        If you're unsure, or cannot answer the question, simply reply "I don't know".
        Please format your output as follows:
        "This document is a [???]. The information that leads to this conclusion is: [???]"
    """

    template = """
        [Introduction]
        Basé sur les extraits suivants de ce document.
        Répond a la question suivante: De quel type de document s'agit-il? 
        Si tu n'est pas sur de réponse, répond simplement "Je ne sais pas".

        La réponse doit etre formattée en type JSON comme le template suivant avec les cles: [document_type, proof, confidence, key_words]:
        {{
            "document_type": [le type de document],
            "proof": [explications ou preuves concises soutenant la decision], 
            "confidence": [le niveau de confiance dans la prediction sur une echelle de 1 a 5],
            "parties": [personnes physiques ou morales, societes ou cabinets mentionnes],
            "key_words": [5 mots-clés importants sur le document, permettant une recherche et indexation efficaces]
        }}
        
        [Contexte]
        Extraits du document: {context}
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
    chunks = chunk_doc(filepath, chunk_size=chunk_size, overlap=overlap)

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
    file_name = "data/pdfs/CDI Eliott LEGENDRE.pdf"
    # file_name = "data/pdfs/Eliott Legendre CV.pdf"
    # file_name = "data/pdfs/lettre urssaf.pdf"
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)
    
    response = doc_classification_pipeline(file_name, model, embeddings)
    print(f"Answer: {response}")


if __name__ == "__main__":
   main()