import os
import time
from typing import (Dict, List, Tuple)
import uuid 

from dotenv import load_dotenv
import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import (
    DocArrayInMemorySearch,
    InMemoryVectorStore,
)
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC as Pinecone
# from langchain.vectorstores import Pinecone as pc_vector_store


# IN-MEMORY
def create_inmemory_retriever(pages, embeddings):
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    #  select top k chunks here
    retriever = vectorstore.as_retriever()
    return retriever


def embed_with_generic_model(
        inputs: List[str],
        embeddings_model,
    ):
    embeddings = embeddings_model.embed_documents(inputs)
    return embeddings

# PINECONE
def embed_with_pinecone(
        pc_client: Pinecone,
        inputs: List[str],
        model_name: str = "multilingual-e5-large",
        parameters: Dict[str,str] = {"input_type": "passage", "truncate": "END"}

    ):
    embeddings = pc_client.inference.embed(
        inputs=inputs,
        model=model_name,
        parameters=parameters
    )
    return embeddings

def create_index_with_pinecone(pc_client: Pinecone, index_name: str, dimension: int = 1024):
    if pc_client.has_index(index_name):
        pc_client.delete_index(index_name)
    
    pc_client.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    )


def upsert_records_with_pinecone(
        pc_index: Pinecone.Index,
        texts: List[str],
        embeddings: List[np.array],
        namespace: str,
        auto_generate_ids: bool = True,
        ids: List[str] = None,
    ):
    assert len(texts) == len(embeddings), f"len(texts) [{len(texts)}] != [{len(embeddings)}] len(embeddings)"
    if ids:
        assert len(texts) == len(ids), f"len(texts) [{len(texts)}] != [{len(ids)}] len(ids)"
    else:
        if auto_generate_ids:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        else:
            raise ValueError("if 'auto_generate_ids' is False - Please provide a list of ids as List[str]")

    records = []
    for t, e, id in zip(texts, embeddings, ids):
        records.append({
            "id": id,
            "values": e['values'],
            "metadata": {'text': t}
        })

    pc_index.upsert(
        vectors=records,
        namespace=namespace
    )

def query_index_pinecone(
        pc_client: Pinecone,
        pc_index: Pinecone.Index,
        namespace: str,
        query: str,
        top_k: int = 3,
        model_name: str = "multilingual-e5-large",
        include_metadata: bool = True,
    ):

    query_embedding = pc_client.inference.embed(
        model=model_name,
        inputs=[query],
        parameters={
            "input_type": "query"
        }
    )

    results = pc_index.query(
        namespace=namespace,
        vector=query_embedding[0].values,
        top_k=top_k,
        include_values=False,
        include_metadata=include_metadata
    )
    return results

def delete_index_pinecone(pc_client: Pinecone, index_name: str):
    pc_client.delete_index(index_name)


def create_vectorstore_pinecone(pc_index_name: str, embeddings=OpenAIEmbeddings()):
    # 1 - in-memory docarray
    # vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    # 2 - in-memory
    # vectorstore = InMemoryVectorStore.from_texts([text],embedding=embeddings)
    # 3 - Pinecone
    vectorstore = PineconeVectorStore.from_existing_index(index_name=pc_index_name, embedding=embeddings)
    # vectorstore = PineconeVectorStore(pc_index_name, embedding=embeddings)
    return vectorstore


def create_retriever_pinecone(pc_vector_store: PineconeVectorStore, top_k: int=3):
    retriever = pc_vector_store.as_retriever(
        # search_type='mmr',
        # search_kwargs={"k": 1, "fetch_k": 2, "lambda_mult": 0.5},
        search_type='similarity',
        search_kwargs={"k": top_k,}
    )
    return retriever


def main():
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Define a sample dataset where each item has a unique ID and piece of text
    data = [
        {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
        {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
        {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
        {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
        {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
        {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
    ]

    # Initialize a Pinecone client with your API key
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Convert the text into numerical vectors that Pinecone can index
    embeddings = embed_with_pinecone(
            pc,
            inputs=[d['text'] for d in data],
        )
    print(embeddings)


    index_name = "example-index"
    create_index_with_pinecone(pc, index_name=index_name)
    index = pc.Index(index_name)

    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    # UPSERT
    namespace = 'example-namespace'

    upsert_records_with_pinecone(
        pc_index=index,
        texts=[t['text'] for t in data],
        embeddings=embeddings,
        namespace=namespace,
        auto_generate_ids=False,
        ids=[t['text'] for t in data],
    )
    
    # STATS
    time.sleep(10)  # Wait for the upserted vectors to be indexed
    print(index.describe_index_stats())

    # QUERY
    # Define your query
    query = "Tell me about the tech company known as Apple."

    results = query_index_pinecone(
        pc,
        index,
        namespace=namespace,
        query=query,
        top_k=3
    )

    print(results)

    delete_index_pinecone(pc, index_name)

def main_2():
    load_dotenv()
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Define a sample dataset where each item has a unique ID and piece of text
    data = [
        {"id": "vec1", "text": "Apple is a popular fruit known for its sweetness and crisp texture."},
        {"id": "vec2", "text": "The tech company Apple is known for its innovative products like the iPhone."},
        {"id": "vec3", "text": "Many people enjoy eating apples as a healthy snack."},
        {"id": "vec4", "text": "Apple Inc. has revolutionized the tech industry with its sleek designs and user-friendly interfaces."},
        {"id": "vec5", "text": "An apple a day keeps the doctor away, as the saying goes."},
        {"id": "vec6", "text": "Apple Computer Company was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne as a partnership."}
    ]  

    embeddimgs_dim = 1536
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    # Initialize a Pinecone client with your API key
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "example-index"
    create_index_with_pinecone(pc, index_name=index_name, dimension=embeddimgs_dim)
    # pc_index = pc.Index(index_name)

    pc_vectorstore = create_vectorstore_pinecone(index_name, embeddings=embeddings)
    pc_retriever = create_retriever_pinecone(pc_vectorstore)

    documents = [Document(page_content=d['text'], metadata={"id": d['id']}) for d in data]
    pc_vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])

    time.sleep(20)
    print('Indexing docs..')
    query = "Tell me about the tech company known as Apple."
    results = pc_retriever.invoke(query)
    print(results)

    delete_index_pinecone(pc, index_name)

if __name__ == "__main__":
    # main()
    main_2()
