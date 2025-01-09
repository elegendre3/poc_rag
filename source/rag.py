from operator import itemgetter
import os
import time
import uuid 

from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone


from source.text import (
    chunk_pdf,
    load_pdf,
)
from source.model import load_model_and_embeddings
from source.vector_store import (
    create_inmemory_retriever, 
    create_index_with_pinecone,
    upsert_records_with_pinecone,
    embed_with_pinecone,
    create_vectorstore_pinecone,
    create_retriever_pinecone,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o-mini"
# MODEL = "text-embedding-3-small"
# MODEL = "mixtral:8x7b"
MODEL = "llama3.2"



def create_rag_chain(retriever, model):
    prompt = question_answering_prompt()
    parser = StrOutputParser()
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )
    return chain

def pdf_qa_pipeline_inmemory(pages, embeddings, model):

    retriever = create_inmemory_retriever(pages, embeddings)
    chain = create_rag_chain(retriever, model)
    return chain


def question_answering_prompt():
    """
    usage: print(prompt.format(context="Here is some context", question="Here is a question"))
    """
    template = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
    """
    prompt = PromptTemplate.from_template(template)
    return prompt


def main():
    # 1 - basic prompt/chat
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)

    # 1.b - Add Str parser for chat models (OpenAI)
    # llama is completion model so output is str
    parser = StrOutputParser()
    chain = model | parser 
    print(chain.invoke("Tell me a joke"))


    # 2 - Pdf
    pages = load_pdf("../data/CDI Eliott LEGENDRE.pdf")
    # pages[0]


    # 3 - In-Memory vector search engine
    retriever = create_inmemory_retriever(pages, embeddings)


    # 4 - Prompt with context
    prompt = question_answering_prompt()
    # print(prompt.format(context="Here is some context", question="Here is a question"))


    # 5 - simple chain example
    chain = prompt | model | parser
    chain.invoke({"context": "My parents named me Santiago", "question": "What's your name'?"})

    # 6 - chain with retriever
    chain = (
        {
            "context": itemgetter("question") | retriever,
            "question": itemgetter("question"),
        }
        | prompt
        | model
        | parser
    )

    questions = [
        # "Which schools did this person attend and for how long?",
        "What type of document is this?",
        "What is the name of the person?",
        "What is the starting salary?",
        "Is there any evolution planned fo compensation?",
        "Is there any evolution planned fo salary?",
        "What else is important to know from this contract?",
    ]

    for question in questions:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()


def main_2():
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)
    pages = load_pdf("data/CDI Eliott LEGENDRE.pdf")
    chain = pdf_qa_pipeline_inmemory(pages, embeddings, model)

    for question in [ "What type of document is this?"]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()


def main_3():
    
    embeddimgs_dim = 1536
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)

    index_name = "example-index"
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    create_index_with_pinecone(pc, index_name=index_name, dimension=embeddimgs_dim)
    index = pc.Index(index_name)

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    pc_vectorstore = create_vectorstore_pinecone(index_name, embeddings=embeddings)
    pc_retriever = create_retriever_pinecone(pc_vectorstore)

    file_name = "data/CDI Eliott LEGENDRE.pdf"
    chunks = chunk_pdf(file_name)
    documents = [Document(page_content=d, metadata={"title": file_name}) for d in chunks]
    pc_vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])
    
    # check status
    time.sleep(10)  # Wait for the upserted vectors to be indexed
    print(index.describe_index_stats())

    chain = create_rag_chain(pc_retriever, model)

    for question in ["What type of document is this?"]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()

if __name__ == "__main__":
    # main()
#    main_2()
   main_3()