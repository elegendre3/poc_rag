from operator import itemgetter
import os

from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings



load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o-mini"
# MODEL = "text-embedding-3-small"
# MODEL = "mixtral:8x7b"
MODEL = "llama3.2"


def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()


def load_model_and_embeddings(model_name):
    """
    Usage: model.invoke("Tell me a joke"))
    """
    if MODEL.startswith("gpt"):
        model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=MODEL)
        embeddings = OpenAIEmbeddings()
    else:
        model = Ollama(model=MODEL)
        embeddings = OllamaEmbeddings(model=MODEL)
    return model, embeddings

def create_in_memory_pdf_retriever(pages, embeddings):
    vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
    #  select top k chunks here
    retriever = vectorstore.as_retriever()
    return retriever


def pdf_question_answering_pipeline(pages, embeddings, model):

    retriever = create_in_memory_pdf_retriever(pages, embeddings)
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
    model, embeddings = load_model_and_embeddings(MODEL)

    # 1.b - Add Str parser for chat models (OpenAI)
    # llama is completion model so output is str
    parser = StrOutputParser()
    chain = model | parser 
    print(chain.invoke("Tell me a joke"))


    # 2 - Pdf
    pages = load_pdf("../data/CDI Eliott LEGENDRE.pdf")
    # pages[0]


    # 3 - In-Memory vector search engine
    retriever = create_in_memory_pdf_retriever(pages, embeddings)


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
    model, embeddings = load_model_and_embeddings(MODEL)
    pages = load_pdf("data/CDI Eliott LEGENDRE.pdf")
    chain = pdf_question_answering_pipeline(pages, embeddings, model)

    for question in [ "What type of document is this?"]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()

if __name__ == "__main__":
    # main()
   main_2()