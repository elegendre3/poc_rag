import json
from operator import itemgetter
import os
from pathlib import Path
import time
import uuid 

from dotenv import load_dotenv

from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone


from source.doc_classification import doc_classification_pipeline
from source.text import (
    chunk_pdf,
    chunk_doc,
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
    delete_index_pinecone,
)


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o-mini"
# MODEL = "text-embedding-3-small"
# MODEL = "mixtral:8x7b"
MODEL = "llama3.2"



def create_qa_rag_chain(retriever, model):
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

def create_propal_rag_chain(retriever, model):        
    prompt = propal_retriever_prompt()
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
    chain = create_qa_rag_chain(retriever, model)
    return chain


def question_answering_prompt():
    """
    usage: print(prompt.format(context="Here is some context", question="Here is a question"))
    """
    template_eng = """
        Answer the question based on the context below. If you can't 
        answer the question, reply "I don't know".

        Context: {context}

        Question: {question}
    """

    template_fr = """
        [Introduction]
        En tant qu'assistant, et dans le mesure du possible, 
        répond de manière claire et concise à la question posée 
        en te basant sur le contexte fourni ci-dessous.
        Si tu ne peux pas y répondre car l'information n'est pas présente ou
        pour une autre raison, répond simplement "Je ne sais pas".

        [Contexte]: {context}

        [Question]: {question}
    """

    prompt = PromptTemplate.from_template(template_fr)
    return prompt

def propal_retriever_prompt():
    """
    usage: print(prompt.format(context="Here is some context", question="Here is a question"))
    """

    # question_part_prompt = """
    #     Vous recevrez des extraits d'un document en lien avec une proposition de services, 
    #     et devrez extraire: 
    #         la société ou cabinet émetteur,
    #         la société ou cabinet destinataire,
    #         le type de prestations ou produits proposés,
    #         la description de ce que la mission ou l'offre propose,
    #         le montant de la proposition,
    #         les conditions de facturation,
    #         les conditions de règlement,
    # """

    template_propal_json = """
        [Instructions]: {question}
        
        Retournez l'information extraite dans un format JSON.
        Si la réponse n'est pas evidente/présente, repondez "je ne sais pas".
        
        Voici un exemple: {{
        "emetteur": "Société AAA", 
        "proposition": "La societe AAA propose un accompagnement sur le sujet de l'implementation d'un outil type ERP,
        , de conduite du changement et de formation des collaborateurs.",
        "montant": "80,000 euros hors taxes par an pour la license, ainsi que des frais de conseil et de deplacement seront a prevoir.", 
        "conditions de facturation": "La mission sera facturée au temps passé, une facture sera émise à chaque fin de mois pour le nombre de jours effectivement consacrés à la mission.
            Si des déplacements sont nécessaires hors de la région parisienne, les frais afférents (train, avion, hôtel, restaurant) seront refacturés à l’identique.", 
        "conditions de règlement": "A réception de facture.", 
        }}

        [Extraits]: {context}
    """

    prompt = PromptTemplate.from_template(template_propal_json)
    return prompt


def init_model_and_vectorstore(index_name: str = "example-index", top_k: int = 3):
    embeddimgs_dim = 1536
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)

    index_name = "example-index"
    
    pc = Pinecone(api_key=PINECONE_API_KEY)

    create_index_with_pinecone(pc, index_name=index_name, dimension=embeddimgs_dim)
    index = pc.Index(index_name)

    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

    pc_vectorstore = create_vectorstore_pinecone(index_name, embeddings=embeddings)
    pc_retriever = create_retriever_pinecone(pc_vectorstore, top_k=top_k)
    return model, embeddings, pc, index, pc_vectorstore, pc_retriever



# Single doc approach
def qa_rag_chain_inmemory_old():
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


def qa_rag_chain_inmemory():
    model, embeddings = load_model_and_embeddings(MODEL, OPENAI_API_KEY)
    pages = load_pdf("data/CDI Eliott LEGENDRE.pdf")
    chain = pdf_qa_pipeline_inmemory(pages, embeddings, model)

    for question in [ "What type of document is this?"]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()


def qa_rag_chain():
    
    embeddimgs_dim = 1536
    model, embeddings, pc, index, pc_vectorstore, pc_retriever = init_model_and_vectorstore()

    file_name = "data/CDI Eliott LEGENDRE.pdf"
    chunks = chunk_pdf(file_name)
    documents = [Document(page_content=d, metadata={"title": file_name}) for d in chunks]
    pc_vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])
    
    # check status
    time.sleep(10)  # Wait for the upserted vectors to be indexed
    print(index.describe_index_stats())

    chain = create_qa_rag_chain(pc_retriever, model)

    for question in ["What type of document is this?"]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()


# Multiple doc approach
def ensemble_rag_chain():
    index_name = "example-index"
    top_k = 10
    chunk_size, chunk_overlap = 256, 50

    model, embeddings, pc, index, pc_vectorstore, pc_retriever = init_model_and_vectorstore(index_name=index_name, top_k=top_k)

    parser = StrOutputParser()
    chain = model | parser

    documents = []
    files_path = Path("data/alkane")
    c = 0
    for file in files_path.rglob("*"):
        # if c == 10:
        #     break
        if file.is_dir():
            continue
        filename = file.name
        if filename == '.DS_Store':
            continue
        if filename == "Devis SEPTEO signé 24 juin mise en place lien Secib Expert & solution Time Tracking.pdf":
            continue
        print(filename)
        c+=1
        
        # ENRICHMENT
        # Get doc type and keywords
        doc_keywords = []
        response = doc_classification_pipeline(file, model, embeddings)
        try:
            doc_json = json.loads(response)
        except json.decoder.JSONDecodeError as e:
            try:
                fixed_response = chain.invoke(f"Please fix the following json: {response}. Only return the fixed json to be ingested downstream. Your response has to start with '{' and end with '}'")
                # remove any non-json characters
                doc_json = json.loads(fixed_response.replace("\\", ''))
            except Exception as e:
                print(f'Doc [{filename}] could not be classified, output is non-json')
                doc_json = None
        
        if doc_json is not None:   
            for key in ["document_type", "parties", "key_words"]:                
                if key in doc_json:
                    if isinstance(doc_json[key], str):
                        doc_keywords.append(doc_json[key])
                    elif isinstance(doc_json[key], list):
                        doc_keywords.extend(doc_json[key])
                    else:
                        print(f"Key [{key}] is neither str nor list")

        print(doc_keywords)
        # 

        chunks = chunk_doc(file, chunk_size=chunk_size, overlap=chunk_overlap, metadata=doc_keywords)
        documents.extend([Document(page_content=d, metadata={"title": filename}) for d in chunks])

    pc_vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])
    
    # check status
    time.sleep(60)  # Wait for the upserted vectors to be indexed
    print(index.describe_index_stats())

    chain = create_qa_rag_chain(pc_retriever, model)
    # chain = create_propal_rag_chain(pc_retriever, model)

    for question in [
        # needs global view (kmeans)
        "Donne moi une vision globale de ma base de documents, je veux avoir une idée globale du type de documents en présence, je veux en particulier savoir quel type d'activité est evoqué, les clients si certains sont recurrents, et les dates importantes et echeances a avoir en tete pour les prochains mois et annees. Je ne veux pas de strategie pour el faire, je veux extraire les informations a disposition.",
        "Quelles sont les informations disponibles sur le fournisseur: Intapp? Plus précisement je veux savoir, basé sur mes documents, avec les extraits qui le montrent: - avec quels clients ils ont travaillé? - sur quels sujets? - a quelles dates?"
    ]:
        print(f"Question: {question}")
        print(f"Answer: {chain.invoke({'question': question})}")
        print()
    
    delete_index_pinecone(pc, index_name)


if __name__ == "__main__":
#    qa_rag_chain()
   ensemble_rag_chain()