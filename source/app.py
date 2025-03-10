from datetime import datetime
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from ratelimit import limits, sleep_and_retry
import time
from typing import (Any, List, Optional)
import uuid
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from openai import OpenAIError
from pinecone.core.openapi.shared.exceptions import UnauthorizedException
from pinecone.grpc import PineconeGRPC as Pinecone
from pydantic_settings import BaseSettings
import uvicorn 

from source.rag import (
    create_qa_rag_chain,
    create_propal_rag_chain,
    load_pdf,
    load_model_and_embeddings,
    pdf_qa_pipeline_inmemory,
    question_answering_prompt,
)
from source.text import (
    chunk_doc,
    load_pdf,
)
from source.vector_store import (
    create_index_with_pinecone,
    create_vectorstore_pinecone,
    create_retriever_pinecone,
    delete_index_pinecone,
)
from source.doc_classification import doc_classification_pipeline


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    # model_name: str = "llama3.2"
    model_name: str = "gpt-4o-mini"
    embeddings_dim: int = 1536
    chunk_size: int = 512
    chunk_overlap: int = 50
    ratelimit_oneminute: int = 60
    max_calls_per_minute: int = 6
    
    class Config:
        env_file = ".env"


class Model:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.is_ready = False
        self.model = None
        self.embeddings = None
        self.chain = None
        self.propal_chain = None
        self.pc = None
        self.index_name = None
        self.index = None
        self.vectorstore = None
        self.retriever = None
        self.pages = None

    def init_rag_system(self):
        self.load_models()
        logging.info("Models loaded..")
        self.init_vectorstore()
        return {"message": "Model and Vectorstore initialized"}

    def load_models(self):
        try:
            self.model, self.embeddings = load_model_and_embeddings(self.settings.model_name, self.settings.openai_api_key)
        except OpenAIError as e:
            raise HTTPException(status_code=401, detail=f"Unauthorized - Check OpenAI API Key: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")
        return None

    def init_vectorstore(self):
        self.index_name = "single-doc-index"
        try:
            if self.pc is None:
                self.pc = Pinecone(api_key=self.settings.pinecone_api_key)

            create_index_with_pinecone(self.pc, index_name=self.index_name, dimension=self.settings.embeddings_dim)
            self.index = self.pc.Index(self.index_name)

            while not self.pc.describe_index(self.index_name).status['ready']:
                logging.info('waiting for index to be ready..')

            self.vectorstore = create_vectorstore_pinecone(self.index_name, embeddings=self.embeddings)
            self.retriever = create_retriever_pinecone(self.vectorstore)
        except UnauthorizedException as e:
            raise HTTPException(status_code=401, detail=f"Unauthorized - Check Pinecone API Key: {str(e)}")
        
    def load_pdf(self, filepath):
        logging.info(f"Loading [{filepath}]")

        chunks = chunk_doc(filepath, chunk_size=self.settings.chunk_size, overlap=self.settings.chunk_overlap, metadata=None)

        logging.info(f"Vectorizing pdf..")
        self.vectorize_pdf(str(filepath.stem), chunks)
        return {"file_name": filepath.as_posix(), "num_chunks": len(chunks)}
    
    def load_pdf_inmem(self, pdf_filepath):
        logging.info(f"Loading [{pdf_filepath}]")
        self.pages = load_pdf(pdf_filepath)
        logging.info(f"Excerpt: {str(self.pages[0])[:200]}")
        
        logging.info(f"Vectorizing pdf..")
        self.vectorize_pdf()
        return {"file_name": pdf_filepath, "num_pages": len(self.pages)}

    def vectorize_pdf(self, title: str, chunks: List[str]):
        documents = [Document(page_content=d, metadata={"title": title}) for d in chunks]
        
        # reset Pinecone index
        create_index_with_pinecone(self.pc, index_name=self.index_name, dimension=self.settings.embeddings_dim)
        
        init_vector_count = self.index.describe_index_stats()['total_vector_count']
        self.vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])
        logging.info('Indexing doc..')
        while (self.index.describe_index_stats()['total_vector_count'] - init_vector_count) < len(documents):
            time.sleep(1)
        self.is_ready = True

        self.chain = create_qa_rag_chain(self.retriever, self.model)
        self.propal_chain = create_propal_rag_chain(self.retriever, self.model)

    def vectorize_pdf_inmem(self):
        self.chain = pdf_qa_pipeline_inmemory(self.pages, self.embeddings, self.model)
        self.is_ready = True
        return None
    
    def qa_doc(self, questions: str):
        if not self.is_ready:
            raise Exception("Model not ready - either no pdf or no model loaded")
        output = ""
        start = datetime.now()
        for question in questions.split("?"):
                if len(question) > 2:
                    question = question.strip(" ") + "?"
                    output += f"- **Question**: {question}\n"
                    output += f"- **Answer**: {self.chain.invoke({'question': question})}\n\n\n"
        end = datetime.now()
        seconds = int((end - start).total_seconds()) % 60
        all_minutes = int((end - start).total_seconds()) // 60
        hours = all_minutes // 60
        minutes = all_minutes % 60
        logging.info(f"Time elapsed: [{hours}:{minutes}:{seconds}]")
        return {"time_elapsed": f"{hours}:{minutes}:{seconds}", "result": output}

    def propal_extract(self):

        question_part_prompt = """
            Vous recevrez des extraits d'un document en lien avec une proposition de services, 
            et devrez extraire: 
                la société ou cabinet émetteur,
                la société ou cabinet destinataire,
                le type de prestations ou produits proposés,
                la description de ce que la mission ou l'offre propose,
                le montant de la proposition,
                les conditions de facturation,
                les conditions de règlement,
        """
            
        if not self.is_ready:
            raise Exception("Model not ready - either no pdf or no model loaded")
        start = datetime.now()
        output = self.propal_chain.invoke({"question": question_part_prompt})
        end = datetime.now()
        seconds = int((end - start).total_seconds()) % 60
        all_minutes = int((end - start).total_seconds()) // 60
        hours = all_minutes // 60
        minutes = all_minutes % 60
        logging.info(f"Time elapsed: [{hours}:{minutes}:{seconds}]")
        return {"time_elapsed": f"{hours}:{minutes}:{seconds}", "result": output}


    def ask_with_context(self, context: str, question: str):
        if self.model is None:
            logging.info("Model not loaded - waiting..")
            time.sleep(5)
            return {"result": "Model not loaded, retry"}
        prompt = question_answering_prompt() 
        model = self.model 
        parser = StrOutputParser()
        chain = prompt | model | parser
        response = chain.invoke({"context": context, "question": question})
        return {"result": response}

    def set_to_not_ready(self):
        self.is_ready = False
        return None

    def delete_pinecone_index(self, index_name):
        if self.pc is None:
            self.init_rag_system()
        try:
            delete_index_pinecone(self.pc, index_name)
        except UnauthorizedException as e:
            raise HTTPException(status_code=401, detail=f"Unauthorized - Check Pinecone API Key: {str(e)}")

    def are_api_keys_set(self):
        if (self.settings.openai_api_key is None) or (self.settings.pinecone_api_key is None):
            return False
        return True

    def update_openai_api_key(self, api_key):
        self.settings.openai_api_key = api_key
        try:
            self.load_models()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with OPENAI API Key: {str(e)}")
        return None

    def update_pinecone_api_key(self, api_key):
        self.settings.pinecone_api_key = api_key
        try:
            self.init_vectorstore()
        except UnauthorizedException as e:
            raise HTTPException(status_code=401, detail=f"Unauthorized - Check Pinecone API Key: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error with PINECONE API Key: {str(e)}")
        return None

    def classify_document(self, file_name: str):
        start = datetime.now()
        response = doc_classification_pipeline(file_name, self.model, self.embeddings)
        end = datetime.now()
        seconds = int((end - start).total_seconds()) % 60
        all_minutes = int((end - start).total_seconds()) // 60
        hours = all_minutes // 60
        minutes = all_minutes % 60
        logging.info(f"Time elapsed: [{hours}:{minutes}:{seconds}]")
        return {"time_elapsed": f"{hours}:{minutes}:{seconds}", "result": response}


# Define stateful objects
settings = Settings()
pdf_model = Model(settings)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logging.info("Application startup:")
    logging.info("Initializing model..")
    try:
        init_result = pdf_model.init_rag_system()
        logging.info(init_result)
    except Exception as e:
        # raise HTTPException(status_code=500, detail=f"Error initializing models: {str(e)}")
        logging.error(f"Error initializing models - Check API Keys: {str(e)}")
    yield
    # Shutdown event
    try:
        pdf_model.delete_pinecone_index('single_doc_index')
    except UnauthorizedException as e:
            raise HTTPException(status_code=401, detail=f"Unauthorized - Check your API Key: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting Pinecone index: {str(e)}")
    logging.info("Application shutdown..")



app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app"}

@app.get("/health")
async def health_check():
    if pdf_model.pc is None:
       return {"message": "Model has no pinecone client initialized. Check your API Key.",}
    try:
        # Check Pinecone
        index_status = pdf_model.pc.describe_index(pdf_model.index_name).status
        logging.info(f"Index status: {index_status}")
        # Check LLM
        model_response = pdf_model.is_ready
        logging.info(f"Model status: {model_response}")
        
        return {
            "status": "healthy",
            "index_status": "ok",
            "model_status": "Ready" if is_ready else "Not Ready",
        }
    except UnauthorizedException as e:
        raise HTTPException(status_code=401, detail=f"Unauthorized - Check your API Key: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/is_ready")
def is_ready():
    return {"message": pdf_model.is_ready}

@app.get("/status_summary")
def status():
    return {
        "message": f"Status summmary:\
            Model name [{pdf_model.model_name}]\
            Model ready [{pdf_model.is_ready}]"
        }

@app.get("/are_api_keys_set")
def are_api_keys_set():
    return {"message": pdf_model.are_api_keys_set()}
    # return {"message": f"{pdf_model.are_api_keys_set()} - {pdf_model.settings.openai_api_key} - {pdf_model.settings.pinecone_api_key}"}
    # return {"message": False}

@app.post("/update_openai_api_key")
def update_openai_api_key(api_key: str):
    os.environ['OPENAI_API_KEY'] = api_key
    pdf_model.update_openai_api_key(api_key)
    return {"message": f"Updated OpenAI API key"}

@app.post("/update_pinecone_api_key")
def update_pinecone_api_key(api_key: str):
    os.environ['PINECONE_API_KEY'] = api_key
    pdf_model.update_pinecone_api_key(api_key)
    return {"message": f"Updated Pinecone API key"}

@app.post("/set_to_not_ready")
def set_to_not_ready():
    pdf_model.set_to_not_ready()
    return {"message": f"Set model status to [{pdf_model.is_ready}]"}

@app.post("/load_pdf")
def load_pdf_file(pdf_filepath: str):
    try:
        pdf_model.load_pdf(Path(pdf_filepath))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading PDF: {str(e)}")
    return {"message": f"Loaded PDF [{pdf_filepath}]"}

@sleep_and_retry
@limits(calls=settings.max_calls_per_minute, period=settings.ratelimit_oneminute)
@app.post("/qa_doc")
def qa_doc(questions: str):
    try:
        result = pdf_model.qa_doc(questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating PDF chain: {str(e)}")
    return {"message": result["result"]}

@sleep_and_retry
@limits(calls=settings.max_calls_per_minute, period=settings.ratelimit_oneminute)
@app.post("/propal_extract")
def propal_extract():
    try:
        result = pdf_model.propal_extract()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating PDF chain: {str(e)}")
    return {"message": result["result"]}

@sleep_and_retry
@limits(calls=settings.max_calls_per_minute, period=settings.ratelimit_oneminute)
@app.post("/ask_with_context")
def ask_model_with_context(context: str, question: str):
    result = pdf_model.ask_with_context(context, question)
    return {"message": result["result"]}

@sleep_and_retry
@limits(calls=settings.max_calls_per_minute, period=settings.ratelimit_oneminute)
@app.post("/classify_document")
def classify_document(pdf_filepath: str):
    result = pdf_model.classify_document(pdf_filepath)
    return {"message": result["result"]}

@app.post("/delete_index")
def delete_pincecone_index(index_name: str):
    try:
        pdf_model.delete_pinecone_index(index_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting pinecone index: {str(e)}")
    return {"message": "ok"}



if __name__ == "__main__":
    uvicorn.run("app:app",host='0.0.0.0', port=4557, reload=True, workers=1)
