from datetime import datetime
from contextlib import asynccontextmanager
import logging
import os
import time
from typing import List
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_core.documents import Document
from pinecone.grpc import PineconeGRPC as Pinecone
import uvicorn 

from source.rag import (
    load_pdf,
    load_model_and_embeddings,
    pdf_qa_pipeline_inmemory,
    create_rag_chain,
)
from source.text import (
    chunk_pdf,
    load_pdf,
)
from source.vector_store import (
    create_index_with_pinecone,
    create_vectorstore_pinecone,
    create_retriever_pinecone,
    delete_index_pinecone,
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


class Model:
    def __init__(self):
        self.is_ready = False
        self.model = None
        self.model_name = None
        self.embeddings = None
        self.chain = None
        self.pc = None
        self.index_name = None
        self.index = None
        self.embeddings_dim = 1536
        self.vectorstore = None
        self.retriever = None
        self.pages = None

    def init_rag_system(self):
        self.load_model_and_embeddings()
        self.init_vectorstore()

    def load_model_and_embeddings(self):
        self.model_name = "llama3.2"
        self.model, self.embeddings = load_model_and_embeddings(self.model_name)
        return None

    def init_vectorstore(self):
        self.index_name = "example-index"
    
        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        create_index_with_pinecone(self.pc, index_name=self.index_name, dimension=self.embeddings_dim)
        self.index = self.pc.Index(self.index_name)

        while not self.pc.describe_index(self.index_name).status['ready']:
            logging.info('waiting for index to be ready..')

        self.vectorstore = create_vectorstore_pinecone(self.index_name, embeddings=self.embeddings)
        self.retriever = create_retriever_pinecone(self.vectorstore)

    def load_pdf(self, pdf_filepath):
        logging.info(f"Loading [{pdf_filepath}]")
        chunks = chunk_pdf(pdf_filepath)
        logging.info(f"Excerpt: {str(chunks[0])[:200]}")
        
        logging.info(f"Vectorizing pdf..")
        self.vectorize_pdf(str(pdf_filepath), chunks)
        return {"file_name": pdf_filepath, "num_chunks": len(chunks)}
    
    def load_pdf_inmem(self, pdf_filepath):
        logging.info(f"Loading [{pdf_filepath}]")
        self.pages = load_pdf(pdf_filepath)
        logging.info(f"Excerpt: {str(self.pages[0])[:200]}")
        
        logging.info(f"Vectorizing pdf..")
        self.vectorize_pdf()
        return {"file_name": pdf_filepath, "num_pages": len(self.pages)}

    def vectorize_pdf(self, title: str, chunks: List[str]):
        documents = [Document(page_content=d, metadata={"title": title}) for d in chunks]
        init_vector_count = self.index.describe_index_stats()['total_vector_count']
        self.vectorstore.add_documents(documents=documents, ids=[str(uuid.uuid4()) for _ in range(len(documents))])
        logging.info('Indexing doc..')
        while (self.index.describe_index_stats()['total_vector_count'] - init_vector_count) < len(documents):
            time.sleep(1)
        self.is_ready = True
        self.chain = create_rag_chain(self.retriever, self.model)

    def vectorize_pdf_inmem(self):
        self.chain = pdf_qa_pipeline_inmemory(self.pages, self.embeddings, self.model)
        self.is_ready = True
        return None
    
    def ask(self, questions: str):
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

    def set_to_not_ready(self):
        self.is_ready = False
        return None

    def delete_pinecone_index(self):
        delete_index_pinecone(self.pc, self.index_name)


pdf_model = Model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logging.info("Application startup:")
    logging.info("Initializing model..")
    try:
        pdf_model.init_rag_system()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing models: {str(e)}")
    yield
    # Shutdown event
    logging.info("Application shutdown:")
    try:
        pdf_model.delete_pinecone_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting Pinecone index: {str(e)}")



app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app"}

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

@app.post("/set_to_not_ready")
def set_to_not_ready():
    pdf_model.set_to_not_ready()
    return {"message": f"Set model status to [{pdf_model.is_ready}]"}

@app.post("/load_pdf")
def load_pdf_file(pdf_filepath: str):
    pdf_filepath = "data/pdfs/CDI Eliott LEGENDRE.pdf"
    try:
        pdf_model.load_pdf(pdf_filepath)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading PDF: {str(e)}")
    return {"message": f"Loaded PDF [{pdf_filepath}]"}

@app.post("/ask")
def ask_model(questions: str):
    try:
        result = pdf_model.ask(questions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating PDF chain: {str(e)}")
    return {"message": result["result"]}


if __name__ == "__main__":
    uvicorn.run("app:app",host='0.0.0.0', port=4557, reload=True, workers=1)
