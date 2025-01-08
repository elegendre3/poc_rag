from datetime import datetime
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
import uvicorn 

from source.rag import (
    load_pdf,
    load_model_and_embeddings,
    pdf_question_answering_pipeline,
)


class Model:
    def __init__(self):
        self.is_ready = False
        self.model = None
        self.model_name = None
        self.embeddings = None
        self.pages = None
        self.chain = None

    def load_model_and_embeddings(self):
        self.model_name = "llama3.2"
        self.model, self.embeddings = load_model_and_embeddings(self.model_name)
        return None

    def load_pdf(self, pdf_filepath):
        logging.info(f"Loading [{pdf_filepath}]")
        self.pages = load_pdf(pdf_filepath)
        logging.info(f"Excerpt: {str(self.pages[0])[:200]}")
        
        logging.info(f"Vectorizing pdf..")
        self.vectorize_pdf()
        return {"file_name": pdf_filepath, "num_pages": len(self.pages)}

    def vectorize_pdf(self):
        self.chain = pdf_question_answering_pipeline(self.pages, self.embeddings, self.model)
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


pdf_model = Model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup event
    logging.info("Application startup:")
    logging.info("Initializing model..")
    try:
        pdf_model.load_model_and_embeddings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error initializing model: {str(e)}")
    yield
    # Shutdown event
    logging.info("Application shutdown: Perform cleanup tasks here")

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
