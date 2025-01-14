from operator import itemgetter
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings

# MODEL = "gpt-3.5-turbo"
# MODEL = "gpt-4o-mini"
# MODEL = "text-embedding-3-small"
# MODEL = "mixtral:8x7b"
MODEL = "llama3.2"



def load_model_and_embeddings(model_name: str, openai_api_key: str = None):
    """
    Usage: model.invoke("Tell me a joke"))
    """
    if model_name.startswith("gpt"):
        if openai_api_key is not None:
            model = ChatOpenAI(openai_api_key=openai_api_key, model=model_name)
            embeddings = OpenAIEmbeddings()
        else:
            raise ValueError(f"To use GPT models like [{model_name}] you need to pass an OPENAI_API_KEY")
    else:
        model = Ollama(model=model_name)
        # embeddings = OllamaEmbeddings(model=model_name)
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return model, embeddings

