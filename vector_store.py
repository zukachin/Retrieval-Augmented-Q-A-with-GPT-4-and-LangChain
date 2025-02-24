from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()

def create_vector_store(docs, persist_directory="./db"):
    """Creates a vector store from documents and saves it."""
    embedding_model = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(docs, embedding=embedding_model, persist_directory=persist_directory)
    return vector_store

def load_vector_store(persist_directory="./db"):
    """Loads an existing vector store."""
    embedding_model = OpenAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
