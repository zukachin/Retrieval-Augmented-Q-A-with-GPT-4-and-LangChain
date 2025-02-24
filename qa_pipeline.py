from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_response(query, vector_store):
    """Runs retrieval-augmented generation (RAG) to answer a query."""
    llm = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
    
    response = qa_chain.run(query)
    return response
