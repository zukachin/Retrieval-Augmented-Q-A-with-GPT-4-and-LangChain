import gradio as gr
import os
from document_loader import load_and_chunk_pdf
from vector_store import create_vector_store, load_vector_store
from qa_pipeline import get_response

# Ensure the data folder exists
os.makedirs("data", exist_ok=True)

vector_store = None

def process_document(file_path):
    """Loads the document, processes it, and creates vector embeddings."""
    global vector_store
    
    if not os.path.exists(file_path):
        return "File not found. Please upload a valid PDF."
    
    docs = load_and_chunk_pdf(file_path)
    vector_store = create_vector_store(docs)
    
    return "Document processed! You can now ask questions."

def ask_question(query):
    """Retrieves answers based on uploaded document."""
    if vector_store is None:
        return "Please upload and process a document first!"
    
    return get_response(query, vector_store)

# Define Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("#RAG-Based Document Q&A")
    
    with gr.Row():
        file_input = gr.File(label="Upload PDF", type="filepath")
        process_button = gr.Button("Process Document")
    
    status_output = gr.Textbox(label="Status", interactive=False)
    
    process_button.click(process_document, inputs=file_input, outputs=status_output)
    
    question_input = gr.Textbox(label="Ask a question")
    answer_output = gr.Textbox(label="Answer", interactive=False)
    ask_button = gr.Button("Get Answer")
    
    ask_button.click(ask_question, inputs=question_input, outputs=answer_output)

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()
