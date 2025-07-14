# Install Required PDF Libraries
!pip install PyMuPDF

#PDF Text Extraction using PyMuPDF
import fitz  # PyMuPDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

#Text Chunking and Embedding with SentenceTransformer
from sentence_transformers import SentenceTransformer
import numpy as np
model = SentenceTransformer("all-MiniLM-L6-v2")
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
def embed_chunks(chunks):
    return model.encode(chunks)

#Retrieve Most Relevant Chunk using Cosine Similarity 
from sklearn.metrics.pairwise import cosine_similarity
def get_relevant_chunk(question, chunks, chunk_embeddings):
    q_embedding = model.encode([question])
    similarities = cosine_similarity(q_embedding, chunk_embeddings)
    best_chunk_index = similarities.argmax()
    return chunks[best_chunk_index]
  
# Install LLaMA CPP Python Backend & Download Mistral Model using Git LFS
!pip install llama-cpp-python
!sudo apt install git-lfs 
!git lfs install
!git clone https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF

#Generate Answer Locally using Mistral LLaMA Mode
from llama_cpp import Llama
llm = Llama(model_path="/content/drive/MyDrive/mistral-7b-instruct-v0.1.Q4_0.gguf", n_ctx=2048)
def generate_answer_local(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    output = llm(prompt, max_tokens=256, stop=["\n"])
    return output["choices"][0]["text"]

#End-to-End PDF Question Answering Pipeline
pdf_text = extract_text_from_pdf("/content/Phishing Website Detection Using Machine Learning_Akshada Borhade.pdf")
chunks = chunk_text(pdf_text)
embeddings = embed_chunks(chunks)
def answer_question(question):
    context = get_relevant_chunk(question, chunks, embeddings)
    return generate_answer_local(question, context)

# Example
print(answer_question("What are dataset details?"))

# Gradio Interface
!pip install gradio
import gradio as gr
iface = gr.Interface(fn=answer_question,
                     inputs="text",
                     outputs="text",
                     title="PDF based Q&A System")
iface.launch()


