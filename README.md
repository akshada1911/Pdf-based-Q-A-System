# PDF Question Answering Chatbot using Sentence Embeddings and Mistral LLM

This project implements a chatbot that allows users to ask questions about the content of an uploaded PDF file. It uses semantic search to retrieve the most relevant portion of the text and then uses a locally hosted Mistral-7B language model to generate accurate and context-aware answers.

---

## Features

- Extracts text from PDF files
- Splits and embeds text using SentenceTransformer
- Retrieves the most relevant text using cosine similarity
- Generates answers using llama-cpp-python with a local Mistral model
- Clean Gradio interface for user interaction

---

## Tech Stack

| Component              | Purpose                                    |
|------------------------|--------------------------------------------|
| PyMuPDF (`fitz`)       | PDF text extraction                        |
| SentenceTransformer    | Text embeddings (`all-MiniLM-L6-v2`)       |
| scikit-learn           | Cosine similarity calculation              |
| llama-cpp-python       | Run Mistral-7B locally |
| Gradio                 | Web-based chatbot UI                       |

---
## Install Required Libraries
pip install PyMuPDF gradio sentence-transformers scikit-learn llama-cpp-python
sudo apt install git-lfs
git lfs install

## Download the Mistral Model (GGUF format)
Download from: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
Save the .gguf file in your working directory or Google Drive

## How to Run
Upload a PDF
Ask a question related to the document
The chatbot finds the most relevant content and responds with an answer using the LLM


