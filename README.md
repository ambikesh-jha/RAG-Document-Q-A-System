# Talk-to-Doc

# RAG Document Q&A System

A powerful Retrieval-Augmented Generation (RAG) system that enables intelligent question answering from uploaded documents. Upload PDFs or Word files and ask questions - the system intelligently retrieves relevant information from your documents and combines it with local LLM capabilities.

## Features
- Upload PDF and Word documents
- Semantic search using FAISS vector database
- Local LLM integration with Llama 3.2 via Ollama
- Smart chunking and embedding generation
- Real-time document indexing and querying
- Fallback to LLM when no relevant document content found

## Tech Stack
- **Backend**: Flask (Python)
- **Vector Search**: FAISS
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **LLM**: Llama 3.2 1B (Ollama)
- **Document Processing**: pdfplumber, python-docx

## Installation
steps :
1. Clone the repository
   git clone https://github.com/yourusername/rag-document-qa.git
   cd rag-document-qa
2. make the virtual envirnment
3. install the required requirements : 
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

