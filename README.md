# RAGChat — Document Intelligence

A RAG (Retrieval Augmented Generation) chatbot that lets you upload any PDF and ask questions about it. Built with LangChain, ChromaDB, Groq, and Flask.

## Features
- Upload any PDF document
- Ask questions in natural language
- Answers grounded in your document with source page numbers
- Chat history within a session
- Clean purple-themed UI

## Tech Stack
- **LLM:** LLaMA 3.1 via Groq API
- **Embeddings:** HuggingFace all-MiniLM-L6-v2
- **Vector Store:** ChromaDB
- **Framework:** LangChain
- **Backend:** Flask
- **Frontend:** Vanilla HTML/CSS/JS

## Setup

1. Clone the repo
```bash
   git clone https://github.com/YOURUSERNAME/rag-chat.git
   cd rag-chat
```

2. Create virtual environment
```bash
   python -m venv .venv
   .venv\Scripts\activate
```

3. Install dependencies
```bash
   pip install -r requirements.txt
```

4. Add your API keys in a `.env` file

5. Run the app
```bash
   python app.py
```

6. Open `http://localhost:5000`

## How it works
1. PDF is loaded and split into chunks
2. Chunks are embedded using HuggingFace and stored in ChromaDB
3. On each question, relevant chunks are retrieved and passed to LLaMA 3.1
4. LLaMA answers based only on the retrieved context