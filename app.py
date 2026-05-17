from dotenv import load_dotenv
load_dotenv()

import os
import shutil
from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the context below.
If the answer is not in the context, say "I couldn't find that in the document."

Previous conversation:
{history}

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chains = {}
vectorstores={}
chat_histories={}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF files allowed"}), 400

    pdf_id = file.filename.replace(".pdf", "").replace(" ", "_")
    pdf_path = f"./uploads/{pdf_id}.pdf"
    db_path = f"./chroma_dbs/{pdf_id}"

    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./chroma_dbs", exist_ok=True)
    file.save(pdf_path)

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    vectorstores[pdf_id] = vectorstore
    chains[pdf_id] = chain
    chat_histories[pdf_id] = []

    return jsonify({"message": "PDF uploaded successfully", "pdf_id": pdf_id})


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    pdf_id = data.get("pdf_id")
    question = data.get("question")

    if not pdf_id or not question:
        return jsonify({"error": "pdf_id and question are required"}), 400

    if pdf_id not in chains:
        return jsonify({"error": "PDF not found, please upload first"}), 404

    # Get source documents
    retriever = vectorstores[pdf_id].as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    # Format chat history
    history = chat_histories[pdf_id]
    history_text = "\n".join([
        f"Human: {h['question']}\nAssistant: {h['answer']}"
        for h in history[-3:]  # last 3 exchanges only
    ])

    # Build chain with history
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "history": lambda _: history_text
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(question)

    # Save to history
    chat_histories[pdf_id].append({
        "question": question,
        "answer": result
    })

    # Extract page numbers
    pages = list(set([
        doc.metadata.get("page", "unknown") + 1
        for doc in docs
    ]))
    pages.sort()
    source = f"Page {', '.join(str(p) for p in pages)}"

    return jsonify({"answer": result, "source": source})

if __name__ == "__main__":
    app.run(debug=False)