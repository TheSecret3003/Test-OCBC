import os
import pdfplumber
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_folder):
    all_texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                all_texts.append({"text": text, "source": file})
    return all_texts

# Load financial statements from multiple PDFs
pdf_folder = "Datasets/"
documents = extract_text_from_pdfs(pdf_folder)

# Split and store text chunks with source metadata
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

chroma_client = chromadb.PersistentClient(path="chroma_db")
vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

# Add text chunks to ChromaDB
for doc in documents:
    chunks = splitter.split_text(doc["text"])
    metadata = [{"source": doc["source"]} for _ in chunks]
    vector_store.add_texts(chunks, metadatas=metadata)

vector_store.persist()