import os
import pdfplumber
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdfs(pdf_folder):
    """Extract text from all PDFs in the specified folder."""
    all_texts = []
    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, file)
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
                all_texts.append({"text": text, "source": file})
    return all_texts

# Load financial statements from multiple PDFs
pdf_folder = "Datasets/"
documents = extract_text_from_pdfs(pdf_folder)

# Initialize text splitter and embedding model
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client and collection
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="financial_statements")

# Process and add text chunks to ChromaDB
for doc in documents:
    chunks = splitter.split_text(doc["text"])
    ids = [f"{doc['source']}_chunk_{i}" for i in range(len(chunks))]  # Unique IDs for chunks
    metadatas = [{"source": doc["source"]} for _ in chunks]
    
    # Generate embeddings
    embeddings = embedding_model.embed_documents(chunks)
    
    # Add to ChromaDB
    collection.add(
        ids=ids,
        documents=chunks,
        metadatas=metadatas,
        embeddings=embeddings  # Store precomputed embeddings
    )

print("✅ Data successfully added to ChromaDB!")
