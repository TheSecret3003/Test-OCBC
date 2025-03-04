import sys

# Force Python to use pysqlite3 instead of system sqlite3
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    print("pysqlite3 not found. Ensure it's installed in requirements.txt")

import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings

# Create a text-generation pipeline
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="chroma_db")

# Get or create a collection
collection = client.get_or_create_collection(name="financial_statements")  # Change name if needed

def retrieve_relevant_chunks(query):
    # Perform similarity search
    results = collection.query(
        query_texts=[query],
        n_results=3
    )

    sources = []
    for i in range(len(results["documents"][0])):
        sources.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],  # Ensure metadata contains "source"
            "score": results["distances"][0][i]
        })
    
    return sources


def generate_response(query):
    sources = retrieve_relevant_chunks(query)

    # Build context from sources
    context = "\n\n".join([f"Source ({src['source']}): {src['text']}" for src in sources])

    # Create prompt
    prompt = f"User Query: {query}\n\nAnswer:"

    response = pipe(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]

    return response, sources
