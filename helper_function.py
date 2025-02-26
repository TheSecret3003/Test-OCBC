import chromadb
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Create a text-generation pipeline
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_relevant_chunks(query):
    vector_store = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    docs = vector_store.similarity_search_with_relevance_scores(query, k=3)

    sources = []
    for doc, score in docs:
        sources.append({"text": doc.page_content, "source": doc.metadata["source"], "score": score})
    
    return sources


def generate_response(query):
    sources = retrieve_relevant_chunks(query)
    
    # Build context from sources
    context = "\n\n".join([f"Source ({src['source']}): {src['text']}" for src in sources])
    
    # Create prompt
    prompt = f"\n\nUser Query: {query}\n\nAnswer:"
    
    response = pipe(prompt, max_new_tokens=200, do_sample=True)[0]["generated_text"]
    
    return response, sources