import chromadb

# Initialize ChromaDB client
client = chromadb.PersistentClient(path="./chroma_db")

# List current collections
collections = client.list_collections()
print("Existing Collections:", collections)

# Try creating a new collection
new_collection = client.get_or_create_collection(name="test_collection")
print("Created Collection:", new_collection.name)

# List collections again
collections = client.list_collections()
print("Updated Collections:", collections)

import chromadb

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="chroma_db")

# Get list of existing collections
collections = chroma_client.list_collections()
print("Existing Collections:", collections)  # Returns just a list of names

# Delete old 'langchain' collection if it exists
if "langchain" in collections:
    chroma_client.delete_collection(name="langchain")
    print("Deleted 'langchain' collection.")

# Verify updated collections
collections = chroma_client.list_collections()
print("Updated Collections:", collections)

