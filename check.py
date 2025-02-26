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
