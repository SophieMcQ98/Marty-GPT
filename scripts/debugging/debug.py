from chromadb import PersistentClient

chroma_client = PersistentClient(path="../data/chroma_db")
collection = chroma_client.get_or_create_collection(name="marty-messages")
print("Number of documents:", len(collection.get()['documents']))