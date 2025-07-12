import json
import openai
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# CONFIG
OPENAI_API_KEY = "your-openai-api-key"
JSON_FILE = "marty_cleaned.json"
CHROMA_DIR = "marty_chroma_db"
COLLECTION_NAME = "marty-messages"
CHUNK_SIZE = 5  # Number of turns per chunk

openai.api_key = OPENAI_API_KEY

# Set up Chroma
chroma_client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
embedding_fn = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name="text-embedding-3-small")
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)

# Load messages
with open(JSON_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Group messages into chunks
chunks = []
chunk = []

for msg in data:
    chunk.append(f"{msg['sender']}: {msg['text']}")
    if len(chunk) >= CHUNK_SIZE:
        chunks.append("\n".join(chunk))
        chunk = []

# Add leftover messages
if chunk:
    chunks.append("\n".join(chunk))

# Store in ChromaDB
for i, text in enumerate(chunks):
    collection.add(documents=[text], ids=[f"msg-{i}"])

chroma_client.persist()
print(f"✅ Embedded {len(chunks)} chunks into ChromaDB")
