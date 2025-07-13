import os
import json
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# === CONFIG ===
CHUNKS_FILE = "../data/chunks.jsonl"
CHROMA_PATH = "../data/chroma_db"

# === LOAD ENV ===
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# === INIT EMBEDDING + CHROMA ===
embedding_fn = OpenAIEmbeddings(model="text-embedding-3-small")

# Clean start (delete old DB if needed)
if os.path.exists(CHROMA_PATH):
    import shutil
    shutil.rmtree(CHROMA_PATH)

db = Chroma(
    collection_name="marty-messages",
    embedding_function=embedding_fn,
    persist_directory=CHROMA_PATH
)

# === LOAD CHUNKS ===
docs = []
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        doc = Document(page_content=obj["text"], metadata={"chunk_id": obj["chunk_id"]})
        docs.append(doc)

# === ADD TO VECTORSTORE ===
db.add_documents(docs)
db.persist()

print(f"{len(docs)} chunks embedded and stored in ChromaDB.")
