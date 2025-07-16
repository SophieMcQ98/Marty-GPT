import os
import json
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# === CONFIG ===
CHUNKS_FILE = "../data/chunks.jsonl"
ARGUMENTS_FILE = "../data/arguments.json"  # NEW
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

# Load chat history
with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        doc = Document(page_content=obj["text"], metadata={"chunk_id": obj["chunk_id"], "source": "chat"})
        docs.append(doc)

# Load argument memories
with open(ARGUMENTS_FILE, "r", encoding="utf-8") as f:
    arguments = json.load(f)
    print(f)
    # Add each argument 2–3 times to weight them more
    for i, arg in enumerate(arguments):
        for j in range(10):  # Inject each 3x for stronger presence
            doc = Document(
                page_content=arg["text"],
                metadata={"chunk_id": f"argument-{i}-{j}", "source": "argument"}
            )
            docs.append(doc)
print(docs)
# === ADD TO VECTORSTORE ===
db.add_documents(docs)
db.persist()

print(f"{len(docs)} total documents embedded and stored in ChromaDB.")