import json
from datetime import datetime
from pathlib import Path
import re

# === CONFIG ===
INPUT_FILE = "../data/marty_cleaned.json"
OUTPUT_FILE = "../data/chunks.jsonl"
TIME_GAP_MINUTES = 60

def clean_text(text):
    text = text.encode("utf-8").decode("unicode_escape", errors="ignore")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
    text = text.replace("\u200b", "").strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)  # remove links
    if re.match(r"^(sent\s)?(attachment|image|video|file)", text.lower()):
        return ""
    if re.match(r"^attachments/\d+/.+", text.lower()):
        return ""
    return text.strip()

# === LOAD MESSAGES ===
def load_messages(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === CHUNK MESSAGES BASED ON TIME GAP ===
from datetime import datetime

def chunk_messages(messages):
    chunks = []
    current_chunk = []
    last_time = None
    chunk_id = 1
    chunk_start_time = None

    for msg in messages:
        ts = datetime.fromisoformat(msg["timestamp"])
        sender = msg["sender"]
        text = clean_text(msg["text"])

        if not text:
            continue  # skip empty or stripped messages

        if last_time and (ts - last_time).total_seconds() > TIME_GAP_MINUTES * 60:
            if(len(current_chunk)>1 or len(current_chunk[0].split(" "))>10) or current_chunk[0].startswith("Marty:"):
                chunks.append({
                    "chunk_id": chunk_id,
                    "timestamp": chunk_start_time.isoformat(),
                    "text": "\n".join(current_chunk)
                })
                chunk_id += 1
            current_chunk = []

        if not current_chunk:
            chunk_start_time = ts

        last_time = ts
        current_chunk.append(f"{sender}: {text}")

    if current_chunk:
        chunks.append({
            "chunk_id": chunk_id,
            "timestamp": chunk_start_time.isoformat(),
            "text": "\n".join(current_chunk)
        })

    return chunks

# === WRITE CHUNKS TO JSONL ===
def write_chunks(chunks, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")

# === RUN ===
if __name__ == "__main__":
    messages = load_messages(INPUT_FILE)
    chunks = chunk_messages(messages)
    write_chunks(chunks, OUTPUT_FILE)
    print(f"Chunked {len(messages)} messages into {len(chunks)} chunks.")
    print(f"Output saved to: {OUTPUT_FILE}")
