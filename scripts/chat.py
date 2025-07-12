import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# === SETUP ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = Chroma(
    collection_name="marty-messages",
    embedding_function=embedding,
    persist_directory="../data/chroma_db"
)

# === FUNCTION TO GET RELEVANT CHUNKS ===
def retrieve_relevant_chunks(query, k=5):
    results = db.similarity_search(query, k=k)
    return [doc.page_content for doc in results]

# === CHAT FUNCTION ===
def ask_marty(prompt):
    relevant = retrieve_relevant_chunks(prompt)
    context = "\n\n".join(relevant)

    full_prompt = f"""You are Marty, responding in his voice using real messages.
Use the following past texts for context:

{context}

Marty, answer this: {prompt}
"""

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")
    response = llm([HumanMessage(content=full_prompt)])
    return response.content

# === RUN INTERACTIVELY ===
if __name__ == "__main__":
    print("💬 Ask Marty anything. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = ask_marty(user_input)
        print(f"Marty: {reply}\n")
