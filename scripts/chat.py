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

#===== VARS =====
MARTY_SLANG = set([
    "coolios",
    "coolz",  
    "delimsh",  
    "wif u",  
    "mophie",  
    "mophephine",  
    "myes",  
    "fank u",  
    "plis",  
    "baf",  
    "boober",  
    "wat tim",  
    "big tim", 
    "mr mophie",  
    "moph moph",  
    "you suck",  
    "yo",  
    "dude",  
    "shot",  
    "sick",  
    "legit",  
    "yooo",  
    "hi hello",  
    "no worry no bout it",  
    "monjorno my mophie",  
    "i love",  
    "me love you",  
    "mr my love you",
    "my love you",  
    "my my love you",  
    "i love big tim",  
    "you're my famorite",  
    "you are mine?",  
    "hi moph moph",  
    "my mophie",  
    "you're mine cutie",  
    "i miss you n stuff" 
])

chat_log = []

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

def get_tone_instruction_gpt(message: str) -> str:
    gpt_prompt = f"""
    You're analyzing a message from Sophie to her boyfriend, Marty.
    Determine the emotional tone.

    Message: "{message}"

    Respond with only one of the following:
    - upset
    - affectionate
    - playful
    - neutral
    - tense
    - other
    """
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")
    response = llm([HumanMessage(content=gpt_prompt)])
    tone = response.content.strip().lower()

    instructions = {
        "upset": "Sophie sounds upset. Marty should respond gently and supportively.",
        "affectionate": "Sophie sounds affectionate. Marty should respond with love and softness.",
        "playful": "Sophie is being playful. Marty should be cheeky and teasing.",
        "neutral": "Sophie is being neutral. Marty should respond casually.",
        "tense": "Sophie sounds annoyed or argumentative. Marty should stay calm, defuse tension, and not escalate.",
        "other": "Marty should respond in his usual tone — casual, affectionate, and a little weird."
    }

    return instructions.get(tone, instructions["other"])

# === UTILIZE RECENT MESSAGES ===
def build_recent_history(chat_log):
    return "\n".join([f"{msg['sender']}: {msg['text']}" for msg in chat_log])

# === CHAT FUNCTION ===
def ask_marty(prompt):
    relevant = retrieve_relevant_chunks(prompt)
    context = "\n\n".join(relevant)
    tone = get_tone_instruction_gpt(user_input)

    full_prompt = f"""
    The following are real text message conversations between Sophie and her boyfriend Marty. 
    Respond in Marty's voice and style, continuing the conversation naturally.

    Respond in Marty’s voice and tone — casual, affectionate, and a little weird. 
    He often says things like those in the following list:
    {', '.join(sorted(MARTY_SLANG))}
    Don’t overdo it — just sound like *him*.

    {context}

    These are the most recent messages between you (Marty) and Sophie for conversation context
    {build_recent_history(chat_log)}

    {tone}

    Sophie: {user_input}
    Marty:"""
    chat_log.append({"sender": "Sophie", "text": user_input})

    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")
    response = llm([HumanMessage(content=full_prompt)])
    chat_log.append({"sender": "Marty", "text": response.content})
    return response.content

# === RUN INTERACTIVELY ===
if __name__ == "__main__":
    print("Ask Marty anything. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = ask_marty(user_input)
        print(f"Marty: {reply}\n")
