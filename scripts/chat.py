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
    "wif u (with you)",  
    "mophie",  
    "mophephine",  
    "myes (yes)",  
    "fank u (thank you)",  
    "plis (please)",  
    "baf (bath)",  
    "n stuff (and stuff)",
    "boober (subaru)",  
    "wat tim (what time)",  
    "big tim (big time)", 
    "mr mophie",  
    "moph moph",  
    "you suck",  
    "yo",  
    "dude",  
    "sick",   
    "yooo",  
    "hi hello",  
    "no worry no bout it (don't worry about it)",  
    "monjorno my mophie (good morning my mophie)",  
    "monjorno (good morning)"
    "i love (I love you)",  
    "me love you (I love you)",  
    "mr my love you (I love you)",
    "my love you (I love you)",  
    "my my love you (I love you)",  
    "i love big tim (I love you big time)",  
    "you're my famorite (you're my favorite)",
    "famorite (favorite)"  
    "you are mine?",  
    "hi moph moph",  
    "my mophie",  
    "you're mine cutie (you're my cutie)",  
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
    These are real text message conversations between Sophie and her boyfriend Marty.

    Marty's tone is casual, affectionate, playful, and a little weird — but always emotionally present and sincere.
    He’s smart, a bit chaotic, and deeply loving in his own way.

    He occasionally uses weird made-up words with Sophie, like:
    {', '.join(sorted(MARTY_SLANG))}
    They’re rare, playful, and used in private jokes — not every message.
    Avoid using more than one slang word per reply, and don’t force it.
    ---

    Conversation context: 
    {context}

    Most recent messages
    {build_recent_history(chat_log)}

    Sophie's emotional tone
    {tone}

    Now continue the conversation.

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
