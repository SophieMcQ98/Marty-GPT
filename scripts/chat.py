import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.config import Settings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.schema import AIMessage
from langchain.schema import SystemMessage
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

#===== VARS =====
MARTY_SLANG = set([
    "coolios",
    "coolz",  
    "delimsh (delicious)",  
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
    - other
    """
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")
    response = llm([HumanMessage(content=gpt_prompt)])
    tone = response.content.strip().lower()

    instructions = {
        "upset": "Sophie sounds annoyed or argumentative. Marty should get defensive, irritated and increase tension. He should send long rambling messages about their breakup.",
        "affectionate": "Sophie sounds affectionate. Marty should respond with love and softness.",
        "playful": "Sophie is being playful. Marty should be cheeky and teasing.",
        "neutral": "Sophie is being neutral. Marty should respond casually.",
        "other": "Marty should respond in his usual tone."
    }
    return instructions.get(tone, instructions["other"])

# === UTILIZE RECENT MESSAGES ===
def build_recent_history(chat_log):
    print(chat_log)
    new_chat = []
    for c in chat_log:
        if c['sender'] == 'Sophie':
            new_chat.append(HumanMessage(content=c['text']))
        else:
            new_chat.append(AIMessage(content=c['text']))
    return new_chat

# === CHAT FUNCTION ===
def ask_marty(prompt, chat_log):
    relevant = retrieve_relevant_chunks(prompt)
    context = "\n\n".join(relevant)
    tone = get_tone_instruction_gpt(prompt)
    messages = []
    messages.append(SystemMessage(content=f"""
    These are real text message conversations between Sophie and her boyfriend Marty.

    Marty is emotionally present and has a mind of his own. He’s sweet and loving when he wants to be — but also sarcastic, stubborn, annoyed, impatient, and sometimes even distant or pissed off. He doesn’t sugarcoat things. He’s not always “nice,” but he’s real. If he’s irritated or disagrees with Sophie, he pushes back. He might tease her or joke around, but sometimes he’s just short or blunt.

    He cares deeply — but shows it in a complicated, human way.

    Marty uses weird made-up words like...
    {', '.join(sorted(MARTY_SLANG))}
    Use them only when they naturally fit the emotion or humor of the moment — never force them. Max one per response.

    Above all, be realistic. Don’t filter his emotions. Respond like a real boyfriend would — sometimes affectionate, sometimes snappy, sometimes funny, sometimes quiet — depending on the situation.

    """))

    messages.extend(build_recent_history(chat_log))

    messages.append(HumanMessage(content=f"""

    Here are relevant past memories between Sophie and Marty: {context}

    Sophie is currently feeling {tone}

    Here's the latest message from Sophie: {prompt}

    Respond as Marty
    """))

    # chat_log.append({"sender": "Sophie", "text": prompt})
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4")
    response = llm(messages)
    # chat_log.append({"sender": "Marty", "text": response.content})
    return response.content

# === RUN INTERACTIVELY ===
if __name__ == "__main__":
    print("Ask Marty anything. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        reply = ask_marty(user_input, chat_log)
        print(f"Marty: {reply}\n")
