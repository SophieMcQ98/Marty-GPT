import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API Key loaded:", api_key[:8] + "..." if api_key else "Not found")