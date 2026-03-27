from google import genai
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))
chat = client.chats.create(model='gemini-2.0-flash')
h = chat.get_history()
print(f"get_history return type: {type(h)}")
print(f"get_history content: {h}")
