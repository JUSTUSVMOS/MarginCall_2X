import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_KEY)

print("🔍 正在連線 Google 總部，盤點你目前可以用的模型清單...")
print("-" * 50)

try:
    # 暴力列出所有模型對象
    models = client.models.list()
    for m in models:
        # 直接印出 name 屬性即可
        print(f"模型名稱: {m.name}")
except Exception as e:
    print(f"讀取失敗，錯誤訊息: {e}")

print("-" * 50)
print("請從上面清單挑選正確的名稱 (例如: gemini-1.5-flash-8b 等) 填回 main.py")