import os
from google import genai
from dotenv import load_dotenv
import json

load_dotenv()

def list_all_models():
    print("🔍 正在查詢當前 API Key 可用的模型列表...")
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    try:
        # 使用 SDK 的 list 方法
        models = client.models.list()
        for m in models:
            # 嘗試印出物件的所有屬性以確定正確欄位
            # 通常名稱在 m.name
            methods = getattr(m, 'supported_methods', 'Unknown')
            print(f"Name: {m.name} | Methods: {methods}")
    except Exception as e:
        print(f"❌ 查詢失敗: {e}")

if __name__ == "__main__":
    list_all_models()
