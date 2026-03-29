import os
import sys
import random
import time
import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types

# 修正 Windows Unicode 問題
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 引入自定義模組
import fubon
from config import WDT_MESSAGES, system_prompt
import engine_portfolio as portfolio
import engine_market as market
import engine_risk as risk

load_dotenv()

# --- 初始化區 ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if not BOT_TOKEN or not GEMINI_KEY:
    raise ValueError("❌ .env 缺少必要 API KEY")

fubon.init_fubon()
bot = telebot.TeleBot(BOT_TOKEN)
client = genai.Client(api_key=GEMINI_KEY)

# AI 模型清單
AVAILABLE_MODELS = [
    'gemini-3.1-pro-preview',
    'gemini-3.1-flash-lite-preview',
    'gemini-2.0-flash-lite',
    'gemini-flash-latest'
]

def get_dynamic_models():
    models = AVAILABLE_MODELS.copy()
    if market.is_us_market_open() or market.is_tw_market_open():
        models.insert(0, 'gemini-3.1-pro-preview')
    return list(dict.fromkeys(models)) # 去重

def create_agent_chat(model_name, history=None):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[
                portfolio.update_position, portfolio.get_portfolio_raw_data, 
                portfolio.calculate_pnl, portfolio.get_exchange_rate,
                market.get_live_price, market.get_us_realtime_insight,
                market.resolve_symbol_identity, # 【新增】標的身分識別
                market.get_market_sentiment, market.get_stock_news,
                market.get_fundamental_data, market.get_market_history,
                market.get_technical_analysis, # 新增戰略分析工具
                fubon.get_market_hot_stocks, fubon.get_intraday_trend,
                fubon.get_market_trades, fubon.get_price_volumes,
                fubon.get_quote_and_orderbook, fubon.get_historical_stats, 
                fubon.get_txo_sentiment, # 【新增】TXO 戰報工具
                risk.get_global_risk_radar
            ],
            temperature=0.3, 
        ),
        history=history
    )

# 初始化大腦
chat = create_agent_chat(get_dynamic_models()[0])
dead_engines = {}

@bot.message_handler(commands=['reset'])
def reset_memory(message):
    global chat
    chat = create_agent_chat(get_dynamic_models()[0])
    bot.reply_to(message, "🧹 推進器記憶體已排空！大腦已重新裝填。")

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    global chat
    user_text = message.text
    
    # --- 動態時間注入系統提示 ---
    import pytz
    from datetime import datetime
    tw_tz = pytz.timezone('Asia/Taipei')
    us_tz = pytz.timezone('US/Eastern')
    now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
    now_us = datetime.now(us_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    time_context = f"\n【🕒 當前時間環境】\n- 台北時間: {now_tw}\n- 美東時間: {now_us}\n"
    dynamic_prompt = system_prompt + time_context
    
    # 垃圾話表情
    mood = "bad_market" if any(w in user_text for w in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot.send_chat_action(message.chat.id, 'typing')

    # 引擎冷卻管理
    now = time.time()
    current_models = [m for m in get_dynamic_models() if dead_engines.get(m, 0) < now]

    safe_history = chat.get_history() if hasattr(chat, 'get_history') else getattr(chat, 'history', None)

    for model_name in current_models:
        try:
            # 💡 每次對話都注入新的時間 Context 並重建 Chat
            chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_prompt,
                    tools=[
                        portfolio.update_position, portfolio.get_portfolio_raw_data, 
                        portfolio.calculate_pnl, portfolio.get_exchange_rate,
                        market.get_live_price, market.get_us_realtime_insight,
                        market.resolve_symbol_identity, 
                        market.get_market_sentiment, market.get_stock_news,
                        market.get_fundamental_data, market.get_market_history,
                        market.get_technical_analysis, 
                        fubon.get_market_hot_stocks, fubon.get_intraday_trend,
                        fubon.get_market_trades, fubon.get_price_volumes,
                        fubon.get_quote_and_orderbook, fubon.get_historical_stats, 
                        fubon.get_txo_sentiment, # 【新增】TXO 戰報工具
                        risk.get_global_risk_radar
                    ],
                    temperature=0.3, 
                ),
                history=safe_history
            )
            
            response = chat.send_message(user_text)
            final_text = response.text if response.text else "大腦空白，請重試。"
            
            try:
                bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text, parse_mode='Markdown')
            except:
                bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text)
            return

        except Exception as e:
            err = str(e).upper()
            if any(k in err for k in ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE']):
                dead_engines[model_name] = time.time() + 180
                print(f"⏳ {model_name} 進入 CD...")
                continue
            else:
                bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=f"⚠️ 核心出錯: {str(e)[:100]}")
                return

if __name__ == "__main__":
    print("🚀 MarginCall Express 模組化引擎已啟動！")
    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            time.sleep(3)
