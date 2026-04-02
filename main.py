import os
import sys
import random
import time
import telebot
from dotenv import load_dotenv
from google import genai
from google.genai import types
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz

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
MY_USER_ID = 8688853243  # 【防禦力場】授權使用者 ID

if not BOT_TOKEN or not GEMINI_KEY:
    raise ValueError("❌ .env 缺少必要 API KEY")

fubon.init_fubon()
bot = telebot.TeleBot(BOT_TOKEN)
client = genai.Client(api_key=GEMINI_KEY)

# AI 模型清單
AVAILABLE_MODELS = [
    'gemini-3.1-pro-preview',
    'gemini-3.1-flash-lite-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash'
]

# --- AI 核心工具箱統一配置 ---
AGENT_TOOLS = [
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
    fubon.get_txo_sentiment, 
    risk.get_global_risk_radar, risk.get_v_turn_confirmation
]

def get_dynamic_models():
    models = AVAILABLE_MODELS.copy()
    if market.is_us_market_open() or market.is_tw_market_open():
        # 開盤時優先使用 3.1 Pro 進行深度分析
        if 'gemini-3.1-pro-preview' in models:
            models.remove('gemini-3.1-pro-preview')
            models.insert(0, 'gemini-3.1-pro-preview')
    return list(dict.fromkeys(models)) # 去重

def create_agent_chat(model_name, history=None):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=AGENT_TOOLS, # 使用統一工具箱
            temperature=0.3, 
        ),
        history=history
    )

# 初始化大腦
chat = create_agent_chat(get_dynamic_models()[0])
dead_engines = {}

@bot.message_handler(commands=['reset'])
def reset_memory(message):
    if message.from_user.id != MY_USER_ID:
        return
    global chat
    chat = create_agent_chat(get_dynamic_models()[0])
    bot.reply_to(message, "🧹 推進器記憶體已排空！大腦已重新裝填。")

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    if message.from_user.id != MY_USER_ID:
        print(f"🚨 偵測到非法操作！來源 ID: {message.from_user.id}")
        return
    global chat
    user_text = message.text
    
    # --- 動態時間與市場狀態注入 ---
    import pytz
    from datetime import datetime
    tw_tz = pytz.timezone('Asia/Taipei')
    us_tz = pytz.timezone('US/Eastern')
    now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
    now_us = datetime.now(us_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    tw_status = "🟢 開盤中" if market.is_tw_market_open() else "🔴 已收盤/休市"
    us_status = "🟢 開盤中" if market.is_us_market_open() else "🔴 已收盤/休市"
    
    time_context = f"\n【🕒 當前時間環境】\n- 台北時間: {now_tw} ({tw_status})\n- 美東時間: {now_us} ({us_status})\n"
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
                    tools=AGENT_TOOLS, # 這裡也改用統一配置
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
    
    # --- 啟動背景監控排程 (V 轉狙擊手) ---
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
    
    def auto_v_turn_monitor():
        """自動監控 V 轉狀態並推播"""
        try:
            # 【V4 加固】動態時區偵測：判斷夏令/冬令
            us_tz = pytz.timezone('US/Eastern')
            now_us = datetime.now(us_tz)
            is_dst = now_us.dst().total_seconds() != 0
            
            # 夏令收盤 04:00 (台北), 冬令收盤 05:00 (台北)
            # 這裡我們不寫死時間，而是根據排程觸發。
            # 但我們可以在這裡加一個「美股開盤檢查」
            if not market.is_us_market_open() and now_us.hour < 16:
                # 如果不是在收盤前夕，也不是開盤中，則跳過
                pass

            report = risk.get_v_turn_confirmation()
            
            # 判斷是否需要推播：FTD 觸發、破底、護法警戒、或收盤總結
            if "✅ 觸發" in report or "偵測底盤中" in report or "🚨 警戒" in report or "🏁" in report:
                bot.send_message(MY_USER_ID, report, parse_mode='Markdown')
        except Exception as e:
            print(f"📡 監控引擎異常: {e}")

    # 1. 盤中壓力測試 (每 2 小時一次)
    scheduler.add_job(auto_v_turn_monitor, 'interval', hours=2)
    
    # 2. 關鍵狙擊時刻 (動態偵測夏冬令)
    # 夏令時: 03:45 執行; 冬令時: 04:45 執行
    def scheduled_sniper():
        us_tz = pytz.timezone('US/Eastern')
        is_dst = datetime.now(us_tz).dst().total_seconds() != 0
        now_tw = datetime.now(pytz.timezone('Asia/Taipei'))
        
        # 夏令 03:45, 冬令 04:45
        target_hour = 3 if is_dst else 4
        if now_tw.hour == target_hour and now_tw.minute == 45:
            auto_v_turn_monitor()

    scheduler.add_job(scheduled_sniper, 'cron', minute=45, hour='3,4', day_of_week='tue-sat')
    
    # 3. 最終結算 (夏令 04:10, 冬令 05:10)
    def scheduled_settlement():
        us_tz = pytz.timezone('US/Eastern')
        is_dst = datetime.now(us_tz).dst().total_seconds() != 0
        now_tw = datetime.now(pytz.timezone('Asia/Taipei'))
        target_hour = 4 if is_dst else 5
        if now_tw.hour == target_hour and now_tw.minute == 10:
            auto_v_turn_monitor()

    scheduler.add_job(scheduled_settlement, 'cron', minute=10, hour='4,5', day_of_week='tue-sat')
    
    scheduler.start()
    print("🦅 V 轉狙擊手 V4 已就位 (動態時區支援中)")

    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            time.sleep(3)
