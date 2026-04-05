import os
import sys
import random
import time
import json
import logging
import telebot
import subprocess
import threading
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
import pytz

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- .env 檔案的絕對路徑，確保 systemd 正確載入 ---
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# 修正 Windows Unicode 問題
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 引入自定義模組
import fubon
from config import WDT_MESSAGES, system_prompt
import engine_portfolio as portfolio
import engine_market as market
import engine_risk as risk
import engine_router as router

# --- 初始化區 ---
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MY_USER_ID = os.getenv("TELEGRAM_USER_ID")  # 【防禦力場】授權使用者 ID

if not BOT_TOKEN or not GEMINI_KEY or not MY_USER_ID:
    raise ValueError("❌ .env 缺少必要 API KEY 或 TELEGRAM_USER_ID")

MY_USER_ID = int(MY_USER_ID)

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

dead_engines = {}

# --- 1. 本地非阻塞喚醒機制 (非同步回調版) ---
def trigger_nlp_and_callback(symbol, chat_id=None, message_id=None):
    """
    非同步處理：啟動 GPU 運算，完成後主動推送報告。
    """
    def _run():
        try:
            logger.info(f"🚀 [情報局] 正在啟動深度收割: {symbol}")
            python_exe = sys.executable
            worker_path = os.path.join(script_dir, "nlp_worker.py")
            cmd = [python_exe, worker_path, symbol]
            
            # 讓背景進程耐心跑完，不設超時
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                logger.info(f"✅ [情報局] {symbol} 收割完成")
                if chat_id and message_id:
                    bot.edit_message_text(f"✅ {symbol} 情報收割完畢，正在進行最終決策...", chat_id=chat_id, message_id=message_id)
                    
                    # 重新抓取熱騰騰的新資料
                    strat_data = router.fetch_strat_data(symbol)
                    nlp_alpha = strat_data.get("nlp_insights", {})
                    
                    # 生成分析並推送
                    generate_final_report(symbol, strat_data, nlp_alpha, chat_id, message_id)
            else:
                logger.error(f"❌ [情報局] {symbol} 失敗: {stderr[:200]}")
                if chat_id:
                    bot.send_message(chat_id, f"❌ {symbol} 情報收割失敗，請檢查日誌。\n{stderr[:100]}")
        except Exception as e:
            logger.error(f"🚨 [情報局] 異常: {str(e)}")
            if chat_id: bot.send_message(chat_id, f"🚨 系統異常: {str(e)}")

    threading.Thread(target=_run, daemon=True).start()

def generate_final_report(symbol, strat_data, nlp_alpha, chat_id, message_id=None):
    """
    調用 Gemini 模型生成最終戰報。
    """
    alpha_official = nlp_alpha.get("alpha_official", 0)
    analysis_prompt = f"""
你是交易戰友「破產推進器」。請針對以下數據進行深度推論。

【📊 {symbol} 雙重視角數據集】
1. 技術面/即時盤勢:
{json.dumps(strat_data.get('metrics', {}), indent=2, ensure_ascii=False)}

2. NLP 情緒因子 (Alpha Factors):
- 綜合 Alpha: {nlp_alpha.get('nlp_alpha', 0):+.2f}
- 官方/SEC 訊號: {alpha_official:+.2f}
- 散戶情緒: {nlp_alpha.get('alpha_retail', 0):+.2f}
- 語意報告: {nlp_alpha.get('semantic_summary', '無資料')}

【🧠 推論任務】
- 你必須綜合技術面指標與 NLP Alpha 因子給出最終交易建議。
- **🚨 強烈警告規則**: 若官方訊號 (alpha_official) 小於 -0.5，代表內部人拋售或重大利空公告，請在回覆開頭發出「強烈警告」。
- 請給出具體的「戰略方向」（例如：多頭佈局、觀望、或空頭避險）。
"""
    now = time.time()
    current_models = [m for m in get_dynamic_models() if dead_engines.get(m, 0) < now]
    final_text = "分析失敗。"

    for model_name in current_models:
        try:
            response = client.models.generate_content(
                model=model_name, contents=analysis_prompt,
                config=types.GenerateContentConfig(system_instruction=system_prompt)
            )
            if response.text:
                final_text = response.text
                break
        except: continue

    if message_id:
        try:
            bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text, parse_mode='Markdown')
        except:
            bot.send_message(chat_id, final_text, parse_mode='Markdown')
    else:
        bot.send_message(chat_id, final_text, parse_mode='Markdown')

# --- 2. 背景定時巡邏 ---
def daily_nlp_scout():
    watch_list = ["NVDA", "TSLA", "AAPL", "MSFT", "ARM"]
    for stock in watch_list:
        trigger_nlp_and_callback(stock)
        time.sleep(20)

scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
scheduler.add_job(daily_nlp_scout, 'interval', hours=4)
scheduler.start()

# --- 3. Telegram 雙重視角決策 ---
@bot.message_handler(commands=['analyze', 'nlp'])
def handle_deep_analysis(message):
    if message.from_user.id != MY_USER_ID: return
    
    parts = message.text.strip().split()
    if len(parts) < 2:
        bot.reply_to(message, "💡 用法: `/analyze <股票代號>`", parse_mode='Markdown')
        return
    
    symbol = parts[1].upper()
    sent_msg = bot.reply_to(message, f"🔍 正在調閱 {symbol} 雙重視角戰報...\n(GPU 正在運算中，預計 1~2 分鐘，完成後主動通知您)")
    bot.send_chat_action(message.chat.id, 'typing')

    strat_data = router.fetch_strat_data(symbol)
    nlp_alpha = strat_data.get("nlp_insights", {})

    # 若過期或缺失，進入非同步回調模式
    if "error" in nlp_alpha:
        trigger_nlp_and_callback(symbol, message.chat.id, sent_msg.message_id)
        return

    # 若資料庫已有新鮮資料，秒回報告
    generate_final_report(symbol, strat_data, nlp_alpha, message.chat.id, sent_msg.message_id)

@bot.message_handler(commands=['reset'])
def reset_memory(message):
    if message.from_user.id != MY_USER_ID: return
    global chat
    chat = create_agent_chat(get_dynamic_models()[0])
    bot.reply_to(message, "🧹 推進器記憶體已排空！")

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    if message.from_user.id != MY_USER_ID: return
    user_text = message.text
    
    tw_tz = pytz.timezone('Asia/Taipei')
    us_tz = pytz.timezone('US/Eastern')
    now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
    now_us = datetime.now(us_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    tw_status = "🟢 開盤中" if market.is_tw_market_open() else "🔴 已收盤"
    us_status = "🟢 開盤中" if market.is_us_market_open() else "🔴 已收盤"
    
    time_context = f"\n【🕒 當前時間環境】\n- 台北: {now_tw} ({tw_status})\n- 美東: {now_us} ({us_status})\n"
    strat_context = router.get_strat_context(user_text)
    dynamic_prompt = system_prompt + time_context + strat_context
    
    mood = "bad_market" if any(w in user_text for w in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot.send_chat_action(message.chat.id, 'typing')

    now = time.time()
    current_models = [m for m in get_dynamic_models() if dead_engines.get(m, 0) < now]

    for model_name in current_models:
        try:
            chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_prompt,
                    tools=AGENT_TOOLS,
                    temperature=0.3, 
                )
            )
            response = chat.send_message(user_text)
            final_text = response.text if response.text else "大腦空白。"
            bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text, parse_mode='Markdown')
            return
        except Exception as e:
            if any(k in str(e).upper() for k in ['429', 'RESOURCE_EXHAUSTED']):
                dead_engines[model_name] = time.time() + 180
                continue
            else:
                bot.send_message(message.chat.id, f"⚠️ 錯誤: {str(e)[:100]}")
                return

if __name__ == "__main__":
    print("🚀 MarginCall Express 已啟動！")
    scheduler = BackgroundScheduler(timezone=pytz.timezone('Asia/Taipei'))
    
    def auto_v_turn_monitor():
        try:
            if not market.is_us_market_open() and datetime.now(pytz.timezone('US/Eastern')).hour < 16: return
            report = risk.get_v_turn_confirmation()
            if any(k in report for k in ["✅ 觸發", "偵測", "🚨", "🏁"]):
                bot.send_message(MY_USER_ID, report, parse_mode='Markdown')
        except: pass

    scheduler.add_job(auto_v_turn_monitor, 'interval', hours=2)
    scheduler.start()

    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            time.sleep(3)
