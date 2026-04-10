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

# 設定日誌 (只保留 FileHandler 避免雙重輸出)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(process)d] - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log")
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

# AI 模型清單 (暫時將 Flash 移至第一位，避開 Pro 的 429 延遲)
AVAILABLE_MODELS = [
    'gemini-3.1-flash-lite-preview',
    'gemini-3.1-pro-preview',
    'gemini-2.5-pro',
    'gemini-2.5-flash',
    'gemini-2.5-flash-lite',
    'gemma-4-31b-it',
    'gemini-flash-latest'
]

# --- 工具呼叫追蹤器 (用於 Debug 卡住問題) ---
def tool_wrapper(func):
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        arg_str = f"args={args}, kwargs={kwargs}"
        logger.info(f"🛠️ [TOOL_START] 正在呼叫工具: {func.__name__} | 參數: {arg_str[:100]}...")
        start_t = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_t
            logger.info(f"✅ [TOOL_DONE] {func.__name__} 執行完畢 (耗時: {duration:.2f}s)")
            return result
        except Exception as e:
            logger.error(f"❌ [TOOL_ERROR] {func.__name__} 報錯: {e}")
            raise e
    return wrapper

# --- AI 核心工具箱統一配置 (套用追蹤器) ---
# 1. 記帳專用工具 (會修改資料庫)
BOOKKEEPING_TOOLS = [tool_wrapper(portfolio.update_position)]

# 2. 分析與查詢工具 (唯讀)
ANALYTICS_TOOLS = [
    tool_wrapper(portfolio.get_portfolio_raw_data), 
    tool_wrapper(portfolio.calculate_pnl), tool_wrapper(portfolio.get_exchange_rate),
    tool_wrapper(market.get_live_price), tool_wrapper(market.get_us_realtime_insight),
    tool_wrapper(market.resolve_symbol_identity), 
    tool_wrapper(market.get_market_sentiment), tool_wrapper(market.get_stock_news),
    tool_wrapper(market.get_fundamental_data), tool_wrapper(market.get_market_history),
    tool_wrapper(market.get_technical_analysis), 
    tool_wrapper(fubon.get_market_hot_stocks), tool_wrapper(fubon.get_intraday_trend),
    tool_wrapper(fubon.get_market_trades), tool_wrapper(fubon.get_price_volumes),
    tool_wrapper(fubon.get_quote_and_orderbook), tool_wrapper(fubon.get_historical_stats), 
    tool_wrapper(fubon.get_txo_sentiment), 
    tool_wrapper(risk.get_global_risk_radar), tool_wrapper(risk.get_v_turn_confirmation),
    tool_wrapper(risk.get_capital_flow_matrix)
]

# 全能工具箱 (用於 /trade)
FULL_AGENT_TOOLS = BOOKKEEPING_TOOLS + ANALYTICS_TOOLS
# 唯讀工具箱 (用於一般對話)
READ_ONLY_TOOLS = ANALYTICS_TOOLS

def get_dynamic_models():
    models = AVAILABLE_MODELS.copy()
    if market.is_us_market_open() or market.is_tw_market_open():
        # 開盤時優先使用 2.0 Flash 進行快速反應
        if 'gemini-2.0-flash' in models:
            models.remove('gemini-2.0-flash')
            models.insert(0, 'gemini-2.0-flash')
    return list(dict.fromkeys(models)) # 去重

def create_agent_chat(model_name, history=None, tools=READ_ONLY_TOOLS):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=tools, 
            temperature=0.3, 
        ),
        history=history
    )

dead_engines = {}
user_chat_history = [] # --- 全域對話歷史紀錄 ---

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
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 分鐘硬上限
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                logger.error(f"❌ [情報局] {symbol} 超時被強制終止")
                if chat_id:
                    bot.send_message(chat_id, f"❌ {symbol} 情報收割超時 (>5min)，請檢查 Ollama 是否正常運行。")
                return
            
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
        except Exception as te:
            logger.warning(f"Report Markdown parse failed, falling back to plain text: {te}")
            try:
                bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=final_text)
            except:
                bot.send_message(chat_id, final_text)
    else:
        try:
            bot.send_message(chat_id, final_text, parse_mode='Markdown')
        except Exception as te:
            logger.warning(f"Report Markdown parse failed (send), falling back to plain text: {te}")
            bot.send_message(chat_id, final_text)

# --- 2. 背景定時巡邏 ---
def daily_nlp_scout():
    watch_list = ["NVDA", "TSLA", "AAPL", "MSFT", "ARM"]
    for stock in watch_list:
        trigger_nlp_and_callback(stock)
        time.sleep(20)

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
    global user_chat_history
    user_chat_history = []
    bot.reply_to(message, "🧹 推進器記憶體已排空！對話上下文已重置。")

# --- 核心 LLM 調用邏輯 (支援 Tool 切換) ---
def ask_llm(user_text, tools, chat_history=None, system_prompt_override=None):
    """
    統一的 LLM 調用入口，處理模型切換、超時與 Tool 呼叫。
    """
    tw_tz = pytz.timezone('Asia/Taipei')
    us_tz = pytz.timezone('US/Eastern')
    now_tw = datetime.now(tw_tz).strftime('%Y-%m-%d %H:%M:%S')
    now_us = datetime.now(us_tz).strftime('%Y-%m-%d %H:%M:%S')
    
    tw_status = "🟢 開盤中" if market.is_tw_market_open() else "🔴 已收盤"
    us_status = "🟢 開盤中" if market.is_us_market_open() else "🔴 已收盤"
    
    time_context = f"\n【🕒 當前時間環境】\n- 台北: {now_tw} ({tw_status})\n- 美東: {now_us} ({us_status})\n"
    strat_context = router.get_strat_context(user_text)
    dynamic_prompt = (system_prompt_override or system_prompt) + time_context + strat_context
    
    now = time.time()
    current_models = [m for m in get_dynamic_models() if dead_engines.get(m, 0) < now]

    for i, model_name in enumerate(current_models):
        try:
            chat = client.chats.create(
                model=model_name,
                config=types.GenerateContentConfig(
                    system_instruction=dynamic_prompt,
                    tools=tools,
                    temperature=0.3, 
                ),
                history=chat_history
            )
            
            response_container = []
            exception_container = []
            
            def _thread_task():
                try:
                    res = chat.send_message(user_text)
                    response_container.append(res)
                except Exception as ex:
                    exception_container.append(ex)

            llm_thread = threading.Thread(target=_thread_task)
            llm_thread.start()
            llm_thread.join(timeout=45) 

            if llm_thread.is_alive():
                logger.warning(f"Engine {model_name} timeout.")
                dead_engines[model_name] = time.time() + 60
                continue
            
            if exception_container:
                raise exception_container[0]
            
            if not response_container:
                continue

            response = response_container[0]
            # 更新歷史紀錄 (如果是全域歷史)
            if chat_history is not None:
                new_history = chat.get_history()
                chat_history.clear()
                chat_history.extend(new_history[-20:])
                
            return response.text if response.text else "大腦空白。"
        except Exception as e:
            err_str = str(e).upper()
            if any(k in err_str for k in ['429', 'RESOURCE_EXHAUSTED', '503', 'UNAVAILABLE', 'INTERNAL', 'DEADLINE_EXCEEDED']):
                logger.warning(f"Engine {model_name} temp failure: {str(e)}")
                dead_engines[model_name] = time.time() + 60
                continue
            else:
                logger.error(f"Engine {model_name} fatal: {str(e)}")
                return f"⚠️ 模型異常: {str(e)[:100]}"
    return "🧪 所有推進器皆暫時熄火，請稍後再試。"

@bot.message_handler(commands=['trade'])
def handle_trade_command(message):
    if message.from_user.id != MY_USER_ID: return
    
    user_input = message.text.replace('/trade', '').strip()
    if not user_input:
        bot.reply_to(message, "📝 請輸入交易內容，例如：`/trade 買入 10 股 NVDA`", parse_mode='Markdown')
        return

    sent_msg = bot.reply_to(message, "✍️ 【記帳模式啟動】正在寫入帳本...")
    bot.send_chat_action(message.chat.id, 'typing')

    trade_prompt = system_prompt + "\n\n【⚠️ 記帳指令模式】你現在擁有「更新持倉 (update_position)」的權限。請根據使用者輸入精確執行買入、賣出或校正。"
    
    # 記帳模式不使用全域歷史，避免上下文干擾
    result = ask_llm(user_input, tools=FULL_AGENT_TOOLS, system_prompt_override=trade_prompt)
    
    try:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=result, parse_mode='Markdown')
    except:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=result)

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    if message.from_user.id != MY_USER_ID: return
    user_text = message.text
    global user_chat_history
    
    # 1. 立即回應「推進器點火」
    mood = "bad_market" if any(w in user_text for w in ["損益", "賠", "慘"]) else "normal"
    sent_msg = bot.reply_to(message, f"【推進器點火】\n{random.choice(WDT_MESSAGES[mood])}")
    bot.send_chat_action(message.chat.id, 'typing')

    # 2. 調用 LLM (使用唯讀工具箱)
    final_text = ask_llm(user_text, tools=READ_ONLY_TOOLS, chat_history=user_chat_history)
    
    try:
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text, parse_mode='Markdown')
    except Exception as te:
        logger.warning(f"Markdown parse failed, falling back to plain text: {te}")
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_msg.message_id, text=final_text)

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
    scheduler.add_job(daily_nlp_scout, 'interval', hours=4)
    scheduler.start()

    while True:
        try:
            logger.info("📡 正在開啟 Infinity Polling...")
            bot.infinity_polling(timeout=20, long_polling_timeout=10)
        except telebot.apihelper.ApiTelegramException as e:
            if e.error_code == 409:
                logger.error("🛑 偵測到 409 衝突：已有其他 Bot 實例在運行。將於 15 秒後重試...")
                time.sleep(15)
            else:
                logger.error(f"❌ Telegram API 異常: {e}")
                time.sleep(5)
        except Exception as e:
            logger.error(f"🚨 Polling 崩潰: {e}")
            time.sleep(5)
    while True:
        try:
            bot.infinity_polling(timeout=10, long_polling_timeout=5)
        except Exception as e:
            time.sleep(3)
