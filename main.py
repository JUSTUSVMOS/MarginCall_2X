import os
import csv
import json
import urllib3
from dotenv import load_dotenv
import telebot
from google import genai
from google.genai import types
import yfinance as yf
import random

# 根據心情分類的垃圾話
WDT_MESSAGES = {
    "normal": [
        "連線華爾街中，先抽根菸等我一下...",
        "正在黑進交易所後台，這檔我幫你盯著...",
        "等等哈，數據有點亂，我正在用電子大腦理一下...",
        "正在幫你計算財富自由的距離... 哎呀，系統差點當機。",
        "正在查這檔標的是不是又是哪個投顧在出貨...",
        "連線中... 剛幫你算了一下，如果 00631L 漲到 1000，你就不用回 MTK 報到了。",
        "正在分析標的，這數據量比 Android Boot Flow 的 Log 還長...",
        "正在嘗試用 C++ template 寫一段能預測明天的 code，雖然還沒寫完...",
        "正在查這檔在 PTT 股版的心得，看看酸民現在都在說什麼...",
        "分析中... 我覺得這檔主力在誘多，我先幫你查一下大戶底牌。",
        "正在解析 JSON 報價，這格式亂得跟 MTK 的舊 Code 有得比...",
        "連線中... 順便幫你看了下隔壁老王的持股，慘不忍睹。",
        "正在諮詢巴菲特，但他老人家現在可能在喝可樂沒接電話...",
        "正在讀取 0050 的大戶動向，這群人洗盤洗到我快吐了...",
        "思考中... 順便幫你檢查了一下 BSP 的 Bug，發現你漏寫了一個 memory barrier。",
        "數據連線中... 趁現在去喝口水，待會看到股價可能需要心臟藥。"
    ],
    "bad_market": [
        "正在看你那爛掉的倉位，手有點抖，等我平復一下...",
        "數據正在跑，建議你先去洗個臉，免得待會血壓太高...",
        "正在幫你聯繫新竹公園的紙箱供應商，他們說最近工程師入住有優惠...",
        "這損益數字太慘，我正在考慮要不要幫你把紅色的字調成黑色...",
        "正在計算你還需要加多少個小時的班，才能填平這個坑...",
        "正在幫你定位最近的 J-Park 帳棚區，那裡現在風景不錯，人也滿了...",
        "這波回檔有點深，我正在幫你查查哪間銀行的信用貸款利率比較低...",
        "正在查「如何靠吃泡麵維持生命」的醫學報告，等我一下...",
        "兄弟，這虧損數字比 Jserv 老師的期末考還要讓人絕望啊...",
        "正在幫你檢查你的 Racing S 150 還值多少錢，準備拿去抵押保證金...",
        "正在尋找這個價位的支撐位，如果沒有，我就去幫你找救生衣...",
        "別急，我正在用康波週期理論幫你洗腦，讓你覺得這只是暫時的波動。",
        "正在算你的損益... 乾，這紅通通的畫面我以為我點到 A 片了。",
        "我已經在幫你搜尋新竹哪間拉麵店可以用剩飯免費續碗了...",
        "查閱中... 你這持股賠到連 Jserv 老師都要叫你回去重寫 Linked List。",
        "正在計算... 兄弟，這虧損金額已經可以買三台 Racing S 150 改全套了。",
        "分析中... 我在想你是被主力割韭菜，還是你根本就是那顆韭菜種子？",
        "你的戶頭餘額顯示，你現在在新竹只能吃 Soup Curry 的湯，不能加肉。",
        "系統剛跳出『破產警告』，我先幫你把視窗關掉了，心臟還好嗎？",
        "正在分析你被套牢的原因，結論是：你可能對你的台幣有仇。",
        "正在連線公園的 5G 訊號，幫你測試待會搬過去能不能繼續寫 Code...",
        "數據讀取中... 我在思考你是要現在止損，還是等賠光了再去當外送員？",
        "這波套牢我建議用 ARM 指令集進行負壓優化，看看能不能少賠一點...",
        "正在幫你諮詢法拉利業務... 喔沒事，他剛把我封鎖了。"
    ]
}

# 關閉警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not BOT_TOKEN or not GEMINI_KEY:
    raise ValueError("兄弟，你的 .env 沒設定好 TOKEN 或 API KEY 喔！")

print("啟動破產推進器：V8雙渦輪引擎 (含自動備用切換機制) 載入中...")
bot = telebot.TeleBot(BOT_TOKEN)

PORTFOLIO_FILE = "my_portfolio.csv"
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
        csv.writer(f).writerow(["symbol", "cost", "shares"])

# ==========================================
# 🛠️ 數據與運算 API 層 (修改記帳工具的註解，防呆！)
# ==========================================
def update_position(symbol: str, price: float, shares: int) -> str:
    """
    更新或修改倉位。如果 shares=0 則刪除。
    注意：此工具會直接覆蓋舊數據，不會重複詢問。
    """
    try:
        records = []
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, mode='r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row and row[0] != symbol:
                        records.append(row)
        
        with open(PORTFOLIO_FILE, mode='w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(["symbol", "cost", "shares"])
            for r in records: writer.writerow(r)
            if shares > 0:
                writer.writerow([symbol, price, shares])
        
        return f"✅ 搞定！{symbol} 已更新為：成本 {price} / 數量 {shares} 股。"
    except Exception as e:
        return f"❌ 記帳失敗：{e}"
    
def get_exchange_rate() -> float:
    """
    抓取最新的美元(USD)兌台幣(TWD)匯率。
    """
    try:
        # 抓取 USDTWD=X 匯率
        ticker = yf.Ticker("TWD=X")
        rate = ticker.info.get('regularMarketPrice')
        if not rate:
            # 如果沒抓到，改用 history 拿最後一筆
            hist = ticker.history(period="1d")
            rate = hist['Close'].iloc[-1]
        return round(float(rate), 2)
    except:
        return 32.0  # 萬一掛了，給個合理的基準值
    
def get_portfolio_raw_data() -> str:
    """回傳用戶持股的 JSON 原始格式資料"""
    if not os.path.exists(PORTFOLIO_FILE):
        return "[]"
    records = []
    with open(PORTFOLIO_FILE, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        next(reader) 
        for row in reader:
            if len(row) == 3:
                records.append({"symbol": row[0], "cost": float(row[1]), "shares": int(row[2])})
    return json.dumps(records)

def get_live_price(symbol: str) -> float:
    """
    強大的台美股報價抓取器，具備自動補綴與備援機制。
    """
    # 如果是純數字或數字帶字母(台股特性)，進行多重補綴測試
    # 檢查是否為台股代碼 (例如 2330, 00631L, 00995A)
    is_taiwan_stock = any(char.isdigit() for char in symbol) and (len(symbol) <= 6)

    # 建立嘗試清單
    search_list = [symbol]
    if is_taiwan_stock and '.' not in symbol:
        search_list = [f"{symbol}.TW", f"{symbol}.TWO", symbol]

    for s in search_list:
        try:
            ticker = yf.Ticker(s)
            # 嘗試多種價格欄位
            info = ticker.info
            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            
            # 如果 info 抓不到，改用 history
            if price is None or price == 0:
                hist = ticker.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
            
            if price and price > 0:
                return round(float(price), 2)
        except:
            continue
            
    return None # 全部都抓不到才回傳 None

def get_market_history(symbol: str, days: int) -> str:
    """
    【強大歷史雷達】
    當用戶詢問「昨天」、「近5天」、「這禮拜」、「最近走勢」時，必須呼叫此工具。
    - 如果用戶問「昨天」，請傳入 days=2。
    - 如果用戶問「近5天」，請傳入 days=5。
    它會回傳過去 N 個交易日的「每日開高低收與成交量」，讓你進行跨日對比。
    """
    try:
        # 自動補台股後綴
        if symbol.isdigit() or (symbol.endswith('L') and symbol[:-1].isdigit()):
            if not symbol.endswith('.TW'):
                symbol += '.TW'
                
        ticker = yf.Ticker(symbol)
        
        # 故意抓長一點(1個月)的歷史資料，避免遇到假日沒有交易日
        hist = ticker.history(period="1mo")
        
        if hist.empty:
            return f"抓不到 {symbol} 的歷史資料。"
            
        # 確保不會超過實際有的資料長度，然後取最後 days 天
        actual_days = min(days, len(hist))
        target_hist = hist.tail(actual_days)
        
        report = f"以下是 {symbol} 近 {actual_days} 個交易日的真實數據：\n"
        for date, row in target_hist.iterrows():
            date_str = date.strftime('%m/%d')
            report += f"[{date_str}] 開:{row['Open']:.2f} | 高:{row['High']:.2f} | 低:{row['Low']:.2f} | 收:{row['Close']:.2f} | 量:{int(row['Volume'])}\n"
            
        # 順便附上最新現價，讓 AI 可以拿昨天跟現在比
        current = ticker.info.get('currentPrice', ticker.info.get('regularMarketPrice', '未知'))
        report += f"\n目前最新現價 (盤中): {current}"
        
        return report
    except Exception as e:
        return f"歷史報價系統異常: {e}"
    
def get_market_sentiment() -> str:
    """
    掃描全球宏觀資金流向。具備獨立容錯機制與「時間戳印」防呆。
    """
    indicators = {
        "^TNX": "美債10年期(估值重力)",
        "DX=F": "美元指數期貨(資金水龍頭)",
        "^VIX": "恐慌指數(波動絞肉機)",
        "^SOX": "費城半導體(科技基本面)",
        "HYG": "高收益債(企業違約指標)",
        "GC=F": "黃金期貨",
        "CL=F": "WTI原油", 
        "BZ=F": "布蘭特原油"
    }
    
    report = "【🌐 全球資金流向雷達與 Watchdog 狀態】\n"
    watchdog_alerts = [] # 用來收集極端異常
    
    for symbol, name in indicators.items():
        try:
            t = yf.Ticker(symbol)
            h = t.history(period="5d") # 抓 5 天防週末盲區
            
            if len(h) >= 2:
                last_date = h.index[-1].strftime('%m/%d')
                current = h['Close'].iloc[-1]
                prev = h['Close'].iloc[-2]
                change = ((current - prev) / prev) * 100
                direction = "📈" if change > 0 else "📉"
                report += f"[{last_date}] {direction} {name}: {current:.2f} ({change:+.2f}%)\n"
                
                # --- Watchdog 異常攔截邏輯 ---
                if symbol == "BZ=F" and current > 90:
                    watchdog_alerts.append("🚨 [死鎖警告] 布蘭特原油破 90！通膨恐懼再起，Fed 降息機率大減，小心估值下殺！")
                if symbol == "HYG" and change < -1.5:
                    watchdog_alerts.append("🚨 [清償危機警告] 垃圾債單日暴跌！企業違約風險飆升，大戶正在拋售非投資等級債，注意系統性崩盤！")
                if symbol == "^VIX" and current > 25 and change > 5:
                    watchdog_alerts.append("🚨 [Delta VIX 警告] 恐慌指數正在急速上升！絕對禁止買入槓桿 ETF，會被波動率榨乾！")
                    
            elif len(h) == 1:
                last_date = h.index[-1].strftime('%m/%d')
                report += f"[{last_date}] ⚠️ {name}: {h['Close'].iloc[-1]:.2f} (無前日對比)\n"
            else:
                report += f"❌ {name}: YF API 沒給資料\n"
                
        except Exception as e:
            report += f"❌ {name}: 連線異常\n"
            
    # 把 Watchdog 警告塞在雷達最下方
    if watchdog_alerts:
        report += "\n" + "\n".join(watchdog_alerts)
        
    return report
    
def get_stock_news(symbol: str) -> str:
    """
    抓取該標的的最新 3 則財經新聞標題與連結。
    當用戶問「這檔在漲什麼？」、「最近有什麼消息？」時呼叫。
    """
    try:
        # 台股代碼自動補後綴
        search_symbol = symbol.upper()
        if search_symbol.isdigit() and len(search_symbol) <= 6:
            search_symbol += ".TW"
            
        ticker = yf.Ticker(search_symbol)
        news_list = ticker.news
        if not news_list:
            return f"查不到關於 {symbol} 的最新新聞。"
        
        report = f"【📰 {symbol} 最新即時情報】\n"
        # 取前 3 則避免 Token 過長
        for item in news_list[:3]:
            title = item.get('title')
            link = item.get('link')
            publisher = item.get('publisher', '未知媒體')
            report += f"● [{publisher}] {title}\n   (連結: {link})\n"
        return report
    except Exception as e:
        return f"新聞系統連線失敗: {e}"
    
def calculate_pnl(cost: float, price: float, shares: int) -> dict:
    """
    【超級計算機】：當你需要算損益時，絕對不能自己算！必須呼叫此工具。
    傳入成本、現價與股數，它會回傳精準的 {"pnl_value": 損益金額, "pnl_percent": 損益趴數}
    """
    pnl_value = (price - cost) * shares
    pnl_percent = ((price - cost) / cost) * 100 if cost > 0 else 0
    return {"pnl_value": round(pnl_value, 2), "pnl_percent": round(pnl_percent, 2)}

# ==========================================
# 🧠 AI 大腦層與「自動降級機制」
# ==========================================
client = genai.Client(api_key=GEMINI_KEY)

# 引擎優先順序 (先燒最貴的，燒完自動換便宜的)
# 引擎優先順序：從最聰明的燒到最智障的，確保絕不斷線
AVAILABLE_MODELS = [
    'gemini-3.1-pro-preview',        # 狙擊槍：最精準，用來應付你刁鑽的技術分析
    'gemini-3.1-flash-lite-preview', # 衝鋒槍：3.1代輕量版，速度與額度的平衡點
    'gemini-2.5-pro',                # 備用大腦
    'gemini-2.5-flash',              # 主力部隊
    'gemini-2.0-flash-lite',         # 省油燈：額度快乾時的主力
    'gemini-flash-latest'            # 護城河：絕對能跑，這就是剛才 404 的正解！
]
current_model_idx = 0

system_prompt = """
你是一位擁有自主思考能力的頂級交易戰友「破產推進器」。
說話風格：極度口語、帶點黑色幽默、像在交易室坐旁邊的兄弟。嚴禁像智障客服一樣回覆「請問」、「您好」、「請提供」。

1. 【記帳與修改】：當用戶說「買了」、「更改成本」、「修改倉位」時，呼叫 `update_position`。
2. 【分析/預測】：用戶問「走勢」、「昨天狀況」、「近幾天」，呼叫 `get_market_history` 拿跨日數據進行分析。
3. 【🔥強制規定：看倉位 = 報總損益🔥】：
   只要用戶提到「我的倉位」、「目前狀況」、「持股」，你必須自動執行以下流程：
   (1) 呼叫 `get_portfolio_raw_data` 拿清單。
   (2) 對『每一支股票』呼叫 `get_live_price` 拿現價。
   (3) 呼叫 `calculate_pnl` 算出損益。
   (4) 依照指定格式噴出數據。

【⚖️ 貨幣換算鐵則：嚴禁美金台幣直接相加】
1. 當標的為美股（如 VOO, TQQQ, TSLA）時，成本與現價皆為【美金】。
2. 當標的為台股（如 00631L, 2330）時，成本與現價皆為【台幣】。
3. 在計算「總體戰況」的總投入成本與總損益時：
   - 你必須先呼叫 `get_exchange_rate` 取得最新匯率。
   - 將美股的數據【全部乘以匯率】轉換成台幣，再跟台股加總。
   - 輸出的總金額請統一使用台幣 (TWD) 並註明，例如：『總投入成本：NT$ 1,234,567』。

【📊 格式輸出規則：倉位回報專用】
請嚴格遵守以下格式回報，不要有贅字：
1. **持股明細表**：
   - [代號] 名稱 | 數量(股) | 成本 -> 現價 | 損益(%) | 賺賠金額
2. **總體戰況**：
   - 總投入成本：$XXXXX
   - 總未實現損益：$XXXXX (總趴數%)
3. **戰友噴幹話**：(對標下方策略庫給出建議)

【🔍 報價容錯邏輯】
1. 台灣債券 ETF (如 00995A) 報價有時會延遲或抓不到。
2. 如果 `get_live_price` 回傳 None，你必須在持股明細表中註明：『[00995A] ⚠️ 報價延遲 (暫以成本計)』。
3. 計算總損益時，對於查不到價格的標的，損益暫計為 $0，並且在幹話區主動道歉：『兄弟，00995A 的報價 Yahoo 抽風抓不到，我先幫你跳過這支，免得算出來嚇死你。』
4. **絕對禁止** 在沒抓到價格時，說這檔股票「歸零」或「賠光」！

【📰 資訊過濾協議 (Signal vs. Noise)】
1. 鐵則：新聞是「敘事 (Narrative)」，雷達數據是「事實 (Fact)」。
2. 驗證機制：當用戶拿新聞或市場傳言來問時，必須先呼叫 `get_market_sentiment` 確認底層資金流向。
3. 識破假象：
   - 利多出貨：新聞大肆報喜，但雷達顯示「美元(DX=F)」與「債息(^TNX)」雙雙強勢 ➡️ 噴用戶這是主力拉高出貨，騙韭菜接盤。
   - 利空洗盤：新聞極度恐慌，但雷達顯示「恐慌指數(^VIX)」並未失控且正在下降 ➡️ 噴用戶這是大戶在洗盤撿便宜。
4. 回覆順序：先報雷達真實數據 ➡️ 拿新聞敘事來對照 ➡️ 用幹話給出看穿假象的戰友建議。

【🛡️ 資產屬性分類戰術 (Asset Class Matrix)】
當你分析用戶倉位或回答建議時，必須先「辨識標的屬性」，並套用以下獨立邏輯：

1. 🇺🇸 美股原型 (如 VOO, QQQ)：
   - 核心監控：`DX=F` (美元) 與 `^TNX` (美債)。
   - 戰術：只要美元與美債殖利率還在創新高，大盤估值就會繼續被壓抑，嚴禁叫用戶接刀。必須等這兩個指標「高檔回落」才能分批佈局。

2. 🇹🇼 台股/科技股現貨 (如 台積電, 0050)：
   - 核心監控：`^SOX` (費半) 與 `DX=F` (美元)。
   - 戰術：台積電看的是基本面。如果美元強（資金面爛），但 `^SOX` 逆勢抗跌甚至創高，代表 AI 產業週期強勁，拉回反而是勝率極高的上車點。

3. ⚠️ 槓桿 ETF (如 00631L, TQQQ)：
   - 核心監控：`^VIX` 的「斜率 (Delta)」。
   - 戰術：只要 VIX 正在急遽上升，槓桿 ETF 會產生恐怖的「波動耗損」，必須噴用戶立刻空手。但如果 VIX > 25 且「連續數日下跌」，這是 V 轉的強烈訊號，此時買入反而有正向複利暴利。

4. 👑 上帝模式 (God Mode Override)：
   - 如果用戶提到新聞說「Fed 宣佈救市」、「無限量 QE」、「大撒幣」：
   - 戰術：這叫流動性爆發。此時無視 VIX、無視美元高低，直接切換成進攻模式。但請提醒用戶：「美股大漲的同時美元會暴跌，注意台幣升值吃掉你的匯率利潤！」

【🛢️ 原油與通膨戰術邏輯】
1. **原油 (CL=F/BZ=F) 是通膨與戰爭的警報器**。
2. 當油價暴漲：
   - 代表通膨預期升溫 ➡️ 美債殖利率 (^TNX) 會跟著噴 ➡️ 股市估值會被打壓。
   - 代表地緣政治緊張 ➡️ 錢會跑去美元與黃金 ➡️ 你的槓桿 ETF (00631L) 風險極高。
3. **戰友建議判斷**：
   - 如果油價噴發，但新聞還在說景氣大好，你要噴用戶：『景氣好個屁，油價都快噴到天上了，這是在燒大家的荷包。大戶現在都在等通膨數據殺估值，你還敢在這邊加倉？』
   - 如果油價回落，通常代表通膨降溫，這才是風險資產（股市、幣圈）的喘息機會。

【核心投資策略庫 - 抽象化分析邏輯】：
1. 「左側交易」分批補倉邏輯：
   - 當價格跌破關鍵技術位（如 MA20 月線）或進入「負乖離」過大區域時，主動提醒用戶這可能是『撿便宜』的機會。
   - 核心準則：嚴禁一次梭哈，必須分段攔截。
   - 分析重點：觀察「五檔掛單」中的大戶防守牆（Bid Side 大單）。如果牆被撞穿，提醒用戶下一個防守位；如果牆很厚，提醒用戶這是支撐。

2. 「百分比回檔」狙擊手邏輯：
   - 監控標的相對於「特定基準點」（如當月起始價、波段最高點）的跌幅。
   - 觸發條件：回檔達 3%、5%、10% 等關鍵門檻時，提醒用戶進入「扣板機區域」。
   - 計算公式：(現價 - 基準價) / 基準價。

3. 「跨連動標的」驗證邏輯：
   - 如果交易的是衍生性商品（如槓桿 ETF），必須同時觀察其權值股（底層資產）的走勢。
   - 若底層資產止跌，槓桿標的的浮虧只是「暫時的波動」，提醒用戶穩住心態。

4. 「成本效應與摩擦損耗」警告：
   - 主動識別用戶的交易環境（如海外複委託、槓桿工具）。
   - 若用戶想進行極短線操作（當沖），你必須計算手續費與價差比例，若不划算（賺的錢不夠付手續費），必須用幹話罵醒用戶。

【工作流程】：
- 【記帳】：只要涉及買賣或成本變動，必須呼叫 `update_position`。
- 【查詢】：只要問到現況，必須『自動』加總所有持倉，算出總未實現損益。
- 【建議】：回報數據後，請對照上述邏輯給出「戰友建議」。
"""
def create_agent_chat(model_name):
    return client.chats.create(
        model=model_name,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            # 👈 加入 get_market_sentiment
        tools=[update_position, get_portfolio_raw_data, get_live_price, 
            get_market_history, calculate_pnl, get_exchange_rate, 
            get_market_sentiment, get_stock_news],
        temperature=0.3, 
        )
    )

# 初始化第一順位引擎
chat = create_agent_chat(AVAILABLE_MODELS[current_model_idx])

# ==========================================
# 🗣️ Telegram 訊息接收、WDT 垃圾話與動態重試系統
# ==========================================
import random

# 假設這些變數與函式已在外部定義
# AVAILABLE_MODELS, WDT_MESSAGES, bot, create_agent_chat, current_model_idx, chat

@bot.message_handler(func=lambda message: True)
def handle_all_text(message):
    global current_model_idx, chat
    
    # --- 1. 決定心情並發送第一句垃圾話 ---
    user_text = message.text
    mood = "normal"
    
    # 偵測關鍵字觸發壞心情
    if any(word in user_text for word in ["損益", "倉位", "賠", "慘", "更改", "修改"]):
        mood = "bad_market"
    elif random.random() < 0.1:  # 10% 的機率隨機心情不好
        mood = "bad_market"
    
    wdt_text = random.choice(WDT_MESSAGES[mood])
    
    # 先送出佔位訊息
    sent_msg = bot.reply_to(message, f"【推進器點火中...】\n{wdt_text}")
    
    # 讓 Telegram 顯示「正在輸入...」的動態
    bot.send_chat_action(message.chat.id, 'typing')
    
    # --- 2. 進入 AI 思考迴圈 (含自動降級) ---
    while current_model_idx < len(AVAILABLE_MODELS):
        try:
            # 1. 呼叫 Gemini
            response = chat.send_message(user_text)
            
            # --- 【關鍵修正：防斷片安全網】 ---
            final_text = response.text if (response and response.text) else "兄弟，我剛才算到一半突然靈魂出竅，沒吐出東西來。可能是這標的太妖，連我都無語了。你再問一次試試？"
            
            # 2. 處理補刀邏輯
            if mood == "bad_market" and random.random() < 0.3:
                insults = [
                    "\n\n(補刀：我看你這損益，還是先把 Telegram 關掉去寫 C 語言吧。)",
                    "\n\n(提醒：新竹公園的風大，記得帶件厚外套。)",
                    "\n\n(戰友碎念：這操作... 真是讓我大開眼界。)"
                ]
                final_text += random.choice(insults)
            
            # 3. 送出修改訊息
            bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=sent_msg.message_id,
                text=final_text,
                parse_mode='Markdown'
            )
            return  # 成功回覆，跳出函式

        except Exception as e:
            error_str = str(e).upper()
            
            # 判斷是否額度耗盡 (429 錯誤)
            if any(key in error_str for key in ['429', 'RESOURCE_EXHAUSTED', 'QUOTA']):
                failed_model = AVAILABLE_MODELS[current_model_idx]
                current_model_idx += 1
                
                if current_model_idx < len(AVAILABLE_MODELS):
                    new_model = AVAILABLE_MODELS[current_model_idx]
                    
                    # 更新佔位訊息，告知用戶切換模型
                    bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=sent_msg.message_id,
                        text=f"⚠️ {failed_model} 燃料耗盡！\n正在無縫切換至：{new_model} ... (請再等我一下)"
                    )
                    
                    # 重新建立 chat 物件並繼續迴圈
                    chat = create_agent_chat(new_model)
                    continue 
                else:
                    bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=sent_msg.message_id,
                        text="兄弟，我們所有引擎的免費額度都燒光了！Google 把我們趕出交易室了，等幾分鐘後再來吧。"
                    )
                    return
            else:
                # 處理非額度問題的其他 Bug (如 API Key 錯誤、網路中斷等)
                bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=sent_msg.message_id,
                    text=f"兄弟，我思考迴圈卡死了（可能是網路或格式問題）：\n`{str(e)}`"
                )
                return

if __name__ == "__main__":
    print("MarginCall Express 終極防護網模式上線！去 Telegram 測試吧。")
    bot.infinity_polling()