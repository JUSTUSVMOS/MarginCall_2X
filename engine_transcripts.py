import requests
import logging
from bs4 import BeautifulSoup
import yfinance as yf
from yf_session import get_ticker, get_download
import re
import time
import warnings
from datetime import datetime

# 忽略警告
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TranscriptEngine:
    """
    Motley Fool (fool.com) 專屬逐字稿構造引擎。
    採用「網址構造法」避免搜尋引擎限流。
    """

    def __init__(self, ticker_symbol):
        self.ticker_symbol = ticker_symbol.upper().replace('.', '-')
        self.ticker = get_ticker(self.ticker_symbol)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/'
        }

    def get_company_slug(self):
        """從 yfinance 提取公司 Slug (例如 Tesla, Inc. -> tesla)"""
        try:
            info = self.ticker.info
            name = info.get('shortName') or info.get('longName') or ""
            # 取第一個單詞，轉小寫，去除非字母數字字元
            first_word = name.split()[0].split(',')[0].lower()
            slug = re.sub(r'[^a-z0-9]', '', first_word)
            return slug
        except Exception as e:
            logger.warning(f"Error getting company slug for {self.ticker_symbol}: {e}")
            return self.ticker_symbol.lower()

    def get_latest_earnings_info(self):
        """獲取最近一次財報的日期與季度資訊"""
        try:
            dates = self.ticker.earnings_dates
            if dates is None or dates.empty:
                return None
            
            # 移除時區資訊以便比較
            dates.index = dates.index.tz_localize(None)
            
            # 找到距離今天最近的「過去」日期 (法說會通常在當天或隔天)
            now = datetime.now()
            past_dates = dates[dates.index <= now]
            if past_dates.empty:
                latest_date = dates.index[-1] # 如果都是未來的，取列表中的最後一個
            else:
                latest_date = past_dates.index[0] # 取最接近現在的那個
            
            month = latest_date.month
            year = latest_date.year
            
            # 簡易季度判斷 (財報發布月 vs 所屬季度)
            if month <= 2: # 1-2月發布的通常是前一年的 Q4
                q = "q4"
                report_year = year - 1
            elif month <= 5: # 4-5月發布的是 Q1
                q = "q1"
                report_year = year
            elif month <= 8: # 7-8月發布的是 Q2
                q = "q2"
                report_year = year
            else: # 10-11月發布的是 Q3
                q = "q3"
                report_year = year
                
            return {
                "date": latest_date,
                "q": q,
                "year": report_year
            }
        except Exception as e:
            print(f"Earnings Info Error: {e}")
            return None

    def construct_fool_urls(self):
        """構造可能的 Motley Fool 網址 (考慮到 NVDA 這種財報年度超前的怪胎)"""
        info = self.get_latest_earnings_info()
        slug = self.get_company_slug()
        if not info: return []
        
        d = info['date']
        urls = []
        
        # 構造幾種可能的網址格式
        # 格式 1: /YYYY/MM/DD/slug-ticker-qx-year-earnings-call-transcript/
        base = f"https://www.fool.com/earnings/call-transcripts/{d.year}/{d.month:02d}/{d.day:02d}/"
        
        # 考慮到有些公司的 Year 會 +1 (像 NVDA)
        for y_offset in [0, 1]:
            for q_offset in ["q4", "q3", "q2", "q1"]:
                urls.append(f"{base}{slug}-{self.ticker_symbol.lower()}-{q_offset}-{info['year']+y_offset}-earnings-call-transcript/")
        
        return urls

    def scrape_transcript(self):
        """強攻逐字稿內容"""
        urls = self.construct_fool_urls()
        if not urls:
            return "❌ 無法推算財報網址，請確認 Ticker 是否正確。"

        print(f"🕵️ 正在嘗試構造的 {len(urls)} 個網址...")
        
        for url in urls:
            try:
                # 增加隨機延遲，模擬真人
                time.sleep(1.5)
                print(f"📡 嘗試抓取: {url}")
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    print("✅ 擊中目標！正在解析全文...")
                    soup = BeautifulSoup(response.text, 'html.parser')
                    content = soup.find('div', class_='article-body')
                    if content:
                        text = content.get_text(separator='\n')
                        clean_text = re.sub(r'\n\s*\n', '\n\n', text).strip()
                        if len(clean_text) > 1000:
                            return clean_text
                elif response.status_code == 429:
                    print("🛑 遭到 429 限流，請更換 User-Agent 或稍後再試。")
                    return "❌ 被網站擋住了 (429 Too Many Requests)。"
            except Exception as e:
                continue
                
        return "❌ 找遍了所有可能的網址構造，都沒看到逐字稿。可能還沒發布，或 Slug 不對。"

if __name__ == "__main__":
    # 以 TSLA 為例，因為它的 Slug 很穩定
    engine = TranscriptEngine("TSLA")
    result = engine.scrape_transcript()
    print(f"\n--- 抓取結果 (前 500 字) ---\n{result[:500]}...")
