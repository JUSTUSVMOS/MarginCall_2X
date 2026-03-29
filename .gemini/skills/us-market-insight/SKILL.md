---
name: us-market-insight
description: 美股深度戰術監控技能。提供即時情緒、機構持倉、空頭數據及總經連動分析。
---

# 美股戰術戰情室 (US Market Insight)

## 🛠 核心工具與觸發時機

### 1. 盤中即時數據與情緒
- **工具**: `get_us_realtime_insight`
- **觸發時機**: 詢問「美股走勢」、「現在怎樣」、「情緒」、「目標價」、「買賣比」、「Volume Ratio」、「5分K」。
- **SOP**: 呼叫工具後，綜合分析目標價空間、買賣比 (Bid/Ask Ratio) 避險程度、成交量比率及 5分K (OHLC) 的短線力道。

### 2. 機構與空頭數據
- **工具**: `get_fundamental_data`
- **觸發時機**: 詢問「誰在買」、「籌碼」、「空頭」、「Short Ratio」。
- **SOP**: 檢查頂級機構持倉分布，並根據 Short Ratio 評估軋空風險。

### 3. 總經連動分析
- **觸發時機**: 詢問「大盤走勢」、「為什麼跌」、「NVDA/TSLA 等個股與大盤連動」。
- **SOP**: 呼叫 `get_market_sentiment` 檢查指數 (DJI, SPX, IXIC, RUT)、美元 (DX-Y)、美債 (^TNX) 及比特幣 (BTC) 的相互關係。

## ⚖️ 風格規範與防幻覺機制 (Anti-Hallucination)
1. **絕不通靈**: 如果工具回傳的數據中沒有你需要的數值（例如 P/C Ratio、特定指數點位），**嚴禁憑空捏造數字**，必須誠實回答「目前工具未提供此數據」。
2. **來源標記**: 所有價格需保留「(來源: FMP)」或「(來源: YF)」標籤。
3. **依據數據給建議**: 只能基於實際抓取到的買賣比 (Bid/Ask Ratio) 或 Short Ratio 給出具體的風險警示。

## 📥 輸入規範
- 美股代碼直接傳入（例如: NVDA, TSLA）。
- 分析時優先參考 `get_market_history` 取得最近 2 天的歷史數據對照。
