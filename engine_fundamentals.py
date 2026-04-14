import yfinance as yf
from yf_session import get_ticker, get_download
import pandas as pd
import logging
from typing import Dict, Any, Optional
from src.tools import tool

logger = logging.getLogger(__name__)

class FundamentalEngine:
    """
    Alice-Level 全武裝基本面分析引擎 (深度時間序列與預估版)。
    嚴格執行：100% 涵蓋 OpenAlice 要求的模組，並補齊資產/現金流的動態趨勢(QoQ)與 EPS 未來預估/歷史驚喜。
    """
    def __init__(self, symbol: str):
        self.symbol = symbol.upper().replace('.', '-')
        if self.symbol.isdigit() and len(self.symbol) <= 6:
            self.symbol += ".TW"
        self.ticker = get_ticker(self.symbol, cache_level="daily")
        self._info = None

    @property
    def info(self) -> dict:
        if self._info is None:
            try:
                self._info = self.ticker.info
            except:
                self._info = {}
        return self._info

    def _extract_last_5(self, df: Optional[pd.DataFrame], row_name: str) -> str:
        """安全擷取最新一季的數據與 QoQ 變化，供 LLM 分析 (壓縮版)"""
        if df is None or df.empty or row_name not in df.index:
            return "N/A"
        try:
            vals = df.loc[row_name].head(2).tolist()
            if not vals or pd.isna(vals[0]):
                return "N/A"
            latest = vals[0]
            if len(vals) > 1 and pd.notna(vals[1]) and vals[1] != 0:
                qoq = ((latest - vals[1]) / abs(vals[1])) * 100
                if abs(latest) >= 1e9:
                    return f"{latest/1e9:.1f}B ({qoq:+.1f}% QoQ)"
                else:
                    return f"{latest/1e6:.1f}M ({qoq:+.1f}% QoQ)"
            else:
                if abs(latest) >= 1e9:
                    return f"{latest/1e9:.1f}B"
                else:
                    return f"{latest/1e6:.1f}M"
        except:
            return "N/A"

    def _extract_last_5_ratio(self, df_num: Optional[pd.DataFrame], row_num: str, df_den: Optional[pd.DataFrame], row_den: str) -> str:
        """安全計算兩行數據的相除比率，並回傳最新一季與 QoQ 變化 (壓縮版)"""
        if df_num is None or df_den is None or df_num.empty or df_den.empty:
            return "N/A"
        if row_num not in df_num.index or row_den not in df_den.index:
            return "N/A"
        try:
            # 確保兩者的欄位(時間)對齊
            common_cols = df_num.columns.intersection(df_den.columns)
            if common_cols.empty:
                return "N/A"

            num_vals = df_num.loc[row_num, common_cols].head(2)
            den_vals = df_den.loc[row_den, common_cols].head(2)

            res_arr = []
            for n, d in zip(num_vals, den_vals):
                if pd.isna(n) or pd.isna(d) or d == 0:
                    res_arr.append(None)
                else:
                    res_arr.append((n/d)*100)

            if not res_arr or res_arr[0] is None:
                return "N/A"
            latest = res_arr[0]
            if len(res_arr) > 1 and res_arr[1] is not None:
                qoq = latest - res_arr[1]
                return f"{latest:.1f}% ({qoq:+.1f}% QoQ)"
            else:
                return f"{latest:.1f}%"
        except:
            return "N/A"
    # ==========================================
    # 保留最原始的方法 (回傳 DataFrame)，防止破壞其他依賴
    # ==========================================
    def get_income_statement(self, freq: str = "yearly") -> Optional[pd.DataFrame]:
        return self.ticker.income_stmt if freq == "yearly" else self.ticker.quarterly_income_stmt

    def get_balance_sheet(self, freq: str = "yearly") -> Optional[pd.DataFrame]:
        return self.ticker.balance_sheet if freq == "yearly" else self.ticker.quarterly_balance_sheet

    def get_cash_flow(self, freq: str = "yearly") -> Optional[pd.DataFrame]:
        return self.ticker.cashflow if freq == "yearly" else self.ticker.quarterly_cashflow

    # ==========================================
    # OpenAlice 對標 1. 公司概況 (Profile)
    # ==========================================
    def get_company_profile(self) -> Dict[str, Any]:
        info = self.info
        return {
            "公司名稱": info.get('longName', info.get('shortName', 'N/A')),
            "產業": f"{info.get('sector', 'N/A')} / {info.get('industry', 'N/A')}",
            "CEO": info.get('companyOfficers', [{}])[0].get('name', 'N/A') if info.get('companyOfficers') else 'N/A',
            "員工數": f"{info.get('fullTimeEmployees', 0):,}" if info.get('fullTimeEmployees') else 'N/A',
            "官方網站": info.get('website', 'N/A'),
            "公司簡介": info.get('longBusinessSummary', '無描述')[:150] + "..."
        }

    # ==========================================
    # OpenAlice 對標 2. 關鍵估值指標 (Metrics)
    # ==========================================
    def get_valuation_metrics(self) -> Dict[str, Any]:
        info = self.info
        inst_own = info.get('heldPercentInstitutions')
        return {
            "市值": f"{info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else 'N/A',
            "本益比(PE)": info.get('trailingPE', 'N/A'),
            "預估PE(Forward)": info.get('forwardPE', 'N/A'),
            "市淨率(PB)": info.get('priceToBook', 'N/A'),
            "企業倍數(EV/EBITDA)": info.get('enterpriseToEbitda', 'N/A'),
            "股息殖利率": f"{info.get('dividendYield', 0)*100:.2f}%" if info.get('dividendYield') else "0.00%",
            "空頭回補天數": info.get('shortRatio', 'N/A'),
            "機構持倉比": f"{inst_own*100:.1f}%" if inst_own is not None else "N/A"
        }

    # ==========================================
    # OpenAlice 對標 3. 財務報表 (Financial Statements) + 5 期時間序列陣列
    # ==========================================
    def get_financial_statements_summary(self) -> Dict[str, Any]:
        try:
            inc = self.ticker.quarterly_income_stmt
            bal = self.ticker.quarterly_balance_sheet
            cf = self.ticker.quarterly_cashflow
            
            res = {}
            if inc is not None and not inc.empty:
                rev_arr = self._extract_last_5(inc, 'Total Revenue')
                gp_arr = self._extract_last_5(inc, 'Gross Profit')
                ni_key = 'Net Income' if 'Net Income' in inc.index else ('Net Income Common Stockholders' if 'Net Income Common Stockholders' in inc.index else None)
                ni_arr = self._extract_last_5(inc, ni_key) if ni_key else "N/A"
                res['損益表'] = f"營收: {rev_arr} | 毛利: {gp_arr} | 淨利: {ni_arr}"
            
            if bal is not None and not bal.empty:
                cash_arr = self._extract_last_5(bal, 'Cash And Cash Equivalents')
                debt_arr = self._extract_last_5(bal, 'Total Debt')
                assets_arr = self._extract_last_5(bal, 'Total Assets')
                res['資產負債表'] = f"現金: {cash_arr} | 總債務: {debt_arr} | 總資產: {assets_arr}"

            if cf is not None and not cf.empty:
                ocf_arr = self._extract_last_5(cf, 'Operating Cash Flow')
                fcf_arr = self._extract_last_5(cf, 'Free Cash Flow')
                res['現金流量表'] = f"營運現金流: {ocf_arr} | 自由現金流: {fcf_arr}"

            return res
        except Exception as e:
            return {"error": f"報表解析失敗: {e}"}

    # ==========================================
    # OpenAlice 對標 4. 財務比率 (Ratios) - 升級為 5 季歷史陣列
    # ==========================================
    def get_quality_ratios(self) -> Dict[str, Any]:
        info = self.info
        res = {
            "最新ROE": f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
            "最新毛利率": f"{info.get('grossMargins', 0)*100:.2f}%" if info.get('grossMargins') else 'N/A',
            "最新淨利率": f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
            "流動比率(Current Ratio)": info.get('currentRatio', 'N/A'),
            "債務權益比(D/E)": info.get('debtToEquity', 'N/A')
        }
        
        # 動態計算 5 季歷史比率陣列
        inc = self.ticker.quarterly_income_stmt
        bal = self.ticker.quarterly_balance_sheet
        
        if inc is not None and not inc.empty:
            ni_key = 'Net Income' if 'Net Income' in inc.index else ('Net Income Common Stockholders' if 'Net Income Common Stockholders' in inc.index else None)
            
            # 毛利率 (Gross Profit / Total Revenue)
            res['毛利率(近5季)'] = self._extract_last_5_ratio(inc, 'Gross Profit', inc, 'Total Revenue')
            # 營業利益率 (Operating Income / Total Revenue)
            if 'Operating Income' in inc.index:
                res['營業利益率(近5季)'] = self._extract_last_5_ratio(inc, 'Operating Income', inc, 'Total Revenue')
            # 淨利率 (Net Income / Total Revenue)
            if ni_key:
                res['淨利率(近5季)'] = self._extract_last_5_ratio(inc, ni_key, inc, 'Total Revenue')
                
            # ROE & ROA (需結合資產負債表)
            if bal is not None and not bal.empty:
                eq_key = 'Stockholders Equity' if 'Stockholders Equity' in bal.index else ('Total Equity Gross Minority Interest' if 'Total Equity Gross Minority Interest' in bal.index else None)
                if eq_key and ni_key:
                    res['ROE權益報酬率(近5季)'] = self._extract_last_5_ratio(inc, ni_key, bal, eq_key)
                
                if 'Total Assets' in bal.index and ni_key:
                    res['ROA資產報酬率(近5季)'] = self._extract_last_5_ratio(inc, ni_key, bal, 'Total Assets')
                    
                # 流動比率 (Current Assets / Current Liabilities)
                if 'Current Assets' in bal.index and 'Current Liabilities' in bal.index:
                    # 這是倍數不是百分比，需要另外處理
                    try:
                        ca = bal.loc['Current Assets'].head(5)
                        cl = bal.loc['Current Liabilities'].head(5)
                        cr_arr = [f"{n/d:.2f}x" if pd.notna(n) and pd.notna(d) and d!=0 else "N/A" for n, d in zip(ca, cl)]
                        res['流動比率(近5季)'] = "[" + ", ".join(cr_arr) + "]"
                    except: pass

        return res

    # ==========================================
    # [新增] OpenAlice 對標： EPS 預估與財報驚喜 (Estimates & Surprises)
    # ==========================================
    def get_earnings_and_estimates(self) -> Dict[str, Any]:
        info = self.info
        res = {
            "近四季EPS(Trailing)": info.get('trailingEps', 'N/A'),
            "預估EPS(Forward)": info.get('forwardEps', 'N/A')
        }
        
        try:
            dates = self.ticker.earnings_dates
            if dates is not None and not dates.empty:
                # 過濾出已經發布的財報 (確保有 Reported EPS)
                past_earnings = dates.dropna(subset=['Reported EPS'])
                if not past_earnings.empty:
                    latest = past_earnings.iloc[0]
                    est = latest.get('Estimate', 'N/A')
                    rep = latest.get('Reported EPS', 'N/A')
                    surp = latest.get('Surprise(%)', 0)
                    # 處理百分比顯示 (有些來源給小數，有些給整數)
                    surp_pct = surp * 100 if isinstance(surp, (int, float)) and abs(surp) < 10 else surp
                    
                    res['最新財報驚喜'] = f"預估: {est} | 實際: {rep} | 驚喜差距: {surp_pct:+.2f}%" if isinstance(surp_pct, (int, float)) else f"預估: {est} | 實際: {rep}"
                else:
                    res['最新財報驚喜'] = "無歷史開獎資料"
        except:
            res['最新財報驚喜'] = "解析失敗"
            
        return res

    # ==========================================
    # OpenAlice 對標 5. 日曆事件 (與原始的最新新聞/評等)
    # ==========================================
    def get_events_and_opinions(self) -> Dict[str, Any]:
        res = {}
        try:
            dates = self.ticker.earnings_dates
            res['預計財報日'] = dates.index[0].strftime('%Y-%m-%d') if dates is not None and not dates.empty else "未知"
        except: res['預計財報日'] = "未知"
        
        try:
            recs = self.ticker.recommendations
            if recs is not None and not recs.empty:
                latest = recs.iloc[-1]
                res['分析師共識'] = f"{latest.get('strongBuy', 0)}強買, {latest.get('buy', 0)}買, {latest.get('hold', 0)}持有, {latest.get('sell', 0)}賣"
            else:
                res['分析師共識'] = "無資料"
        except: res['分析師共識'] = "解析失敗"

        try:
            news = self.ticker.news
            if news:
                res['最新頭條'] = f"[{news[0].get('publisher', 'N/A')}] {news[0].get('title', 'N/A')}"
        except: pass

        return res

    # ==========================================
    # OpenAlice 對標 6. 內部人士交易 (Insider Trading)
    # ==========================================
    def get_insider_trading(self) -> str:
        try:
            insider = self.ticker.insider_transactions
            if insider is not None and not insider.empty:
                recent = insider.head(3)
                lines = []
                for idx, row in recent.iterrows():
                    date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx).split(' ')[0]
                    name = row.get('Insider Purchases', row.get('Name', '高管'))
                    shares = row.get('Shares', 0)
                    val = row.get('Value', 0)
                    lines.append(f"{date_str} | {name} | 股數: {shares} | 價值: {val}")
                return "\n    ".join(lines)
            return "近期無申報紀錄"
        except Exception as e:
            return "數據獲取失敗 (台股不適用或無資料)"

    def get_full_fundamental_report(self) -> str:
        profile = self.get_company_profile()
        val = self.get_valuation_metrics()
        stmts = self.get_financial_statements_summary()
        ratios = self.get_quality_ratios()
        ests = self.get_earnings_and_estimates()
        events = self.get_events_and_opinions()
        insider = self.get_insider_trading()

        report = f"💎 === {self.symbol} 深度基本面全武裝 (含時間序列與預估) ===\n"
        report += f"【🏢 1. 公司概況】\n- 名稱: {profile['公司名稱']} | 產業: {profile['產業']}\n- CEO: {profile['CEO']} | 員工數: {profile['員工數']} | 網站: {profile.get('官方網站')}\n- 簡介: {profile['公司簡介']}\n"
        
        report += f"【💰 2. 關鍵估值】\n- 市值: {val['市值']} | PE: {val['本益比(PE)']} | 預估PE: {val['預估PE(Forward)']}\n- PB: {val['市淨率(PB)']} | EV/EBITDA: {val['企業倍數(EV/EBITDA)']} | 股息: {val['股息殖利率']}\n- 空頭回補天數: {val['空頭回補天數']} | 機構持倉: {val['機構持倉比']}\n"
        
        report += f"【📊 3. 財務報表 (動態 QoQ 追蹤)】\n"
        if "error" not in stmts:
            report += f"- {stmts.get('損益表', '無損益表')}\n- {stmts.get('資產負債表', '無資產負債表')}\n- {stmts.get('現金流量表', '無現金流量表')}\n"
        else:
            report += f"- {stmts['error']}\n"
            
        report += f"【⚖️ 4. 財務比率 (動態 5 季歷史)】\n- 最新指標: ROE {ratios.get('最新ROE', 'N/A')} | 毛利率 {ratios.get('最新毛利率', 'N/A')} | 淨利率 {ratios.get('最新淨利率', 'N/A')}\n- 毛利率(近5季): {ratios.get('毛利率(近5季)', 'N/A')}\n- 營業利益率(近5季): {ratios.get('營業利益率(近5季)', 'N/A')}\n- 淨利率(近5季): {ratios.get('淨利率(近5季)', 'N/A')}\n- ROE(近5季): {ratios.get('ROE權益報酬率(近5季)', 'N/A')}\n- 流動比率(近5季): {ratios.get('流動比率(近5季)', 'N/A')}\n- 債務權益比(D/E): {ratios.get('債務權益比(D/E)', 'N/A')}\n"
        
        report += f"【🎯 5. 華爾街預估與歷史開獎 (Estimates & Surprises)】\n- 近四季EPS: {ests['近四季EPS(Trailing)']} | 預估EPS: {ests['預估EPS(Forward)']}\n- 上季開獎: {ests.get('最新財報驚喜', '無資料')}\n"
        
        report += f"【📅 6. 日曆與事件】\n- 預計財報日: {events.get('預計財報日')}\n- 分析師共識: {events.get('分析師共識')}\n- 最新頭條: {events.get('最新頭條', '無')}\n"
        
        report += f"【🕵️ 7. 內部人士交易】\n    {insider}\n"
        
        return report

@tool()
def get_deep_fundamentals(symbol: str) -> str:
    """
    Retrieves a comprehensive fundamental analysis report for a stock.
    Includes company profile, valuation metrics, financial statements (5-quarter trend),
    earnings estimates, and insider trading activity.
    """
    return FundamentalEngine(symbol).get_full_fundamental_report()

if __name__ == "__main__":
    print(get_deep_fundamentals("NVDA"))
