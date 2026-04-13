import os
import time
import json
import requests
import pandas as pd
import finnhub
from lxml import etree
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

class FinnhubNewsDownloader:
    """
    從 FinNLP 提取並簡化後的 Finnhub 新聞抓取器。
    整合了 FinNLP_Downloader -> News_Downloader -> Finnhub_Date_Range 的核心邏輯。
    """
    def __init__(self, token: str):
        self.token = token
        self.finnhub_client = finnhub.Client(api_key=token)
        self.dataframe = pd.DataFrame()

    def _request_get(self, url, headers=None, verify=None, params=None):
        if headers is None:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/112.0"
            }
        try:
            response = requests.get(url=url, headers=headers, verify=verify, params=params, timeout=10)
            if response.status_code == 200:
                return response
        except Exception as e:
            print(f"Request Error: {url} - {e}")
        return None

    def download_date_range_stock(self, start_date, end_date, stock="AAPL"):
        """抓取指定日期範圍內的新聞列表"""
        self.date_list = pd.date_range(start_date, end_date)
        self.dataframe = pd.DataFrame()

        days_each_time = 4
        date_list = self.date_list
        
        # 計算總步數
        total = (len(date_list) // days_each_time) + (1 if len(date_list) % days_each_time != 0 else 0)

        with tqdm(total=total, desc=f"Downloading {stock} Titles") as bar:
            while len(date_list):
                tmp_date_list = date_list[:days_each_time]
                date_list = date_list[days_each_time:]
                tmp_start_date = tmp_date_list[0].strftime("%Y-%m-%d")
                tmp_end_date = tmp_date_list[-1].strftime("%Y-%m-%d")
                res = self._gather_one_part(tmp_start_date, tmp_end_date, stock=stock)
                self.dataframe = pd.concat([self.dataframe, res])
                bar.update(1)

        if not self.dataframe.empty and "datetime" in self.dataframe.columns:
            self.dataframe.datetime = pd.to_datetime(self.dataframe.datetime, unit="s")
        self.dataframe = self.dataframe.reset_index(drop=True)

    def _gather_one_part(self, start_date, end_date, stock="AAPL", delay=1):
        res = self.finnhub_client.company_news(stock, _from=start_date, to=end_date)
        time.sleep(delay)
        return pd.DataFrame(res)

    def gather_content(self, delay=0.01):
        """抓取新聞內容 (內建各媒體 Xpath)"""
        if self.dataframe.empty:
            return
        pbar = tqdm(total=self.dataframe.shape[0], desc="Gathering news contents")
        self.dataframe["content"] = self.dataframe.apply(lambda x: self._gather_content_apply(x, pbar, delay), axis=1)

    def _gather_content_apply(self, x, pbar, delay=0.01):
        time.sleep(delay)
        url = x.url
        source = x.source
        response = self._request_get(url=url)
        pbar.update(1)
        
        if response is None:
            return "Connection Error"
        
        page = etree.HTML(response.text)
        try:
            if source == "Yahoo":
                xpath_query = "/html/body/div[3]/div[1]/div/main/div[1]/div/div/div/div/article/div/div/div/div/div/div[2]/div[4]"
                elements = page.xpath(xpath_query)
                if not elements: # Fallback for different Yahoo layouts
                    elements = page.xpath("//div[contains(@class, 'caas-body')]")
                content = elements[0].xpath(".//text()")
                return "\n".join(content)
            
            elif source == "Reuters":
                elements = page.xpath("/html/body/div[1]/div[3]/div/main/article/div[1]/div[2]/div/div/div[2]")
                if not elements:
                    elements = page.xpath("//div[contains(@class, 'article-body')]")
                content = elements[0].xpath(".//text()")
                return "\n".join(content)
            
            elif source == "SeekingAlpha" or source == "Seeking Alpha":
                elements = page.xpath("//div[@data-test-id='article-content']")
                if not elements:
                    elements = page.xpath("/html/body/div[2]/div/div[1]/main/div/div[2]/div/article/div/div/div[2]/div/section[1]/div/div/div")
                if elements:
                    content = elements[0].xpath(".//text()")
                    return "\n".join(content)
                return "Not supported yet (SA layout changed)"

            elif source == "MarketWatch":
                elements = page.xpath('//*[@id="js-article__body"]')
                if elements:
                    content = "".join(elements[0].xpath(".//text()"))
                    content = content.replace("  ", " ").replace("\n \n", " ").replace("\n  ", " ")
                    return content
                return "Not supported yet"

            elif source == "CNBC":
                elements = page.xpath("//div[contains(@class, 'ArticleBody-articleBody')]")
                if elements:
                    content = "\n".join(elements[0].xpath(".//text()"))
                    return content
                return "Not supported yet"
            
            # 其餘媒體維持原 FinNLP 邏輯...
            else:
                return "Not supported yet"
        except Exception:
            return "Parsing Error"

if __name__ == "__main__":
    # 簡單測試
    TOKEN = os.getenv("FINNHUB_API_KEY")
    if TOKEN:
        downloader = FinnhubNewsDownloader(TOKEN)
        downloader.download_date_range_stock("2024-04-10", "2024-04-12", stock="NVDA")
        print(downloader.dataframe.head())
    else:
        print("Please set FINNHUB_API_KEY for testing.")
