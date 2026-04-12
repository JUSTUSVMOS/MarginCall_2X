from playwright.sync_api import sync_playwright
import time
import os

def generate_sa_auth():
    with sync_playwright() as p:
        # 啟動帶有介面的瀏覽器
        print("🚀 正在啟動瀏覽器...")
        browser = p.chromium.launch(headless=False) 
        
        # 模擬真實瀏覽器環境
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        )

        page = context.new_page()
        print("🔗 正在前往 Seeking Alpha 登入頁面...")
        page.goto("https://seekingalpha.com/login", wait_until="networkidle")

        print("\n" + "="*50)
        print("📢 請在彈出的瀏覽器視窗中完成登入。")
        print("💡 提示：支援 Google 登入或電郵登入。")
        print("✅ 登入成功後，腳本會自動偵測並儲存 auth.json。")
        print("="*50 + "\n")

        # 監控 URL 或特定元素來判斷登入成功
        # Seeking Alpha 登入後通常會跳轉到首頁或跟隨 r= 參數
        login_success = False
        timeout = 300  # 5 分鐘寬限期
        start_time = time.time()

        while time.time() - start_time < timeout:
            current_url = page.url
            # 判斷登入成功的標誌：不再是 login 頁面且包含某些登入後的特徵
            if "seekingalpha.com/login" not in current_url and "seekingalpha.com" in current_url:
                # 額外檢查：看看是否有登入後的 Cookie
                cookies = context.cookies()
                if any(c['name'] == 'machine_id' for c in cookies):
                    print(f"🎉 偵測到登入成功！當前頁面: {current_url}")
                    login_success = True
                    break
            
            time.sleep(2)
            if (int(time.time() - start_time) % 30 == 0):
                print(f"⏳ 正在等待登入... (剩餘 {int(timeout - (time.time() - start_time))} 秒)")

        if login_success:
            # 確保內容完全加載
            page.wait_for_timeout(3000)
            # 儲存狀態
            context.storage_state(path="auth.json")
            print("\n✅ [成功] 登入狀態已儲存至 auth.json")
            print("📦 您現在可以使用這個檔案進行自動化爬取了。")
        else:
            print("\n❌ [逾時] 未能偵測到登入成功的狀態。")

        browser.close()

if __name__ == "__main__":
    generate_sa_auth()
