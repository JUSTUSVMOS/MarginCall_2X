import asyncio
import os
from dotenv import load_dotenv
load_dotenv()
import fubon

fubon.init_fubon()

if fubon.fubon_ready:
    print("--- Technical ---")
    print(fubon.get_fubon_technical("2330"))
    print("--- Price Volumes ---")
    print(fubon.get_price_volumes("2330"))
    print("--- Intraday Trend ---")
    print(fubon.get_intraday_trend("2330"))
else:
    print("Fubon not ready")
