import sys
sys.path.insert(0, '/home/margincaller/MarginCall_2X')
import engine_market as market
from yf_session import get_ticker

def test_new_logic(symbol):
    clean_sym = symbol.replace('.TW', '').replace('.TWO', '').replace('_ESOP', '').replace('_TRUST', '')
    query_sym = market._normalize_lookup_symbol(clean_sym)
    ticker = get_ticker(query_sym, cache_level="daily")
    name = ticker.info.get('shortName') or ticker.info.get('longName') or clean_sym
    print(f"Original: {symbol}, query_sym: {query_sym}, name: {name}")

test_new_logic("2330")
test_new_logic("00981A")
test_new_logic("2330.TW")
test_new_logic("TSM")
