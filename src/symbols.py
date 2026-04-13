MARKET_SUFFIXES = (
    ".TW",
    ".TWO",
    ".HK",
    ".SS",
    ".SZ",
    ".L",
    ".DE",
    ".AS",
    ".AX",
    ".T",
    ".PA",
    ".MI",
    ".TO",
    ".V",
)


def normalize_ticker(symbol: str) -> str:
    symbol = symbol.upper().strip()
    is_taiwan = any(char.isdigit() for char in symbol) and (
        len(symbol.replace(".TW", "").replace(".TWO", "")) <= 6
    )
    if not is_taiwan and "." in symbol and not symbol.endswith(MARKET_SUFFIXES):
        return symbol.replace(".", "-")
    return symbol
