from yf_session import get_ticker

def test():
    t = get_ticker("VOO")
    h1 = t.get_holdings()
    print(f"Fetch 1: {len(h1)} items")
    h2 = t.get_holdings()
    print(f"Fetch 2 (Cache): {len(h2)} items")
    assert len(h1) > 0 and h1 == h2
    print("Test passed!")

if __name__ == "__main__":
    test()
