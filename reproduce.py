from engine_technical import build_technical_snapshot
try:
    print(build_technical_snapshot("CRWV"))
except Exception as e:
    import traceback
    traceback.print_exc()
