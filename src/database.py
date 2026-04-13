import sqlite3
import threading
from contextlib import contextmanager

from config import DB_FILE


db_lock = threading.Lock()


def get_connection(check_same_thread: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_FILE), check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def get_db_connection(check_same_thread: bool = False) -> sqlite3.Connection:
    return get_connection(check_same_thread=check_same_thread)


@contextmanager
def locked_connection(check_same_thread: bool = False):
    with db_lock:
        conn = get_connection(check_same_thread=check_same_thread)
        try:
            yield conn
        finally:
            conn.close()
