"""
Tracked tests for trade_outcome_checkpoints schema and enqueue scaffold.

These tests use an isolated file-based SQLite DB (created per test, deleted in tearDown)
and patch get_connection / db_lock so they remain independent from the live portfolio.db.

init_db() calls conn.close() internally, so each patched get_connection call returns a
fresh connection to the same file, keeping init_db's lifecycle separate from the test's
own reader connection (self.conn).
"""

import os
import sqlite3
import unittest
from unittest.mock import MagicMock, patch

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


class TestTradeOutcomeCheckpointsSchema(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(_TESTS_DIR, f"_test_journal_{id(self)}.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.mock_lock = MagicMock()

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _get_conn(self, **kwargs):
        """Return a fresh connection to the test DB file each time (init_db closes it)."""
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_portfolio_db(self):
        import engine_portfolio
        with patch.object(engine_portfolio, "db_lock", self.mock_lock), \
             patch.object(engine_portfolio, "get_connection", self._get_conn):
            engine_portfolio.init_db()

    def test_trade_outcome_checkpoints_table_created(self):
        """init_db() must create the trade_outcome_checkpoints table."""
        self._init_portfolio_db()

        cur = self.conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trade_outcome_checkpoints'"
        )
        self.assertIsNotNone(
            cur.fetchone(),
            "trade_outcome_checkpoints table should exist after init_db()",
        )

    def test_trade_outcome_checkpoints_has_required_columns(self):
        """trade_outcome_checkpoints must include benchmark and sector price columns."""
        self._init_portfolio_db()

        cur = self.conn.cursor()
        cur.execute("PRAGMA table_info(trade_outcome_checkpoints)")
        cols = {row["name"] for row in cur.fetchall()}

        for required in (
            "id",
            "trade_log_id",
            "horizon_label",
            "symbol",
            "action",
            "entry_price",
            "due_date",
            "benchmark_symbol",
            "benchmark_entry_price",
            "sector_symbol",
            "sector_entry_price",
            "outcome_price",
            "status",
            "created_at",
        ):
            self.assertIn(required, cols, f"Column '{required}' is missing")

    def test_unique_index_on_trade_log_id_horizon(self):
        """A unique constraint on (trade_log_id, horizon_label) must be enforced."""
        self._init_portfolio_db()

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO trade_outcome_checkpoints
                (trade_log_id, horizon_label, symbol, action, entry_price, due_date,
                 benchmark_symbol, sector_symbol)
            VALUES (1, 'T+5', 'AAPL', 'buy', 150.0, '2025-01-10', 'SPY', 'XLK')
            """
        )
        self.conn.commit()

        with self.assertRaises(sqlite3.IntegrityError):
            cur.execute(
                """
                INSERT INTO trade_outcome_checkpoints
                    (trade_log_id, horizon_label, symbol, action, entry_price, due_date,
                     benchmark_symbol, sector_symbol)
                VALUES (1, 'T+5', 'AAPL', 'buy', 150.0, '2025-01-10', 'SPY', 'XLK')
                """
            )


class TestEnqueueTradeOutcomeCheckpoints(unittest.TestCase):
    def setUp(self):
        self.db_path = os.path.join(_TESTS_DIR, f"_test_journal_{id(self)}.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.mock_lock = MagicMock()

    def tearDown(self):
        self.conn.close()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _get_conn(self, **kwargs):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        return c

    def _init_db_and_insert_trade(self, action="buy", symbol="AAPL", price=150.0):
        import engine_portfolio
        with patch.object(engine_portfolio, "db_lock", self.mock_lock), \
             patch.object(engine_portfolio, "get_connection", self._get_conn):
            engine_portfolio.init_db()

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
            (symbol, action, price, 10),
        )
        self.conn.commit()
        return cur.lastrowid

    def _enqueue(self, trade_log_ids, mock_price=155.0):
        import engine_journal
        with patch.object(engine_journal, "db_lock", self.mock_lock), \
             patch.object(engine_journal, "get_connection", self._get_conn), \
             patch.object(engine_journal, "_load_price_on_or_after", return_value=mock_price):
            engine_journal.enqueue_trade_outcome_checkpoints(trade_log_ids)

    def test_buy_creates_t5_and_t20_checkpoints(self):
        """Enqueueing a buy trade must create exactly T+5 and T+20 rows."""
        trade_id = self._init_db_and_insert_trade(action="buy")
        self._enqueue([trade_id])

        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM trade_outcome_checkpoints WHERE trade_log_id = ?",
            (trade_id,),
        )
        rows = cur.fetchall()

        self.assertEqual(len(rows), 2, "Expected exactly 2 checkpoint rows for a buy trade")
        labels = {row["horizon_label"] for row in rows}
        self.assertEqual(labels, {"T+5", "T+20"})

    def test_checkpoint_rows_have_benchmark_and_sector(self):
        """Each checkpoint row must carry non-NULL benchmark and sector symbols/prices."""
        trade_id = self._init_db_and_insert_trade(action="buy")
        self._enqueue([trade_id], mock_price=200.0)

        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM trade_outcome_checkpoints WHERE trade_log_id = ?",
            (trade_id,),
        )
        for row in cur.fetchall():
            self.assertIsNotNone(row["benchmark_symbol"], "benchmark_symbol must not be NULL")
            self.assertIsNotNone(row["benchmark_entry_price"], "benchmark_entry_price must not be NULL")
            self.assertIsNotNone(row["sector_symbol"], "sector_symbol must not be NULL")

    def test_sync_buy_creates_checkpoints(self):
        """sync_buy is also an eligible action and must produce checkpoints."""
        trade_id = self._init_db_and_insert_trade(action="sync_buy")
        self._enqueue([trade_id])

        cur = self.conn.cursor()
        cur.execute(
            "SELECT count(*) as cnt FROM trade_outcome_checkpoints WHERE trade_log_id = ?",
            (trade_id,),
        )
        self.assertEqual(cur.fetchone()["cnt"], 2)

    def test_sell_action_skipped(self):
        """sell is not eligible; enqueueing it must produce zero checkpoint rows."""
        trade_id = self._init_db_and_insert_trade(action="sell")
        self._enqueue([trade_id])

        cur = self.conn.cursor()
        cur.execute(
            "SELECT count(*) as cnt FROM trade_outcome_checkpoints WHERE trade_log_id = ?",
            (trade_id,),
        )
        self.assertEqual(cur.fetchone()["cnt"], 0)

    def test_enqueue_idempotent_for_duplicate_call(self):
        """Calling enqueue twice for the same trade must not duplicate rows."""
        trade_id = self._init_db_and_insert_trade(action="buy")
        self._enqueue([trade_id])
        self._enqueue([trade_id])

        cur = self.conn.cursor()
        cur.execute(
            "SELECT count(*) as cnt FROM trade_outcome_checkpoints WHERE trade_log_id = ?",
            (trade_id,),
        )
        self.assertEqual(cur.fetchone()["cnt"], 2, "Duplicate enqueue must remain idempotent")

    def test_empty_list_is_noop(self):
        """Passing an empty list must not raise and must leave the table empty."""
        self._init_db_and_insert_trade(action="buy")
        self._enqueue([])

        cur = self.conn.cursor()
        cur.execute("SELECT count(*) as cnt FROM trade_outcome_checkpoints")
        self.assertEqual(cur.fetchone()["cnt"], 0)


if __name__ == "__main__":
    unittest.main()
