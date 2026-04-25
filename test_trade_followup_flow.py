import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import engine_portfolio
import fubon
from src import database
from src import bot as bot_module
from src import scheduler as scheduler_runtime


class TradeFollowupFlowTests(unittest.TestCase):
    def setUp(self):
        engine_portfolio.init_db()
        with database.locked_connection() as conn:
            conn.execute("DELETE FROM trade_followups")
            conn.execute("DELETE FROM trade_log")
            conn.commit()
        self.original_bot = bot_module.bot
        self.original_user_id = bot_module.AUTHORIZED_USER_ID
        bot_module.bot = None
        bot_module.AUTHORIZED_USER_ID = None

    def tearDown(self):
        bot_module.bot = self.original_bot
        bot_module.AUTHORIZED_USER_ID = self.original_user_id

    def test_prompt_marks_followup_as_prompted(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2330", "sync_buy", "pending", "pending", "Broker sync detected", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        fake_bot = SimpleNamespace(send_message=Mock())
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123

        sent_count = bot_module.send_pending_trade_followups()

        self.assertEqual(sent_count, 1)
        fake_bot.send_message.assert_called_once()
        sent_text = fake_bot.send_message.call_args.args[1]
        self.assertIn("2330", sent_text)
        self.assertIn("原因", sent_text)
        self.assertIn("目標", sent_text)

        with database.locked_connection() as conn:
            row = conn.execute(
                "SELECT status, prompt_state FROM trade_followups WHERE id = ?",
                (followup_id,),
            ).fetchone()

        self.assertEqual(row, ("pending", "prompted"))

    def test_pending_followups_are_claimed_before_send(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2330", "sync_buy", "pending", "pending", "Broker sync detected", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        observed_states = []

        def send_message_side_effect(chat_id, text):
            with database.locked_connection() as conn:
                observed_states.append(
                    conn.execute(
                        "SELECT prompt_state FROM trade_followups WHERE id = ?",
                        (followup_id,),
                    ).fetchone()[0]
                )

        fake_bot = SimpleNamespace(send_message=Mock(side_effect=send_message_side_effect))
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123

        sent_count = bot_module.send_pending_trade_followups()

        self.assertEqual(sent_count, 1)
        self.assertEqual(observed_states, ["sending"])
        with database.locked_connection() as conn:
            row = conn.execute(
                "SELECT prompt_state FROM trade_followups WHERE id = ?",
                (followup_id,),
            ).fetchone()[0]
        self.assertEqual(row, "prompted")

    def test_send_failure_returns_claimed_followup_to_pending(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2330", "sync_buy", "pending", "pending", "Broker sync detected", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        observed_states = []

        def send_message_side_effect(chat_id, text):
            with database.locked_connection() as conn:
                observed_states.append(
                    conn.execute(
                        "SELECT prompt_state FROM trade_followups WHERE id = ?",
                        (followup_id,),
                    ).fetchone()[0]
                )
            raise RuntimeError("telegram down")

        fake_bot = SimpleNamespace(send_message=Mock(side_effect=send_message_side_effect))
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123

        with self.assertRaises(RuntimeError):
            bot_module.send_pending_trade_followups()

        self.assertEqual(observed_states, ["sending"])
        with database.locked_connection() as conn:
            row = conn.execute(
                "SELECT prompt_state FROM trade_followups WHERE id = ?",
                (followup_id,),
            ).fetchone()[0]
        self.assertEqual(row, "pending")

    def test_batch_send_failure_reverts_remaining_claimed_followups(self):
        with database.locked_connection() as conn:
            followup_ids = []
            for symbol in ("2330", "2317", "2454"):
                conn.execute(
                    "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                    (symbol, "sync_buy", 520.0, 50.0),
                )
                trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
                conn.execute(
                    """
                    INSERT INTO trade_followups (
                        trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (trade_log_id, symbol, "sync_buy", "pending", "pending", f"Followup for {symbol}", 0),
                )
                followup_ids.append(conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0])
            conn.commit()

        call_count = 0

        def send_message_side_effect(chat_id, text):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("telegram down")

        fake_bot = SimpleNamespace(send_message=Mock(side_effect=send_message_side_effect))
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123

        with self.assertRaises(RuntimeError):
            bot_module.send_pending_trade_followups()

        with database.locked_connection() as conn:
            rows = [
                conn.execute(
                    "SELECT prompt_state FROM trade_followups WHERE id = ?",
                    (followup_id,),
                ).fetchone()[0]
                for followup_id in followup_ids
            ]

        self.assertEqual(rows, ["prompted", "pending", "pending"])

    def test_fubon_sync_job_sends_prompt_for_new_pending_followup(self):
        fake_result = {"synced": True, "event_count": 1, "followup_count": 1, "events": [], "message": "ok"}

        with patch.object(fubon, "fubon_ready", True), patch.object(
            scheduler_runtime, "backup_database", return_value=None
        ), patch.object(
            scheduler_runtime, "daily_portfolio_review", return_value=None
        ), patch.object(
            scheduler_runtime, "macro_brain_heartbeat", return_value=None
        ), patch.object(
            scheduler_runtime, "auto_v_turn_monitor", return_value=None
        ), patch.object(
            scheduler_runtime, "daily_nlp_scout", return_value=None
        ), patch.object(
            engine_portfolio, "sync_fubon_portfolio_state", return_value=fake_result
        ) as mock_sync, patch.object(
            bot_module, "send_pending_trade_followups", return_value=1
        ) as mock_send:
            result = scheduler_runtime.fubon_portfolio_sync(source="scheduler")

        self.assertEqual(result, fake_result)
        mock_sync.assert_called_once_with(source="scheduler", sync_memory=True)
        mock_send.assert_called_once_with()

    def test_handle_all_text_records_skip_for_pending_followup(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2330", "sync_buy", "pending", "prompted", "Broker sync detected", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        fake_bot = SimpleNamespace(
            reply_to=Mock(return_value=SimpleNamespace(message_id=999)),
            edit_message_text=Mock(),
            send_chat_action=Mock(),
        )
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123
        message = SimpleNamespace(text="跳過", chat=SimpleNamespace(id=321), from_user=SimpleNamespace(id=123))

        with patch.object(bot_module, "ask_agent") as mock_ask_agent:
            bot_module.handle_all_text(message)

        mock_ask_agent.assert_not_called()
        fake_bot.send_chat_action.assert_not_called()
        fake_bot.reply_to.assert_called_once()
        confirmation_text = fake_bot.reply_to.call_args.args[1]
        self.assertIn("已略過", confirmation_text)

        with database.locked_connection() as conn:
            row = conn.execute(
                """
                SELECT status, prompt_state, skipped, user_reason, target_price, stop_price, responded_at
                FROM trade_followups WHERE id = ?
                """,
                (followup_id,),
            ).fetchone()

        self.assertEqual(row[0], "resolved")
        self.assertEqual(row[1], "resolved")
        self.assertEqual(row[2], 1)
        self.assertIsNone(row[3])
        self.assertIsNone(row[4])
        self.assertIsNone(row[5])
        self.assertIsNotNone(row[6])

    def test_handle_all_text_skip_clears_stale_followup_fields(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text,
                    user_reason, target_price, stop_price, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_log_id,
                    "2330",
                    "sync_buy",
                    "pending",
                    "prompted",
                    "Broker sync detected",
                    "舊理由",
                    650.0,
                    500.0,
                    0,
                ),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        fake_bot = SimpleNamespace(
            reply_to=Mock(return_value=SimpleNamespace(message_id=999)),
            edit_message_text=Mock(),
            send_chat_action=Mock(),
        )
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123
        message = SimpleNamespace(text="跳過", chat=SimpleNamespace(id=321), from_user=SimpleNamespace(id=123))

        with patch.object(bot_module, "ask_agent") as mock_ask_agent:
            bot_module.handle_all_text(message)

        mock_ask_agent.assert_not_called()
        with database.locked_connection() as conn:
            row = conn.execute(
                """
                SELECT status, prompt_state, skipped, user_reason, target_price, stop_price
                FROM trade_followups WHERE id = ?
                """,
                (followup_id,),
            ).fetchone()

        self.assertEqual(row[0], "resolved")
        self.assertEqual(row[1], "resolved")
        self.assertEqual(row[2], 1)
        self.assertIsNone(row[3])
        self.assertIsNone(row[4])
        self.assertIsNone(row[5])

    def test_handle_all_text_records_reason_target_and_stop(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2317", "sync_sell", 120.0, 10.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2317", "sync_sell", "pending", "prompted", "Need context", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        fake_bot = SimpleNamespace(
            reply_to=Mock(return_value=SimpleNamespace(message_id=999)),
            edit_message_text=Mock(),
            send_chat_action=Mock(),
        )
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123
        message = SimpleNamespace(
            text="因為籌碼轉弱，目標價 135，停損 118",
            chat=SimpleNamespace(id=321),
            from_user=SimpleNamespace(id=123),
        )

        with patch.object(bot_module, "ask_agent") as mock_ask_agent:
            bot_module.handle_all_text(message)

        mock_ask_agent.assert_not_called()
        fake_bot.send_chat_action.assert_not_called()
        confirmation_text = fake_bot.reply_to.call_args.args[1]
        self.assertIn("已記錄", confirmation_text)

        with database.locked_connection() as conn:
            row = conn.execute(
                """
                SELECT status, prompt_state, skipped, user_reason, target_price, stop_price, responded_at
                FROM trade_followups WHERE id = ?
                """,
                (followup_id,),
            ).fetchone()

        self.assertEqual(row[0], "resolved")
        self.assertEqual(row[1], "resolved")
        self.assertEqual(row[2], 0)
        self.assertEqual(row[3], "因為籌碼轉弱")
        self.assertEqual(row[4], 135.0)
        self.assertEqual(row[5], 118.0)
        self.assertIsNotNone(row[6])

    def test_handle_all_text_falls_through_for_unstructured_message_with_pending_followup(self):
        with database.locked_connection() as conn:
            conn.execute(
                "INSERT INTO trade_log (symbol, action, price, shares) VALUES (?, ?, ?, ?)",
                ("2330", "sync_buy", 520.0, 50.0),
            )
            trade_log_id = conn.execute("SELECT id FROM trade_log ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.execute(
                """
                INSERT INTO trade_followups (
                    trade_log_id, symbol, action, status, prompt_state, prompt_text, skipped
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trade_log_id, "2330", "sync_buy", "pending", "prompted", "Broker sync detected", 0),
            )
            followup_id = conn.execute("SELECT id FROM trade_followups ORDER BY id DESC LIMIT 1").fetchone()[0]
            conn.commit()

        fake_bot = SimpleNamespace(
            reply_to=Mock(return_value=SimpleNamespace(message_id=999)),
            edit_message_text=Mock(),
            send_chat_action=Mock(),
        )
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123
        message = SimpleNamespace(
            text="幫我看一下台積電今天有沒有新聞",
            chat=SimpleNamespace(id=321),
            from_user=SimpleNamespace(id=123),
        )

        with patch.object(bot_module, "ask_agent", return_value="agent reply") as mock_ask_agent:
            bot_module.handle_all_text(message)

        mock_ask_agent.assert_called_once()
        fake_bot.send_chat_action.assert_called_once()
        fake_bot.reply_to.assert_called_once()

        with database.locked_connection() as conn:
            row = conn.execute(
                """
                SELECT status, prompt_state, skipped, user_reason, target_price, stop_price, responded_at
                FROM trade_followups WHERE id = ?
                """,
                (followup_id,),
            ).fetchone()

        self.assertEqual(row[0], "pending")
        self.assertEqual(row[1], "prompted")
        self.assertEqual(row[2], 0)
        self.assertIsNone(row[3])
        self.assertIsNone(row[4])
        self.assertIsNone(row[5])
        self.assertIsNone(row[6])

    def test_handle_all_text_falls_through_without_pending_followup(self):
        fake_bot = SimpleNamespace(
            reply_to=Mock(return_value=SimpleNamespace(message_id=999)),
            edit_message_text=Mock(),
            send_chat_action=Mock(),
        )
        bot_module.bot = fake_bot
        bot_module.AUTHORIZED_USER_ID = 123
        message = SimpleNamespace(text="今天天氣不錯", chat=SimpleNamespace(id=321), from_user=SimpleNamespace(id=123))

        with patch.object(bot_module, "ask_agent", return_value="agent reply") as mock_ask_agent:
            bot_module.handle_all_text(message)

        mock_ask_agent.assert_called_once()
        fake_bot.send_chat_action.assert_called_once()
        fake_bot.reply_to.assert_called_once()


if __name__ == "__main__":
    unittest.main()
