import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import engine_memory
import engine_router
import engine_risk
import fubon
import src.bot as bot_module


VALID_NOTE = (
    "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"
    "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"
    "Portfolio Health: Keep leverage light until event risk clears.\n"
    "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
)


class FollowupProposalChecks(unittest.TestCase):
    def test_risk_model_keeps_fixed_dix_offset_inside_multiplicative_framework(self):
        _, low_plain, low_offset = engine_risk._score_risk_multiplier(1.6)
        _, low_dix, low_dix_offset = engine_risk._score_risk_multiplier(1.6, dix_support_active=True)
        _, high_plain, _ = engine_risk._score_risk_multiplier(2.4)
        _, high_dix, high_dix_offset = engine_risk._score_risk_multiplier(2.4, dix_support_active=True)

        self.assertEqual(low_offset, 0)
        self.assertEqual(low_dix_offset, -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(high_dix_offset, -engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(low_plain - low_dix, engine_risk.DIX_SUPPORT_OFFSET_POINTS)
        self.assertEqual(high_plain - high_dix, engine_risk.DIX_SUPPORT_OFFSET_POINTS)

    def test_frontal_lobe_commit_timestamp_matches_hashed_payload_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            original_paths = {
                "BRAIN_DIR": engine_memory.BRAIN_DIR,
                "BRAIN_FILE": engine_memory.BRAIN_FILE,
                "FRONTAL_LOBE_FILE": engine_memory.FRONTAL_LOBE_FILE,
                "EMOTION_FILE": engine_memory.EMOTION_FILE,
                "MARKET_REGIME_FILE": engine_memory.MARKET_REGIME_FILE,
                "HEARTBEAT_FILE": engine_memory.HEARTBEAT_FILE,
                "SNAPSHOT_FILE": engine_memory.SNAPSHOT_FILE,
            }
            try:
                engine_memory.BRAIN_DIR = root
                engine_memory.BRAIN_FILE = root / "commit.json"
                engine_memory.FRONTAL_LOBE_FILE = root / "frontal-lobe.md"
                engine_memory.EMOTION_FILE = root / "emotion-log.json"
                engine_memory.MARKET_REGIME_FILE = root / "market-regime.md"
                engine_memory.HEARTBEAT_FILE = root / "heartbeat.json"
                engine_memory.SNAPSHOT_FILE = root / "snapshot.json"

                brain = engine_memory.Brain()
                captured = {}
                real_generate_commit_hash = engine_memory.generate_commit_hash

                def capture_payload(payload):
                    captured["timestamp"] = payload["timestamp"]
                    return real_generate_commit_hash(payload)

                with patch.object(engine_memory, "generate_commit_hash", side_effect=capture_payload):
                    result = brain.update_frontal_lobe(VALID_NOTE)

                self.assertTrue(result["success"])
                self.assertEqual(brain.commits[-1]["timestamp"], captured["timestamp"])
                self.assertTrue(captured["timestamp"].endswith("Z"))
            finally:
                for name, value in original_paths.items():
                    setattr(engine_memory, name, value)

    def test_router_alerts_flow_through_callback_without_bot_global(self):
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 9.5]}))
        captured_alerts = []

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(
            engine_router, "fetch_nlp_alpha", return_value={"nlp_alpha": -0.2}
        ), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(
            engine_router, "get_ticker", return_value=fake_ticker
        ), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=-0.95
        ), patch.object(
            engine_router.market,
            "build_technical_snapshot",
            return_value={
                "divergence": {"label": "🔴 頂背離", "bearish_divergence": True},
                "adx": {"value": 31.2, "trend_regime": "trending"},
                "obv": {"signal": "📉 價跌量弱，空方主導"},
                "mtf_rsi": {"signal_label": "🔴 強過熱共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
            },
        ), patch.object(
            engine_router.market, "build_technical_report", return_value="TECH"
        ), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 1.60"
        ), patch.object(
            engine_router.market, "build_option_volatility_context", return_value={"summary": "N/A", "signal": "⚪ 無期權波動資料"}
        ):
            engine_router.set_alert_callback(lambda message: captured_alerts.append(message))
            data = engine_router.fetch_strat_data("test")
            engine_router.set_alert_callback(None)

        self.assertEqual(data["leading_indicators"]["rsi_divergence"], "🔴 頂背離")
        self.assertEqual(len(captured_alerts), 1)
        self.assertIn("硬體中斷", captured_alerts[0])

    def test_bot_runtime_registers_router_callback_to_telegram_delivery(self):
        fake_bot = SimpleNamespace(send_message=Mock())
        original_bot = bot_module.bot
        original_user = bot_module.AUTHORIZED_USER_ID
        original_runtime = bot_module._runtime_initialized
        original_callback = getattr(bot_module.router, "_alert_callback", None)
        try:
            bot_module.bot = fake_bot
            bot_module.AUTHORIZED_USER_ID = 456
            bot_module._runtime_initialized = False
            bot_module.router.set_alert_callback(None)

            with patch.object(fubon, "init_fubon") as mock_init_fubon, patch.object(
                bot_module.market, "set_fubon_provider"
            ) as mock_provider:
                bot_module.initialize_bot_runtime()
                bot_module.router._alert_callback("router alert")

            mock_init_fubon.assert_called_once_with()
            mock_provider.assert_called_once_with(fubon)
            fake_bot.send_message.assert_called_once_with(456, "router alert")
        finally:
            bot_module.bot = original_bot
            bot_module.AUTHORIZED_USER_ID = original_user
            bot_module._runtime_initialized = original_runtime
            bot_module.router.set_alert_callback(original_callback)


if __name__ == "__main__":
    unittest.main()
