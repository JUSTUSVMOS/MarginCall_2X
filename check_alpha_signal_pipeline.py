import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

import engine_router
import nlp_worker
from src import agent as agent_module


class TrinityAlphaLogicTests(unittest.TestCase):
    def test_score_sentiment_groups_keeps_dimensions_independent(self):
        groups = {
            "SEC": ["SEC: subpoena risk called out in filing"],
            "Macro": ["Macro: analysts cut estimates on slowing demand"],
            "Retail": [
                "Reddit: to the moon",
                "StockTwits: this thing only goes up",
                "Reddit: gamma squeeze incoming",
                "StockTwits: loading calls",
                "Reddit: diamond hands forever",
            ],
        }

        def fake_extract(text, _symbol):
            if text.startswith("SEC:"):
                return {
                    "institutional": {"sentiment": "strong_bearish", "insights": ["sec-risk"]},
                    "retail": {"sentiment": "strong_bullish", "insights": ["ignored-sec-retail"]},
                }
            if text.startswith("Macro:"):
                return {
                    "institutional": {"sentiment": "mild_bearish", "insights": ["macro-headwind"]},
                    "retail": {"sentiment": "strong_bullish", "insights": ["ignored-macro-retail"]},
                }
            return {
                "institutional": {"sentiment": "strong_bullish", "insights": ["ignored-retail-inst"]},
                "retail": {"sentiment": "strong_bullish", "insights": ["retail-hype"]},
            }

        with patch.object(nlp_worker, "extract_insight_parallel", side_effect=fake_extract):
            scores, tags = nlp_worker.score_sentiment_groups(groups, "TEST")

        self.assertEqual(scores["sec"], -1.0)
        self.assertEqual(scores["macro"], -0.3)
        self.assertEqual(scores["retail"], -0.3)
        self.assertEqual(tags["SEC"], ["sec-risk"])
        self.assertEqual(tags["Macro"], ["macro-headwind"])
        self.assertEqual(tags["Retail"], ["retail-hype"])

    def test_compose_alpha_signal_reweights_missing_dimensions(self):
        self.assertEqual(nlp_worker.compose_alpha_signal(-1.0, 1, 0.0, 0, 0.0, 0), -1.0)

        blended = nlp_worker.compose_alpha_signal(-1.0, 1, -0.3, 3, 0.0, 0)
        expected = ((-1.0 * 0.45) + (-0.3 * 0.30)) / (0.45 + 0.30)
        self.assertAlmostEqual(blended, expected, places=4)


class RouterPayloadTests(unittest.TestCase):
    def test_decode_nlp_summary_payload_supports_v2_and_legacy_rows(self):
        payload = json.dumps(
            {
                "signal_pack": {"sec_stance": "bearish", "nuclear_alert": True},
                "semantic_summary": "SEC risk summary",
            },
            ensure_ascii=False,
        )

        signal_pack, semantic_summary = engine_router._decode_nlp_summary_payload(payload)
        legacy_pack, legacy_summary = engine_router._decode_nlp_summary_payload("legacy summary text")

        self.assertEqual(signal_pack["sec_stance"], "bearish")
        self.assertTrue(signal_pack["nuclear_alert"])
        self.assertEqual(semantic_summary, "SEC risk summary")
        self.assertIsNone(legacy_pack)
        self.assertEqual(legacy_summary, "legacy summary text")

    def test_fetch_strat_data_keeps_nlp_score_immutable_and_adds_leading_indicators(self):
        seed_nlp = {
            "nlp_alpha": -0.55,
            "alpha_official": -0.8,
            "signal_pack": {"divergence": "無", "nuclear_alert": False},
        }
        fake_ticker = SimpleNamespace(history=lambda period, interval=None: pd.DataFrame({"Close": [10.0, 9.5]}))

        with patch.object(engine_router.market, "normalize_ticker", return_value="TEST"), patch.object(
            engine_router.market, "get_asset_profile", return_value={"asset_type": "Tech_Momentum"}
        ), patch.object(engine_router, "fetch_nlp_alpha", return_value=dict(seed_nlp)), patch.object(
            engine_router, "get_relative_move", return_value=("NORMAL", 0.0)
        ), patch.object(engine_router, "get_ticker", return_value=fake_ticker), patch.object(
            engine_router.risk, "calculate_buying_pressure", return_value=-0.6
        ), patch.object(
            engine_router.market,
            "build_technical_snapshot",
            return_value={
                "divergence": {"label": "⚪ 無明顯背離", "bearish_divergence": False},
                "adx": {"value": 18.4, "trend_regime": "ranging"},
                "obv": {"signal": "⚪ 量價中性"},
                "mtf_rsi": {"signal_label": "🟢 強超賣共振", "confluence_strength": 2, "signal_reliability": "HIGH"},
            },
        ), patch.object(engine_router.market, "build_technical_report", return_value="TECH"), patch.object(
            engine_router.market, "build_realtime_insight", return_value="P/C Ratio: 1.60"
        ), patch.object(
            engine_router.market,
            "build_option_volatility_context",
            return_value={"summary": "ATM IV 40.0% | RV30 25.0% | VRP +15.0pt (🔥 恐慌定價)", "signal": "🔥 恐慌定價", "vrp": 15.0},
        ), patch.object(engine_router, "_alert_callback", None):
            data = engine_router.fetch_strat_data("test")

        self.assertEqual(data["nlp_insights"]["nlp_alpha"], -0.55)
        self.assertEqual(data["leading_indicators"]["cvd_signal"], "🔴 拋壓")
        self.assertEqual(data["leading_indicators"]["pc_signal"], "🔴 避險")
        self.assertEqual(data["leading_indicators"]["pc_ratio"], 1.6)
        self.assertEqual(data["leading_indicators"]["signal_reliability"], "HIGH")
        self.assertIn("恐慌避險定價", data["leading_indicators"]["pc_context"])

    def test_get_strat_context_includes_leading_indicators(self):
        fake_data = {
            "asset_type": "Tech_Momentum",
            "metrics": {"technical_analysis": "TECH"},
            "leading_indicators": {"cvd": -0.6, "pc_signal": "🔴 避險"},
            "relative_move": {"risk_type": "NORMAL", "excess_return": 0.0},
            "nlp_insights": {"nlp_alpha": -0.4},
        }

        with patch.object(engine_router, "detect_symbols", return_value=["TEST"]), patch.object(
            engine_router, "fetch_strat_data", return_value=fake_data
        ):
            context = engine_router.get_strat_context("看看 TEST")

        self.assertIn('"leading_indicators":{"cvd":-0.6,"pc_signal":"🔴 避險"}', context)
        self.assertIn('"relative_move":{"risk_type":"NORMAL","excess_return":0.0}', context)


class AgentPromptTests(unittest.TestCase):
    def test_generate_final_report_uses_structured_signal_pack(self):
        captured = {}

        def fake_quick_call(prompt, **_kwargs):
            captured["prompt"] = prompt
            return "分析完成"

        nlp_data = {
            "nlp_alpha": -0.72,
            "signal_pack": {
                "sec_stance": "bearish",
                "sec_detail": ["subpoena disclosed", "insider sale picked up"],
                "macro_stance": "neutral",
                "macro_detail": ["macro mixed"],
                "retail_stance": "bullish",
                "retail_detail": ["retail still chasing"],
                "divergence": "⚠️ 散戶情緒看多 vs SEC 官方偏空 -> 散戶陷阱風險",
                "nuclear_alert": True,
                "source_counts": {"sec": 2, "macro": 3, "retail": 8},
            },
        }
        strat_data = {
            "metrics": {"technical_analysis": "bearish"},
            "leading_indicators": {
                "cvd": -0.6,
                "cvd_signal": "🔴 拋壓",
                "pc_ratio": 1.7,
                "pc_signal": "🔴 避險",
                "pc_context": "🟡 P/C 偏高且權利金昂貴，偏向恐慌避險定價",
                "volatility_context": "ATM IV 40.0% | RV30 25.0% | VRP +15.0pt (🔥 恐慌定價)",
                "mtf_rsi_signal": "🟢 強超賣共振",
                "mtf_rsi_strength": 2,
                "signal_reliability": "HIGH",
            },
        }

        with patch.object(agent_module, "quick_call", side_effect=fake_quick_call):
            result = agent_module.generate_final_report("TEST", strat_data, nlp_data)

        self.assertEqual(result, "分析完成")
        prompt = captured["prompt"]
        self.assertIn("SEC/官方立場: bearish", prompt)
        self.assertIn("subpoena disclosed", prompt)
        self.assertIn("核彈級警報已核實", prompt)
        self.assertIn("P/C Ratio: 1.70 🔴 避險", prompt)
        self.assertIn("P/C + 波動定價: 🟡 P/C 偏高且權利金昂貴，偏向恐慌避險定價", prompt)
        self.assertIn("多時間框 RSI: 🟢 強超賣共振 (強度 2 / 可靠度 HIGH)", prompt)
        self.assertIn("多空矛盾偵測: ⚠️ 散戶情緒看多 vs SEC 官方偏空 -> 散戶陷阱風險", prompt)


if __name__ == "__main__":
    unittest.main()
