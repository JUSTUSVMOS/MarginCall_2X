import json
import tempfile
import unittest
from pathlib import Path

import engine_memory as memory


class BrainMemoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)

        self.original_paths = {
            "BRAIN_DIR": memory.BRAIN_DIR,
            "BRAIN_FILE": memory.BRAIN_FILE,
            "FRONTAL_LOBE_FILE": memory.FRONTAL_LOBE_FILE,
            "EMOTION_FILE": memory.EMOTION_FILE,
            "MARKET_REGIME_FILE": memory.MARKET_REGIME_FILE,
            "HEARTBEAT_FILE": memory.HEARTBEAT_FILE,
            "SNAPSHOT_FILE": memory.SNAPSHOT_FILE,
        }

        memory.BRAIN_DIR = self.temp_path
        memory.BRAIN_FILE = self.temp_path / "commit.json"
        memory.FRONTAL_LOBE_FILE = self.temp_path / "frontal-lobe.md"
        memory.EMOTION_FILE = self.temp_path / "emotion-log.json"
        memory.MARKET_REGIME_FILE = self.temp_path / "market-regime.md"
        memory.HEARTBEAT_FILE = self.temp_path / "heartbeat.json"
        memory.SNAPSHOT_FILE = self.temp_path / "snapshot.json"

    def tearDown(self):
        for name, value in self.original_paths.items():
            setattr(memory, name, value)
        self.tempdir.cleanup()

    def test_market_regime_persists_and_writes_readable_views(self):
        brain = memory.Brain()
        brain.update_frontal_lobe("Watch CPI and keep leverage light.")
        result = brain.update_market_regime(
            summary="高利率壓力仍在，但暗池承接提供短線緩衝。",
            regime="🟡 整理",
            risk_score=42,
            watchpoints=["CPI", "VIX", "SPX 20MA"],
            reasons=["⚠️ 殖利率曲線倒掛", "🟢 暗池吸籌"],
            signals={"yieldCurve10Y2Y": -0.21, "spx": 5200.4},
            source="unit_test",
            updated_at="2026-01-02T03:04:05Z"
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["changed"])

        reloaded = memory.Brain()
        market = reloaded.get_market_regime(max_age_minutes=999999)
        self.assertEqual(reloaded.get_frontal_lobe(), "Watch CPI and keep leverage light.")
        self.assertEqual(market["state"], "🟡 整理")
        self.assertEqual(market["riskScore"], 42)
        self.assertEqual(market["source"], "unit_test")
        self.assertIn("高利率壓力仍在", market["summary"])
        self.assertTrue(memory.MARKET_REGIME_FILE.exists())
        self.assertIn("Persistent Macro Regime", memory.MARKET_REGIME_FILE.read_text(encoding="utf-8"))
        self.assertTrue(memory.SNAPSHOT_FILE.exists())

    def test_sync_market_snapshot_deduplicates_identical_heartbeat_updates(self):
        snapshot = {
            "generatedAt": "2026-01-02T03:04:05Z",
            "riskScore": 55,
            "state": "🔴 警戒",
            "summary": "風險進入警戒帶，先控槓桿。",
            "reasons": ["🔴 資金緊縮", "📰 新聞極度偏空"],
            "signals": {
                "yieldCurve10Y2Y": -0.33,
                "gexBillions": -1.4,
                "sentimentLabel": "偏空",
                "sentimentScore": -0.5,
                "spx": 5001.1,
                "spx20Ma": 5050.0,
                "spx200Ma": 5200.0
            }
        }

        brain = memory.Brain()
        first = brain.sync_market_snapshot(snapshot, source="test_heartbeat")
        commit_count = len(brain.commits)
        second = brain.sync_market_snapshot(snapshot, source="test_heartbeat")

        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.state["heartbeat"]["lastSyncStatus"], "no_change")

    def test_legacy_state_loads_with_new_market_fields(self):
        legacy_payload = {
            "state": {
                "frontalLobe": "Legacy memory",
                "emotion": "neutral"
            },
            "commits": [],
            "head": None
        }
        memory.BRAIN_FILE.write_text(json.dumps(legacy_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        brain = memory.Brain()
        snapshot = brain.get_brain_snapshot(max_age_minutes=999999)

        self.assertEqual(snapshot["state"]["frontalLobe"], "Legacy memory")
        self.assertEqual(snapshot["state"]["emotion"], "neutral")
        self.assertIn("marketRegime", snapshot["state"])
        self.assertIn("heartbeat", snapshot["state"])
        self.assertEqual(snapshot["state"]["marketRegime"]["state"], "未初始化")


if __name__ == "__main__":
    unittest.main()
