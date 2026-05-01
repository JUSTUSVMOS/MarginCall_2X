"""
Regression tests for frontal lobe structured defaults and no-op writes (system-integrity slice).

These tests patch persistence paths to an isolated temp dir under tests/ to avoid touching repo .brain files.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
class TestBrainMemorySystemIntegrity(unittest.TestCase):
    def setUp(self):
        # create isolated directory inside tests to avoid /tmp and keep repo-local
        self.workdir = tempfile.mkdtemp(prefix="brain_test_", dir=_TESTS_DIR)

        # Before importing engine_memory, record repo-root .brain contents (if any)
        repo_root = Path(__file__).resolve().parents[1]
        repo_brain_dir = repo_root / ".brain"
        if repo_brain_dir.exists():
            pre_import_listing = set(p.name for p in repo_brain_dir.iterdir())
        else:
            pre_import_listing = None

        # import here (should NOT create persistence files at import time)
        import importlib
        mem = importlib.import_module("engine_memory")

        # Verify import did not create or modify the repo .brain directory
        if pre_import_listing is None:
            self.assertFalse(repo_brain_dir.exists(), "Import unexpectedly created repo .brain directory")
        else:
            post = set(p.name for p in repo_brain_dir.iterdir())
            self.assertEqual(pre_import_listing, post, "Import modified repo .brain contents")

        # Patch persistence paths to our isolated dir
        mem.BRAIN_DIR = Path(self.workdir)
        mem.BRAIN_FILE = mem.BRAIN_DIR / "commit.json"
        mem.FRONTAL_LOBE_FILE = mem.BRAIN_DIR / "frontal-lobe.md"
        mem.EMOTION_FILE = mem.BRAIN_DIR / "emotion-log.json"
        mem.MARKET_REGIME_FILE = mem.BRAIN_DIR / "market-regime.md"
        mem.HEARTBEAT_FILE = mem.BRAIN_DIR / "heartbeat.json"
        mem.SNAPSHOT_FILE = mem.BRAIN_DIR / "snapshot.json"

        # Create a fresh Brain instance bound to our patched paths
        mem._global_brain = mem.Brain()
        self.mem = mem

    def tearDown(self):
        # remove created dir
        try:
            shutil.rmtree(self.workdir)
        except Exception:
            pass

    def test_default_state_starts_with_structured_frontal_lobe_template(self):
        note = self.mem.get_frontal_lobe()
        self.assertIsInstance(note, str)
        # Should include labeled sections
        for label in ("Market View:", "Core Levels:", "Portfolio Health:", "Next Round:"):
            self.assertIn(label, note)

    def test_update_lobe_section_skips_identical_normalized_content(self):
        brain = self.mem._global_brain
        VALID_NOTE = (
            "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"

            "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"

            "Portfolio Health: Keep leverage light until event risk clears.\n"

            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
        )
        self.assertTrue(brain.update_frontal_lobe(VALID_NOTE)["success"])
        commit_count = len(brain.commits)

        result = brain.update_lobe_section(
            "Market View",
            "  Neutral - CPI is the next catalyst while breadth remains mixed.  ",
            source="unit_test",
        )

        self.assertTrue(result["success"])
        self.assertTrue(result["unchanged"])
        self.assertEqual(result["message"], "Frontal lobe section 'Market View' unchanged.")
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.get_frontal_lobe(), VALID_NOTE)

    def test_update_frontal_lobe_skips_identical_normalized_content(self):
        brain = self.mem._global_brain
        VALID_NOTE = (
            "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"

            "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"

            "Portfolio Health: Keep leverage light until event risk clears.\n"

            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
        )
        self.assertTrue(brain.update_frontal_lobe(VALID_NOTE)["success"])
        commit_count = len(brain.commits)

        result = brain.update_frontal_lobe(
            "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"

            "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"

            "Portfolio Health: Keep leverage light until event risk clears.\n"

            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure.\n"

        )

        self.assertTrue(result["success"])
        self.assertTrue(result.get("unchanged", True))
        self.assertEqual(result["message"], "Frontal lobe unchanged; skipped commit.")
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.get_frontal_lobe(), VALID_NOTE)

    def test_rejecting_placeholder_note_preserves_default_template(self):
        brain = self.mem._global_brain
        initial = brain.get_frontal_lobe()

        # attempt to write a placeholder-quality note
        res = brain.update_frontal_lobe("觀望")
        # preserved message must be exact when rejected via public API; here we call the method
        # which returns a dict with message
        self.assertIn("message", res)
        self.assertEqual(res["message"], "Rejected: content is too vague to persist.")

        # ensure frontal lobe remains the structured default template
        current = brain.get_frontal_lobe()
        self.assertEqual(current, initial)


    def test_update_market_regime_signal_only_refresh_updates_state_without_commit(self):
        brain = self.mem.Brain()
        first = brain.update_market_regime(
            summary="Macro backdrop unchanged; wait for confirmation.",
            regime="🟡 整理",
            risk_score=33,
            watchpoints=["SPX 20MA"],
            reasons=["Macro steady"],
            signals={"spx": 5200.4},
            source="unit_test",
            updated_at="2026-01-02T03:04:05Z",
        )
        first_change_at = brain.state["heartbeat"]["lastMacroChangeAt"]
        commit_count = len(brain.commits)

        second = brain.update_market_regime(
            summary="Macro backdrop unchanged; wait for confirmation.",
            regime="🟡 整理",
            risk_score=33,
            watchpoints=["SPX 20MA"],
            reasons=["Macro steady"],
            signals={"spx": 5198.8, "yieldCurve10Y2Y": -0.21},
            source="unit_test",
            updated_at="2026-01-02T03:09:05Z",
        )

        self.assertTrue(first["changed"])
        self.assertFalse(second["changed"])
        self.assertEqual(len(brain.commits), commit_count)
        self.assertEqual(brain.state["marketRegime"]["signals"]["spx"], 5198.8)
        self.assertEqual(brain.state["marketRegime"]["signals"]["yieldCurve10Y2Y"], -0.21)
        self.assertEqual(brain.state["heartbeat"]["lastSyncStatus"], "no_change")
        self.assertEqual(brain.state["heartbeat"]["lastMacroChangeAt"], first_change_at)

    def test_commit_history_is_capped_at_200_entries(self):
        brain = self.mem.Brain()

        for idx in range(205):
            result = brain.update_emotion(
                "cautious" if idx % 2 else "neutral",
                f"reason {idx}"
            )
            self.assertTrue(result["success"])

        self.assertEqual(len(brain.commits), 200)
        self.assertEqual(brain.head, brain.commits[-1]["hash"])
        self.assertEqual(brain.commits[0]["delta"]["reason"], "reason 5")

        reloaded = self.mem.Brain()
        self.assertEqual(len(reloaded.commits), 200)
        self.assertEqual(reloaded.head, reloaded.commits[-1]["hash"])

if __name__ == "__main__":
    unittest.main()
