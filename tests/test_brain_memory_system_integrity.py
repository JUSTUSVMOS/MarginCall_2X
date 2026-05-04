"""
Regression tests for frontal lobe structured defaults and no-op writes (system-integrity slice).

These tests patch persistence paths to an isolated temp dir under tests/ to avoid touching repo .brain files.
"""

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from src import llm
_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Test fixtures for structured frontal lobe state
VALID_THESIS = {
    "market_view": "Bearish - Market shows signs of reversal after SPX rejected 5250 resistance.",
    "core_levels": "Watch SPX 5200 support and 5250 resistance.",
    "next_round": "If SPX breaks below 5180, I will cut exposure and wait for confirmation before re-adding.",
    "context_note": "CPI report next week may be catalyst."
}

LEGACY_NOTE = (
    "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"
    "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"
    "Portfolio Health: Keep leverage light until event risk clears.\n"
    "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
)
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

    def test_default_state_uses_structured_frontal_lobe_and_portfolio_health(self):
        """Verify default state contains dict-backed frontalLobe and portfolioHealth."""
        brain = self.mem._global_brain
        
        # Check frontalLobe structure
        frontal_lobe = brain.state["frontalLobe"]
        self.assertIsInstance(frontal_lobe, dict)
        self.assertEqual(frontal_lobe["market_view"], "")
        self.assertEqual(frontal_lobe["core_levels"], "")
        self.assertEqual(frontal_lobe["next_round"], "")
        self.assertEqual(frontal_lobe["context_note"], "")
        self.assertIsNone(frontal_lobe["updated_at"])
        
        # Check portfolioHealth structure
        portfolio_health = brain.state["portfolioHealth"]
        self.assertIsInstance(portfolio_health, dict)
        self.assertIsNone(portfolio_health["nav_twd"])
        self.assertIsNone(portfolio_health["pnl_pct"])
        self.assertIsNone(portfolio_health["top3_concentration"])
        self.assertIsNone(portfolio_health["drawdown_pct"])
        self.assertIsNone(portfolio_health["risk_state"])
        self.assertIsNone(portfolio_health["gross_scale"])
        self.assertIsNone(portfolio_health["updated_at"])

    def test_load_migrates_legacy_string_frontal_lobe(self):
        """Verify that legacy string frontal lobe is migrated to structured dict on load."""
        brain = self.mem._global_brain
        
        # Manually set legacy string state and save
        brain.state["frontalLobe"] = LEGACY_NOTE
        brain._save()
        
        # Reload brain to trigger migration
        reloaded = self.mem.Brain()
        frontal_lobe = reloaded.state["frontalLobe"]
        
        # Should be migrated to dict
        self.assertIsInstance(frontal_lobe, dict)
        self.assertIn("Neutral - CPI is the next catalyst", frontal_lobe["market_view"])
        self.assertIn("5200", frontal_lobe["core_levels"])
        self.assertIn("5250", frontal_lobe["next_round"])
        
        # Portfolio Health should have been extracted from legacy note and moved to portfolioHealth state
        # The frontalLobe dict should NOT have a portfolio_health field anymore
        self.assertNotIn("portfolio_health", frontal_lobe)

    def test_update_frontal_lobe_accepts_structured_payload_and_skips_identical_write(self):
        """Verify update_frontal_lobe accepts dict payload and skips duplicate writes."""
        brain = self.mem._global_brain
        
        # First write
        result1 = brain.update_frontal_lobe(dict(VALID_THESIS))
        self.assertTrue(result1["success"])
        commit_count = len(brain.commits)
        
        # Second identical write should be skipped
        result2 = brain.update_frontal_lobe(dict(VALID_THESIS))
        self.assertTrue(result2["success"])
        self.assertTrue(result2.get("unchanged", False))
        self.assertEqual(result2["message"], "Frontal lobe unchanged; skipped commit.")
        self.assertEqual(len(brain.commits), commit_count)

    def test_update_frontal_lobe_tool_schema_uses_named_parameters(self):
        """Verify update_frontal_lobe tool schema has named parameters, not 'content'."""
        tools = llm._convert_to_openai_tools([self.mem.update_frontal_lobe])
        self.assertEqual(len(tools), 1)
        
        function_schema = tools[0]["function"]
        params = function_schema["parameters"]
        
        # Required parameters
        self.assertIn("market_view", params["required"])
        self.assertIn("core_levels", params["required"])
        self.assertIn("next_round", params["required"])
        
        # Optional parameter
        self.assertIn("context_note", params["properties"])
        
        # Old 'content' parameter should NOT exist
        self.assertNotIn("content", params["properties"])

    def test_default_state_starts_with_structured_frontal_lobe_template(self):
        note = self.mem.get_frontal_lobe()
        self.assertIsInstance(note, str)
        # Should include labeled sections (Portfolio Health removed from frontal lobe)
        for label in ("Market View:", "Core Levels:", "Next Round:"):
            self.assertIn(label, note)

    def test_update_lobe_section_skips_identical_normalized_content(self):
        brain = self.mem._global_brain
        VALID_NOTE = (
            "Market View: Neutral - CPI is the next catalyst while breadth remains mixed.\n"
            "Core Levels: Watch SPX 5200 support and 5250 resistance.\n"
            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
        )
        # Write as dict payload (new structured format)
        payload = {
            "market_view": "Neutral - CPI is the next catalyst while breadth remains mixed.",
            "core_levels": "Watch SPX 5200 support and 5250 resistance.",
            "next_round": "If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure.",
            "context_note": ""
        }
        self.assertTrue(brain.update_frontal_lobe(payload)["success"])
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
            "Next Round: If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure."
        )
        # First write as dict payload (new structured format)
        payload = {
            "market_view": "Neutral - CPI is the next catalyst while breadth remains mixed.",
            "core_levels": "Watch SPX 5200 support and 5250 resistance.",
            "next_round": "If CPI cools and SPX reclaims 5250, add risk; if 5200 breaks, cut exposure.",
            "context_note": ""
        }
        self.assertTrue(brain.update_frontal_lobe(payload)["success"])
        commit_count = len(brain.commits)

        # Second write with identical content
        result = brain.update_frontal_lobe(dict(payload))

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

    def test_duplicate_write_detection_normalizes_whitespace_differences(self):
        """
        Regression test: whitespace-only differences in structured payloads should NOT create extra commits.
        This ensures _frontal_lobe_write_is_unchanged() compares normalized values, not raw strings.
        """
        brain = self.mem.Brain()
        
        # First write with clean values
        payload1 = {
            "market_view": "Bearish - SPX rejected 5250 resistance.",
            "core_levels": "Watch SPX 5200 support.",
            "next_round": "If SPX breaks 5180, cut exposure.",
            "context_note": "CPI next week."
        }
        result1 = brain.update_frontal_lobe(payload1)
        self.assertTrue(result1["success"])
        commit_count = len(brain.commits)
        
        # Second write with identical content but extra whitespace
        payload2 = {
            "market_view": "  Bearish - SPX rejected 5250 resistance.  ",
            "core_levels": "  Watch SPX 5200 support.  ",
            "next_round": "  If SPX breaks 5180, cut exposure.  ",
            "context_note": "  CPI next week.  "
        }
        result2 = brain.update_frontal_lobe(payload2)
        
        # Should be detected as unchanged
        self.assertTrue(result2["success"])
        self.assertTrue(result2.get("unchanged", False), 
                       "Whitespace-only differences should be detected as no-op")
        self.assertEqual(result2["message"], "Frontal lobe unchanged; skipped commit.")
        self.assertEqual(len(brain.commits), commit_count,
                        "No extra commit should be created for whitespace-only changes")

    def test_concise_valid_structured_update_is_accepted(self):
        """
        Regression test: concise but valid structured payloads should be accepted.
        This ensures _is_placeholder_content() doesn't over-reject legitimate concise updates.
        """
        brain = self.mem.Brain()
        
        # Concise but valid structured payload (all sections have clear content)
        concise_payload = {
            "market_view": "Bullish - breakout confirmed",
            "core_levels": "SPX 5300 support",
            "next_round": "Add if momentum holds",
            "context_note": ""
        }
        
        result = brain.update_frontal_lobe(concise_payload)
        
        # Should be accepted (not rejected as placeholder)
        self.assertTrue(result["success"])
        self.assertFalse(result.get("unchanged", False))
        self.assertNotEqual(result["message"], "Rejected: content is too vague to persist.",
                          "Concise valid content should not be rejected as placeholder")
        self.assertGreater(len(brain.commits), 0, "Valid content should create a commit")

    def test_sparse_placeholder_content_is_still_rejected(self):
        """
        Regression test: sparse/vague placeholder content should still be rejected.
        This ensures the fix to _is_placeholder_content() doesn't break placeholder detection.
        """
        brain = self.mem.Brain()
        
        # Sparse payload with vague placeholder-quality content
        sparse_payload = {
            "market_view": "觀望",
            "core_levels": "",
            "next_round": "wait",
            "context_note": ""
        }
        
        result = brain.update_frontal_lobe(sparse_payload)
        
        # Should be rejected (success=False when rejected)
        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "Rejected: content is too vague to persist.")

    def test_update_lobe_section_true_change_creates_commit(self):
        """
        Regression test: update_lobe_section() with a real content change must:
        - return unchanged=False
        - create one additional commit
        - persist the new section value
        
        Reproduces bug where live frontalLobe dict is mutated before unchanged check,
        causing real changes to be incorrectly detected as no-op.
        """
        brain = self.mem.Brain()
        
        # Create initial state
        initial_payload = {
            'market_view': 'Bearish - test',
            'core_levels': 'SPX 5200',
            'next_round': 'Cut risk',
            'context_note': ''
        }
        result1 = brain.update_frontal_lobe(initial_payload)
        self.assertTrue(result1["success"])
        initial_commit_count = len(brain.commits)
        
        # Verify initial state persisted
        self.assertEqual(brain.state['frontalLobe']['market_view'], 'Bearish - test')
        
        # Now update one section with different content
        result2 = brain.update_lobe_section('Market View', 'Bullish - changed', source='verify')
        
        # Should detect this as a real change
        self.assertTrue(result2["success"], "update_lobe_section should succeed")
        self.assertFalse(result2.get("unchanged", False), 
                        "Real content change should NOT be marked as unchanged")
        
        # Should create one additional commit
        self.assertEqual(len(brain.commits), initial_commit_count + 1,
                        "Real content change should create exactly one commit")
        
        # Should persist the new value
        self.assertEqual(brain.state['frontalLobe']['market_view'], 'Bullish - changed',
                        "New section value should be persisted")

if __name__ == "__main__":
    unittest.main()
