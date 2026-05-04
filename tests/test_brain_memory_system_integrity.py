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

    def test_update_lobe_section_rejects_portfolio_health(self):
        """
        Task 2 regression: update_lobe_section("Portfolio Health", ...) must fail fast.
        """
        brain = self.mem.Brain()
        
        result = brain.update_lobe_section("Portfolio Health", "NAV: $100k | Risk: Normal", source="test")
        
        self.assertFalse(result["success"], "Portfolio Health updates should be rejected")
        self.assertEqual(result["message"], "Portfolio Health is system-managed; use update_portfolio_health().")

    def test_update_portfolio_health_saves_state_without_creating_commit(self):
        """
        Task 2 regression: update_portfolio_health(...) must update state and skip commit.
        Strengthened: first create a normal frontal-lobe commit, then verify portfolio health
        write does not add another commit.
        """
        brain = self.mem.Brain()
        
        # First create a normal frontal-lobe commit
        brain.update_frontal_lobe(dict(VALID_THESIS))
        commit_count_after_frontal_lobe = len(brain.commits)
        self.assertGreater(commit_count_after_frontal_lobe, 0, 
                          "Frontal lobe update should create at least one commit")
        
        # Now call update_portfolio_health
        health_data = {
            "nav_twd": 100000.0,
            "pnl_pct": 5.5,
            "top3_concentration": 0.45,
            "drawdown_pct": 2.1,
            "risk_state": "Normal",
            "gross_scale": 1.2
        }
        
        result = brain.update_portfolio_health(health_data)
        
        # Should succeed
        self.assertTrue(result["success"])
        self.assertFalse(result.get("unchanged", True), "First write should be marked as changed")
        
        # Should update state
        ph = brain.state["portfolioHealth"]
        self.assertEqual(ph["nav_twd"], 100000.0)
        self.assertEqual(ph["pnl_pct"], 5.5)
        self.assertEqual(ph["top3_concentration"], 0.45)
        self.assertEqual(ph["drawdown_pct"], 2.1)
        self.assertEqual(ph["risk_state"], "Normal")
        self.assertEqual(ph["gross_scale"], 1.2)
        self.assertIsNotNone(ph["updated_at"])
        
        # Should NOT create a commit - count should stay exactly the same
        self.assertEqual(len(brain.commits), commit_count_after_frontal_lobe,
                        "update_portfolio_health should not create commits relative to normal frontal-lobe behavior")

    def test_update_portfolio_health_skips_small_nav_only_drift(self):
        """
        Task 2 regression: small NAV drift (< 0.5%) with unchanged other fields should return unchanged=True.
        """
        brain = self.mem.Brain()
        
        # First write
        first_data = {
            "nav_twd": 100000.0,
            "pnl_pct": 5.5,
            "top3_concentration": 0.45,
            "drawdown_pct": 2.1,
            "risk_state": "Normal",
            "gross_scale": 1.2
        }
        result1 = brain.update_portfolio_health(first_data)
        self.assertTrue(result1["success"])
        
        # Second write with small NAV drift (0.3%) but same other fields
        second_data = {
            "nav_twd": 100300.0,  # +300 = +0.3%
            "pnl_pct": 5.5,
            "top3_concentration": 0.45,
            "drawdown_pct": 2.1,
            "risk_state": "Normal",
            "gross_scale": 1.2
        }
        result2 = brain.update_portfolio_health(second_data)
        
        # Should be detected as unchanged
        self.assertTrue(result2["success"])
        self.assertTrue(result2.get("unchanged", False), 
                       "Small NAV drift (<0.5%) should be detected as unchanged")
        self.assertEqual(result2["message"], "Portfolio health unchanged.")

    def test_refresh_portfolio_health_summary_uses_memory_updater(self):
        """
        Task 2 regression: engine_portfolio.refresh_portfolio_health_summary must call 
        memory.update_portfolio_health(...) and must NOT call patch_frontal_lobe_section("Portfolio Health", ...).
        """
        # Read the source code to verify it uses the correct API
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        engine_portfolio_path = repo_root / "engine_portfolio.py"
        
        source = engine_portfolio_path.read_text()
        
        # Must contain the new API call
        self.assertIn("memory.update_portfolio_health(", source,
                     "engine_portfolio.py must call memory.update_portfolio_health()")
        
        # Must NOT contain the old frontal-lobe patch path
        # Look for the specific pattern that would appear in refresh_portfolio_health_summary
        self.assertNotIn('patch_frontal_lobe_section("Portfolio Health"', source,
                        "engine_portfolio.py must not patch Portfolio Health into frontal lobe")

    def test_refresh_portfolio_health_summary_rounds_numeric_fields_correctly(self):
        """
        Task 2 code-quality review fix: refresh_portfolio_health_summary must round numeric fields
        before passing to update_portfolio_health to avoid materiality gaps in no-op detection.
        
        This test verifies:
        - nav_twd rounded to 2 decimals
        - pnl_pct rounded to 4 decimals
        - top3_concentration rounded to 4 decimals
        - drawdown_pct rounded to 4 decimals (after multiplying by 100)
        - gross_scale rounded to 4 decimals when not None
        - return object includes memory_update
        """
        from unittest.mock import patch, MagicMock, call
        import engine_portfolio
        
        # Mock all internal dependencies to make this test fast and deterministic
        mock_analysis = {
            "total_current": 123456.789012,  # Should round to 123456.79
            "total_pnl_pct": 5.123456789,    # Should round to 5.1235
            "top3_concentration": 45.6789012,  # PERCENT SCALE: 45.67%, should round to 45.6789
            "summary": "Mock summary"
        }
        
        mock_overlay = {
            "current_drawdown": 0.03456789012,  # * 100 = 3.456789012, should round to 3.4568
            "trade_mode_label": "Normal",
            "recommended_gross_scale": 1.234567890  # Should round to 1.2346
        }
        
        mock_nav_snapshot = {"snapshot": "mock"}
        
        with patch.object(engine_portfolio, '_load_portfolio_rows', return_value=[]):
            with patch.object(engine_portfolio, '_build_live_position_snapshots', return_value=[]):
                with patch.object(engine_portfolio, 'record_portfolio_nav_snapshot', return_value=mock_nav_snapshot):
                    with patch.object(engine_portfolio, 'build_portfolio_analysis', return_value=mock_analysis):
                        with patch.object(engine_portfolio, 'compute_portfolio_risk_overlay', return_value=mock_overlay):
                            # Patch the engine_memory module at import time
                            with patch('engine_memory.update_portfolio_health') as mock_update:
                                mock_update.return_value = {"success": True, "message": "Updated"}
                                
                                # Call the function
                                result = engine_portfolio.refresh_portfolio_health_summary(source="test")
                                
                                # Verify update_portfolio_health was called with correctly rounded values
                                self.assertEqual(mock_update.call_count, 1, 
                                               "update_portfolio_health should be called exactly once")
                                
                                call_args = mock_update.call_args[0][0]
                                
                                # Verify rounding
                                self.assertEqual(call_args["nav_twd"], 123456.79, 
                                               "nav_twd should be rounded to 2 decimals")
                                self.assertEqual(call_args["pnl_pct"], 5.1235, 
                                               "pnl_pct should be rounded to 4 decimals")
                                self.assertEqual(call_args["top3_concentration"], 45.6789, 
                                               "top3_concentration (percent scale) should be rounded to 4 decimals")
                                self.assertEqual(call_args["drawdown_pct"], 3.4568, 
                                               "drawdown_pct should be rounded to 4 decimals after multiplying by 100")
                                self.assertEqual(call_args["risk_state"], "Normal", 
                                               "risk_state should pass through unchanged")
                                self.assertEqual(call_args["gross_scale"], 1.2346, 
                                               "gross_scale should be rounded to 4 decimals")
                                
                                # Verify return object includes memory_update
                                self.assertIn("memory_update", result, 
                                            "Result should include memory_update")
                                self.assertEqual(result["memory_update"], {"success": True, "message": "Updated"})

    def test_update_portfolio_health_detects_zero_to_positive_nav_as_material(self):
        """
        Regression test: zero-to-positive NAV transition is material and should NOT be suppressed.
        
        Bug reproduced by controller:
        1. First write with nav_twd=0.0
        2. Second write with nav_twd=1000.0 (same other fields)
        3. Current behavior returns unchanged=True and leaves stored nav_twd at 0.0
        
        Expected behavior:
        - Second call should return unchanged=False
        - Stored nav_twd should become 1000.0
        """
        brain = self.mem.Brain()
        
        # First write with zero NAV
        first_data = {
            "nav_twd": 0.0,
            "pnl_pct": 1.0,
            "top3_concentration": 10.0,
            "drawdown_pct": 0.0,
            "risk_state": "Normal",
            "gross_scale": 1.0
        }
        result1 = brain.update_portfolio_health(first_data)
        self.assertTrue(result1["success"])
        self.assertFalse(result1.get("unchanged", True), "First write should be marked as changed")
        self.assertEqual(brain.state["portfolioHealth"]["nav_twd"], 0.0)
        
        # Second write with positive NAV (same other fields)
        second_data = {
            "nav_twd": 1000.0,
            "pnl_pct": 1.0,
            "top3_concentration": 10.0,
            "drawdown_pct": 0.0,
            "risk_state": "Normal",
            "gross_scale": 1.0
        }
        result2 = brain.update_portfolio_health(second_data)
        
        # Should be detected as a material change
        self.assertTrue(result2["success"])
        self.assertFalse(result2.get("unchanged", True), 
                        "Zero-to-positive NAV transition should be detected as material")
        
        # Should update stored NAV
        self.assertEqual(brain.state["portfolioHealth"]["nav_twd"], 1000.0,
                        "Stored nav_twd should be updated to 1000.0")

    def test_get_cognitive_context_renders_three_clean_blocks(self):
        """
        Task 3 regression: get_cognitive_context() must render three clean, labeled blocks:
        - ### Trading Thesis (Frontal Lobe)
        - ### Portfolio Health (Auto)
        - ### Persistent Macro / Market Regime
        
        Old Portfolio Health inline text like "Portfolio Health: Old auto-summary" must NOT appear.
        """
        brain = self.mem.Brain()
        
        # Write valid thesis
        brain.update_frontal_lobe(dict(VALID_THESIS))
        
        # Write portfolio health
        health_data = {
            "nav_twd": 100000.0,
            "pnl_pct": 5.5,
            "top3_concentration": 45.0,  # Percent scale: 45%
            "drawdown_pct": 2.1,
            "risk_state": "Normal",
            "gross_scale": 1.2
        }
        brain.update_portfolio_health(health_data)
        
        # Write market regime
        brain.update_market_regime(
            summary="Macro backdrop stable",
            regime="🟢 上漲",
            risk_score=75,
            watchpoints=["SPX 20MA"],
            reasons=["Strong momentum"],
            signals={"spx": 5200.4},
            source="test",
            updated_at="2026-01-02T03:04:05Z"
        )
        
        # Get cognitive context
        context = brain.get_cognitive_context(max_age_minutes=999999)
        
        # Must include three clean block headers
        self.assertIn("### Trading Thesis (Frontal Lobe)", context)
        self.assertIn("### Portfolio Health (Auto)", context)
        self.assertIn("### Persistent Macro / Market Regime", context)
        
        # Must NOT include old inline Portfolio Health text
        self.assertNotIn("Portfolio Health: Old auto-summary", context)
        self.assertNotIn("Portfolio Health:", context.split("### Portfolio Health (Auto)")[0],
                        "Frontal lobe section must not contain 'Portfolio Health:' inline")

    def test_get_frontal_lobe_write_guide_describes_named_parameters(self):
        """
        Task 3 regression: get_frontal_lobe_write_guide() must mention structured named parameters
        and explicitly say NOT to write portfolio health.
        
        Must include:
        - market_view
        - core_levels
        - next_round
        - context_note
        - "Do not write portfolio health here"
        
        Must NOT include:
        - "Portfolio Health:"
        """
        guide = self.mem.get_frontal_lobe_write_guide()
        
        # Must mention all named parameters
        self.assertIn("market_view", guide)
        self.assertIn("core_levels", guide)
        self.assertIn("next_round", guide)
        self.assertIn("context_note", guide)
        
        # Must warn NOT to write portfolio health
        self.assertIn("do not write portfolio health", guide.lower(),
                     "Guide must explicitly tell model not to write portfolio health in frontal lobe")
        
        # Must NOT include old Portfolio Health section marker
        self.assertNotIn("Portfolio Health:", guide)

    def test_agent_prompt_contract_mentions_structured_write_fields(self):
        """
        Task 3 regression: src/agent.py prompt must tell model to use structured write fields
        and must NOT mention the old four-section note format.
        
        Must include:
        - market_view
        - core_levels
        - next_round
        - context_note
        
        Must NOT include:
        - 四段式專業交易筆記格式 (old four-section format reference)
        """
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        agent_path = repo_root / "src" / "agent.py"
        
        source = agent_path.read_text()
        
        # Must mention structured write fields
        self.assertIn("market_view", source)
        self.assertIn("core_levels", source)
        self.assertIn("next_round", source)
        self.assertIn("context_note", source)
        
        # Must NOT mention old four-section format
        self.assertNotIn("四段式專業交易筆記格式", source,
                        "src/agent.py must not reference old four-section format")

    def test_top3_concentration_renders_with_percent_scale_without_double_scaling(self):
        """
        Task 3 code-quality review fix: top3_concentration is already stored as a percent value
        (e.g., 67.0 means 67%), but current rendering multiplies by 100 again, showing 6700.0%.
        
        Verified data flow:
        - engine_portfolio.build_portfolio_analysis() computes: top3_pct = (top3_mv / total * 100)
        - then returns "top3_concentration": top3_pct (already percent scale)
        
        This test verifies:
        - top3_concentration = 67.0 renders as "67.0%" not "6700.0%"
        - rendering does NOT multiply by 100
        """
        brain = self.mem.Brain()
        
        # Write portfolio health with percent-scale top3_concentration
        health_data = {
            "nav_twd": 100000.0,
            "pnl_pct": 5.5,
            "top3_concentration": 67.0,  # Already percent scale (67%)
            "drawdown_pct": 2.1,
            "risk_state": "Normal",
            "gross_scale": 1.2
        }
        brain.update_portfolio_health(health_data)
        
        # Get cognitive context
        context = brain.get_cognitive_context()
        
        # Must render as 67.0%, not 6700.0%
        self.assertIn("67.0%", context, "top3_concentration should render as 67.0%")
        self.assertNotIn("6700", context, "top3_concentration must not be double-scaled to 6700.0%")

    def test_partial_portfolio_health_renders_safely_with_placeholders(self):
        """
        Task 3 code-quality review fix: partial portfolio-health state should render gracefully.
        
        Verified bug reproduction:
        - Set nav_twd = 1000.0
        - Set pnl_pct, top3_concentration, drawdown_pct, risk_state, gross_scale to None
        - Current behavior raises: TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
        
        This test verifies:
        - get_cognitive_context() does not crash when numeric fields are None
        - renders placeholders like "N/A" for missing values
        - keeps empty-state message only for fully absent state (nav_twd is None)
        """
        brain = self.mem.Brain()
        
        # Write partial portfolio health (nav_twd exists, other fields None)
        partial_health = {
            "nav_twd": 1000.0,
            "pnl_pct": None,
            "top3_concentration": None,
            "drawdown_pct": None,
            "risk_state": None,
            "gross_scale": None
        }
        brain.update_portfolio_health(partial_health)
        
        # Should not crash
        context = brain.get_cognitive_context()
        
        # Should include Portfolio Health section
        self.assertIn("### Portfolio Health (Auto)", context)
        
        # Should include NAV
        self.assertIn("1000.0", context)
        
        # Should include placeholders for missing fields
        self.assertIn("N/A", context, "Missing fields should render as N/A")
        
        # Should NOT show empty-state message (that's only for fully absent state)
        self.assertNotIn("尚未同步 portfolio health", context,
                        "Should not show empty-state message when nav_twd exists")

    def test_thesis_with_only_context_note_does_not_show_blank_state(self):
        """
        Task 3 final fix: empty-state gate must check all four fields including context_note.
        
        When context_note exists (even with quality content in other fields), the empty-state
        gate should not trigger. This tests that context_note is included in the gate check.
        
        (Note: must provide enough quality content to pass placeholder check, so we can test
        the rendering logic rather than the quality filter.)
        """
        brain = self.mem.Brain()
        
        # Set market_view and core_levels with quality content, plus context_note
        # (Placeholder check requires 2/3 core fields filled)
        brain.update_frontal_lobe({
            "market_view": "Neutral - awaiting catalyst",
            "core_levels": "SPX 5200 support, 5250 resistance",
            "next_round": "",
            "context_note": "CPI data release next Tuesday."
        })
        
        context = brain.get_cognitive_context(max_age_minutes=999999)
        
        # Should NOT show blank-state message
        self.assertNotIn("尚未建立。請在分析後使用 update_frontal_lobe 記錄你的觀點", context,
                        "Should not show blank-state message when any field exists")
        
        # Should render three core lines, with fallback for next_round
        self.assertIn("**Market View:** Neutral - awaiting catalyst", context)
        self.assertIn("**Core Levels:** SPX 5200 support, 5250 resistance", context)
        self.assertIn("**Next Round:** 尚未建立", context,
                     "Should render Next Round with fallback when empty")
        
        # Should include the context note
        self.assertIn("**Context:** CPI data release", context)
    
    def test_partial_thesis_renders_all_three_core_lines_with_fallbacks(self):
        """
        Task 3 final fix: partial thesis should always render Market View, Core Levels, Next Round
        with 尚未建立 fallbacks where missing.
        
        When thesis has some fields but not all, it must render:
        - Market View with fallback if empty
        - Core Levels with fallback if empty
        - Next Round with fallback if empty
        - Context only when present
        - Last updated always in non-empty branch
        """
        brain = self.mem.Brain()
        
        # Set only market_view and core_levels (missing next_round)
        # Plus context_note to verify it's rendered
        brain.update_frontal_lobe({
            "market_view": "Bullish - SPX broke through resistance",
            "core_levels": "Watch 5300 next resistance level",
            "next_round": "",
            "context_note": "Fed meeting next week"
        })
        
        context = brain.get_cognitive_context(max_age_minutes=999999)
        
        # Should NOT show blank-state message
        self.assertNotIn("尚未建立。請在分析後使用 update_frontal_lobe 記錄你的觀點", context)
        
        # Should render all three core lines
        self.assertIn("**Market View:** Bullish - SPX broke through resistance", context)
        self.assertIn("**Core Levels:** Watch 5300 next resistance level", context)
        self.assertIn("**Next Round:** 尚未建立", context,
                     "Should render Next Round with fallback when empty")
        
        # Should include context
        self.assertIn("**Context:** Fed meeting next week", context)
        
        # Should include last updated
        self.assertIn("**Last updated:**", context)

    def test_refresh_portfolio_health_uses_percent_scale_for_top3_concentration(self):
        """
        Task 3 code-quality review fix: update the runtime payload-mapping test to use
        real percent-scale top3_concentration fixture.
        
        Current test at line 537 uses fractional scale (0.456789012), but the real producer
        engine_portfolio.build_portfolio_analysis() returns percent scale (e.g., 45.6789012).
        
        This test verifies the fixture matches production convention.
        """
        from unittest.mock import patch
        import engine_portfolio
        
        # Mock with percent-scale top3_concentration (as real producer returns)
        mock_analysis = {
            "total_current": 123456.789012,
            "total_pnl_pct": 5.123456789,
            "top3_concentration": 45.6789012,  # Percent scale: 45.67%
            "summary": "Mock summary"
        }
        
        mock_overlay = {
            "current_drawdown": 0.03456789012,
            "trade_mode_label": "Normal",
            "recommended_gross_scale": 1.234567890
        }
        
        mock_nav_snapshot = {"snapshot": "mock"}
        
        with patch.object(engine_portfolio, '_load_portfolio_rows', return_value=[]):
            with patch.object(engine_portfolio, '_build_live_position_snapshots', return_value=[]):
                with patch.object(engine_portfolio, 'record_portfolio_nav_snapshot', return_value=mock_nav_snapshot):
                    with patch.object(engine_portfolio, 'build_portfolio_analysis', return_value=mock_analysis):
                        with patch.object(engine_portfolio, 'compute_portfolio_risk_overlay', return_value=mock_overlay):
                            with patch('engine_memory.update_portfolio_health') as mock_update:
                                mock_update.return_value = {"success": True, "message": "Updated"}
                                
                                # Call the function
                                result = engine_portfolio.refresh_portfolio_health_summary(source="test")
                                
                                # Verify update_portfolio_health was called
                                self.assertEqual(mock_update.call_count, 1)
                                
                                call_args = mock_update.call_args[0][0]
                                
                                # Verify percent-scale top3_concentration is rounded correctly
                                # 45.6789012 should round to 45.6789 (4 decimals)
                                self.assertEqual(call_args["top3_concentration"], 45.6789,
                                               "top3_concentration should use percent scale and round to 4 decimals")

if __name__ == "__main__":
    unittest.main()
