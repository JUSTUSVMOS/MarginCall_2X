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
        # import here to patch module vars after import
        import engine_memory as mem

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
        initial = brain.get_frontal_lobe()
        sections = self.mem._coerce_frontal_lobe_sections(initial)
        before_commits = len(brain.commits)

        # update with identical content for a section
        res = brain.update_lobe_section("Core Levels", sections.get("Core Levels", ""), source="test")
        after_commits = len(brain.commits)

        # unchanged writes should indicate unchanged and not create a commit
        self.assertIn("success", res)
        # allow both unchanged True or absent (older behavior) but prefer unchanged True
        self.assertTrue(res.get("unchanged", True))
        self.assertEqual(before_commits, after_commits)

    def test_update_frontal_lobe_skips_identical_normalized_content(self):
        brain = self.mem._global_brain
        initial = brain.get_frontal_lobe()
        before_commits = len(brain.commits)

        res = brain.update_frontal_lobe(initial)
        after_commits = len(brain.commits)

        self.assertIn("success", res)
        self.assertTrue(res.get("unchanged", True))
        self.assertEqual(before_commits, after_commits)

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


if __name__ == "__main__":
    unittest.main()
