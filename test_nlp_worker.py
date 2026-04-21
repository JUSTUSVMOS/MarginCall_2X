import unittest
from datetime import datetime, timedelta, timezone
from nlp_worker import (
    parse_form4_insider,
    extract_section,
    adjust_retail_score,
    _time_decay_weight,
    _effective_group_count,
    compose_alpha_signal,
    _annotate_macro_candidate,
    _merge_macro_candidates,
    _build_signal_pack,
)

class TestNLPWorker(unittest.TestCase):
    def test_parse_form4_insider_empty(self):
        """Test parsing empty Form 4 content."""
        result = parse_form4_insider(None)
        self.assertEqual(result, "無內容")

    def test_parse_form4_insider_no_transactions(self):
        """Test parsing Form 4 content without transaction tags."""
        xml_content = "<ownershipDocument></ownershipDocument>"
        result = parse_form4_insider(xml_content)
        self.assertEqual(result, "【內部人變動】未發現實質交易標籤。")

    def test_extract_section(self):
        """Test the string extraction utility."""
        text = "HEADER\nINTRODUCTION\nSTART_KEY\nImportant data\nSTOP_KEY\nFOOTER"
        result = extract_section(text, "START_KEY", ["STOP_KEY"])
        self.assertIn("IMPORTANT DATA", result.upper())

    def test_adjust_retail_score(self):
        """Test retail score adjustment logic."""
        # Low sample size
        self.assertEqual(adjust_retail_score(0.8, 3), 0.0)
        # Extreme positive (>= 0.8)
        self.assertEqual(adjust_retail_score(0.9, 10), -0.3)
        # Extreme negative (<= -0.7)
        self.assertEqual(adjust_retail_score(-0.9, 10), 0.3)
        # Neutral/moderate (-0.3 to 0.3)
        self.assertEqual(adjust_retail_score(0.2, 10), 0.0)
        # Slight positive/negative
        self.assertEqual(adjust_retail_score(0.4, 10), 0.2)

    def test_time_decay_weight_prefers_recent_news(self):
        fresh = datetime.now(timezone.utc) - timedelta(hours=1)
        stale = datetime.now(timezone.utc) - timedelta(hours=36)
        fresh_weight, fresh_hours = _time_decay_weight(fresh)
        stale_weight, stale_hours = _time_decay_weight(stale)

        self.assertGreater(fresh_weight, stale_weight)
        self.assertLess(fresh_hours, stale_hours)

    def test_time_decay_weight_accepts_unix_timestamp(self):
        now = datetime.now(timezone.utc)
        unix_ts = int((now - timedelta(hours=12)).timestamp())
        weight, hours = _time_decay_weight(unix_ts)

        self.assertTrue(0 < weight < 1)
        self.assertAlmostEqual(hours, 12.0, delta=0.5)

    def test_effective_group_count_uses_embedded_weights(self):
        texts = [
            "Macro(Reuters | 1.0h | w=0.95): demand rebounds",
            "Macro(CNBC | 18.0h | w=0.41): supply chain warning",
            "SEC: filing risk factor updated",
        ]

        self.assertAlmostEqual(_effective_group_count(texts), 2.36, places=2)

    def test_compose_alpha_signal_accepts_fractional_confidence_counts(self):
        blended = compose_alpha_signal(-1.0, 1.0, -0.3, 1.5, 0.0, 0.0)
        expected = ((-1.0 * 0.45) + (-0.3 * 0.15)) / (0.45 + 0.15)

        self.assertAlmostEqual(blended, expected, places=4)

    # New helper-level regression tests for Task 1
    def test_annotate_macro_candidate_sets_defaults_for_normal_item(self):
        candidate = {
            'headline': 'Chip demand steady into next quarter',
            'summary': 'Analysts expect routine seasonal demand normalization.',
            'lane': 'macro',
            'source': 'Reuters',
            'published_at': datetime.now(timezone.utc),
        }
        annotated = _annotate_macro_candidate(candidate.copy())
        self.assertEqual(annotated.get('event_class'), 'normal')
        self.assertEqual(annotated.get('event_type'), 'normal')
        self.assertFalse(annotated.get('must_keep'))
        self.assertEqual(annotated.get('event_window_days'), 3)

    def test_annotate_macro_candidate_marks_acquisition_rule_from_table(self):
        candidate = {
            'headline': 'Amazon to acquire Globalstar in all-cash deal',
            'summary': 'The tie-up would expand Amazon satellite ambitions with Globalstar assets.',
            'lane': 'macro',
            'source': 'Reuters',
            'published_at': datetime.now(timezone.utc)
        }
        annotated = _annotate_macro_candidate(candidate.copy())
        self.assertEqual(annotated.get('event_class'), 'major_event')
        self.assertEqual(annotated.get('event_type'), 'acquisition')
        self.assertTrue(annotated.get('must_keep') is True)
        self.assertEqual(annotated.get('event_window_days'), 14)

    def test_annotate_macro_candidate_marks_partnership_rule_from_table(self):
        candidate = {
            'headline': 'Nvidia enters strategic partnership with TSMC on advanced packaging',
            'summary': 'The companies announced a joint collaboration to expand AI chip capacity.',
            'lane': 'macro',
            'source': 'Reuters',
            'published_at': datetime.now(timezone.utc),
        }
        annotated = _annotate_macro_candidate(candidate.copy())
        self.assertEqual(annotated.get('event_class'), 'major_event')
        self.assertEqual(annotated.get('event_type'), 'partnership')
        self.assertTrue(annotated.get('must_keep'))
        self.assertEqual(annotated.get('event_window_days'), 10)

    def test_merge_macro_candidates_prioritizes_event_and_formats_exact_must_mention(self):
        published_at = datetime.now(timezone.utc)
        primary = [{
            'headline': 'Amazon to acquire Globalstar in all-cash deal!!!',
            'summary': 'Primary lane duplicate that should lose priority after dedupe.',
            'lane': 'macro',
            'source': 'Reuters',
            'published_at': published_at,
            'event_class': 'normal',
            'event_type': 'normal',
            'must_keep': False,
            'event_window_days': 3,
        }]
        event_candidate = {
            'headline': 'Amazon to acquire Globalstar in all-cash deal',
            'summary': 'The tie-up would expand Amazon satellite ambitions with Globalstar assets.',
            'lane': 'event',
            'source': 'Reuters',
            'published_at': published_at,
            'event_class': 'major_event',
            'event_type': 'acquisition',
            'must_keep': True,
            'event_window_days': 7,
        }
        event_candidates = [event_candidate]
        merged = _merge_macro_candidates(primary, event_candidates, max_macro_items=5, max_must_keep=2)
        self.assertEqual(len(merged['selected_candidates']), 1)
        self.assertEqual(merged['selected_candidates'][0]['lane'], 'event')
        self.assertEqual(
            merged['must_mention_events'],
            ['acquisition: Amazon to acquire Globalstar in all-cash deal'],
        )

    def test_build_signal_pack_matches_task1_contract(self):
        must_mention_events = [
            'partnership: Nvidia enters strategic partnership with TSMC',
            'customer: Microsoft wins multi-year cloud contract',
        ]
        packed = _build_signal_pack(
            sec_dir='neutral',
            a_sec=0.0049,
            sec_detail=[
                '10-Q updated guidance',
                '8-K disclosed risk factor changes',
                'Form 4 insider buy',
                'Extra SEC item dropped by cap',
            ],
            mac_dir='bullish',
            a_mac=0.505,
            macro_detail=[
                'Macro(Reuters | 1.0h | w=0.95): Nvidia enters strategic partnership with TSMC',
                'Macro(Bloomberg | 3.0h | w=0.75): Microsoft wins multi-year cloud contract',
                'Macro(CNBC | 5.0h | w=0.60): Financing round expands runway',
                'Macro(WSJ | 6.0h | w=0.40): Extra macro item dropped by cap',
            ],
            ret_dir='neutral',
            a_retail=-0.334,
            retail_detail=[
                'Retail buzz muted',
                'Retail chatter skeptical',
                'Retail dip-buying persists',
                'Extra retail item dropped by cap',
            ],
            divergence_alert='',
            nuclear_confirmed=True,
            groups={
                'SEC': ['10-Q updated guidance', '8-K disclosed risk factor changes'],
                'Macro': [
                    'Nvidia enters strategic partnership with TSMC',
                    'Microsoft wins multi-year cloud contract',
                    'Financing round expands runway',
                    'Regulatory filing approved',
                ],
                'Retail': ['Retail buzz muted', 'Retail chatter skeptical', 'Retail dip-buying persists'],
            },
            effective_counts={'sec': 1.0, 'macro': 1.95, 'retail': 1.0},
            nlp_alpha=0.2149,
            must_mention_events=must_mention_events,
        )
        self.assertEqual(
            packed,
            {
                'sec_stance': 'neutral',
                'sec_score': 0.0,
                'sec_detail': [
                    '10-Q updated guidance',
                    '8-K disclosed risk factor changes',
                    'Form 4 insider buy',
                ],
                'macro_stance': 'bullish',
                'macro_score': 0.51,
                'macro_detail': [
                    'Macro(Reuters | 1.0h | w=0.95): Nvidia enters strategic partnership with TSMC',
                    'Macro(Bloomberg | 3.0h | w=0.75): Microsoft wins multi-year cloud contract',
                    'Macro(CNBC | 5.0h | w=0.60): Financing round expands runway',
                ],
                'retail_stance': 'neutral',
                'retail_score': -0.33,
                'retail_detail': [
                    'Retail buzz muted',
                    'Retail chatter skeptical',
                    'Retail dip-buying persists',
                ],
                'divergence': '無',
                'nuclear_alert': True,
                'source_counts': {'sec': 2, 'macro': 4, 'retail': 3},
                'effective_counts': {'sec': 1.0, 'macro': 1.95, 'retail': 1.0},
                'composite_alpha': 0.21,
                'must_mention_events': must_mention_events,
            },
        )

    def test_annotate_macro_candidate_marks_amazon_globalstar_as_major_event(self):
        return self.test_annotate_macro_candidate_marks_acquisition_rule_from_table()

    def test_merge_macro_candidates_keeps_major_event_even_when_primary_lane_is_empty(self):
        """Real regression: primary lane empty but an event candidate is kept and formatted."""
        published_at = datetime.now(timezone.utc)
        primary = []
        event_candidate = {
            'headline': 'Amazon to acquire Globalstar in all-cash deal',
            'summary': 'The tie-up would expand Amazon satellite ambitions with Globalstar assets.',
            'lane': 'event',
            'source': 'Reuters',
            'published_at': published_at,
            'event_class': 'major_event',
            'event_type': 'acquisition',
            'must_keep': True,
            'event_window_days': 7,
        }
        event_candidates = [event_candidate]
        merged = _merge_macro_candidates(primary, event_candidates, max_macro_items=5, max_must_keep=2)
        self.assertEqual(len(merged['selected_candidates']), 1)
        self.assertEqual(merged['selected_candidates'][0]['lane'], 'event')
        self.assertEqual(
            merged['must_mention_events'],
            ['acquisition: Amazon to acquire Globalstar in all-cash deal'],
        )

    def test_build_signal_pack_includes_must_mention_events(self):
        return self.test_build_signal_pack_matches_task1_contract()

if __name__ == '__main__':
    unittest.main()
