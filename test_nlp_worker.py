import unittest
from nlp_worker import parse_form4_insider, extract_section, adjust_retail_score

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

if __name__ == '__main__':
    unittest.main()
