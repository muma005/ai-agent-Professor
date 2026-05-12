"""Unit tests for rubric extraction regex patterns."""

def test_extract_points_from_parentheses():
    """'Clinical Relevance (25 points)' → name='Clinical Relevance', weight=25"""
    import re
    pattern = r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&]+?)\s*\((?:\d+\s*(?:points?|pts)\s*[-|]?\s*)?(\d+)\s*(?:points?|pts)\)'
    # Adjusted to match the deterministic extraction
    pattern = r'(?:^|\n)\s*\d*\.?\s*([A-Z][A-Za-z\s&]+?)\s*\((\d+)\s*(?:points?|pts)\)'
    text = "1. Clinical Relevance (25 points)"
    matches = re.findall(pattern, text)
    assert len(matches) == 1
    assert matches[0] == ("Clinical Relevance", "25")

def test_extract_word_limit():
    text = "Your Writeup should not exceed 2,000 words."
    import re
    match = re.search(r'(?:not exceed|under|maximum|max)\s*(\d{1,5})\s*words', text.lower().replace(",", ""))
    assert match
    assert int(match.group(1)) == 2000

def test_extract_prizes():
    text = "1st Place: $5,000\n2nd Place: $3,000"
    import re
    match = re.search(r'(?:1st|first)\s*(?:place)?\s*[:\-—]?\s*\$?([\d,]+)', text.lower())
    assert match
    assert match.group(1).replace(",", "") == "5000"
