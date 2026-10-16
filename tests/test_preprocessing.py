"""
SentimentIQ - Unit Tests for Text Preprocessing
"""

import pytest
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import TextPreprocessor, clean_text, NEGATION_WORDS


@pytest.fixture
def preprocessor():
    return TextPreprocessor(remove_stopwords=True)


def test_clean_text_normal():
    raw = "The headset sound quality is absolutely amazing!"
    cleaned = clean_text(raw)
    assert "amazing" in cleaned
    assert "headset" in cleaned


def test_html_tag_removal(preprocessor):
    raw = "<p>Great experience! <br/>Food was <b>delicious</b>.</p>"
    cleaned = preprocessor.clean_text(raw)
    assert "<p>" not in cleaned
    assert "<br/>" not in cleaned
    assert "<b>" not in cleaned
    assert "delicious" in cleaned


def test_url_and_mention_removal(preprocessor):
    raw = "Check this out @company http://example.com/item #awesome"
    cleaned = preprocessor.clean_text(raw)
    assert "@company" not in cleaned
    assert "http" not in cleaned
    assert "awesome" in cleaned  # Hashtag token preserved without '#'


def test_negation_word_preservation(preprocessor):
    """
    CRITICAL: Verify that negation words like 'not', 'never', 'no'
    are NEVER stripped by the stopword filter.
    """
    raw = "This was not good and never worked."
    cleaned = preprocessor.clean_text(raw)
    assert "not" in cleaned
    assert "never" in cleaned
    assert "worked" in cleaned


def test_contractions_expansion(preprocessor):
    raw = "I can't recommend this and wouldn't buy again."
    cleaned = preprocessor.clean_text(raw)
    assert "not" in cleaned


def test_empty_and_none_input(preprocessor):
    assert preprocessor.clean_text("") == ""
    assert preprocessor.clean_text(None) == ""
    assert preprocessor.clean_text("   \n\t  ") == ""


def test_non_string_input(preprocessor):
    assert preprocessor.clean_text(12345) == ""
    assert preprocessor.clean_text({}) == ""


def test_pipeline_steps(preprocessor):
    raw = "<b>Not bad</b> at all! Visit https://test.com"
    steps = preprocessor.get_pipeline_steps(raw)
    assert "raw" in steps
    assert "html_stripped" in steps
    assert "urls_mentions_normalized" in steps
    assert "not" in steps["negation_preserved_filtered"]
