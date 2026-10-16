"""
SentimentIQ - NLP Text Preprocessing Module
Provides reproducible, sentiment-aware text normalization and tokenization.
Preserves critical sentiment signals such as negations ("not", "never", "no").
"""

import re
import html
from typing import List, Optional, Set

# Negation words that dramatically invert sentiment polarity
# NEVER strip these out during stopword filtering
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nowhere",
    "hardly", "scarcely", "barely", "cannot", "cant", "can't", "dont",
    "don't", "didnt", "didn't", "wont", "won't", "wouldnt", "wouldn't",
    "shouldnt", "shouldn't", "couldnt", "couldn't", "isnt", "isn't",
    "arent", "aren't", "wasnt", "wasn't", "werent", "weren't", "hasnt",
    "hasn't", "havent", "haven't", "hadnt", "hadn't", "without"
}

# Default standard English stopwords (excluding negations)
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "me", "more", "most", "my", "myself",
    "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should",
    "so", "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
} - NEGATION_WORDS


class TextPreprocessor:
    """
    Sentiment-aware NLP Preprocessor.
    Ensures identical text transformation across training and inference.
    """

    def __init__(self, remove_stopwords: bool = True, custom_stopwords: Optional[Set[str]] = None):
        self.remove_stopwords = remove_stopwords
        if custom_stopwords is not None:
            self.stopwords = custom_stopwords - NEGATION_WORDS
        else:
            self.stopwords = DEFAULT_STOPWORDS

    def clean_text(self, text: Optional[str]) -> str:
        """
        Full text cleaning pipeline:
        1. Type checking and fallback for None / non-string
        2. HTML entity unescaping and tag stripping
        3. URL, email, and social mention normalization
        4. Contraction expansion / standardizing
        5. Punctuation handling preserving sentiment indicators
        6. Whitespace normalization
        7. Configurable stopword removal with negation preservation
        """
        if text is None or not isinstance(text, str):
            return ""

        # Step 1: Decode HTML entities (&amp;, &lt;, etc.)
        cleaned = html.unescape(text)

        # Step 2: Remove HTML markup tags (<p>, <br />, </div>)
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        # Step 3: Normalize URLs and mentions
        cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
        cleaned = re.sub(r"@\w+", " ", cleaned)  # Strip @mentions
        cleaned = re.sub(r"#(\w+)", r"\1", cleaned)  # Keep hashtag words without #

        # Step 4: Expand common contractions to preserve negation signals
        contractions = {
            r"\bcan\'t\b": "can not",
            r"\bcannot\b": "can not",
            r"\bwon\'t\b": "will not",
            r"\bn\'t\b": " not",
            r"\b\'re\b": " are",
            r"\b\'s\b": " is",
            r"\b\'d\b": " would",
            r"\b\'ll\b": " will",
            r"\b\'ve\b": " have",
            r"\b\'m\b": " am",
        }
        for pattern, replacement in contractions.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)

        # Step 5: Convert to lowercase
        cleaned = cleaned.lower()

        # Step 6: Retain alphanumeric characters, spaces, and hyphens (word boundaries)
        # Strip out symbols like @, #, $, %, ^, &, *, ~, `, quotes, brackets
        cleaned = re.sub(r"[^\w\s-]", " ", cleaned)
        cleaned = re.sub(r"[-_]+", " ", cleaned)

        # Step 7: Normalize whitespace
        tokens = cleaned.split()

        # Step 8: Optional stopword filtering (preserving negation words)
        if self.remove_stopwords:
            tokens = [w for w in tokens if w not in self.stopwords and len(w) > 1 or w in NEGATION_WORDS]

        return " ".join(tokens)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize cleaned text into word tokens."""
        cleaned = self.clean_text(text)
        return cleaned.split()

    def get_pipeline_steps(self, text: str) -> dict:
        """
        Returns step-by-step transformation for UI inspection and educational visualizer.
        """
        raw = text if text is not None else ""
        step1_html = re.sub(r"<[^>]+>", " ", html.unescape(raw))
        step2_urls = re.sub(r"https?://\S+|www\.\S+", " ", step1_html)
        step2_urls = re.sub(r"@\w+", " ", step2_urls)
        step2_urls = re.sub(r"#(\w+)", r"\1", step2_urls)

        step3_contractions = step2_urls
        for pat, rep in [(r"\bcan\'t\b", "can not"), (r"\bn\'t\b", " not")]:
            step3_contractions = re.sub(pat, rep, step3_contractions, flags=re.IGNORECASE)

        step4_lower = step3_contractions.lower()
        step5_symbols = re.sub(r"[^\w\s-]", " ", step4_lower)
        step5_symbols = re.sub(r"[-_]+", " ", step5_symbols)
        step6_tokens = step5_symbols.split()
        step7_filtered = [w for w in step6_tokens if (w not in self.stopwords and len(w) > 1) or w in NEGATION_WORDS]

        return {
            "raw": raw,
            "html_stripped": step1_html.strip(),
            "urls_mentions_normalized": step2_urls.strip(),
            "contractions_expanded": step3_contractions.strip(),
            "lowercased": step4_lower.strip(),
            "clean_tokens": step6_tokens,
            "negation_preserved_filtered": " ".join(step7_filtered)
        }


# Global preprocessor instance for fast reuse
preprocessor = TextPreprocessor(remove_stopwords=True)


def clean_text(text: Optional[str]) -> str:
    """Convenience functional wrapper."""
    return preprocessor.clean_text(text)
