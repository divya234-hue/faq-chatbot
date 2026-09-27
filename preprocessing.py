"""
preprocessing.py
-----------------
Text preprocessing utilities for the FAQ Chatbot.
"""

import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
    """
    Makes sure required NLTK data packages are downloaded.
    Runs automatically the first time, then never again.
    """
    required_packages = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in required_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


_ensure_nltk_data()

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    """
    Cleans and normalizes text for NLP similarity matching.
    Steps: lowercase -> remove punctuation -> tokenize ->
           remove stopwords -> lemmatize -> join.
    """
    if not text or not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords (and single-character junk tokens)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]

    # 5. Lemmatize
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    # 6. Join back into a cleaned string
    return " ".join(tokens)


if __name__ == "__main__":
    sample = "How can I register for an internship?"
    print("Input :", sample)
    print("Output:", preprocess_text(sample))