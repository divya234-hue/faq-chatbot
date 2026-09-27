import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


def _ensure_nltk_data():
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
    if not text or not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words and len(t) > 1]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)
if __name__ == "__main__":
    sample = "How can I register for an internship?"
    print("Input :", sample)
    print("Output:", preprocess_text(sample))