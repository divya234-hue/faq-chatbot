import re
import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Preprocess raw user/FAQ text into a normalized form suitable
    for TF-IDF vectorization.

    Pipeline: lowercase -> remove punctuation -> tokenize ->
    remove stopwords -> lemmatize -> rejoin.

    Example:
        "How can I reset my Password?" -> "reset password"
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in _stop_words]
    tokens = [_lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)
