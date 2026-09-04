import json
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.services.preprocessing import clean_text

_DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "faqs.json"


class MatchResult:
    def __init__(self, answer: Optional[str], confidence: float,
                 matched_question: Optional[str], category: Optional[str]):
        self.answer = answer
        self.confidence = confidence
        self.matched_question = matched_question
        self.category = category


class FAQMatcher:
    """
    Loads the FAQ dataset and builds a TF-IDF matrix once at startup.
    Subsequent queries are matched against the precomputed matrix
    using cosine similarity, without refitting the vectorizer.
    """

    def __init__(self, data_path: Path = _DATA_PATH):
        with open(data_path, "r", encoding="utf-8-sig") as f:
            self.faqs = json.load(f)

        self._raw_questions = [faq["question"] for faq in self.faqs]
        self._cleaned_questions = [clean_text(q) for q in self._raw_questions]

        self.vectorizer = TfidfVectorizer()
        self.faq_matrix = self.vectorizer.fit_transform(self._cleaned_questions)

    def match(self, user_query: str, threshold: float) -> MatchResult:
        cleaned_query = clean_text(user_query)
        query_vector = self.vectorizer.transform([cleaned_query])

        similarities = cosine_similarity(query_vector, self.faq_matrix)[0]
        best_index = similarities.argmax()
        best_score = float(similarities[best_index])

        if best_score >= threshold:
            best_faq = self.faqs[best_index]
            return MatchResult(
                answer=best_faq["answer"],
                confidence=round(best_score, 4),
                matched_question=best_faq["question"],
                category=best_faq["category"],
            )

        return MatchResult(
            answer="Sorry, I couldn't find a relevant answer to your question.",
            confidence=round(best_score, 4),
            matched_question=None,
            category=None,
        )

