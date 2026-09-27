import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_text

CONFIDENCE_THRESHOLD = 0.25

FALLBACK_MESSAGE = (
    "Sorry, I couldn't find a relevant answer to your question. "
    "Please try asking about registration, internship, projects, "
    "certificates, submissions, account, payment, or technical support."
)


class FAQChatbot:
    def __init__(self, csv_path: str = "faq_data.csv", threshold: float = CONFIDENCE_THRESHOLD):
        self.csv_path = csv_path
        self.threshold = threshold
        self.df = None
        self.vectorizer = None
        self.faq_vectors = None
        self._load_data()
        self._build_vectors()
    def _load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"FAQ data file not found at '{self.csv_path}'. "
                "Please make sure faq_data.csv exists in the project folder."
            )

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise ValueError(f"Could not read '{self.csv_path}'. Invalid CSV format: {e}")

        required_columns = {"id", "question", "answer", "category"}
        if not required_columns.issubset(set(df.columns)):
            raise ValueError(
                f"faq_data.csv must contain columns: {required_columns}. "
                f"Found: {set(df.columns)}"
            )

        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
        if df.empty:
            raise ValueError("faq_data.csv has no valid rows after cleaning.")

        self.df = df

    def _build_vectors(self):
        # Preprocess every FAQ question once, at startup
        self.df["clean_question"] = self.df["question"].apply(preprocess_text)

        # ngram_range=(1, 2) lets the vectorizer also look at word pairs
        # (e.g. "forgot password"), which makes paraphrased questions
        # match better than single words alone.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.faq_vectors = self.vectorizer.fit_transform(self.df["clean_question"])

    def get_response(self, user_question: str) -> dict:
        if not user_question or not user_question.strip():
            return {
                "answer": "Please type a question so I can help you.",
                "matched_question": None,
                "category": None,
                "confidence": 0.0,
                "is_match": False,
            }

        clean_query = preprocess_text(user_question)

        if not clean_query:
            return {
                "answer": FALLBACK_MESSAGE,
                "matched_question": None,
                "category": None,
                "confidence": 0.0,
                "is_match": False,
            }

        query_vector = self.vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vector, self.faq_vectors).flatten()

        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score >= self.threshold:
            row = self.df.iloc[best_idx]
            return {
                "answer": row["answer"],
                "matched_question": row["question"],
                "category": row["category"],
                "confidence": round(best_score * 100, 2),
                "is_match": True,
            }

        return {
            "answer": FALLBACK_MESSAGE,
            "matched_question": None,
            "category": None,
            "confidence": round(best_score * 100, 2),
            "is_match": False,
        }


if __name__ == "__main__":
    bot = FAQChatbot(csv_path="faq_data.csv")

    test_questions = [
        "How do I register?",
        "Where can I apply for internship?",
        "How can I submit my project?",
        "When do I get my certificate?",
        "I forgot my account password.",
        "Can I update my profile?",
        "What is the internship process?",
        "My payment failed, what should I do?",
        "The site is not loading",
        "What is today's weather?",
        "Who is the Prime Minister?",
        "Tell me a joke.",
    ]

    for q in test_questions:
        result = bot.get_response(q)
        print(f"\nUser: {q}")
        print(f"Bot : {result['answer']}")
        if result["is_match"]:
            print(f"  Category   : {result['category']}")
            print(f"  Matched FAQ: {result['matched_question']}")
        print(f"  Confidence : {result['confidence']}%")