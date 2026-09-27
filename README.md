# 🤖 AI-Powered FAQ Chatbot (NLP, No External API)

## 1. Project Overview
A fully local FAQ chatbot for an internship/online-learning platform. It matches
user questions to a predefined FAQ dataset using TF-IDF + Cosine Similarity —
no external API, no API key, no backend server.

## 2. Features
- 100% local, no API key needed
- Real NLP similarity matching (not hardcoded if/else)
- Confidence score + category shown with every answer
- Configurable threshold with graceful fallback
- Simple Streamlit chat UI with history, suggested questions, clear chat

## 3. Technologies
Python, Streamlit, Pandas, NLTK, Scikit-learn (TF-IDF + Cosine Similarity)

## 4. Project Structure
faq-chatbot/
├── app.py
├── chatbot.py
├── preprocessing.py
├── faq_data.csv
├── requirements.txt
└── README.md


## 5. NLP Preprocessing
Lowercase → remove punctuation → tokenize → remove stopwords → lemmatize.
Example: "How can I register for an internship?" → "register internship"

## 6. TF-IDF
Converts text into numeric vectors, giving more weight to distinctive words
and less weight to common words that appear in almost every question.

## 7. Cosine Similarity
Measures how close two TF-IDF vectors are (0 = unrelated, 1 = identical topic),
regardless of sentence length.

## 8. Confidence Threshold
Best match must score ≥ 0.35 (35%) to be trusted. Below that, the chatbot
returns a fallback message instead of guessing.

## 9. How the Chatbot Works

User Question → Preprocessing → TF-IDF → Cosine Similarity →
Best Matching FAQ → Confidence Check → Answer / Fallback


## 10. Installation

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt


## 11. How to Run

streamlit run app.py


## 12. Example Questions
"How can I register?", "Where can I apply for internship?",
"How can I submit my project?", "I forgot my password."

## 13. Future Improvements
- Sentence-embeddings for better semantic matching
- Spelling correction
- Multi-turn conversation memory
- Admin panel to manage FAQs
STEP 7 — PowerShell Commands
powershell
mkdir faq-chatbot
cd faq-chatbot

python -m venv venv

.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

streamlit run app.py