import streamlit as st
from chatbot import FAQChatbot

st.set_page_config(page_title="AI FAQ Chatbot", page_icon="🤖", layout="centered")

CSV_PATH = "faq_data.csv"

SUGGESTED_QUESTIONS = [
    "How can I register for an internship?",
    "How do I submit my project?",
    "When will I receive my certificate?",
    "How can I reset my password?",
    "How do I contact technical support?",
]


@st.cache_resource(show_spinner="Loading FAQ knowledge base...")
def load_chatbot():
    return FAQChatbot(csv_path=CSV_PATH)


def render_message(role, content, meta=None):
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content)
        if meta and meta.get("is_match"):
            st.markdown(f"**Category:** {meta['category']}  \n**Confidence:** {meta['confidence']}%")
            st.caption(f"🔎 Matched FAQ: {meta['matched_question']}")


def handle_user_question(question: str):
    question = question.strip()
    if not question:
        return
    st.session_state.messages.append({"role": "user", "content": question, "meta": None})
    result = st.session_state.bot.get_response(question)
    st.session_state.messages.append({"role": "bot", "content": result["answer"], "meta": result})


def main():
    st.title("🤖 AI FAQ Chatbot")
    st.caption("Ask questions about internships and online learning.")

    if "bot" not in st.session_state:
        try:
            st.session_state.bot = load_chatbot()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()
        except ValueError as e:
            st.error(f"There is a problem with the FAQ data file: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Something went wrong while starting the chatbot: {e}")
            st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.subheader("💡 Suggested Questions")
        for q in SUGGESTED_QUESTIONS:
            if st.button(q, use_container_width=True):
                handle_user_question(q)

        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    for msg in st.session_state.messages:
        render_message(msg["role"], msg["content"], msg.get("meta"))

    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("Type your question here...", key="user_input", label_visibility="collapsed")
    with col2:
        send_clicked = st.button("Send", use_container_width=True)

    if send_clicked and user_input:
        handle_user_question(user_input)
        st.rerun()


if __name__ == "__main__":
    main()