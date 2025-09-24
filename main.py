
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever
from dotenv import load_dotenv

# Load API key
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# Set page config
st.set_page_config(page_title="Barai Relata Chatbot", page_icon="👨‍👩‍👧‍👦", layout="centered")
st.title("👨‍👩‍👧‍👦 Barai Relata Chatbot")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;} /* Hide hamburger menu */
    header {visibility: hidden;}    /* Hide Streamlit header */
    footer {visibility: hidden;}    /* Hide default footer */

    /* Custom sticky footer */
    .custom-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: gray;
        border-top: 1px solid #ddd;
        z-index: 9999;
    }
    </style>

    <div class="custom-footer">
        © 2025 Barai – Built with ❤️ by <b>M-Ashraf-Barai</b>
    </div>
    """,
    unsafe_allow_html=True
)

# Model setup
model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=api_key,
    streaming=True  # enable streaming
)


# Prompt template
template = """
You are a family knowledge assistant.
    - Respond in a fun, casual, and witty tone. Add light jokes when possible, but keep answers short and clear.
    - Only answer questions using the information in the family_data.txt knowledge base.
    - Do NOT invent or assume information.
    - Do NOT discuss anything about car or motorcycle unless the user specifically asks about car or motorcycle.
    - Do NOT include phrases like:
        "The search results did not provide..."
        "Based on the search results..."
    - Answer should not exceed 25 words
    - If user talks about "ali ba" or "ali", take it as "Haji Ali Muhammad".
    - When the user mentions "iqbal chacha", always interpret it as "Iqbal Husain".
    - Husain means Ghulam Husain
    - If user uses word "eat", take it as "consume"
    - abba = Muhammad Qasim
    - bapa = Muhammad Qasim
    - qasim = Muhammad Qasim
    - ali ba = Haji Ali Muhammad
    - ali = Haji Ali Muhammad
    - rasheed = Haroon Rasheed
    - aziz = Abdul Azeez
    - iqbal = Iqbal Husain
    - kashu = Kashif
    - If you don't find answer see first all lines with "=", and even after this you don't find answer then return sorry! I don't know.
    Keep responses short and direct.

{reviews}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Keep chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input box
if question := st.chat_input("Ask about your family..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve facts
    reviews = retriever.invoke(question)

    # Build chain
    chain = prompt | model

    # Show assistant message with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        for chunk in chain.stream({"reviews": reviews, "question": question}):
            if chunk.content:
                full_response += chunk.content
                message_placeholder.markdown(full_response + "▌")

        message_placeholder.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
