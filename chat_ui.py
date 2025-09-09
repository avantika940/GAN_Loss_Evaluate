import streamlit as st
import requests
import time
import os

# Backend API URL

# CHAT_API_URL = os.getenv("CHAT_API_URL", "http://chat_agent:8000/chat")
CHAT_API_URL = os.getenv("CHAT_API_URL", "http://chat_agent:8000/moderate")
# CHAT_API_URL = "http://root_agent:8000/query"
VALID_TOKEN = "hub-agents"

# Streamlit page config
st.set_page_config(page_title="HR Chatbot", layout="centered")
st.title("🤖 HR Assistant Chatbot")

# Auth state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# Token prompt if not authenticated
if not st.session_state.authenticated:
    token_input = st.text_input("🔐 Enter Access Token", type="password")
    if st.button("Login"):
        if token_input == VALID_TOKEN:
            st.success("✅ Access granted.")
            st.session_state.authenticated = True
        else:
            st.error("❌ Invalid token.")
    st.stop()  # Stop further execution if not authenticated

# Chat history and session ID setup
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = None  # Will be set after first response

# Display chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Input box
query = st.chat_input("Ask your HR assistant...")


# If user submits a query
if query:
    # Show user query
    st.session_state.chat_history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Prepare request payload with session_id
    payload = {"query": query,
                "user_id": "user-123",
                "org_id": "org-xyz"
                }
    if st.session_state.session_id:
        payload["session_id"] = st.session_state.session_id  # 🔁 maintain session

    # Assistant response block
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💬 *Typing...*")

        try:
            response = requests.post(CHAT_API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                print(data)
                final_answer = data.get("chats", "No response.")
                session_id = data.get("session_id")

                # Save session_id from first response
                if session_id and not st.session_state.session_id:
                    st.session_state.session_id = session_id  #  maintain session
            else:
                final_answer = f" Error: {response.status_code} - {response.text}"
        except Exception as e:
            final_answer = f" Request failed: {str(e)}"

        # Typing animation
        full_response = ""
        for char in final_answer:
            full_response += char
            placeholder.markdown(full_response + "▌")
            time.sleep(0.02)
        placeholder.markdown(full_response)

        # Save assistant reply
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
