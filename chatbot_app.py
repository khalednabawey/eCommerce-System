import streamlit as st
import requests
from threading import Thread
from chatbot_api import run_api

# Run the API in a separate thread to avoid blocking Streamlit


def start_api():
    run_api()


# Start the FastAPI server in a new thread
api_thread = Thread(target=start_api)
api_thread.start()

# Define the base URL of your Chatbot API
API_URL = "http://127.0.0.1:8000/generate-response"

# Title and description
st.title("💬 Olist E-Commerce Chatbot")
st.caption("🚀 E-commerce Customer Assist chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "How can I assist you with your order today?"}
    ]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# User input
if prompt := st.chat_input(placeholder="Type your question here..."):

    # Add user's message to the session state and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Make a request to the FastAPI chatbot API
    try:
        with st.spinner("Generating response..."):
            response = requests.post(
                API_URL,
                json={"prompt": prompt}
            )

        # Parse the response from API
        if response.status_code == 200:
            msg = response.json().get("response", "Sorry, I couldn't understand that.")
            st.session_state.messages.append(
                {"role": "assistant", "content": msg})
            st.chat_message("assistant").write(msg)
        else:
            st.error("Error: Unable to get a response from the API.")
    except Exception as e:
        st.error(f"Exception: {str(e)}. Could not connect to the API.")
