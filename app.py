import streamlit as st
import pandas as pd
import requests
from threading import Thread
import torch
from pricing_model import GenAIPricingStrategy  # Import your pricing strategy class
from chatbot_api import run_api  # Import your FastAPI function

# Function to start the FastAPI server in a separate thread
def start_api():
    run_api()

# Start the FastAPI server in a new thread
api_thread = Thread(target=start_api)
api_thread.start()

# Define the base URL of your Chatbot API
API_URL = "http://127.0.0.1:8000/generate-response"

# Initialize the pricing strategy
pricing_strategy = GenAIPricingStrategy()

# Streamlit sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Pricing Strategy", "Chatbot"])

# Pricing Strategy Page
if page == "Pricing Strategy":
    st.title("Pricing Strategy Predictor")

    # File uploader for CSV
    uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if uploaded_file:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.write(df.head())  # Display the first few rows of the dataframe

        # Preprocess and prepare features
        df_processed = pricing_strategy.preprocess_features(df)
        X = pricing_strategy.prepare_features(df_processed)

        # Load the model (ensure input_dim is set correctly based on your data)
        input_dim = X.shape[1]  # Get number of features
        pricing_strategy.load_model(input_dim)

        # Get predictions for the first sample
        sample_features = X[0:1]  # Example: first product in dataset
        sample_features_tensor = torch.tensor(sample_features, dtype=torch.float32).to(pricing_strategy.device)

        # Get recommendations from the model
        recommendations = pricing_strategy.model(sample_features_tensor)

        # Extracting predictions
        market_condition_tensor, price_elasticity_tensor, recommended_prices_tensor = recommendations  # Assuming the second tensor is not used

        # Handle market_condition tensor
        market_condition = market_condition_tensor.item() if market_condition_tensor.numel() == 1 else market_condition_tensor.mean().item()
        
        # Handle price_elasticity tensor
        price_elasticity = price_elasticity_tensor.item() if price_elasticity_tensor.numel() == 1 else price_elasticity_tensor.mean().item()

        # Pricing Strategy Recommendations
        st.subheader("Pricing Strategy Recommendations")
        st.write(f"Market Condition: {market_condition:.2f}")
        st.write(f"Price Elasticity: {price_elasticity:.2f}")

        # Handle recommended prices tensor
        recommended_prices = [price.item() for price in recommended_prices_tensor]

        # Display recommended prices
        st.subheader("Recommended Prices:")
        strategies = ['Strategy']  # Modify this list based on your strategies
        for strategy, price in zip(strategies, recommended_prices):
            st.write(f"{strategy}: ${price:.2f}")
        
        

# Chatbot Page
elif page == "Chatbot":
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
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
            else:
                st.error("Error: Unable to get a response from the API.")
        except Exception as e:
            st.error(f"Exception: {str(e)}. Could not connect to the API.")
            
            
            


