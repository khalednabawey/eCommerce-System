from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from forecasting_utils import LSTMModel, TimeSeriesPreprocessor, predict_next_month

app = FastAPI()

# Load your model and scaler paths
MODEL_PATH = './artifacts/lstm_model.h5'
CSV_PATH = './dataset/df_sampled_olist.csv'

# Initialize the preprocessor and load the dataset
olist_df = pd.read_csv("./dataset/df_sampled_olist.csv")
preprocessor = TimeSeriesPreprocessor(olist_df)
preprocessor.preprocess()

# Initialize model
lstm_model = LSTMModel(seq_length=7)
lstm_model.load_model(MODEL_PATH)


@app.post("/forecast")
async def forecast():
    # Extract the last observed values from the preprocessed data
    last_row = preprocessor.daily_metrics.iloc[-1]
    last_observed = [
        last_row['daily_orders'],
        last_row['payment_value'],
        last_row['price'],
    ]

    # Predict the next 30 days based on the last observed data
    predictions = predict_next_month(last_observed, lstm_model)

    # Prepare the response
    forecasted_data = {
        "daily_orders": predictions[:, 0].tolist(),
        "payment_value": predictions[:, 1].tolist(),
        "price": predictions[:, 2].tolist(),
    }
    return forecasted_data
