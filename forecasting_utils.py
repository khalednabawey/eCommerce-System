import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dense, Dropout, Input, Attention


class TimeSeriesPreprocessor:
    def __init__(self, olist_df):
        self.olist_df = olist_df
        self.daily_metrics = None
        self.scaler = None

    def preprocess(self):
        time_df = self.olist_df.copy()
        time_df['order_purchase_timestamp'] = pd.to_datetime(
            time_df['order_purchase_timestamp'])

        time_df['year'] = time_df['order_purchase_timestamp'].dt.year
        time_df['month'] = time_df['order_purchase_timestamp'].dt.month
        time_df['day'] = time_df['order_purchase_timestamp'].dt.day
        time_df['day_of_week'] = time_df['order_purchase_timestamp'].dt.dayofweek
        time_df['is_weekend'] = time_df['day_of_week'].apply(
            lambda x: 1 if x >= 5 else 0)

        # Set index and resample
        time_df.set_index('order_purchase_timestamp', inplace=True)
        self.daily_metrics = time_df.resample('D').agg(
            {'order_id': 'count', 'payment_value': 'sum', 'price': 'mean'})
        self.daily_metrics.rename(
            columns={'order_id': 'daily_orders'}, inplace=True)

        # Handle missing values
        self.daily_metrics.ffill(inplace=True)  # Forward fill

    def create_sequences(self, seq_length):
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.daily_metrics)

        X, y = [], []
        for i in range(len(scaled_data) - seq_length):
            X.append(scaled_data[i:i + seq_length])
            y.append(scaled_data[i + seq_length])
        return np.array(X), np.array(y)


class LSTMModel:
    def __init__(self, seq_length):
        self.seq_length = seq_length
        self.model = None

    def build_model_with_attention(self, input_shape):
        inputs = Input(shape=input_shape)
        lstm_out = LSTM(64, return_sequences=True)(inputs)
        # Query and value from the same LSTM output
        attention = Attention()([lstm_out, lstm_out])
        lstm_out = LSTM(32)(attention)
        outputs = Dense(3)(lstm_out)  # Output layer for three metrics
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = load_model(model_path)

    def evaluate(self, y_test, predictions):
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        # Calculate RMSE and MAE for each metric
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        return rmse, mae


def run_time_series_analysis(olist_df, seq_length=7, epochs=50, batch_size=32,
                             model_path='./artifacts/lstm_model.h5'):
    # Initialize and preprocess the data
    preprocessor = TimeSeriesPreprocessor(olist_df)
    preprocessor.preprocess()

    # Create sequences
    X, y = preprocessor.create_sequences(seq_length)

    # Split data
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Build and train LSTM model
    lstm_model = LSTMModel(seq_length)
    lstm_model.build_model_with_attention(
        input_shape=(X_train.shape[1], X_train.shape[2]))
    lstm_model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the model
    lstm_model.save_model(model_path)

    # Make predictions
    predictions = lstm_model.predict(X_test)

    # Evaluate model and return results as DataFrame
    evaluation_results = lstm_model.evaluate(y_test, predictions)
    return evaluation_results


def predict_next_month(last_sequence, seq_length=7):
    """Predicts the next 30 days using the last available sequence."""
    all_predictions = []

    for _ in range(30):  # Predict for the next 30 days
        # Make a prediction
        prediction = lstm_model.predict(
            last_sequence[np.newaxis, ...])  # Add batch dimension
        all_predictions.append(prediction)

        # Update the last_sequence: remove the oldest observation and add the new prediction
        last_sequence = np.append(last_sequence[1:], prediction, axis=0)

    # Convert predictions to a numpy array and return
    return np.array(all_predictions)
