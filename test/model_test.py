import torch
from pricing_model import GenAIPricingTransformer, GenAIPricingStrategy
import pandas as pd

df = pd.read_csv("dataset/df_sampled_olist.csv")
print(f"Loading model with input_dim: {df.shape}")

# Initialize the pricing strategy
pricing_strategy = GenAIPricingStrategy()

        
df_processed = pricing_strategy.preprocess_features(df)
print(f"Loading model with input_dim: {df_processed.shape}")
X = pricing_strategy.prepare_features(df_processed)

        # Load the model (ensure input_dim is set correctly based on your data)
input_dim = X.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenAIPricingTransformer(input_dim=input_dim).to(device)

try:
    state_dict = torch.load('best_model.pth', map_location=device)
    print("State dict loaded. Keys:", state_dict.keys())
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

