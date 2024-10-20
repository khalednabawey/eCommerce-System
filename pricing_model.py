# Import Libraries
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import GPT2Config, GPT2Model
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


def normalize_prices(prices, min_price=1e-8):
    """Normalize prices to prevent numerical instability."""
    return np.clip(prices, min_price, None)

# Model Architecture
class GenAIPricingTransformer(nn.Module):
    def __init__(self, input_dim, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=1,
            n_positions=input_dim,
            n_ctx=input_dim,
            n_embd=256,
            n_layer=n_layers,
            n_head=n_heads,
            n_inner=1024,
            activation_function='gelu',
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            scale_attn_weights=True
        )
        
        self.feature_embedding = nn.Linear(input_dim, 256)
        self.transformer = GPT2Model(self.config)
        
        self.price_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self.market_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )
        
        self.elasticity_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        embedded = self.feature_embedding(x)
        transformer_out = self.transformer(inputs_embeds=embedded.unsqueeze(1)).last_hidden_state
        
        price_pred = self.price_head(transformer_out.squeeze(1))
        market_pred = self.market_head(transformer_out.squeeze(1))
        elasticity_pred = self.elasticity_head(transformer_out.squeeze(1))
        
        return price_pred, market_pred, elasticity_pred


# Pricing Strategy Class
class GenAIPricingStrategy:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_price = 1e-8

    def preprocess_features(self, df):
        processed_df = df.copy()

        # Normalize prices first
        price_columns = ['price', 'payment_value']
        for col in price_columns:
            if col in processed_df.columns:
                processed_df[col] = normalize_prices(processed_df[col], self.min_price)

        # Handle datetime features
        datetime_cols = ['order_purchase_timestamp', 'order_approved_at',
                         'order_delivered_carrier_date', 'order_delivered_customer_date']

        for col in datetime_cols:
            if col in processed_df.columns:
                processed_df[col] = pd.to_datetime(processed_df[col])
                processed_df[f'{col}_hour'] = processed_df[col].dt.hour
                processed_df[f'{col}_day'] = processed_df[col].dt.day
                processed_df[f'{col}_month'] = processed_df[col].dt.month

        # Create basic features with numerical stability
        processed_df['product_volume'] = (
            processed_df['product_length_cm'].clip(lower=0.1) *
            processed_df['product_height_cm'].clip(lower=0.1) *
            processed_df['product_width_cm'].clip(lower=0.1)
        )

        # Market features
        processed_df['market_density'] = processed_df.groupby('product_category_name')['seller_id'].transform('nunique')
        processed_df['category_demand'] = processed_df.groupby('product_category_name')['order_id'].transform('count')

        # Price features with numerical stability
        processed_df['price_per_weight'] = (
            processed_df['price'] / processed_df['product_weight_g'].clip(lower=0.1)
        ).clip(lower=self.min_price)

        processed_df['price_per_volume'] = (
            processed_df['price'] / processed_df['product_volume'].clip(lower=0.1)
        ).clip(lower=self.min_price)

        # Log transform price-related features
        price_related_cols = ['price', 'price_per_weight', 'price_per_volume']
        for col in price_related_cols:
            if col in processed_df.columns:
                processed_df[f'{col}_log'] = np.log1p(processed_df[col])

        # Encode categorical variables
        cat_columns = ['product_category_name', 'customer_state', 'seller_state']
        for col in cat_columns:
            if col in processed_df.columns:
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                self.label_encoders[col] = le

        return processed_df

    def prepare_features(self, df):
        feature_columns = [
            'payment_sequential', 'payment_installments', 'product_weight_g',
            'price_per_weight_log', 'price_per_volume_log', 'category_demand',
            'market_density', 'product_photos_qty', 'product_volume',
            'product_category_name'
        ]

        # Check for missing columns
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
        
        available_columns = [col for col in feature_columns if col in df.columns]
        X = df[available_columns].values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        return X

    def load_model(self, input_dim):
        print(f"Loading model with input_dim: {input_dim}")

        self.model = GenAIPricingTransformer(input_dim=input_dim).to(self.device)
        try:
            state_dict = torch.load('artifacts/best_model.pth', map_location=self.device)
            print("State dict loaded successfully. Keys: ", state_dict.keys())
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except RuntimeError as e:
            print(f"Runtime error while loading model: {str(e)}")
            raise
        except FileNotFoundError:
            print("Model file not found. Please ensure 'best_model.pth' exists.")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")



