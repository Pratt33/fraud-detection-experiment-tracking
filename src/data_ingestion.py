# Simple script to download credit card fraud dataset
# Perfect for beginners learning fraud detection!

import pandas as pd
import kagglehub
import os

# Download the dataset from Kaggle
dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

# Load the data
data = pd.read_csv(os.path.join(dataset_path, "creditcard.csv"))

# Create a simple folder structure
os.makedirs("data/raw", exist_ok=True)

# Save the dataset locally
output_file = "data/raw/creditcard.csv"
data.to_csv(output_file, index=False)

# Quick peek at the data
fraud_count = data['Class'].sum()
normal_count = len(data) - fraud_count