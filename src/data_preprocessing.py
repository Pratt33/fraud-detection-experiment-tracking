import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Step 1: Load the data
data = pd.read_csv("data/raw/creditcard.csv")

# Step 2: Basic data exploration
fraud_cases = data['Class'].sum()
normal_cases = len(data) - fraud_cases

# Step 3: feature preparation

# Fix extreme values in Amount (cap at 99th percentile)
amount_cap = data['Amount'].quantile(0.99)
data['Amount'] = np.minimum(data['Amount'], amount_cap)

# Transform Amount using log (makes the data more normal)
data['Amount_log'] = np.log1p(data['Amount'])

# Convert Time to hours
data['Time_hours'] = data['Time'] / 3600

# Step 4: Scale the features (make them similar ranges)

# Features to use for modeling
feature_columns = ['Time_hours', 'Amount_log'] + [f'V{i}' for i in range(1, 29)]

# Separate features (X) and target (y)
X = data[feature_columns]
y = data['Class']

# Scale the features to have similar ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=feature_columns)

# Step 5: Split into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2,  # 20% for testing
    random_state=42,  # for reproducible results
    stratify=y  # keep fraud ratio same in both sets
)

# Step 6: Balance the training data using SMOTE

# SMOTE creates synthetic fraud examples to balance the data
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Step 7: Save the processed data
# Save training data
pd.DataFrame(X_train_balanced, columns=feature_columns).to_csv('data/processed/X_train.csv', index=False)
pd.DataFrame(y_train_balanced, columns=['Class']).to_csv('data/processed/y_train.csv', index=False)

# Save test data (original distribution)
X_test.to_csv('data/processed/X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Class']).to_csv('data/processed/y_test.csv', index=False)

# Save the scaler for later use
import pickle
with open('data/processed/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)