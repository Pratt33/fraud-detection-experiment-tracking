# Simple Credit Card Fraud Detection - Model Training
# Easy to understand version for beginners

import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Load the preprocessed data
X_train = pd.read_csv('data/processed/X_train.csv')
y_train = pd.read_csv('data/processed/y_train.csv')['Class']
X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')['Class']

print(f"Training data: {len(X_train):,} transactions")
print(f"Test data: {len(X_test):,} transactions")
print(f"Training fraud rate: {y_train.sum()/len(y_train)*100:.1f}%")
print(f"Test fraud rate: {y_test.sum()/len(y_test)*100:.1f}%")

# Step 2: Train different models
# Dictionary to store our models
models = {}
results = {}

# Model 1: Logistic Regression (simple and fast)
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
models['Logistic Regression'] = lr_model

# Model 2: Random Forest (more powerful)
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model

# Step 3: Evaluate each model

for model_name, model in models.items():
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    results[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    # Show results
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  Precision: {precision:.3f} (How many predicted frauds were actually fraud)")
    print(f"  Recall:    {recall:.3f} (How many actual frauds were caught)")
    print(f"  F1-Score:  {f1:.3f} (Balance of precision and recall)")
    print(f"  AUC:       {auc:.3f} (Overall model performance)")

# Step 4: Find the best model
best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])

print(f"Best model: {best_model_name} (F1-Score: {results[best_model_name]['f1_score']:.3f})")

# Create a simple comparison chart
model_names = list(results.keys())
f1_scores = [results[name]['f1_score'] for name in model_names]
precisions = [results[name]['precision'] for name in model_names]
recalls = [results[name]['recall'] for name in model_names]

plt.figure(figsize=(10, 6))
x = range(len(model_names))
width = 0.25

plt.bar([i - width for i in x], f1_scores, width, label='F1-Score', alpha=0.8)
plt.bar(x, precisions, width, label='Precision', alpha=0.8)
plt.bar([i + width for i in x], recalls, width, label='Recall', alpha=0.8)

plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, model_names)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Step 5: Save the best model
# Create models directory
import os
os.makedirs('models', exist_ok=True)

# Save all models
for model_name, model in models.items():
    filename = f"models/{model_name.lower().replace(' ', '_')}_model.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
# Save results
with open('visualizations/results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Save results as CSV for easy viewing
results_df = pd.DataFrame(results).T
results_df.to_csv('visualizations/model_results.csv')

# Step 6: Test the best model with some examples

best_model = models[best_model_name]

# Get some fraud and normal examples
fraud_examples = X_test[y_test == 1].head(3)
normal_examples = X_test[y_test == 0].head(3)

fraud_probabilities = best_model.predict_proba(fraud_examples)[:, 1]
normal_probabilities = best_model.predict_proba(normal_examples)[:, 1]