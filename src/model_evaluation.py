# Simple Credit Card Fraud Detection - Model Evaluation
# Detailed analysis of model performance

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, f1_score
from dvclive import Live

# Step 1: Load test data and models

X_test = pd.read_csv('data/processed/X_test.csv')
y_test = pd.read_csv('data/processed/y_test.csv')['Class']

# Load models
with open('models/logistic_regression_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('models/random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model
}

print(f"Loaded {len(models)} models")
print(f"Test data: {len(X_test):,} transactions ({y_test.sum()} frauds)")

# Step 2: Detailed evaluation for each model
with Live() as live:
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"{model_name.upper()} DETAILED RESULTS")
        print(f"{'='*50}")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Confusion Matrix:")
        print(f"  True Negatives (Correct Normal):  {tn:,}")
        print(f"  False Positives (Wrong Fraud):    {fp:,}")
        print(f"  False Negatives (Missed Fraud):   {fn:,}")
        print(f"  True Positives (Caught Fraud):    {tp:,}")
        
        # Business Impact Metrics
        print(f"\nBusiness Impact:")
        if tp + fn > 0:  # Avoid division by zero
            fraud_catch_rate = tp / (tp + fn) * 100
            print(f"  Fraud Catch Rate: {fraud_catch_rate:.1f}% ({tp} out of {tp + fn} frauds caught)")
        
        if fp + tn > 0:
            false_alarm_rate = fp / (fp + tn) * 100
            print(f"  False Alarm Rate: {false_alarm_rate:.1f}% ({fp} normal transactions flagged)")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

        # Calculate and log evaluation metrics with model prefix
        test_accuracy = (y_pred == y_test).mean()
        test_f1 = f1_score(y_test, y_pred)
        
        # Add model prefix to avoid conflicts
        prefix = model_name.lower().replace(' ', '_')
        live.log_metric(f"{prefix}_test_accuracy", test_accuracy)
        live.log_metric(f"{prefix}_test_f1", test_f1)
        
        # Generate and save confusion matrix plot (for visualization folder only)
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Fraud'], 
                    yticklabels=['Normal', 'Fraud'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'{model_name} Confusion Matrix')
        plot_filename = f'visualizations/{prefix}_confusion_matrix.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        # REMOVED: live.log_image() call

# Step 3: Create comprehensive visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Comprehensive Model Evaluation', fontsize=16, fontweight='bold')

# Chart 1: Confusion Matrices
for i, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'],
                ax=axes[0, i])
    axes[0, i].set_title(f'{model_name}\nConfusion Matrix')
    axes[0, i].set_xlabel('Predicted')
    axes[0, i].set_ylabel('Actual')

# Chart 2: ROC Curves
ax_roc = axes[0, 2]
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    auc = np.trapz(tpr, fpr)
    ax_roc.plot(fpr, tpr, label=f'{model_name} (AUC = {abs(auc):.3f})', linewidth=2)

ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('ROC Curves')
ax_roc.legend()
ax_roc.grid(True, alpha=0.3)

# Chart 3: Precision-Recall Curves
ax_pr = axes[1, 0]
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    ax_pr.plot(recall, precision, label=model_name, linewidth=2)

ax_pr.set_xlabel('Recall (Fraud Catch Rate)')
ax_pr.set_ylabel('Precision (Accuracy of Fraud Alerts)')
ax_pr.set_title('Precision-Recall Curves')
ax_pr.legend()
ax_pr.grid(True, alpha=0.3)

# Chart 4: Feature Importance (for Random Forest)
ax_feat = axes[1, 1]
if 'Random Forest' in models:
    rf_model = models['Random Forest']
    feature_names = X_test.columns
    importances = rf_model.feature_importances_
    
    # Get top 10 most important features
    indices = np.argsort(importances)[-10:]
    
    ax_feat.barh(range(len(indices)), importances[indices])
    ax_feat.set_yticks(range(len(indices)))
    ax_feat.set_yticklabels([feature_names[i] for i in indices])
    ax_feat.set_xlabel('Feature Importance')
    ax_feat.set_title('Top 10 Most Important Features\n(Random Forest)')

# Chart 5: Prediction Confidence Distribution
ax_conf = axes[1, 2]
for model_name, model in models.items():
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Separate probabilities for fraud and normal cases
    fraud_probs = y_pred_proba[y_test == 1]
    normal_probs = y_pred_proba[y_test == 0]
    
    ax_conf.hist(normal_probs, bins=50, alpha=0.7, label=f'{model_name} - Normal', density=True)
    ax_conf.hist(fraud_probs, bins=50, alpha=0.7, label=f'{model_name} - Fraud', density=True)

ax_conf.set_xlabel('Fraud Probability')
ax_conf.set_ylabel('Density')
ax_conf.set_title('Prediction Confidence Distribution')
ax_conf.legend()
ax_conf.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/detailed_evaluation.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 4: Fraud Detection Analysis
best_model_name = 'Random Forest'  # Usually performs better
best_model = models[best_model_name]

y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Analyze different threshold values
print(f"\nThreshold Analysis for {best_model_name}:")
print("Threshold | Frauds Caught | False Alarms | Precision | Recall")
print("-" * 60)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_test, y_pred_thresh)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"   {threshold:.1f}   |    {tp:3d}     |     {fp:3d}     | {precision:.3f}   | {recall:.3f}")

# Step 5: Save detailed results
# Create evaluation summary
evaluation_summary = {}
for model_name, model in models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    evaluation_summary[model_name] = {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'fraud_catch_rate': float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
        'false_alarm_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
        'confusion_matrix': cm.tolist()
    }

# Save detailed results
with open('models/detailed_evaluation.pkl', 'wb') as f:
    pickle.dump(evaluation_summary, f)

# Save as JSON for easy reading
import json
with open('visualizations/evaluation_summary.json', 'w') as f:
    json.dump(evaluation_summary, f, indent=2)
# Final recommendations
best_f1_model = max(models.keys(), key=lambda x: evaluation_summary[x]['fraud_catch_rate'])
print(f"Best Overall Model: {best_f1_model}")
print(f"Fraud Catch Rate: {evaluation_summary[best_f1_model]['fraud_catch_rate']:.1%}")
print(f"False Alarm Rate: {evaluation_summary[best_f1_model]['false_alarm_rate']:.1%}")