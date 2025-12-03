# svm_stock_real_denoised_v2.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)

# =====================================================
# 1. Load dataset saham sebenar
# =====================================================
base_path: str = r"C:\\Users\\AMJAD\\PycharmProjects\\fyp1test\\Lib\\Price Stock Dataset"
ticker: str = "AAPL"
date_suffix: str = "2025-10-08"

file_path = os.path.join(base_path, f"{ticker}_historical_data_{date_suffix}.csv")

df = pd.read_csv(file_path)

# Pastikan nama kolum betul
df.columns = [c.strip() for c in df.columns]

# Tukar Volume kepada numeric (buang koma)
df["Volume"] = df["Volume"].astype(str).str.replace(",", "").astype(float)

# Convert Date → datetime
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)

# Sort ikut tarikh (ascending)
df = df.sort_values("Date").reset_index(drop=True)

# =====================================================
# 2. Denoising harga asas
# =====================================================
# Savitzky-Golay untuk Close
df["Close_Smooth"] = savgol_filter(df["Close"], window_length=7, polyorder=2)

# SMA smoothing untuk Open, High, Low, Volume
df["Open_Smooth"] = df["Open"].rolling(window=5, min_periods=1).mean()
df["High_Smooth"] = df["High"].rolling(window=5, min_periods=1).mean()
df["Low_Smooth"] = df["Low"].rolling(window=5, min_periods=1).mean()
df["Volume_Smooth"] = df["Volume"].rolling(window=5, min_periods=1).mean()

# Gantikan original dengan versi smooth
df["Open"] = df["Open_Smooth"]
df["High"] = df["High_Smooth"]
df["Low"] = df["Low_Smooth"]
df["Close"] = df["Close_Smooth"]
df["Volume"] = df["Volume_Smooth"]

# =====================================================
# 3. Feature Engineering (Technical Indicators)
# =====================================================
df["SMA5"] = df["Close"].rolling(window=5).mean()
df["SMA10"] = df["Close"].rolling(window=10).mean()
df["EMA10"] = df["Close"].ewm(span=10, adjust=False).mean()
df["Return"] = df["Close"].pct_change()
df["Momentum"] = df["Close"].diff(3)

# ====== Denoising Features ======
df["SMA5"] = df["SMA5"].rolling(window=3, min_periods=1).mean()
df["SMA10"] = df["SMA10"].rolling(window=3, min_periods=1).mean()
df["EMA10"] = savgol_filter(df["EMA10"], window_length=7, polyorder=2)
df["Return"] = df["Return"].rolling(window=3, min_periods=1).mean()
df["Momentum"] = df["Momentum"].rolling(window=3, min_periods=1).mean()

# Target: arah harga esok (1 = naik, 0 = turun)
df["Tomorrow_Close"] = df["Close"].shift(-1)
df["Direction"] = np.where(df["Tomorrow_Close"] > df["Close"], 1, 0)

# Buang missing values
df = df.dropna()

# =====================================================
# 4. Split Feature & Target
# =====================================================
X = df.drop(columns=[
    "Date", "Tomorrow_Close", "Direction",
    "Open_Smooth", "High_Smooth", "Low_Smooth",
    "Close_Smooth", "Volume_Smooth"
])
y = df["Direction"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =====================================================
# 5. Baseline Model (SVM)
# =====================================================
pipe_baseline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", random_state=42))
])

pipe_baseline.fit(X_train, y_train)
y_pred_base = pipe_baseline.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)

print("=== Baseline SVM (Denoised) ===")
print("Accuracy:", base_acc)
print(classification_report(y_test, y_pred_base))

# =====================================================
# 6. Tuned Model (GridSearchCV)
# =====================================================
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=42))
])

param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf", "poly", "sigmoid"],
    "svm__gamma": ["scale", "auto"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

y_pred_tuned = grid.predict(X_test)
y_pred_proba = grid.predict_proba(X_test)[:, 1]
tuned_acc = accuracy_score(y_test, y_pred_tuned)

print("\n=== Tuned SVM (Denoised) ===")
print("Best Parameters:", grid.best_params_)
print("Accuracy:", tuned_acc)
print(classification_report(y_test, y_pred_tuned))

# =====================================================
# TRAINING OUTPUT (ADDED)
# =====================================================
y_train_pred = grid.predict(X_train)
y_train_pred_proba = grid.predict_proba(X_train)[:, 1]
train_acc = accuracy_score(y_train, y_train_pred)

print("\n=== TRAINING PERFORMANCE ===")
print("Training Accuracy:", train_acc)
print(classification_report(y_train, y_train_pred))

# =====================================================
# 7. Save ALL Output into Folder
# =====================================================
output_dir = os.path.join(base_path, f"svm_{ticker}_output")
os.makedirs(output_dir, exist_ok=True)

# 7A — Save model
model_path = os.path.join(output_dir, "svm_tuned_model.pkl")
joblib.dump(grid.best_estimator_, model_path)

# 7B — Save classification report (train + test)
report_path = os.path.join(output_dir, "classification_report.txt")
with open(report_path, "w") as f:
    f.write("Best Parameters:\n")
    f.write(str(grid.best_params_) + "\n\n")

    f.write("=== TRAINING ===\n")
    f.write("Accuracy: " + str(train_acc) + "\n\n")
    f.write(classification_report(y_train, y_train_pred))
    f.write("\n")

    f.write("=== TESTING ===\n")
    f.write("Accuracy: " + str(tuned_acc) + "\n\n")
    f.write(classification_report(y_test, y_pred_tuned))

# 7C — Save predictions CSV
pred_df = pd.DataFrame({
    "Date": df["Date"].iloc[len(X_train):].values,
    "Actual": y_test.values,
    "Predicted": y_pred_tuned,
    "Predicted_Probability_Up": y_pred_proba
})
pred_csv_path = os.path.join(output_dir, "predictions.csv")
pred_df.to_csv(pred_csv_path, index=False)

# 7D — Save TEST confusion matrix
cm_test = confusion_matrix(y_test, y_pred_tuned)
disp = ConfusionMatrixDisplay(cm_test, display_labels=["Down", "Up"])
disp.plot(cmap="Greens")
plt.title("Confusion Matrix - TEST (Tuned SVM)")
cm_test_path = os.path.join(output_dir, "confusion_matrix_test.png")
plt.savefig(cm_test_path, dpi=300)
plt.close()

# 7E — Save TRAIN confusion matrix (ADDED)
cm_train = confusion_matrix(y_train, y_train_pred)
disp = ConfusionMatrixDisplay(cm_train, display_labels=["Down", "Up"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - TRAIN (Tuned SVM)")
cm_train_path = os.path.join(output_dir, "confusion_matrix_train.png")
plt.savefig(cm_train_path, dpi=300)
plt.close()

# =====================================================
# 8. ADDITIONAL OUTPUTS - Performance Metrics & Visualizations
# =====================================================

# 8A — Calculate comprehensive metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, dataset_name):
    return {
        "Dataset": dataset_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_pred_proba),
        "Samples": len(y_true)
    }

train_metrics = calculate_metrics(y_train, y_train_pred, y_train_pred_proba, "Training")
test_metrics = calculate_metrics(y_test, y_pred_tuned, y_pred_proba, "Test")

metrics_df = pd.DataFrame([train_metrics, test_metrics])
metrics_csv_path = os.path.join(output_dir, "performance_metrics.csv")
metrics_df.to_csv(metrics_csv_path, index=False)

print("\n=== PERFORMANCE METRICS ===")
print(metrics_df.to_string(index=False))

# 8B — Save ROC Curve
fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr_train, tpr_train, label=f'Train (AUC = {train_metrics["ROC-AUC"]:.4f})', linewidth=2)
plt.plot(fpr_test, tpr_test, label=f'Test (AUC = {test_metrics["ROC-AUC"]:.4f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend()
plt.grid(True, alpha=0.3)
roc_path = os.path.join(output_dir, "roc_curve.png")
plt.savefig(roc_path, dpi=300)
plt.close()

# 8C — Save Feature Importance (for linear kernel)
best_model = grid.best_estimator_.named_steps['svm']
feature_names = X.columns.tolist()

# Check if the best model uses linear kernel for feature importance
if hasattr(best_model, 'coef_') and best_model.kernel == 'linear':
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': best_model.coef_[0],
        'Absolute_Importance': np.abs(best_model.coef_[0])
    }).sort_values('Absolute_Importance', ascending=False)
else:
    # For non-linear kernels, we can't get coefficients directly
    # So we'll use a placeholder or alternative importance measure
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': np.nan,
        'Absolute_Importance': np.nan
    })
    print("Note: Feature importance not available for non-linear kernel")

feature_importance_path = os.path.join(output_dir, "feature_importance.csv")
feature_importance.to_csv(feature_importance_path, index=False)

# 8D — Plot Feature Importance (if available)
if hasattr(best_model, 'coef_') and best_model.kernel == 'linear':
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(10)
    bars = plt.barh(top_features['Feature'], top_features['Absolute_Importance'])
    plt.xlabel('Absolute Coefficient Value')
    plt.title('Top 10 Most Important Features (SVM - Linear Kernel)')
    plt.gca().invert_yaxis()

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2,
                 f'{width:.4f}', ha='left', va='center')

    feature_plot_path = os.path.join(output_dir, "feature_importance_plot.png")
    plt.tight_layout()
    plt.savefig(feature_plot_path, dpi=300)
    plt.close()
else:
    print("Skipping feature importance plot - non-linear kernel used")

# 8E — Save Prediction Distribution
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(y_train_pred_proba, bins=50, alpha=0.7, color='blue', label='Train')
plt.xlabel('Predicted Probability of Up Movement')
plt.ylabel('Frequency')
plt.title('Training Set: Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_pred_proba, bins=50, alpha=0.7, color='green', label='Test')
plt.xlabel('Predicted Probability of Up Movement')
plt.ylabel('Frequency')
plt.title('Test Set: Prediction Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

pred_dist_path = os.path.join(output_dir, "prediction_distribution.png")
plt.tight_layout()
plt.savefig(pred_dist_path, dpi=300)
plt.close()

# 8F — Save Actual vs Predicted over Time
plt.figure(figsize=(15, 8))
dates_test = df["Date"].iloc[len(X_train):].values

plt.plot(dates_test, y_test.values, label='Actual', marker='o', markersize=3, linewidth=1)
plt.plot(dates_test, y_pred_tuned, label='Predicted', marker='x', markersize=3, linewidth=1)
plt.xlabel('Date')
plt.ylabel('Direction (0=Down, 1=Up)')
plt.title('Actual vs Predicted Stock Direction Over Time (SVM)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()

time_series_path = os.path.join(output_dir, "actual_vs_predicted_timeseries.png")
plt.savefig(time_series_path, dpi=300)
plt.close()

# 8G — Save Model Parameters Summary
params_summary = {
    'Best Parameters': str(grid.best_params_),
    'Best CV Score': f"{grid.best_score_:.4f}",
    'Test Accuracy': f"{tuned_acc:.4f}",
    'Train Accuracy': f"{train_acc:.4f}",
    'Number of Features': len(feature_names),
    'Train Samples': len(X_train),
    'Test Samples': len(X_test),
    'Feature Names': ', '.join(feature_names),
    'Kernel Type': best_model.kernel if hasattr(best_model, 'kernel') else 'N/A'
}

params_df = pd.DataFrame(list(params_summary.items()), columns=['Parameter', 'Value'])
params_csv_path = os.path.join(output_dir, "model_parameters_summary.csv")
params_df.to_csv(params_csv_path, index=False)

# 8H — Baseline vs Tuned Comparison
comparison_df = pd.DataFrame({
    'Model': ['Baseline SVM', 'Tuned SVM'],
    'Test Accuracy': [base_acc, tuned_acc],
    'Best Parameters': ['rbf (default)', str(grid.best_params_)]
})
comparison_path = os.path.join(output_dir, "model_comparison.csv")
comparison_df.to_csv(comparison_path, index=False)

print(f"\n✅ All outputs saved into: {output_dir}")
print(f" - Model: {model_path}")
print(f" - Report: {report_path}")
print(f" - Predictions: {pred_csv_path}")
print(f" - Test Confusion Matrix: {cm_test_path}")
print(f" - Train Confusion Matrix: {cm_train_path}")
print(f" - Performance Metrics: {metrics_csv_path}")
print(f" - ROC Curve: {roc_path}")
print(f" - Feature Importance: {feature_importance_path}")
if hasattr(best_model, 'coef_') and best_model.kernel == 'linear':
    print(f" - Feature Importance Plot: {feature_plot_path}")
print(f" - Prediction Distribution: {pred_dist_path}")
print(f" - Time Series Plot: {time_series_path}")
print(f" - Model Parameters: {params_csv_path}")
print(f" - Model Comparison: {comparison_path}")

# Print final summary
print(f"\n🎯 FINAL SUMMARY for {ticker}")
print(f"Training Accuracy: {train_metrics['Accuracy']:.4f}")
print(f"Test Accuracy: {test_metrics['Accuracy']:.4f}")
print(f"Test Precision: {test_metrics['Precision']:.4f}")
print(f"Test Recall: {test_metrics['Recall']:.4f}")
print(f"Test F1-Score: {test_metrics['F1-Score']:.4f}")
print(f"Test ROC-AUC: {test_metrics['ROC-AUC']:.4f}")
print(f"Best Parameters: {grid.best_params_}")
print(f"Kernel Type: {best_model.kernel if hasattr(best_model, 'kernel') else 'N/A'}")