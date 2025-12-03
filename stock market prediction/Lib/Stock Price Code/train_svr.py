import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# ----------------------------
# Force Determinism
# ----------------------------
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ----------------------------
# Config
# ----------------------------
base_path = r"C:\Users\AMJAD\PycharmProjects\fyp1test\Lib\Price Stock Dataset"
ticker = "AAPL"
date_suffix = "2025-06-16"

# Use glob to find the exact CSV file
file_pattern = os.path.join(base_path, f"{ticker}*historical_data*{date_suffix}.csv")
matching_files = glob.glob(file_pattern)

if not matching_files:
    raise FileNotFoundError(f"No CSV file found matching pattern: {file_pattern}")

file_path = matching_files[0]  # take the first match
output_dir = f"svr_{ticker}_output"
os.makedirs(output_dir, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(file_path)
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.sort_values("date").reset_index(drop=True)

# Convert numeric columns
numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace(",", "")
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=numeric_cols).reset_index(drop=True)

# ----------------------------
# Features & Target
# ----------------------------
features = ["open", "high", "low", "close", "adj_close", "volume"]
target = "close"

data = df[features].values
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# ----------------------------
# Sequence builder
# ----------------------------
def create_sequences(data, target_col_idx, window=30, horizon=1):
    X, y = [], []
    for i in range(len(data) - window - horizon):
        X.append(data[i:i + window])
        y.append(data[i + window + horizon - 1, target_col_idx])
    return np.array(X), np.array(y)

target_col_idx = features.index(target)
X_all, y_all = create_sequences(scaled_data, target_col_idx, window=30, horizon=1)
split = int(len(X_all) * 0.8)
X_train, X_test = X_all[:split], X_all[split:]
y_train, y_test = y_all[:split], y_all[split:]

# ----------------------------
# Flatten for SVR (2D)
# ----------------------------
X_train_2d = X_train.reshape(X_train.shape[0], -1)
X_test_2d = X_test.reshape(X_test.shape[0], -1)

# ----------------------------
# Hyperparameter Tuning with GridSearchCV
# ----------------------------
param_grid = {
    "kernel": ["rbf", "poly", "sigmoid"],
    "C": [1, 10, 100],
    "gamma": ["scale", "auto", 0.01, 0.001],
    "epsilon": [0.001, 0.01, 0.1, 1]
}

print("\nRunning GridSearchCV for SVR...")
grid_search = GridSearchCV(
    SVR(),
    param_grid,
    scoring="neg_mean_squared_error",
    cv=3,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_train_2d, y_train)

print("\nBest Parameters found:")
print(grid_search.best_params_)

# Save best parameters to CSV
best_params_df = pd.DataFrame([grid_search.best_params_])
best_params_df.to_csv(os.path.join(output_dir, f"{ticker}_best_params.csv"), index=False)

best_svr = grid_search.best_estimator_

# ----------------------------
# Predict & inverse scaling
# ----------------------------
y_train_pred = best_svr.predict(X_train_2d)
y_test_pred = best_svr.predict(X_test_2d)

def invert_scale(y_scaled, scaler, target_col_idx):
    y_scaled = y_scaled.reshape(-1)
    dummy = np.zeros((len(y_scaled), len(features)))
    dummy[:, target_col_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, target_col_idx]

y_train_inv = invert_scale(y_train, scaler, target_col_idx)
y_train_pred_inv = invert_scale(y_train_pred, scaler, target_col_idx)
y_test_inv = invert_scale(y_test, scaler, target_col_idx)
y_test_pred_inv = invert_scale(y_test_pred, scaler, target_col_idx)

# ----------------------------
# Evaluation Metrics
# ----------------------------
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (np.maximum(np.abs(y_true), eps)))) * 100
    return rmse, mae, r2, mape

rmse_train, mae_train, r2_train, mape_train = compute_metrics(y_train_inv, y_train_pred_inv)
rmse_test, mae_test, r2_test, mape_test = compute_metrics(y_test_inv, y_test_pred_inv)

metrics_df = pd.DataFrame([
    {"dataset": "train", "rmse": rmse_train, "mae": mae_train, "r2": r2_train, "mape": mape_train},
    {"dataset": "test", "rmse": rmse_test, "mae": mae_test, "r2": r2_test, "mape": mape_test},
])
metrics_df.to_csv(os.path.join(output_dir, f"{ticker}_model_metrics.csv"), index=False)

# ----------------------------
# Correlation Matrix
# ----------------------------
corr = df[features].corr()
plt.figure(figsize=(8, 6))
plt.imshow(corr, interpolation='nearest', aspect='auto')
plt.colorbar()
plt.xticks(range(len(features)), features, rotation=45)
plt.yticks(range(len(features)), features)
plt.title(f"{ticker} - Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f"{ticker}_correlation_matrix.png"))
plt.close()

# ----------------------------
# Price vs Time Step & Predicted vs Actual
# ----------------------------
for dataset, y_true, y_pred, label in [("train", y_train_inv, y_train_pred_inv, "Tuned"),
                                      ("test", y_test_inv, y_test_pred_inv, "Tuned")]:
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label="Actual")
    plt.plot(y_pred, label=f"Predicted ({label})")
    plt.title(f"{label} - Price vs Time Step ({dataset.capitalize()}) for {ticker}")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ticker}_price_vs_timestep_{dataset}.png"))
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, s=10)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--')
    plt.title(f"{label} - Predicted vs Actual ({dataset.capitalize()}) for {ticker}")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ticker}_pred_vs_actual_{dataset}.png"))
    plt.close()

print(f"All outputs saved in folder: {output_dir}")
