"""
hybrid_pipeline_multi_ticker_weighted.py

Improved multi-ticker hybrid pipeline with WEIGHTED decision system + sentiment override.
- Combines LSTM, Logistic Regression, and FinBERT sentiment using weighted scoring
- NEW: If >50% of news are negative → Force SELL
- Loops through multiple tickers
- Produces PNG dashboard per ticker (without LSTM forecast annotations) and summary CSV
- Handles missing models/files gracefully
"""

import os
import sys
import traceback
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta, datetime
from collections import Counter
import warnings

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification, Trainer

# ===========================
# CONFIG - EDIT THESE
# ===========================
BASE_PATH = os.path.normpath(
    "C:/Users/AMJAD/PycharmProjects/fyp1test/Lib/Price Stock Dataset"
)
NEWS_PATH = os.path.normpath(
    "C:/Users/AMJAD/PycharmProjects/fyp1test/Lib/News Stock Dataset"
)
DATE_SUFFIX = "2025-10-08"
TICKERS = ["AAPL", "AMD", "AMZN", "GOOG", "INTC", "META", "MSFT", "NFLX", "NVDA", "TSLA"]

# Model patterns
LSTM_MODEL_PATTERN = "best_lstm_{ticker}_fixed.h5"
LOGREG_MODEL_PATTERN = "logreg_{ticker}.pkl"

# Optional FinBERT checkpoint
FINBERT_CHECKPOINT = os.path.normpath(
    "C:/Users/AMJAD/PycharmProjects/fyp1test/Lib/Price Stock Dataset/finbert_results/checkpoint-486"
)
FINBERT_HF_ID = "yiyanghkust/finbert-tone"

OUT_DIR = os.path.join("outputs", DATE_SUFFIX)
os.makedirs(OUT_DIR, exist_ok=True)

# Features used for LSTM + target mapping
FEATURES = ["open", "high", "low", "close", "adj_close", "volume"]
TARGET = "close"
WINDOW = 30
FORECAST_DAYS = 7  # business days

# ===========================
# WEIGHT CONFIGURATION
# ===========================
WEIGHTS = {"lstm": 0.6, "logreg": 0.3, "sentiment": 0.1}


# ===========================
# Helper functions
# ===========================
def calculate_model_confidence(lstm_predictions, logreg_probs, sentiments):
    """Calculate confidence scores for each model component"""
    # LSTM Confidence (based on prediction stability)
    lstm_confidence = 0.0
    if len(lstm_predictions) > 1:
        lstm_std = np.std(lstm_predictions)
        lstm_mean = np.mean(lstm_predictions)
        if lstm_mean != 0:
            lstm_confidence = max(0, 1 - (lstm_std / abs(lstm_mean)))

    # Logistic Regression Confidence
    logreg_confidence = 0.0
    if logreg_probs is not None and len(logreg_probs) == 2:
        logreg_confidence = max(logreg_probs)

    # Sentiment Confidence
    sentiment_confidence = 0.0
    if sentiments:
        c = Counter(sentiments)
        total = sum(c.values())
        majority_ratio = max(c.values()) / total
        sentiment_confidence = majority_ratio

    return {
        "lstm_confidence": lstm_confidence,
        "logreg_confidence": logreg_confidence,
        "sentiment_confidence": sentiment_confidence
    }


def calculate_risk_metrics(df, forecast_prices):
    """Calculate comprehensive risk metrics"""
    if len(df) < 30:
        return {}

    current_price = df["close"].iloc[-1]

    # Historical volatility (30-day)
    returns = df["close"].pct_change().dropna()
    historical_volatility = returns.std() * np.sqrt(252) * 100  # Annualized %

    # Forecast volatility
    if len(forecast_prices) > 1:
        forecast_returns = np.diff(forecast_prices) / forecast_prices[:-1]
        forecast_volatility = np.std(forecast_returns) * np.sqrt(252) * 100 if len(forecast_returns) > 0 else 0
    else:
        forecast_volatility = 0

    # Maximum Drawdown (recent)
    recent_prices = df["close"].tail(60)
    peak = recent_prices.expanding().max()
    drawdown = (recent_prices - peak) / peak
    max_drawdown = drawdown.min() * 100

    # Support and Resistance levels
    resistance = df["high"].tail(30).max()
    support = df["low"].tail(30).min()

    # Price position relative to support/resistance
    price_position = (current_price - support) / (resistance - support) * 100 if resistance != support else 50

    return {
        "historical_volatility": historical_volatility,
        "forecast_volatility": forecast_volatility,
        "max_drawdown": max_drawdown,
        "support_level": support,
        "resistance_level": resistance,
        "price_position_pct": price_position,
        "distance_to_support_pct": ((current_price - support) / support) * 100,
        "distance_to_resistance_pct": ((resistance - current_price) / current_price) * 100
    }


def decide_action(
        y_future_prices, last_close, direction_label, sentiments,
        logreg_model=None, last_features_df=None,
        weights=WEIGHTS
):
    """Weighted scoring decision system combining LSTM, LogReg, and FinBERT"""

    # Mean of LSTM future predictions
    mean_future = float(np.mean(y_future_prices)) if len(y_future_prices) > 0 else last_close

    # pct_change = % change predicted by LSTM from last close
    pct_change = (mean_future - last_close) / last_close * 100.0  # %

    # Sentiment score (FinBERT)
    c = Counter(sentiments)
    total = sum(c.values()) if c else 0
    sent_score = 0.0
    if total > 0:
        sent_score = (c.get("positive", 0) - c.get("negative", 0)) / total  # -1 to +1

    # Logistic Regression confidence score
    logreg_score = 0.0
    logreg_probs = None
    try:
        if (logreg_model is not None) and hasattr(logreg_model, "predict_proba") and (last_features_df is not None):
            probs = logreg_model.predict_proba(last_features_df)[0]
            logreg_probs = probs
            logreg_score = probs[1] - probs[0]  # UP positive, DOWN negative
    except Exception:
        pass

    # Normalize LSTM score (linear scale from -1 to +1)
    if pct_change > 5:
        lstm_score = 1.0
    elif pct_change < -5:
        lstm_score = -1.0
    else:
        lstm_score = pct_change / 5.0

    # Weighted total score
    total_score = (
            lstm_score * weights["lstm"]
            + logreg_score * weights["logreg"]
            + sent_score * weights["sentiment"]
    )

    # Calculate model confidences
    confidences = calculate_model_confidence(y_future_prices, logreg_probs, sentiments)

    # Enhanced decision with confidence thresholds
    if total_score > 0.2 and confidences["lstm_confidence"] > 0.6:
        action = "STRONG BUY"
    elif total_score > 0.2:
        action = "BUY"
    elif total_score < -0.2 and confidences["lstm_confidence"] > 0.6:
        action = "STRONG SELL"
    elif total_score < -0.2:
        action = "SELL"
    else:
        action = "HOLD"

    # Prepare detailed reason string
    reason = (
        f"Final Score: {total_score:.3f}\n"
        f"LSTM: {lstm_score:.3f} (Conf: {confidences['lstm_confidence']:.1%})\n"
        f"LogReg: {logreg_score:.3f} (Conf: {confidences['logreg_confidence']:.1%})\n"
        f"Sentiment: {sent_score:.3f} (Conf: {confidences['sentiment_confidence']:.1%})\n"
        f"Predicted Change: {pct_change:.2f}%\n"
        f"Model Weights: LSTM({weights['lstm'] * 100:.0f}%), LR({weights['logreg'] * 100:.0f}%), Sent({weights['sentiment'] * 100:.0f}%)"
    )

    return action, reason, total_score, confidences


def check_sentiment_override(sentiments, current_action):
    """If >50% of news are negative → Force SELL."""
    if not sentiments:
        return current_action, ""
    c = Counter(sentiments)
    total = sum(c.values())
    neg_ratio = c.get("negative", 0) / total if total > 0 else 0
    if neg_ratio > 0.5:
        return "SELL", f"⚠️ Sentiment override: {neg_ratio * 100:.1f}% negative news → Forced SELL"
    return current_action, ""


def create_sequences(data, window=WINDOW):
    if data.shape[0] < window:
        raise ValueError(f"Not enough rows for window={window}")
    return np.array([data[-window:]])


def invert_scale_array(y_scaled, scaler, target_col_idx, features_len=len(FEATURES)):
    dummy = np.zeros((len(y_scaled), features_len))
    dummy[:, target_col_idx] = y_scaled
    return scaler.inverse_transform(dummy)[:, target_col_idx]


def generate_trading_recommendation(action, risk_metrics, total_score):
    """Generate detailed trading recommendation based on action and risk metrics"""

    recommendations = {
        "STRONG BUY": {
            "risk_level": "Low",
            "position_size": "Large (70-100% of allocated capital)",
            "time_horizon": "Short to Medium term (1-4 weeks)",
            "stop_loss": f"{risk_metrics.get('support_level', 0):.2f} (-{max(5, risk_metrics.get('distance_to_support_pct', 0)):.1f}%)",
            "take_profit": f"{risk_metrics.get('resistance_level', 0):.2f} (+{risk_metrics.get('distance_to_resistance_pct', 0):.1f}%)"
        },
        "BUY": {
            "risk_level": "Medium",
            "position_size": "Medium (30-70% of allocated capital)",
            "time_horizon": "Short term (1-2 weeks)",
            "stop_loss": f"{risk_metrics.get('support_level', 0):.2f} (-{max(8, risk_metrics.get('distance_to_support_pct', 0)):.1f}%)",
            "take_profit": f"{risk_metrics.get('resistance_level', 0):.2f} (+{risk_metrics.get('distance_to_resistance_pct', 0):.1f}%)"
        },
        "HOLD": {
            "risk_level": "Variable",
            "position_size": "Maintain current position",
            "time_horizon": "Wait for clearer signals",
            "stop_loss": "Not applicable",
            "take_profit": "Not applicable"
        },
        "SELL": {
            "risk_level": "High",
            "position_size": "Reduce position (50-100%)",
            "time_horizon": "Immediate to Short term",
            "stop_loss": f"{risk_metrics.get('resistance_level', 0):.2f} (if holding)",
            "take_profit": "Not applicable"
        },
        "STRONG SELL": {
            "risk_level": "Very High",
            "position_size": "Close entire position",
            "time_horizon": "Immediate",
            "stop_loss": "Not applicable",
            "take_profit": "Not applicable"
        }
    }

    base_rec = recommendations.get(action, recommendations["HOLD"])

    # Adjust based on score magnitude
    if "BUY" in action and total_score > 0.5:
        base_rec["position_size"] = "Very Large (80-100% of allocated capital)"
    elif "SELL" in action and total_score < -0.5:
        base_rec["position_size"] = "Close position + Consider short"

    return base_rec


def create_compact_dashboard(ticker, df, y_future, future_dates, risk_metrics,
                           logreg_probs, direction_label, sentiments,
                           action, reason, total_score, confidences, trading_rec):
    """Create compact dashboard without wasted space"""

    # Create figure with optimized layout
    fig = plt.figure(figsize=(16, 10))

    # Use GridSpec for better control
    gs = plt.GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Plot 1: Price Analysis (top left, 2x2)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df["date"].iloc[-60:], df["close"].iloc[-60:], label="Historical",
             color="blue", linewidth=1.5)
    if len(y_future) > 0:
        ax1.plot(future_dates, y_future, "ro--", label="Forecast", markersize=4, linewidth=1)

    # Add technical indicators
    ax1.plot(df["date"].iloc[-60:], df["SMA5"].iloc[-60:], label="SMA5",
             linestyle="--", alpha=0.6, linewidth=1)
    ax1.plot(df["date"].iloc[-60:], df["SMA10"].iloc[-60:], label="SMA10",
             linestyle="--", alpha=0.6, linewidth=1)

    # Support and Resistance
    if risk_metrics.get('support_level'):
        ax1.axhline(y=risk_metrics['support_level'], color='green', linestyle=':',
                   label=f'Support: ${risk_metrics["support_level"]:.2f}', alpha=0.7)
    if risk_metrics.get('resistance_level'):
        ax1.axhline(y=risk_metrics['resistance_level'], color='red', linestyle=':',
                   label=f'Resistance: ${risk_metrics["resistance_level"]:.2f}', alpha=0.7)

    ax1.set_title(f'{ticker} Price Analysis', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45, labelsize=8)

    # Plot 2: Model Confidences (top right 1)
    ax2 = fig.add_subplot(gs[0, 2])
    models = ['LSTM', 'LogReg', 'Sentiment']
    confidence_scores = [confidences['lstm_confidence'], confidences['logreg_confidence'],
                        confidences['sentiment_confidence']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax2.bar(models, confidence_scores, color=colors, alpha=0.8)
    ax2.set_ylim(0, 1)
    ax2.set_title('Model Confidence', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Confidence', fontsize=9)
    ax2.tick_params(axis='both', labelsize=8)
    for bar, score in zip(bars, confidence_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.0%}', ha='center', va='bottom', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Plot 3: LogReg Probabilities (top right 2)
    ax3 = fig.add_subplot(gs[0, 3])
    ax3.bar(["DOWN", "UP"], logreg_probs if len(logreg_probs) == 2 else [0.5, 0.5],
            color=["#d62728", "#2ca02c"], alpha=0.8)
    ax3.set_ylim(0, 1)
    ax3.set_title(f'LogReg: {direction_label or "N/A"}', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Probability', fontsize=9)
    ax3.tick_params(axis='both', labelsize=8)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sentiment Analysis (middle left)
    ax4 = fig.add_subplot(gs[1, 0])
    if sentiments:
        c = Counter(sentiments)
        labels = ["Positive", "Neutral", "Negative"]
        values = [c.get("positive", 0), c.get("neutral", 0), c.get("negative", 0)]
        colors_pie = ['#2ca02c', '#ff7f0e', '#d62728']
        wedges, texts, autotexts = ax4.pie(values, labels=labels, autopct="%1.0f%%",
                                          colors=colors_pie, startangle=90, textprops={'fontsize': 8})
        ax4.set_title("Sentiment Analysis", fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, "No News\nAvailable", ha="center", va="center",
                transform=ax4.transAxes, fontsize=9)
        ax4.set_title("Sentiment Analysis", fontsize=10, fontweight='bold')

    # Plot 5: Risk Metrics (middle)
    ax5 = fig.add_subplot(gs[1, 1])
    risk_data = {
        'Volatility': risk_metrics.get('historical_volatility', 0),
        'Drawdown': abs(risk_metrics.get('max_drawdown', 0))
    }
    bars = ax5.bar(risk_data.keys(), risk_data.values(),
                   color=['#9467bd', '#8c564b'], alpha=0.8)
    ax5.set_title('Risk Metrics (%)', fontsize=10, fontweight='bold')
    ax5.set_ylabel('Percentage', fontsize=9)
    ax5.tick_params(axis='x', rotation=45, labelsize=8)
    ax5.tick_params(axis='y', labelsize=8)
    for bar, value in zip(bars, risk_data.values()):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=8)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Support/Resistance Levels (middle right)
    ax6 = fig.add_subplot(gs[1, 2:])
    current_price = df["close"].iloc[-1]
    levels_data = {
        'Support': risk_metrics.get('support_level', 0),
        'Current': current_price,
        'Resistance': risk_metrics.get('resistance_level', 0)
    }
    colors_level = ['green', 'blue', 'red']
    bars = ax6.bar(levels_data.keys(), levels_data.values(), color=colors_level, alpha=0.7)
    ax6.set_title('Price Levels', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Price ($)', fontsize=9)
    ax6.tick_params(axis='both', labelsize=8)
    for bar, value in zip(bars, levels_data.values()):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'${value:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # Plot 7: Final Decision & Trading Recommendation (bottom, full width)
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis("off")

    # Color coding for action
    color_map = {
        "STRONG BUY": "darkgreen",
        "BUY": "green",
        "HOLD": "orange",
        "SELL": "red",
        "STRONG SELL": "darkred"
    }
    action_color = color_map.get(action, "black")

    # Main decision
    ax7.text(0.02, 0.85, f"FINAL DECISION: {action}",
            fontsize=16, fontweight="bold", color=action_color)
    ax7.text(0.02, 0.70, f"Overall Score: {total_score:.3f} • Confidence: {confidences['lstm_confidence']:.1%}",
            fontsize=11, fontweight="bold")

    # Trading recommendation in two columns
    ax7.text(0.02, 0.55, "TRADING RECOMMENDATION:", fontsize=11, fontweight="bold")
    ax7.text(0.02, 0.45, f"• Risk: {trading_rec['risk_level']}", fontsize=9)
    ax7.text(0.02, 0.40, f"• Position: {trading_rec['position_size']}", fontsize=9)
    ax7.text(0.02, 0.35, f"• Horizon: {trading_rec['time_horizon']}", fontsize=9)

    ax7.text(0.25, 0.45, f"• Stop Loss: {trading_rec['stop_loss']}", fontsize=9)
    ax7.text(0.25, 0.40, f"• Take Profit: {trading_rec['take_profit']}", fontsize=9)

    # Model details on the right
    ax7.text(0.55, 0.85, "MODEL ANALYSIS:", fontsize=11, fontweight="bold")

    # Split reason into lines and display compactly
    reason_lines = reason.split('\n')
    for i, line in enumerate(reason_lines[:4]):  # Show first 4 lines
        ax7.text(0.55, 0.75 - i*0.08, line, fontsize=8, wrap=True)

    # Add override reason if exists
    override_reason = ""
    if sentiments:
        c = Counter(sentiments)
        total = sum(c.values())
        neg_ratio = c.get("negative", 0) / total if total > 0 else 0
        if neg_ratio > 0.5:
            override_reason = f"⚠️ Sentiment Override: {neg_ratio:.1%} Negative News"
            ax7.text(0.55, 0.45, override_reason, fontsize=9,
                    style='italic', color='red', weight='bold')

    # Add timestamp
    ax7.text(0.85, 0.05, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            fontsize=7, alpha=0.7, ha='right')

    plt.tight_layout()
    return fig


# ===========================
# Load FinBERT
# ===========================
print("Loading FinBERT model (best-effort)...")
try:
    tokenizer = BertTokenizer.from_pretrained(FINBERT_HF_ID)
    if os.path.exists(FINBERT_CHECKPOINT):
        finbert_model = BertForSequenceClassification.from_pretrained(FINBERT_CHECKPOINT)
    else:
        finbert_model = BertForSequenceClassification.from_pretrained(FINBERT_HF_ID)
    finbert_model.eval()
    trainer = Trainer(model=finbert_model, tokenizer=tokenizer)
    finbert_available = True
    print("FinBERT loaded successfully.")
except Exception:
    print("⚠️ Warning: FinBERT failed to load. Sentiment analysis will be skipped.")
    finbert_available = False
    tokenizer = None
    finbert_model = None
    trainer = None

# ===========================
# MAIN LOOP
# ===========================
summary_rows = []
detailed_analysis = []

for TICKER in TICKERS:
    print(f"\n--- Processing {TICKER} ---")
    try:
        price_file = os.path.join(BASE_PATH, f"{TICKER}_historical_data_{DATE_SUFFIX}.csv")
        news_file = os.path.join(NEWS_PATH, f"{TICKER}_news_{DATE_SUFFIX}.csv")

        if not os.path.exists(price_file):
            print(f"⚠️ Price file not found for {TICKER}. Skipping.")
            summary_rows.append({"ticker": TICKER, "action": "SKIP", "reason": "price file missing"})
            continue

        df = pd.read_csv(price_file)
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df["date"] = pd.to_datetime(df.get("date", pd.date_range(end=pd.Timestamp(DATE_SUFFIX), periods=len(df))),
                                    errors="coerce")
        df = df.sort_values("date").reset_index(drop=True)

        numeric_cols = ["open", "high", "low", "close", "adj_close", "volume"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)

        # Technical Indicators
        df["SMA5"] = df["close"].rolling(5).mean()
        df["SMA10"] = df["close"].rolling(10).mean()
        df["EMA10"] = df["close"].ewm(span=10).mean()
        df["Return"] = df["close"].pct_change()
        df["Momentum"] = df["close"] - df["close"].shift(4)
        df = df.dropna().reset_index(drop=True)

        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[FEATURES].values)
        target_idx = FEATURES.index(TARGET)
        X_input = create_sequences(scaled_data, WINDOW)

        # LSTM Prediction
        lstm_path = os.path.join(os.getcwd(), LSTM_MODEL_PATTERN.format(ticker=TICKER.lower()))
        lstm_model = load_model(lstm_path) if os.path.exists(lstm_path) else None
        y_future = []
        if lstm_model is not None:
            future_preds_scaled = []
            X_curr = X_input.copy()
            for _ in range(FORECAST_DAYS):
                pred_scaled = float(lstm_model.predict(X_curr, verbose=0)[0][0])
                future_preds_scaled.append(pred_scaled)
                last_row_scaled = X_curr[0, -1, :].copy()
                last_row_scaled[target_idx] = pred_scaled
                X_curr = np.append(X_curr[:, 1:, :], [[last_row_scaled]], axis=1)
            y_future = invert_scale_array(future_preds_scaled, scaler, target_idx, len(FEATURES))

        # Future Dates
        last_date = df["date"].iloc[-1]
        future_dates = []
        d = last_date
        while len(future_dates) < FORECAST_DAYS:
            d += timedelta(days=1)
            if d.weekday() < 5:
                future_dates.append(d)

        # Logistic Regression
        logreg_path = os.path.join(os.getcwd(), LOGREG_MODEL_PATTERN.format(ticker=TICKER.lower()))
        logreg_model = joblib.load(logreg_path) if os.path.exists(logreg_path) else None
        last_row = df.tail(1).copy()
        rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close",
                      "adj_close": "Adj Close", "volume": "Volume", "SMA5": "SMA5",
                      "SMA10": "SMA10", "EMA10": "EMA10", "Return": "Return", "Momentum": "Momentum"}
        last_features_df = last_row.rename(columns=rename_map).drop(columns=["date"], errors="ignore")

        direction_label, logreg_probs = None, [0.0, 0.0]
        try:
            if logreg_model is not None:
                direction_pred = logreg_model.predict(last_features_df)[0]
                direction_label = "UP" if direction_pred == 1 else "DOWN"
                logreg_probs = logreg_model.predict_proba(last_features_df)[0]
                print(
                    f"LogReg Prediction for {TICKER}: {direction_label} | Probabilities: DOWN={logreg_probs[0]:.2f}, UP={logreg_probs[1]:.2f}")
        except Exception as e:
            print(f"LogReg prediction failed for {TICKER}: {e}")

        # FinBERT Sentiment
        sentiments = []
        if os.path.exists(news_file) and finbert_available:
            try:
                news_df = pd.read_csv(news_file)
                if "title" in news_df.columns and "content" in news_df.columns:
                    news_df["text"] = news_df["title"].fillna("") + ". " + news_df["content"].fillna("")
                else:
                    txt_col = news_df.columns[0]
                    news_df["text"] = news_df[txt_col].astype(str)
                if "date" in news_df.columns:
                    news_df["date"] = pd.to_datetime(news_df["date"], errors="coerce")
                    news_df = news_df[news_df["date"].dt.date == pd.to_datetime(DATE_SUFFIX).date()]
                latest_news = news_df["text"].dropna().tolist()
                if latest_news:
                    inputs = tokenizer(latest_news, padding=True, truncation=True, max_length=128, return_tensors="pt")
                    with torch.no_grad():
                        outputs = finbert_model(**inputs)
                    preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                    labels_map = {0: "positive", 1: "neutral", 2: "negative"}
                    sentiments = [labels_map[int(p)] for p in preds]
            except Exception as e:
                print(f"FinBERT failed for {TICKER}: {e}")

        # Calculate Risk Metrics
        risk_metrics = calculate_risk_metrics(df, y_future)

        # Decision with enhanced output
        last_close = float(df["close"].iloc[-1])
        action, reason, total_score, confidences = decide_action(
            y_future if len(y_future) > 0 else [last_close],
            last_close, direction_label or "N/A",
            sentiments, logreg_model, last_features_df,
            weights=WEIGHTS
        )

        # Sentiment Override
        action, override_reason = check_sentiment_override(sentiments, action)
        if override_reason:
            reason += "\n" + override_reason

        # Generate Trading Recommendation
        trading_rec = generate_trading_recommendation(action, risk_metrics, total_score)

        # Calculate predicted change percentage
        if len(y_future) > 0:
            predicted_change_pct = ((np.mean(y_future) - last_close) / last_close * 100)
        else:
            predicted_change_pct = 0.0

        # Store detailed analysis
        ticker_analysis = {
            "ticker": TICKER,
            "timestamp": datetime.now(),
            "current_price": last_close,
            "final_action": action,
            "total_score": total_score,
            "predicted_change_pct": predicted_change_pct,
            "lstm_confidence": confidences["lstm_confidence"],
            "logreg_confidence": confidences["logreg_confidence"],
            "sentiment_confidence": confidences["sentiment_confidence"],
            "risk_level": trading_rec["risk_level"],
            "position_size": trading_rec["position_size"],
            "historical_volatility": risk_metrics.get("historical_volatility", 0),
            "forecast_volatility": risk_metrics.get("forecast_volatility", 0),
            "max_drawdown": risk_metrics.get("max_drawdown", 0),
            "support_level": risk_metrics.get("support_level", 0),
            "resistance_level": risk_metrics.get("resistance_level", 0),
            "positive_news": sentiments.count("positive"),
            "neutral_news": sentiments.count("neutral"),
            "negative_news": sentiments.count("negative"),
            "news_total": len(sentiments)
        }
        detailed_analysis.append(ticker_analysis)

        # Create COMPACT dashboard
        try:
            fig = create_compact_dashboard(
                TICKER, df, y_future, future_dates, risk_metrics,
                logreg_probs, direction_label, sentiments,
                action, reason, total_score, confidences, trading_rec
            )

            fig.savefig(os.path.join(OUT_DIR, f"{TICKER}_compact_dashboard.png"),
                       dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            print(f"✅ Compact dashboard saved for {TICKER}")
        except Exception as e:
            print(f"❌ Dashboard creation failed for {TICKER}: {e}")

        # Summary data
        forecast_mean = float(np.mean(y_future)) if len(y_future) > 0 else None

        summary_rows.append({
            "ticker": TICKER,
            "action": action,
            "reason": reason,
            "total_score": total_score,
            "last_close": last_close,
            "forecast_mean": forecast_mean,
            "predicted_change_pct": predicted_change_pct,
            "logreg_dir": direction_label,
            "lstm_confidence": confidences["lstm_confidence"],
            "logreg_confidence": confidences["logreg_confidence"],
            "sentiment_confidence": confidences["sentiment_confidence"],
            "sent_pos": sentiments.count("positive") if sentiments else 0,
            "sent_neu": sentiments.count("neutral") if sentiments else 0,
            "sent_neg": sentiments.count("negative") if sentiments else 0,
            "risk_level": trading_rec["risk_level"],
            "historical_volatility": risk_metrics.get("historical_volatility", 0),
            "max_drawdown": risk_metrics.get("max_drawdown", 0)
        })

        print(f"✅ Successfully processed {TICKER}: {action} (Score: {total_score:.3f})")

    except Exception as exc:
        print(f"❌ Unexpected failure for {TICKER}: {exc}")
        traceback.print_exc()
        summary_rows.append({
            "ticker": TICKER,
            "action": "ERROR",
            "reason": str(exc),
            "total_score": None,
            "last_close": None,
            "forecast_mean": None,
            "predicted_change_pct": None,
            "logreg_dir": None,
            "lstm_confidence": None,
            "logreg_confidence": None,
            "sentiment_confidence": None,
            "sent_pos": 0,
            "sent_neu": 0,
            "sent_neg": 0,
            "risk_level": "Unknown",
            "historical_volatility": None,
            "max_drawdown": None
        })

# Save summary and detailed analysis
summary_df = pd.DataFrame(summary_rows)
summary_csv = os.path.join(OUT_DIR, "summary_actions_weighted.csv")
summary_df.to_csv(summary_csv, index=False)

detailed_df = pd.DataFrame(detailed_analysis)
detailed_csv = os.path.join(OUT_DIR, "detailed_hybrid_analysis.csv")
detailed_df.to_csv(detailed_csv, index=False)

# Generate Comprehensive Portfolio Summary
print(f"\n{'=' * 80}")
print("HYBRID MODEL PORTFOLIO SUMMARY")
print(f"{'=' * 80}")

# Action Distribution
action_counts = summary_df['action'].value_counts()
print("\n📊 ACTION DISTRIBUTION:")
for action, count in action_counts.items():
    percentage = (count / len(summary_df)) * 100
    print(f"  {action}: {count} stocks ({percentage:.1f}%)")

# Performance Summary
successful_analysis = summary_df[summary_df['action'].isin(['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'])]
if not successful_analysis.empty:
    print(f"\n✅ SUCCESSFUL ANALYSES: {len(successful_analysis)} stocks")

    # Buy vs Sell recommendations
    strong_buy = successful_analysis[successful_analysis['action'] == 'STRONG BUY']
    buy = successful_analysis[successful_analysis['action'] == 'BUY']
    hold = successful_analysis[successful_analysis['action'] == 'HOLD']
    sell = successful_analysis[successful_analysis['action'] == 'SELL']
    strong_sell = successful_analysis[successful_analysis['action'] == 'STRONG SELL']

    print(f"  🟢 STRONG BUY: {len(strong_buy)}")
    print(f"  🟢 BUY: {len(buy)}")
    print(f"  🟡 HOLD: {len(hold)}")
    print(f"  🔴 SELL: {len(sell)}")
    print(f"  🔴 STRONG SELL: {len(strong_sell)}")

    # Total bullish vs bearish
    total_bullish = len(strong_buy) + len(buy)
    total_bearish = len(sell) + len(strong_sell)
    print(f"\n  📈 Bullish (BUY): {total_bullish} stocks")
    print(f"  📉 Bearish (SELL): {total_bearish} stocks")
    print(f"  ⚖️  Neutral (HOLD): {len(hold)} stocks")

    # Average scores
    if 'total_score' in successful_analysis.columns:
        avg_score = successful_analysis['total_score'].mean()
        max_score = successful_analysis['total_score'].max()
        min_score = successful_analysis['total_score'].min()
        print(f"\n  🎯 Score Statistics:")
        print(f"    Average: {avg_score:.3f}")
        print(f"    Maximum: {max_score:.3f}")
        print(f"    Minimum: {min_score:.3f}")

# Risk Analysis
if not detailed_df.empty:
    avg_volatility = detailed_df['historical_volatility'].mean()
    avg_drawdown = detailed_df['max_drawdown'].mean()
    max_volatility = detailed_df['historical_volatility'].max()
    max_volatility_stock = detailed_df.loc[detailed_df['historical_volatility'].idxmax()]
    max_drawdown_stock = detailed_df.loc[detailed_df['max_drawdown'].idxmax()]

    print(f"\n⚠️  PORTFOLIO RISK METRICS:")
    print(f"  📊 Average Volatility: {avg_volatility:.2f}%")
    print(f"  📊 Maximum Volatility: {max_volatility:.2f}% ({max_volatility_stock['ticker']})")
    print(f"  📉 Average Max Drawdown: {avg_drawdown:.2f}%")
    print(f"  📉 Worst Drawdown: {max_drawdown_stock['max_drawdown']:.2f}% ({max_drawdown_stock['ticker']})")

    # High Confidence Recommendations
    high_confidence = detailed_df[detailed_df['lstm_confidence'] > 0.7]
    if not high_confidence.empty:
        print(f"\n🎯 HIGH CONFIDENCE RECOMMENDATIONS (LSTM Confidence > 70%):")
        for _, stock in high_confidence.iterrows():
            action_icon = "🟢" if "BUY" in stock['final_action'] else "🔴" if "SELL" in stock['final_action'] else "🟡"
            print(f"  {action_icon} {stock['ticker']}: {stock['final_action']} (Score: {stock['total_score']:.3f}, Conf: {stock['lstm_confidence']:.1%})")

    # Best and Worst Performers by Score
    if 'total_score' in detailed_df.columns:
        best_performer = detailed_df.loc[detailed_df['total_score'].idxmax()]
        worst_performer = detailed_df.loc[detailed_df['total_score'].idxmin()]

        print(f"\n🏆 BEST PERFORMER:")
        print(f"  {best_performer['ticker']}: {best_performer['final_action']} (Score: {best_performer['total_score']:.3f})")

        print(f"📉 WORST PERFORMER:")
        print(f"  {worst_performer['ticker']}: {worst_performer['final_action']} (Score: {worst_performer['total_score']:.3f})")

# Sentiment Analysis Summary
if not detailed_df.empty and 'positive_news' in detailed_df.columns:
    total_positive = detailed_df['positive_news'].sum()
    total_neutral = detailed_df['neutral_news'].sum()
    total_negative = detailed_df['negative_news'].sum()
    total_news = total_positive + total_neutral + total_negative

    if total_news > 0:
        print(f"\n📰 SENTIMENT ANALYSIS SUMMARY:")
        print(f"  Positive News: {total_positive} ({total_positive/total_news*100:.1f}%)")
        print(f"  Neutral News: {total_neutral} ({total_neutral/total_news*100:.1f}%)")
        print(f"  Negative News: {total_negative} ({total_negative/total_news*100:.1f}%)")
        print(f"  Total News Analyzed: {total_news}")

# Error Summary
error_stocks = summary_df[summary_df['action'] == 'ERROR']
if not error_stocks.empty:
    print(f"\n❌ STOCKS WITH ERRORS: {len(error_stocks)}")
    for _, stock in error_stocks.iterrows():
        # Truncate long error messages
        error_msg = stock['reason']
        if len(error_msg) > 100:
            error_msg = error_msg[:100] + "..."
        print(f"  {stock['ticker']}: {error_msg}")

# Generate Portfolio Allocation Recommendation
if not successful_analysis.empty:
    print(f"\n💼 SUGGESTED PORTFOLIO ALLOCATION:")

    strong_buy_allocation = len(strong_buy) / len(successful_analysis) * 100
    buy_allocation = len(buy) / len(successful_analysis) * 100
    hold_allocation = len(hold) / len(successful_analysis) * 100

    print(f"  🟢 STRONG BUY: {strong_buy_allocation:.1f}% of portfolio")
    print(f"  🟢 BUY: {buy_allocation:.1f}% of portfolio")
    print(f"  🟡 HOLD: {hold_allocation:.1f}% of portfolio")
    print(f"  🔴 AVOID: {100 - (strong_buy_allocation + buy_allocation + hold_allocation):.1f}% of portfolio")

# Model Performance Summary
if not detailed_df.empty:
    avg_lstm_conf = detailed_df['lstm_confidence'].mean()
    avg_logreg_conf = detailed_df['logreg_confidence'].mean()
    avg_sentiment_conf = detailed_df['sentiment_confidence'].mean()

    print(f"\n🤖 MODEL PERFORMANCE SUMMARY:")
    print(f"  LSTM Average Confidence: {avg_lstm_conf:.1%}")
    print(f"  Logistic Regression Average Confidence: {avg_logreg_conf:.1%}")
    print(f"  Sentiment Analysis Average Confidence: {avg_sentiment_conf:.1%}")

print(f"\n📁 OUTPUT FILES:")
print(f"  ✅ Summary CSV: {summary_csv}")
print(f"  ✅ Detailed Analysis: {detailed_csv}")
print(f"  ✅ Compact Dashboards: {OUT_DIR}/*_compact_dashboard.png")

print(f"\n🎯 HYBRID MODEL ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"   Total Stocks Processed: {len(TICKERS)}")
print(f"   Successful: {len(successful_analysis) if not successful_analysis.empty else 0}")
print(f"   Errors: {len(error_stocks)}")
print(f"   Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Create a quick portfolio overview chart
if not successful_analysis.empty:
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart of actions
        action_counts = successful_analysis['action'].value_counts()
        colors = ['#2ecc71', '#27ae60', '#f39c12', '#e74c3c', '#c0392b']  # Original colors

        # Tukar warna STRONG SELL saja - guna dark red (#8b0000)
        color_list = []
        for action in action_counts.index:
            if action == 'STRONG SELL':
                color_list.append('#8b0000')  # Dark red untuk STRONG SELL
            else:
                # Cari warna asal untuk action lain
                color_idx = ['STRONG BUY', 'BUY', 'HOLD', 'SELL', 'STRONG SELL'].index(action)
                color_list.append(colors[color_idx])

        ax1.pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%',
                colors=color_list, startangle=90)
        ax1.set_title('Portfolio Action Distribution', fontweight='bold')

        # Bar chart of scores (tetap sama)
        successful_sorted = successful_analysis.sort_values('total_score', ascending=True)
        colors_bar = ['#e74c3c' if x < -0.2 else '#f39c12' if x < 0.2 else '#2ecc71' for x in
                      successful_sorted['total_score']]
        bars = ax2.barh(successful_sorted['ticker'], successful_sorted['total_score'], color=colors_bar, alpha=0.7)
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax2.axvline(x=0.2, color='green', linestyle='--', alpha=0.5, label='Buy Threshold')
        ax2.axvline(x=-0.2, color='red', linestyle='--', alpha=0.5, label='Sell Threshold')
        ax2.set_xlabel('Total Score')
        ax2.set_title('Stock Scores Distribution', fontweight='bold')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "portfolio_overview.png"), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Portfolio Overview Chart: {os.path.join(OUT_DIR, 'portfolio_overview.png')}")
    except Exception as e:
        print(f"  ⚠️  Could not create portfolio chart: {e}")

print(f"\n{'=' * 80}")
print("ANALYSIS COMPLETE - READY FOR TRADING DECISIONS!")
print(f"{'=' * 80}")