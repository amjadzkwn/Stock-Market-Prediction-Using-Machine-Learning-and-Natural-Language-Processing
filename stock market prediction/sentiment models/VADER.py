# vader_sentiment.py
import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# VADER
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
except Exception as e:
    raise ImportError(
        "nltk or VADER not found. Install with: pip install nltk\n"
        "Then in Python run:\n"
        "  import nltk\n"
        "  nltk.download('vader_lexicon')\n"
    ) from e

# ---------------------------
# Debug / Environment Info
# ---------------------------
print("=== DEBUG INFO ===")
print("Python executable:", sys.executable)
print("Current working dir:", os.getcwd())

# Fix seed for reproducibility of train/test split
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------
# 1. Load Dataset
# ---------------------------
path = r"C:\Users\AMJAD\PycharmProjects\fyp1test\Lib\News Stock Dataset\Ground Truth Sentiment Analysis\all-data.csv"
print("\nLoading dataset from:", path)

# If your file has no header, use header=None as in your BERT script
df = pd.read_csv(path, encoding="ISO-8859-1", header=None)
df.columns = ["label", "text"]
print("Dataset shape:", df.shape)
print("First rows:\n", df.head(5))

# Standardize labels (same mapping as your BERT script)
df["label"] = df["label"].astype(str).str.lower().str.strip()
label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}
# filter any rows with labels not in mapping (safety)
df = df[df["label"].isin(label2id.keys())].copy()
df["label_id"] = df["label"].map(label2id)
print("After filtering label distribution:\n", df["label"].value_counts())

# ---------------------------
# 2. Train-test split
# ---------------------------
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["label"]  # keep same class proportions
)
print("\nTrain size:", len(train_df), "Test size:", len(test_df))
print("Train label distribution:\n", train_df["label"].value_counts())
print("Test label distribution:\n", test_df["label"].value_counts())

# ---------------------------
# 3. Initialize VADER
# ---------------------------
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

def vader_label_from_text(text, pos_thresh=0.05, neg_thresh=-0.05):
    """
    Standard VADER compound thresholds:
      compound >= 0.05 -> positive
      compound <= -0.05 -> negative
      else -> neutral
    Returns label string and id (consistent with label2id mapping).
    """
    if not isinstance(text, str):
        text = ""
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= pos_thresh:
        lbl = "positive"
    elif compound <= neg_thresh:
        lbl = "negative"
    else:
        lbl = "neutral"
    return lbl, label2id[lbl], scores

# ---------------------------
# 4. Run VADER on test set
# ---------------------------
print("\nRunning VADER on test set...")
pred_labels = []
pred_label_ids = []
pred_scores = []

for txt in test_df["text"].tolist():
    lbl, lbl_id, scores = vader_label_from_text(txt)
    pred_labels.append(lbl)
    pred_label_ids.append(lbl_id)
    pred_scores.append(scores)

test_df = test_df.reset_index(drop=True)
test_df["vader_label"] = pred_labels
test_df["vader_label_id"] = pred_label_ids
# Expand score dicts into columns
scores_df = pd.DataFrame(pred_scores)
test_df = pd.concat([test_df, scores_df], axis=1)

# ---------------------------
# 5. Evaluation
# ---------------------------
y_true = test_df["label_id"].tolist()
y_pred = test_df["vader_label_id"].tolist()

acc = accuracy_score(y_true, y_pred)
print("\n=== VADER Evaluation ===")
print("Accuracy:", acc)
print("\nClassification report:")
target_names = ["positive", "neutral", "negative"]
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("VADER - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ---------------------------
# 6. Optional: Save results
# ---------------------------
out_csv = "vader_test_results.csv"
test_df.to_csv(out_csv, index=False, encoding="utf-8")
print(f"\nSaved detailed test results to: {out_csv}")

# ---------------------------
# 7. Optional: Quick analysis
# ---------------------------
# Show some examples where VADER disagreed with ground truth (useful for inspection)
mismatch = test_df[test_df["label_id"] != test_df["vader_label_id"]].sample(n=min(10, len(test_df[test_df["label_id"] != test_df["vader_label_id"]])), random_state=SEED)
if not mismatch.empty:
    print("\nSample mismatches (true_label -> vader_label -> compound):")
    for idx, row in mismatch.iterrows():
        print(f"- True: {row['label']} | VADER: {row['vader_label']} | compound={row['compound']:.3f} | text_snippet: {row['text'][:120]!r}")
else:
    print("\nNo mismatches found in sampled selection (unexpected if dataset > tiny).")

# End
print("\nDone.")
