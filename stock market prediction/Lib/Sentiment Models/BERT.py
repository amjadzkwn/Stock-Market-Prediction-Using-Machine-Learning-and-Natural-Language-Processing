# bert_finetune_sentiment.py
import os
import sys
import pandas as pd
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, __version__ as transformers_version
)
from datasets import Dataset

# =====================================================
# Debug Environment
# =====================================================
print("=== DEBUG INFO ===")
print("Python executable:", sys.executable)
print("Transformers version:", transformers_version)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Current working dir:", os.getcwd())

# =====================================================
# 1. Load Dataset
# =====================================================
path = r"C:\Users\AMJAD\PycharmProjects\fyp1test\Lib\News Stock Dataset\Ground Truth Sentiment Analysis\all-data.csv"
print("\nLoading dataset from:", path)

df = pd.read_csv(path, encoding="ISO-8859-1", header=None)
df.columns = ["label", "text"]
print("Dataset shape:", df.shape)
print("First rows:\n", df.head())

# Standardize labels
df["label"] = df["label"].str.lower().str.strip()
label2id = {"positive": 0, "neutral": 1, "negative": 2}
id2label = {v: k for k, v in label2id.items()}
df["label"] = df["label"].map(label2id)

# Train-test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(), df["label"].tolist(),
    test_size=0.2, random_state=42, stratify=df["label"]
)
print("Train size:", len(train_texts), "Test size:", len(test_texts))

# =====================================================
# 2. Tokenization
# =====================================================
print("\nLoading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

print("Tokenizing datasets...")
train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
print("Train dataset format:", train_dataset)

# =====================================================
# 3. Load Model
# =====================================================
print("\nLoading BERT-base model...")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# =====================================================
# 4. Define Metrics
# =====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc}

# =====================================================
# 5. Training Arguments
# =====================================================
print("\nSetting training arguments...")
training_args = TrainingArguments(
    output_dir="./bert_results",
    do_eval=True,
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs_bert",
    logging_steps=100,
    report_to=[]  # disable wandb/tensorboard
)

# =====================================================
# 6. Trainer
# =====================================================
print("\nInitializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =====================================================
# 7. Train & Evaluate
# =====================================================
print("\nStarting training...")
trainer.train()

print("\nRunning prediction on test set...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_labels

# =====================================================
# 8. Evaluation Report
# =====================================================
print("\n=== BERT-base (Fine-Tuned) Evaluation ===")
print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=label2id.keys(), digits=4))

# Confusion Matrix
labels = ["positive", "neutral", "negative"]
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("BERT-base Fine-Tuned - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
