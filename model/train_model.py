import os
import re
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -------------------------------
# 1. Text Cleaning
# -------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# -------------------------------
# 2. Load Dataset
# -------------------------------

DATA_PATH = "./datasets/Resume.csv"

df = pd.read_csv(DATA_PATH)
df.drop(columns=["Resume_html"], errors="ignore", inplace=True)

df["cleaned_resume"] = df["Resume_str"].apply(clean_text)

X_text = df["cleaned_resume"]
y = df["Category"]

# -------------------------------
# 3. Train/Test Split
# -------------------------------

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# 4. TF-IDF Vectorizer
# -------------------------------

vectorizer = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 3),
    min_df=2
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# -------------------------------
# 5. Model Training
# -------------------------------

model = LinearSVC(class_weight="balanced", C=1.0)
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluation
# -------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print("Accuracy:", accuracy)

# -------------------------------
# 7. Create Output Folder
# -------------------------------

timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join("./artifacts", timestamp)
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# 8. Save Model & Vectorizer
# -------------------------------

joblib.dump(model, os.path.join(output_dir, "resume_model.pkl"))
joblib.dump(vectorizer, os.path.join(output_dir, "vectorizer.pkl"))

# -------------------------------
# 9. Save Classification Report
# -------------------------------

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy}\n\n")
    f.write(classification_report(y_test, y_pred))

# Save JSON version
with open(os.path.join(output_dir, "metrics.json"), "w") as f:
    json.dump({
        "accuracy": accuracy,
        "classification_report": report
    }, f, indent=4)

# -------------------------------
# 10. Confusion Matrix Plot
# -------------------------------

cm = confusion_matrix(y_test, y_pred)
labels = model.classes_

plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

print(f"\nTraining artifacts saved in: {output_dir}")