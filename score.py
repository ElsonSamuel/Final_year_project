
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Load the model and vectorizer
model = joblib.load("trained_model.sav")
vectorizer = joblib.load("vectorizer.pkl")

# Load the dataset
df = pd.read_csv("f:/Project/new/dataset/tanglish.csv")
df.columns = ["text", "label", "binary_label"]
df["binary_label"] = df["label"].apply(lambda x: 1 if x.lower() == "positive" else 0)

# Transform the text
X_tfidf = vectorizer.transform(df["text"])

# Make predictions
y_pred = model.predict(X_tfidf)

# Compute accuracy and F1-score
accuracy = accuracy_score(df["binary_label"], y_pred)
f1_positive = f1_score(df["binary_label"], y_pred, pos_label=1)
f1_negative = f1_score(df["binary_label"], y_pred, pos_label=0)

print("Accuracy:", accuracy)
print("F1-Score (Positive):", f1_positive)
print("F1-Score (Negative):", f1_negative)
