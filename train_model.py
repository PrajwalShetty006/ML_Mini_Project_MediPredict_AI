import numpy as np
import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Load Dataset ---
df = pd.read_csv("data/dataset.csv")
df.fillna(0, inplace=True)

# --- Build Symptom Set (exactly like your Colab) ---
symptoms = set()
for col in df.columns[1:]:
    symptoms.update(df[col].unique())
symptoms.discard(0)

# --- Build Binary Matrix ---
new_df = pd.DataFrame(columns=list(symptoms))
new_df["Disease"] = df["Disease"]
new_df.fillna(0, inplace=True)

for i in range(len(df)):
    for col in df.columns[1:]:
        symptom = df.iloc[i][col]
        if symptom != 0:
            new_df.at[i, symptom] = 1

new_df.fillna(0, inplace=True)

# --- Move Disease column to first ---
cols = list(new_df.columns)
cols.remove("Disease")
new_df = new_df[["Disease"] + cols]

# --- Split Features and Target ---
X = new_df.drop("Disease", axis=1)
y = new_df["Disease"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# --- Train Model (exact same params as Colab) ---
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=54
)
model.fit(X_train, y_train)

# --- Evaluate ---
y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# --- Save Model and Symptoms List ---
os.makedirs("models", exist_ok=True)

symptoms_list = list(X.columns)  # Save column order for app.py

with open("models/rf_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/symptoms_list.pkl", "wb") as f:
    pickle.dump(symptoms_list, f)

print("Saved: models/rf_model.pkl and models/symptoms_list.pkl")
print(f"Total unique symptoms: {len(symptoms_list)}")