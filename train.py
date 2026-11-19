import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Project folder path
BASE_DIR = Path(__file__).resolve().parent

# Load dataset
csv_path = BASE_DIR / "dataset" / "student.csv"

print("CSV file path:", csv_path)
print("File exists:", csv_path.exists())

if csv_path.exists():
    print("File size:", csv_path.stat().st_size, "bytes")

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        print("CSV content:", repr(f.read(500)))

data = pd.read_csv(csv_path)
# Remove rows with missing target values
data = data.dropna(subset=["result"])

# Separate features and target
X = data.drop(columns=["result"]).select_dtypes(include="number")
X = X.fillna(X.median())

y = data["result"].astype(str).str.strip()

# Experiment configurations
runs = [
    (0.5, 10),
    (0.8, 50),
    (0.3, 100)
]

# MLflow experiment
mlflow.set_experiment("students")

best_acc = -1
best_model = None

for i, (train_size, n_estimators) in enumerate(runs, start=1):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(
        run_name=f"train={train_size}_trees={n_estimators}"
    ):

        # Train Random Forest model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # MLflow tracking
        mlflow.log_param("train_size", train_size)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", accuracy)

        print(f"\nRun {i}")
        print("Train size:", train_size)
        print("Number of trees:", n_estimators)
        print("Accuracy:", accuracy)

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Find best model
        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model

# Create models folder
model_dir = BASE_DIR / "models"
os.makedirs(model_dir, exist_ok=True)

# Save best model
model_path = model_dir / "student_models.pkl"

joblib.dump(best_model, model_path)

print("\nBest Accuracy:", best_acc)
print("Best model saved successfully!")
print("Model location:", model_path)