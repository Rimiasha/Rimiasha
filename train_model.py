import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from feature_extraction import FEATURE_NAMES

FEATURES_CSV = "features.csv"
MODEL_PATH = "potato_disease_model.pkl"
SCALER_PATH = "feature_scaler.pkl"


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} samples from {FEATURES_CSV}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    X = df[FEATURE_NAMES]   # feature columns only
    y = df["label"]          # target labels: Healthy / Early_Blight / Late_Blight

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=model.classes_),
        index=model.classes_,
        columns=model.classes_,
    ))

    importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES)
    print("\nTop 10 most important features:")
    print(importances.sort_values(ascending=False).head(10))

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved feature scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
