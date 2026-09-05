import sys

import joblib
import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

MODEL_PATH = "potato_disease_model.pkl"
SCALER_PATH = "feature_scaler.pkl"


def predict(image_path):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    features = extract_features(image_path)
    X = pd.DataFrame([features], columns=FEATURE_NAMES)
    X_scaled = scaler.transform(X)

    predicted_class = model.predict(X_scaled)[0]
    probabilities = model.predict_proba(X_scaled)[0]

    return predicted_class, dict(zip(model.classes_, probabilities))


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py path/to/leaf_image.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    predicted_class, probabilities = predict(image_path)

    print(f"\nPrediction: {predicted_class}")
    print("\nConfidence breakdown:")
    for class_name, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        print(f"  {class_name}: {prob * 100:.2f}%")

    print("\nNote: this is an educational model based on hand-crafted image "
          "features, not a substitute for expert agricultural diagnosis.")


if __name__ == "__main__":
    main()
