import os

import pandas as pd

from feature_extraction import extract_features, FEATURE_NAMES

DATA_DIR = "dataset"
OUTPUT_CSV = "features.csv"
CLASS_NAMES = ["Healthy", "Early_Blight", "Late_Blight"]

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def build_dataset():
    rows = []

    for class_name in CLASS_NAMES:
        class_dir = os.path.join(DATA_DIR, class_name)

        if not os.path.isdir(class_dir):
            print(f"Warning: folder not found, skipping: {class_dir}")
            continue

        image_files = [
            f for f in os.listdir(class_dir) if f.lower().endswith(VALID_EXTENSIONS)
        ]
        print(f"Processing {len(image_files)} images in '{class_name}'...")

        for img_name in image_files:
            img_path = os.path.join(class_dir, img_name)
            try:
                features = extract_features(img_path)
            except Exception as e:
                print(f"  Skipping {img_path} (error: {e})")
                continue

            features["label"] = class_name
            features["filename"] = img_name
            rows.append(features)


    df = pd.DataFrame(rows, columns=["filename", "label"] + FEATURE_NAMES)

    print(f"\nTotal images processed: {len(df)}")
    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nFeature summary (first few columns):")
    print(df[FEATURE_NAMES[:6]].describe())

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved features to {OUTPUT_CSV}")


if __name__ == "__main__":
    build_dataset()
