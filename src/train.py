import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.preprocess import load_dataset
from src.preprocess import  split_data
from src.preprocess import create_vectorizer


# Configuration
DATASET_PATH = "dataset/product_attribute_dataset.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGETS = [
    "category",
    "silhouette",
    "fabric",
    "neckline",
    "sleeve",
    "length",
    "embellishment",
    "color"
]


def main():
    # Load Dataset
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)
    train_df, test_df = split_data(df)

    # TF-IDF
    print("Creating TF-IDF...")
    vectorizer = create_vectorizer()
    X_train = vectorizer.fit_transform(train_df["description"])
    X_test = vectorizer.transform(test_df["description"])
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf.pkl"))
    print("TF-IDF Saved")

    # Train Models
    results = []
    for target in TARGETS:
        print(f"\nTraining {target}")

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(train_df[target])
        y_test = encoder.transform(test_df[target])
        joblib.dump(encoder, os.path.join(MODEL_DIR, f"{target}_encoder.pkl"))

        model = LinearSVC()
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)

        acc = accuracy_score(y_test, prediction)
        precision = precision_score(y_test, prediction, average="weighted", zero_division=0)
        recall = recall_score(y_test, prediction, average="weighted", zero_division=0)
        f1 = f1_score(y_test, prediction, average="weighted", zero_division=0)

        results.append({
            "Attribute": target,
            "Accuracy": round(acc, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4)
        })

        joblib.dump(model, os.path.join(MODEL_DIR, f"{target}.pkl"))
        print(f"{target} model saved.")

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_DIR, "evaluation.csv"), index=False)

    print("\nTraining Complete\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
