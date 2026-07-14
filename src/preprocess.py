import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Text Cleaning
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)              # Remove HTML
    text = re.sub(r"http\S+|www\S+", " ", text)      # Remove URLs
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\d+", " ", text)                 # Remove numbers
    text = re.sub(r"\s+", " ", text).strip()         # Collapse whitespace
    return text

# Load Dataset
def load_dataset(path):
    df = pd.read_csv(path)
    df["description"] = df["description"].apply(clean_text)
    return df

# TF-IDF
def create_vectorizer():
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english"
    )
    return vectorizer

# Split Dataset
def split_data(df, test_size=0.20, random_state=42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train_df, test_df

