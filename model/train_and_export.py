#!/usr/bin/env python3
"""
train_and_export.py - Serializes the optimal trained models into model/ for application inference.

Artifacts saved to model/:
1. tfidf_vectorizer.joblib
2. emotion_model.joblib (12-class Logistic Regression with probability estimates)
3. sentiment_model.joblib (Bullish/Bearish Logistic Regression with probability estimates)
4. metadata.json (labels, mapping dictionaries, evaluation metrics)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Set random seed
np.random.seed(42)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(WORKSPACE_DIR, "model")
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")
RESULTS_PATH = os.path.join(ANALYSIS_DIR, "results.json")

os.makedirs(MODEL_DIR, exist_ok=True)

def export_models():
    print("[train_and_export] Loading prepared dataset...")
    df = pd.read_csv(PREPARED_CSV)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    # 1. Fit TF-IDF Vectorizer on train split
    print("[train_and_export] Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["processed"])
    
    # 2. Train Emotion Classifier (12 classes)
    # Using C=5.0 which was selected as optimal on validation split
    print("[train_and_export] Training 12-class Emotion Classifier (Logistic Regression, C=5.0)...")
    emo_model = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    emo_model.fit(X_train, train_df["emo_encoded"])
    
    # 3. Train Sentiment Classifier (Bullish vs Bearish)
    # Using C=1.0 which was selected as optimal on validation split
    print("[train_and_export] Training Sentiment Classifier (Logistic Regression, C=1.0)...")
    senti_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    senti_model.fit(X_train, train_df["senti_encoded"])
    
    # Save artifacts
    tfidf_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
    emo_path = os.path.join(MODEL_DIR, "emotion_model.joblib")
    senti_path = os.path.join(MODEL_DIR, "sentiment_model.joblib")
    meta_path = os.path.join(MODEL_DIR, "metadata.json")
    
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(emo_model, emo_path)
    joblib.dump(senti_model, senti_path)
    
    # Prepare metadata
    emo_labels = sorted(df["emo_label"].unique().tolist())
    emo_to_id = {emo: i for i, emo in enumerate(emo_labels)}
    id_to_emo = {i: emo for i, emo in enumerate(emo_labels)}
    
    senti_labels = ["bearish", "bullish"]
    senti_to_id = {"bearish": 0, "bullish": 1}
    id_to_senti = {0: "bearish", 1: "bullish"}
    
    emotion_groups = {
        "positive": ["optimism", "excitement", "amusement", "belief"],
        "negative": ["anxiety", "anger", "panic", "depression", "disgust"],
        "neutral": ["ambiguous", "confusion", "surprise"]
    }
    
    metadata = {
        "emotion_labels": emo_labels,
        "emotion_to_id": emo_to_id,
        "id_to_emotion": id_to_emo,
        "sentiment_labels": senti_labels,
        "sentiment_to_id": senti_to_id,
        "id_to_sentiment": id_to_senti,
        "emotion_groups": emotion_groups,
        "model_architecture": {
            "feature_extractor": "TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)",
            "emotion_classifier": "LogisticRegression(C=5.0, solver='lbfgs', multi_class='auto')",
            "sentiment_classifier": "LogisticRegression(C=1.0, solver='lbfgs')"
        }
    }
    
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[train_and_export] All models successfully saved to {MODEL_DIR}")

if __name__ == "__main__":
    export_models()
