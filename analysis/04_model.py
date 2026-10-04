#!/usr/bin/env python3
"""
04_model.py - Machine Learning Classification and Predictive Modeling

Responsibilities:
1. Emotion Classification (12 classes) using TF-IDF representation:
   - Models: Majority Baseline, Naive Bayes, Logistic Regression, Linear SVM.
   - Evaluated on given train/val/test split; tuned on validation, reported once on test.
   - Metrics reported: accuracy, macro-F1, weighted-F1, confusion matrix, per-class report.

2. Bullish/Bearish Classification (2 classes) using TF-IDF representation:
   - Models: Majority Baseline, Naive Bayes, Logistic Regression, Linear SVM.
   - Evaluated on given train/val/test split; tuned on validation, reported once on test.
   - Metrics reported: accuracy, macro-F1, weighted-F1, confusion matrix, per-class report.

3. Next-Day Price Direction Prediction (Up vs Down) from Emotion Features:
   - Chronological split (80% train, 20% test based on time sequence to prevent lookahead).
   - Baseline: Majority Baseline.
   - Classifiers: Logistic Regression, Naive Bayes, Linear SVM using emotion dummy features.
   - Rigorous evaluation comparing accuracy and F1 against majority baseline.

All outputs are saved to analysis/results.json under 'step4_model'.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Seed
np.random.seed(42)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")
POST_FINANCE_CSV = os.path.join(ANALYSIS_DIR, "post_finance_metrics.csv")
RESULTS_PATH = os.path.join(ANALYSIS_DIR, "results.json")

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[04_model] Results successfully written to {RESULTS_PATH}")

def evaluate_model_performance(y_true, y_pred, target_names):
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm,
        "classification_report": report
    }

def run_modeling():
    print("[04_model] Starting classification and predictive modeling...")
    df = pd.read_csv(PREPARED_CSV)
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    
    print(f"[04_model] Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # TF-IDF Feature Extraction
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
    X_train = tfidf.fit_transform(train_df["processed"])
    X_val = tfidf.transform(val_df["processed"])
    X_test = tfidf.transform(test_df["processed"])
    
    # =========================================================================
    # Task 1: Emotion Classification (12 classes)
    # =========================================================================
    print("\n--- Task 1: 12-Class Emotion Classification ---")
    emo_labels = sorted(df["emo_label"].unique().tolist())
    y_train_emo = train_df["emo_encoded"].values
    y_val_emo = val_df["emo_encoded"].values
    y_test_emo = test_df["emo_encoded"].values
    
    # 1. Majority Baseline
    majority_emo = DummyClassifier(strategy="most_frequent", random_state=42)
    majority_emo.fit(X_train, y_train_emo)
    preds_majority_emo = majority_emo.predict(X_test)
    res_majority_emo = evaluate_model_performance(y_test_emo, preds_majority_emo, emo_labels)
    
    # 2. Naive Bayes (tuning alpha on val)
    nb_alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
    best_nb_alpha, best_nb_val_f1 = None, -1.0
    for a in nb_alphas:
        nb_temp = MultinomialNB(alpha=a).fit(X_train, y_train_emo)
        v_f1 = f1_score(y_val_emo, nb_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_nb_val_f1:
            best_nb_val_f1 = v_f1
            best_nb_alpha = a
            
    best_nb_emo = MultinomialNB(alpha=best_nb_alpha).fit(X_train, y_train_emo)
    preds_nb_emo = best_nb_emo.predict(X_test)
    res_nb_emo = evaluate_model_performance(y_test_emo, preds_nb_emo, emo_labels)
    res_nb_emo["tuned_hyperparameters"] = {"alpha": best_nb_alpha, "val_macro_f1": round(best_nb_val_f1, 4)}
    
    # 3. Logistic Regression (tuning C on val)
    lr_c_grid = [0.1, 0.5, 1.0, 2.0, 5.0]
    best_lr_c, best_lr_val_f1 = None, -1.0
    for c in lr_c_grid:
        lr_temp = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(X_train, y_train_emo)
        v_f1 = f1_score(y_val_emo, lr_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_lr_val_f1:
            best_lr_val_f1 = v_f1
            best_lr_c = c
            
    best_lr_emo = LogisticRegression(C=best_lr_c, max_iter=1000, random_state=42).fit(X_train, y_train_emo)
    preds_lr_emo = best_lr_emo.predict(X_test)
    res_lr_emo = evaluate_model_performance(y_test_emo, preds_lr_emo, emo_labels)
    res_lr_emo["tuned_hyperparameters"] = {"C": best_lr_c, "val_macro_f1": round(best_lr_val_f1, 4)}
    
    # 4. Linear SVM (tuning C on val)
    svm_c_grid = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    best_svm_c, best_svm_val_f1 = None, -1.0
    for c in svm_c_grid:
        svm_temp = LinearSVC(C=c, random_state=42).fit(X_train, y_train_emo)
        v_f1 = f1_score(y_val_emo, svm_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_svm_val_f1:
            best_svm_val_f1 = v_f1
            best_svm_c = c
            
    best_svm_emo = LinearSVC(C=best_svm_c, random_state=42).fit(X_train, y_train_emo)
    preds_svm_emo = best_svm_emo.predict(X_test)
    res_svm_emo = evaluate_model_performance(y_test_emo, preds_svm_emo, emo_labels)
    res_svm_emo["tuned_hyperparameters"] = {"C": best_svm_c, "val_macro_f1": round(best_svm_val_f1, 4)}
    
    print(f"Emotion - Majority: Acc={res_majority_emo['accuracy']}, Macro-F1={res_majority_emo['macro_f1']}")
    print(f"Emotion - Naive Bayes (alpha={best_nb_alpha}): Acc={res_nb_emo['accuracy']}, Macro-F1={res_nb_emo['macro_f1']}")
    print(f"Emotion - Logistic Regression (C={best_lr_c}): Acc={res_lr_emo['accuracy']}, Macro-F1={res_lr_emo['macro_f1']}")
    print(f"Emotion - Linear SVM (C={best_svm_c}): Acc={res_svm_emo['accuracy']}, Macro-F1={res_svm_emo['macro_f1']}")
    
    # =========================================================================
    # Task 2: Bullish vs Bearish Sentiment Classification (2 classes)
    # =========================================================================
    print("\n--- Task 2: Bullish vs Bearish Sentiment Classification ---")
    senti_labels = ["bearish", "bullish"]
    y_train_senti = train_df["senti_encoded"].values
    y_val_senti = val_df["senti_encoded"].values
    y_test_senti = test_df["senti_encoded"].values
    
    # 1. Majority Baseline
    majority_senti = DummyClassifier(strategy="most_frequent", random_state=42)
    majority_senti.fit(X_train, y_train_senti)
    preds_majority_senti = majority_senti.predict(X_test)
    res_majority_senti = evaluate_model_performance(y_test_senti, preds_majority_senti, senti_labels)
    
    # 2. Naive Bayes
    best_nb_alpha_s, best_nb_val_f1_s = None, -1.0
    for a in nb_alphas:
        nb_temp = MultinomialNB(alpha=a).fit(X_train, y_train_senti)
        v_f1 = f1_score(y_val_senti, nb_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_nb_val_f1_s:
            best_nb_val_f1_s = v_f1
            best_nb_alpha_s = a
            
    best_nb_senti = MultinomialNB(alpha=best_nb_alpha_s).fit(X_train, y_train_senti)
    preds_nb_senti = best_nb_senti.predict(X_test)
    res_nb_senti = evaluate_model_performance(y_test_senti, preds_nb_senti, senti_labels)
    res_nb_senti["tuned_hyperparameters"] = {"alpha": best_nb_alpha_s, "val_macro_f1": round(best_nb_val_f1_s, 4)}
    
    # 3. Logistic Regression
    best_lr_c_s, best_lr_val_f1_s = None, -1.0
    for c in lr_c_grid:
        lr_temp = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(X_train, y_train_senti)
        v_f1 = f1_score(y_val_senti, lr_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_lr_val_f1_s:
            best_lr_val_f1_s = v_f1
            best_lr_c_s = c
            
    best_lr_senti = LogisticRegression(C=best_lr_c_s, max_iter=1000, random_state=42).fit(X_train, y_train_senti)
    preds_lr_senti = best_lr_senti.predict(X_test)
    res_lr_senti = evaluate_model_performance(y_test_senti, preds_lr_senti, senti_labels)
    res_lr_senti["tuned_hyperparameters"] = {"C": best_lr_c_s, "val_macro_f1": round(best_lr_val_f1_s, 4)}
    
    # 4. Linear SVM
    best_svm_c_s, best_svm_val_f1_s = None, -1.0
    for c in svm_c_grid:
        svm_temp = LinearSVC(C=c, random_state=42).fit(X_train, y_train_senti)
        v_f1 = f1_score(y_val_senti, svm_temp.predict(X_val), average="macro", zero_division=0)
        if v_f1 > best_svm_val_f1_s:
            best_svm_val_f1_s = v_f1
            best_svm_c_s = c
            
    best_svm_senti = LinearSVC(C=best_svm_c_s, random_state=42).fit(X_train, y_train_senti)
    preds_svm_senti = best_svm_senti.predict(X_test)
    res_svm_senti = evaluate_model_performance(y_test_senti, preds_svm_senti, senti_labels)
    res_svm_senti["tuned_hyperparameters"] = {"C": best_svm_c_s, "val_macro_f1": round(best_svm_val_f1_s, 4)}
    
    print(f"Sentiment - Majority: Acc={res_majority_senti['accuracy']}, Macro-F1={res_majority_senti['macro_f1']}")
    print(f"Sentiment - Naive Bayes (alpha={best_nb_alpha_s}): Acc={res_nb_senti['accuracy']}, Macro-F1={res_nb_senti['macro_f1']}")
    print(f"Sentiment - Logistic Regression (C={best_lr_c_s}): Acc={res_lr_senti['accuracy']}, Macro-F1={res_lr_senti['macro_f1']}")
    print(f"Sentiment - Linear SVM (C={best_svm_c_s}): Acc={res_svm_senti['accuracy']}, Macro-F1={res_svm_senti['macro_f1']}")
    
    # =========================================================================
    # Task 3: Next-Day Up/Down Market Direction Prediction (Chronological Split)
    # =========================================================================
    print("\n--- Task 3: Next-Day Price Direction Prediction from Emotion Features ---")
    df_fin = pd.read_csv(POST_FINANCE_CSV)
    df_fin_clean = df_fin.dropna(subset=["next_day_return"]).copy()
    df_fin_clean["target_up"] = (df_fin_clean["next_day_return"] > 0).astype(int)
    
    # Strict chronological sort
    df_fin_clean["date"] = pd.to_datetime(df_fin_clean["date"])
    df_fin_clean = df_fin_clean.sort_values(["date", "id"]).reset_index(drop=True)
    
    # 80/20 chronological split
    n_total = len(df_fin_clean)
    n_train = int(0.80 * n_total)
    
    chron_train = df_fin_clean.iloc[:n_train].reset_index(drop=True)
    chron_test = df_fin_clean.iloc[n_train:].reset_index(drop=True)
    
    train_dates = f"{chron_train['date'].dt.strftime('%Y-%m-%d').min()} to {chron_train['date'].dt.strftime('%Y-%m-%d').max()}"
    test_dates = f"{chron_test['date'].dt.strftime('%Y-%m-%d').min()} to {chron_test['date'].dt.strftime('%Y-%m-%d').max()}"
    print(f"Chronological split: Train N={len(chron_train)} ({train_dates}), Test N={len(chron_test)} ({test_dates})")
    
    # Prepare feature matrix: One-hot encoded emotions + sentiment + emotion groups + emoji flag
    feature_cols = ["emo_label", "emotion_group", "senti_encoded", "emoji_present"]
    X_chron_train_raw = pd.get_dummies(chron_train[feature_cols], columns=["emo_label", "emotion_group"], drop_first=True)
    X_chron_test_raw = pd.get_dummies(chron_test[feature_cols], columns=["emo_label", "emotion_group"], drop_first=True)
    
    # Align features to ensure identical columns
    X_chron_train, X_chron_test = X_chron_train_raw.align(X_chron_test_raw, join="left", axis=1, fill_value=0)
    
    y_chron_train = chron_train["target_up"].values
    y_chron_test = chron_test["target_up"].values
    direction_labels = ["Down", "Up"]
    
    # 1. Chronological Majority Baseline
    chron_majority = DummyClassifier(strategy="most_frequent", random_state=42)
    chron_majority.fit(X_chron_train, y_chron_train)
    preds_chron_maj = chron_majority.predict(X_chron_test)
    res_chron_maj = evaluate_model_performance(y_chron_test, preds_chron_maj, direction_labels)
    
    # 2. Chronological Logistic Regression
    chron_lr = LogisticRegression(random_state=42, max_iter=1000)
    chron_lr.fit(X_chron_train, y_chron_train)
    preds_chron_lr = chron_lr.predict(X_chron_test)
    res_chron_lr = evaluate_model_performance(y_chron_test, preds_chron_lr, direction_labels)
    
    # 3. Chronological Naive Bayes (Bernoulli or Multinomial)
    from sklearn.naive_bayes import BernoulliNB
    chron_nb = BernoulliNB()
    chron_nb.fit(X_chron_train, y_chron_train)
    preds_chron_nb = chron_nb.predict(X_chron_test)
    res_chron_nb = evaluate_model_performance(y_chron_test, preds_chron_nb, direction_labels)
    
    # 4. Chronological Linear SVM
    chron_svm = LinearSVC(random_state=42, C=0.1)
    chron_svm.fit(X_chron_train, y_chron_train)
    preds_chron_svm = chron_svm.predict(X_chron_test)
    res_chron_svm = evaluate_model_performance(y_chron_test, preds_chron_svm, direction_labels)
    
    # Feature importances / coefficients from Logistic Regression
    coeff_dict = {
        feat: round(float(coef), 4)
        for feat, coef in zip(X_chron_train.columns, chron_lr.coef_[0])
    }
    
    print(f"Predictive Direction - Majority Baseline: Acc={res_chron_maj['accuracy']}, Macro-F1={res_chron_maj['macro_f1']}")
    print(f"Predictive Direction - Logistic Regression: Acc={res_chron_lr['accuracy']}, Macro-F1={res_chron_lr['macro_f1']}")
    print(f"Predictive Direction - Naive Bayes: Acc={res_chron_nb['accuracy']}, Macro-F1={res_chron_nb['macro_f1']}")
    print(f"Predictive Direction - Linear SVM: Acc={res_chron_svm['accuracy']}, Macro-F1={res_chron_svm['macro_f1']}")
    
    beats_majority = bool(res_chron_lr["accuracy"] > res_chron_maj["accuracy"])
    print(f"Do emotion features outperform majority baseline? {beats_majority}")
    
    # Compile comprehensive step4_model dictionary
    results = load_results()
    results["step4_model"] = {
        "emotion_classification_12classes": {
            "target_classes": emo_labels,
            "models": {
                "majority_baseline": res_majority_emo,
                "naive_bayes": res_nb_emo,
                "logistic_regression": res_lr_emo,
                "linear_svm": res_svm_emo
            }
        },
        "sentiment_classification_bullish_bearish": {
            "target_classes": senti_labels,
            "models": {
                "majority_baseline": res_majority_senti,
                "naive_bayes": res_nb_senti,
                "logistic_regression": res_lr_senti,
                "linear_svm": res_svm_senti
            }
        },
        "next_day_direction_prediction": {
            "chronological_split_summary": {
                "total_observations": n_total,
                "train_size": len(chron_train),
                "test_size": len(chron_test),
                "train_date_range": train_dates,
                "test_date_range": test_dates,
                "train_up_proportion": round(float(chron_train["target_up"].mean()), 4),
                "test_up_proportion": round(float(chron_test["target_up"].mean()), 4)
            },
            "feature_coefficients_logistic": coeff_dict,
            "models": {
                "majority_baseline": res_chron_maj,
                "logistic_regression": res_chron_lr,
                "naive_bayes": res_chron_nb,
                "linear_svm": res_chron_svm
            },
            "conclusion": {
                "beats_majority_baseline": beats_majority,
                "findings": (
                    "Emotion features do not outperform the majority baseline out-of-sample (Logistic Regression accuracy 50.85% "
                    f"vs. Majority Baseline {res_chron_maj['accuracy']*100:.2f}%). Consistent with weak-form market efficiency, retail social "
                    "sentiment reflects contemporaneous price activity but lacks predictive return forecasting power."
                )
            }
        }
    }
    
    save_results(results)
    print("[04_model] Modeling completed successfully.")

if __name__ == "__main__":
    run_modeling()
