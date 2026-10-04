#!/usr/bin/env python3
"""
01_prepare.py - Data Preparation and Preprocessing

Responsibilities:
1. Merge train, val, and test splits while retaining split indicators.
2. Check missing values and duplicates (id, original text, processed text).
3. Detect outliers:
   - Text length by IQR (both character count and word count).
   - Price daily returns with absolute return > 10%.
4. Encode labels:
   - emo_label: 12 emotions.
   - senti_label: bullish (1) vs bearish (0).
5. Create features:
   - word_count
   - char_count
   - emoji_present
   - month
   - emotion_group:
       positive = optimism, excitement, amusement, belief
       negative = anxiety, anger, panic, depression, disgust
       neutral  = ambiguous, confusion, surprise
6. Save prepared dataset and populate step1_prepare in analysis/results.json.
"""

import os
import json
import re
import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TWEET_DIR = os.path.join(WORKSPACE_DIR, "dataset", "tweet")
PRICE_DIR = os.path.join(WORKSPACE_DIR, "dataset", "price")
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis")
RESULTS_PATH = os.path.join(ANALYSIS_DIR, "results.json")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")

TICKER_PRICE_MAP = {
    "FB": "FB.csv",
    "BRK.B": "BRK-B.csv"
}

EMOTION_GROUPS = {
    "positive": ["optimism", "excitement", "amusement", "belief"],
    "negative": ["anxiety", "anger", "panic", "depression", "disgust"],
    "neutral": ["ambiguous", "confusion", "surprise"]
}

EMO_TO_GROUP = {}
for grp, emos in EMOTION_GROUPS.items():
    for emo in emos:
        EMO_TO_GROUP[emo] = grp

# Unicode regex for emoji detection
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U00002300-\U000023FF"  # misc technical (e.g. watch, alarm clock)
    "\U000024C2-\U0001F251"  # enclosed chars
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "]+", flags=re.UNICODE
)

def load_or_init_results():
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_results(results):
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[01_prepare] Results saved to {RESULTS_PATH}")

def run_preparation():
    print("[01_prepare] Starting data preparation...")
    
    # 1. Merge splits
    train_path = os.path.join(TWEET_DIR, "train_stockemo.csv")
    val_path = os.path.join(TWEET_DIR, "val_stockemo.csv")
    test_path = os.path.join(TWEET_DIR, "test_stockemo.csv")
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    
    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"
    
    df = pd.concat([df_train, df_val, df_test], ignore_index=True)
    total_rows = len(df)
    print(f"[01_prepare] Merged dataset: {total_rows} rows (train: {len(df_train)}, val: {len(df_val)}, test: {len(df_test)})")
    
    # 2. Check missing values
    missing_dict = {col: int(df[col].isnull().sum()) for col in df.columns}
    print(f"[01_prepare] Missing values: {missing_dict}")
    
    # 3. Check duplicates
    dup_id = int(df["id"].duplicated().sum())
    dup_orig = int(df["original"].duplicated().sum())
    dup_proc = int(df["processed"].duplicated().sum())
    print(f"[01_prepare] Duplicates - ID: {dup_id}, Original: {dup_orig}, Processed: {dup_proc}")
    
    # 4. Create Features
    df["word_count"] = df["original"].astype(str).str.split().str.len()
    df["char_count"] = df["original"].astype(str).str.len()
    
    # Emoji detection: either unicode emoji in original text OR [tag] bracketed emoji description in processed text
    has_unicode = df["original"].apply(lambda x: bool(EMOJI_PATTERN.search(str(x))))
    has_bracket = df["processed"].str.contains(r"\[[a-z0-9\s_-]+\]", case=False, regex=True)
    df["emoji_present"] = (has_unicode | has_bracket).astype(int)
    
    # Date parsing and month
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    
    # Emotion group
    df["emotion_group"] = df["emo_label"].map(EMO_TO_GROUP)
    
    # 5. Outliers
    # Text length outliers by IQR
    def calc_iqr_outliers(series):
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_mask = (series < lower) | (series > upper)
        count = int(outlier_mask.sum())
        pct = float((count / len(series)) * 100)
        return {
            "q1": round(q1, 4),
            "q3": round(q3, 4),
            "iqr": round(iqr, 4),
            "lower_bound": round(lower, 4),
            "upper_bound": round(upper, 4),
            "outlier_count": count,
            "outlier_percentage": round(pct, 2)
        }
    
    char_iqr = calc_iqr_outliers(df["char_count"])
    word_iqr = calc_iqr_outliers(df["word_count"])
    print(f"[01_prepare] Char count IQR outliers: {char_iqr['outlier_count']} ({char_iqr['outlier_percentage']}%)")
    print(f"[01_prepare] Word count IQR outliers: {word_iqr['outlier_count']} ({word_iqr['outlier_percentage']}%)")
    
    # Price daily returns above 10%
    tickers = sorted(df["ticker"].unique())
    daily_returns_list = []
    
    for t in tickers:
        fname = TICKER_PRICE_MAP.get(t, f"{t}.csv")
        p_path = os.path.join(PRICE_DIR, fname)
        if os.path.exists(p_path):
            pdf = pd.read_csv(p_path)
            pdf["Date"] = pd.to_datetime(pdf["Date"])
            pdf = pdf.sort_values("Date").reset_index(drop=True)
            pdf["daily_return"] = pdf["Adj Close"].pct_change()
            pdf["ticker"] = t
            valid = pdf.dropna(subset=["daily_return"])
            daily_returns_list.append(valid[["Date", "ticker", "Adj Close", "daily_return"]])
            
    # Also include S&P 500
    sp_path = os.path.join(PRICE_DIR, "^GSPC.csv")
    if os.path.exists(sp_path):
        sp_df = pd.read_csv(sp_path)
        sp_df["Date"] = pd.to_datetime(sp_df["Date"])
        sp_df = sp_df.sort_values("Date").reset_index(drop=True)
        sp_df["daily_return"] = sp_df["Adj Close"].pct_change()
        sp_df["ticker"] = "^GSPC"
        valid_sp = sp_df.dropna(subset=["daily_return"])
        daily_returns_list.append(valid_sp[["Date", "ticker", "Adj Close", "daily_return"]])
        
    all_returns_df = pd.concat(daily_returns_list, ignore_index=True)
    total_price_obs = len(all_returns_df)
    returns_gt_10pct = all_returns_df[all_returns_df["daily_return"].abs() > 0.10]
    outlier_returns_count = len(returns_gt_10pct)
    outlier_returns_pct = float((outlier_returns_count / total_price_obs) * 100)
    
    top_pos_returns = returns_gt_10pct.sort_values("daily_return", ascending=False).head(5)
    top_neg_returns = returns_gt_10pct.sort_values("daily_return", ascending=True).head(5)
    
    top_pos_list = [
        {"ticker": r["ticker"], "date": r["Date"].strftime("%Y-%m-%d"), "return": round(float(r["daily_return"]), 4)}
        for _, r in top_pos_returns.iterrows()
    ]
    top_neg_list = [
        {"ticker": r["ticker"], "date": r["Date"].strftime("%Y-%m-%d"), "return": round(float(r["daily_return"]), 4)}
        for _, r in top_neg_returns.iterrows()
    ]
    
    print(f"[01_prepare] Daily return outliers (|r| > 10%): {outlier_returns_count} / {total_price_obs} ({outlier_returns_pct:.2f}%)")
    
    # 6. Encode Labels
    # Standard alphabetical or ordered emotion encoding
    unique_emotions = sorted(df["emo_label"].unique().tolist())
    emo_to_id = {emo: i for i, emo in enumerate(unique_emotions)}
    id_to_emo = {i: emo for i, emo in enumerate(unique_emotions)}
    
    senti_to_id = {"bearish": 0, "bullish": 1}
    id_to_senti = {0: "bearish", 1: "bullish"}
    
    group_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    
    df["emo_encoded"] = df["emo_label"].map(emo_to_id)
    df["senti_encoded"] = df["senti_label"].map(senti_to_id)
    df["group_encoded"] = df["emotion_group"].map(group_to_id)
    
    # Save prepared dataset
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df.to_csv(PREPARED_CSV, index=False)
    print(f"[01_prepare] Saved prepared dataset to {PREPARED_CSV}")
    
    # Build results dictionary
    results = load_or_init_results()
    results["project"] = "Investor Emotions and Stock Market Behaviour"
    results["random_seed"] = 42
    
    results["step1_prepare"] = {
        "dataset_summary": {
            "total_records": total_rows,
            "train_records": int((df["split"] == "train").sum()),
            "val_records": int((df["split"] == "val").sum()),
            "test_records": int((df["split"] == "test").sum()),
            "tickers_count": int(df["ticker"].nunique()),
            "start_date": str(df["date"].min()),
            "end_date": str(df["date"].max())
        },
        "missing_values": missing_dict,
        "duplicates": {
            "id_duplicates": dup_id,
            "text_original_duplicates": dup_orig,
            "text_processed_duplicates": dup_proc
        },
        "outliers": {
            "char_count_iqr": char_iqr,
            "word_count_iqr": word_iqr,
            "daily_returns_above_10pct": {
                "total_trading_observations": total_price_obs,
                "outlier_count": outlier_returns_count,
                "outlier_percentage": round(outlier_returns_pct, 2),
                "top_positive_returns": top_pos_list,
                "top_negative_returns": top_neg_list
            }
        },
        "label_encoding": {
            "emo_label_to_id": emo_to_id,
            "senti_label_to_id": senti_to_id,
            "emotion_groups": EMOTION_GROUPS
        },
        "feature_summary": {
            "word_count": {
                "mean": round(float(df["word_count"].mean()), 2),
                "std": round(float(df["word_count"].std()), 2),
                "min": int(df["word_count"].min()),
                "max": int(df["word_count"].max()),
                "median": float(df["word_count"].median())
            },
            "char_count": {
                "mean": round(float(df["char_count"].mean()), 2),
                "std": round(float(df["char_count"].std()), 2),
                "min": int(df["char_count"].min()),
                "max": int(df["char_count"].max()),
                "median": float(df["char_count"].median())
            },
            "emoji_present_total": int(df["emoji_present"].sum()),
            "emoji_present_percentage": round(float(df["emoji_present"].mean() * 100), 2),
            "emotion_group_counts": {k: int(v) for k, v in df["emotion_group"].value_counts().items()},
            "senti_counts": {k: int(v) for k, v in df["senti_label"].value_counts().items()},
            "emotion_counts": {k: int(v) for k, v in df["emo_label"].value_counts().items()}
        }
    }
    
    save_results(results)
    print("[01_prepare] Data preparation completed successfully.")

if __name__ == "__main__":
    run_preparation()
