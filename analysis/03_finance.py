#!/usr/bin/env python3
"""
03_finance.py - Financial Mapping and Statistical Association Analysis

Responsibilities:
1. Map each tweet post to the first trading day on or after its date.
2. Compute financial metrics for each post:
   - same_day_return: close-to-close return on trading day T0.
   - next_day_return: close-to-close return on trading day T1 (T0 + 1).
   - fwd_vol_5d: 5-day forward return standard deviation [T1..T5].
   - intraday_range: (High - Low) / Low on day T0.
   - excess_next_day_return: next_day_return minus S&P 500 next_day_return.
3. Compute averages by emotion with 95% confidence intervals.
4. Perform Welch's t-test and Mann-Whitney U test (positive vs negative emotion groups).
5. Perform Kruskal-Wallis test across 12 emotion categories.
6. Compute correlation of bullish sentiment flag with financial metrics.
7. Analyze daily bullish share vs S&P 500 returns (same-day and next-day).
8. Save results to analysis/results.json and post-level metrics to analysis/post_finance_metrics.csv.
"""

import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats

# Set random seed
np.random.seed(42)

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis")
PRICE_DIR = os.path.join(WORKSPACE_DIR, "dataset", "price")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")
OUTPUT_CSV = os.path.join(ANALYSIS_DIR, "post_finance_metrics.csv")
RESULTS_PATH = os.path.join(ANALYSIS_DIR, "results.json")

TICKER_PRICE_MAP = {
    "FB": "FB.csv",
    "BRK.B": "BRK-B.csv"
}

METRIC_NAMES = [
    "same_day_return",
    "next_day_return",
    "excess_next_day_return",
    "fwd_vol_5d",
    "intraday_range"
]

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[03_finance] Results saved to {RESULTS_PATH}")

def compute_mean_ci(series, confidence=0.95):
    clean = series.dropna()
    n = len(clean)
    if n < 2:
        return {"n": n, "mean": None, "std": None, "ci_lower": None, "ci_upper": None}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    se = std / np.sqrt(n)
    t_val = float(stats.t.ppf((1 + confidence) / 2.0, df=n - 1))
    ci_lower = mean - t_val * se
    ci_upper = mean + t_val * se
    return {
        "n": int(n),
        "mean": round(mean, 6),
        "std": round(std, 6),
        "ci_lower": round(float(ci_lower), 6),
        "ci_upper": round(float(ci_upper), 6)
    }

def run_finance_analysis():
    print("[03_finance] Starting financial mapping and statistical analysis...")
    df_tweets = pd.read_csv(PREPARED_CSV)
    df_tweets["date"] = pd.to_datetime(df_tweets["date"])
    
    # 1. Load S&P 500 benchmark
    sp_path = os.path.join(PRICE_DIR, "^GSPC.csv")
    sp_df = pd.read_csv(sp_path)
    sp_df["Date"] = pd.to_datetime(sp_df["Date"])
    sp_df = sp_df.sort_values("Date").reset_index(drop=True)
    sp_df["sp_daily_return"] = sp_df["Adj Close"].pct_change()
    sp_df["sp_next_day_return"] = sp_df["sp_daily_return"].shift(-1)
    sp_lookup = sp_df.set_index("Date").to_dict("index")
    
    # 2. Load stock price datasets and compute pre-calculated metrics per ticker
    unique_tickers = df_tweets["ticker"].unique()
    price_dict = {}
    
    for t in unique_tickers:
        fname = TICKER_PRICE_MAP.get(t, f"{t}.csv")
        p_path = os.path.join(PRICE_DIR, fname)
        if not os.path.exists(p_path):
            print(f"[03_finance] WARNING: price file missing for {t}")
            continue
        pdf = pd.read_csv(p_path)
        pdf["Date"] = pd.to_datetime(pdf["Date"])
        pdf = pdf.sort_values("Date").reset_index(drop=True)
        pdf["daily_return"] = pdf["Adj Close"].pct_change()
        pdf["next_day_return"] = pdf["daily_return"].shift(-1)
        pdf["intraday_range"] = (pdf["High"] - pdf["Low"]) / pdf["Low"]
        
        # 5-day forward return standard deviation
        fwd_matrix = pd.DataFrame({f"ret_{k}": pdf["daily_return"].shift(-k) for k in range(1, 6)})
        pdf["fwd_vol_5d"] = fwd_matrix.std(axis=1)
        price_dict[t] = pdf
        
    print(f"[03_finance] Loaded price history for {len(price_dict)} tickers.")
    
    # 3. Map each post to first trading day on or after date
    mapped_records = []
    for _, row in df_tweets.iterrows():
        t = row["ticker"]
        d = row["date"]
        pdf = price_dict[t]
        
        # Filter trading days on or after post date
        future_days = pdf[pdf["Date"] >= d]
        if len(future_days) == 0:
            continue
        
        t0_row = future_days.iloc[0]
        t0_date = t0_row["Date"]
        
        # Retrieve SP500 metrics for same day T0
        sp_day = sp_lookup.get(t0_date, {})
        sp_next_ret = sp_day.get("sp_next_day_return", np.nan)
        
        stock_next_ret = t0_row["next_day_return"]
        excess_next_ret = (stock_next_ret - sp_next_ret) if (pd.notnull(stock_next_ret) and pd.notnull(sp_next_ret)) else np.nan
        
        mapped_records.append({
            "id": row["id"],
            "trading_day_t0": t0_date.strftime("%Y-%m-%d"),
            "same_day_return": t0_row["daily_return"],
            "next_day_return": stock_next_ret,
            "excess_next_day_return": excess_next_ret,
            "fwd_vol_5d": t0_row["fwd_vol_5d"],
            "intraday_range": t0_row["intraday_range"]
        })
        
    df_mapped = pd.DataFrame(mapped_records)
    df_merged = df_tweets.merge(df_mapped, on="id")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(f"[03_finance] Mapped {len(df_merged)} posts. Saved to {OUTPUT_CSV}")
    
    # 4. Average by Emotion with 95% Confidence Interval
    all_emotions = sorted(df_merged["emo_label"].unique().tolist())
    emotion_ci_table = {}
    for emo in all_emotions:
        emo_sub = df_merged[df_merged["emo_label"] == emo]
        emotion_ci_table[emo] = {
            metric: compute_mean_ci(emo_sub[metric])
            for metric in METRIC_NAMES
        }
        
    # Group averages (Positive, Negative, Neutral) with 95% CI
    group_ci_table = {}
    for grp in ["positive", "negative", "neutral"]:
        grp_sub = df_merged[df_merged["emotion_group"] == grp]
        group_ci_table[grp] = {
            metric: compute_mean_ci(grp_sub[metric])
            for metric in METRIC_NAMES
        }
        
    # 5. Welch's t-test and Mann-Whitney U test (Positive vs Negative)
    pos_sub = df_merged[df_merged["emotion_group"] == "positive"]
    neg_sub = df_merged[df_merged["emotion_group"] == "negative"]
    
    pos_vs_neg_tests = {}
    for metric in METRIC_NAMES:
        pos_vals = pos_sub[metric].dropna()
        neg_vals = neg_sub[metric].dropna()
        
        tt = stats.ttest_ind(pos_vals, neg_vals, equal_var=False)
        mw = stats.mannwhitneyu(pos_vals, neg_vals, alternative="two-sided")
        
        pos_vs_neg_tests[metric] = {
            "pos_mean": round(float(pos_vals.mean()), 6),
            "neg_mean": round(float(neg_vals.mean()), 6),
            "difference (pos - neg)": round(float(pos_vals.mean() - neg_vals.mean()), 6),
            "welch_t_stat": round(float(tt.statistic), 4),
            "welch_p_val": float(f"{tt.pvalue:.4e}"),
            "mann_whitney_u_stat": round(float(mw.statistic), 1),
            "mann_whitney_p_val": float(f"{mw.pvalue:.4e}"),
            "is_significant_05": bool(tt.pvalue < 0.05)
        }
        
    # 6. Kruskal-Wallis across 12 Emotions
    kruskal_tests = {}
    for metric in METRIC_NAMES:
        groups = [df_merged[df_merged["emo_label"] == emo][metric].dropna() for emo in all_emotions]
        kw = stats.kruskal(*groups)
        kruskal_tests[metric] = {
            "h_statistic": round(float(kw.statistic), 4),
            "p_value": float(f"{kw.pvalue:.4e}"),
            "is_significant_05": bool(kw.pvalue < 0.05)
        }
        
    # 7. Correlation of Bullish Flag with Returns & Volatility
    correlations_senti = {}
    for metric in METRIC_NAMES:
        valid_df = df_merged.dropna(subset=["senti_encoded", metric])
        r_p, p_p = stats.pearsonr(valid_df["senti_encoded"], valid_df[metric])
        r_s, p_s = stats.spearmanr(valid_df["senti_encoded"], valid_df[metric])
        correlations_senti[metric] = {
            "n_obs": int(len(valid_df)),
            "pearson_r": round(float(r_p), 5),
            "pearson_p_val": float(f"{p_p:.4e}"),
            "spearman_rho": round(float(r_s), 5),
            "spearman_p_val": float(f"{p_s:.4e}")
        }
        
    # 8. Daily Bullish Share vs S&P 500 Return
    daily_tweets = df_merged.groupby("trading_day_t0").agg(
        total_posts=("senti_encoded", "count"),
        bullish_posts=("senti_encoded", "sum")
    ).reset_index()
    daily_tweets["trading_day_t0"] = pd.to_datetime(daily_tweets["trading_day_t0"])
    daily_tweets["bullish_share"] = daily_tweets["bullish_posts"] / daily_tweets["total_posts"]
    
    sp_daily = sp_df[["Date", "Adj Close", "sp_daily_return", "sp_next_day_return"]].copy()
    daily_merged = daily_tweets.merge(sp_daily, left_on="trading_day_t0", right_on="Date", how="inner")
    
    # Same-day correlation
    valid_same = daily_merged.dropna(subset=["bullish_share", "sp_daily_return"])
    r_same_p, p_same_p = stats.pearsonr(valid_same["bullish_share"], valid_same["sp_daily_return"])
    r_same_s, p_same_s = stats.spearmanr(valid_same["bullish_share"], valid_same["sp_daily_return"])
    
    # Next-day correlation
    valid_next = daily_merged.dropna(subset=["bullish_share", "sp_next_day_return"])
    r_next_p, p_next_p = stats.pearsonr(valid_next["bullish_share"], valid_next["sp_next_day_return"])
    r_next_s, p_next_s = stats.spearmanr(valid_next["bullish_share"], valid_next["sp_next_day_return"])
    
    # OLS regression of next-day SP return on daily bullish share
    slope, intercept, r_val, p_val, std_err = stats.linregress(valid_next["bullish_share"], valid_next["sp_next_day_return"])
    
    daily_sp_analysis = {
        "trading_days_analyzed": int(len(daily_merged)),
        "same_day_sp_return_correlation": {
            "pearson_r": round(float(r_same_p), 5),
            "pearson_p_val": float(f"{p_same_p:.4e}"),
            "spearman_rho": round(float(r_same_s), 5),
            "spearman_p_val": float(f"{p_same_s:.4e}")
        },
        "next_day_sp_return_correlation": {
            "pearson_r": round(float(r_next_p), 5),
            "pearson_p_val": float(f"{p_next_p:.4e}"),
            "spearman_rho": round(float(r_next_s), 5),
            "spearman_p_val": float(f"{p_next_s:.4e}")
        },
        "next_day_predictive_ols": {
            "slope_beta": round(float(slope), 5),
            "intercept_alpha": round(float(intercept), 5),
            "r_squared": round(float(r_val**2), 5),
            "p_value": float(f"{p_val:.4e}"),
            "std_err": round(float(std_err), 5)
        }
    }
    
    print("\n[03_finance] === KEY STATISTICAL RESULTS ===")
    print(f"Same-day return (Pos vs Neg): Welch t={pos_vs_neg_tests['same_day_return']['welch_t_stat']}, p={pos_vs_neg_tests['same_day_return']['welch_p_val']}")
    print(f"Next-day return (Pos vs Neg): Welch t={pos_vs_neg_tests['next_day_return']['welch_t_stat']}, p={pos_vs_neg_tests['next_day_return']['welch_p_val']}")
    print(f"5-day forward vol (Pos vs Neg): Welch t={pos_vs_neg_tests['fwd_vol_5d']['welch_t_stat']}, p={pos_vs_neg_tests['fwd_vol_5d']['welch_p_val']}")
    print(f"Daily Bullish Share vs SP500 Same-Day: r={r_same_p:.4f} (p={p_same_p:.4e})")
    print(f"Daily Bullish Share vs SP500 Next-Day: r={r_next_p:.4f} (p={p_next_p:.4e})")
    
    # Save to results.json
    results = load_results()
    results["step3_finance"] = {
        "mapping_summary": {
            "total_posts_mapped": int(len(df_merged)),
            "valid_next_day_returns": int(df_merged["next_day_return"].notnull().sum()),
            "valid_excess_returns": int(df_merged["excess_next_day_return"].notnull().sum()),
            "valid_fwd_vol_5d": int(df_merged["fwd_vol_5d"].notnull().sum())
        },
        "emotion_confidence_intervals": emotion_ci_table,
        "group_confidence_intervals": group_ci_table,
        "welch_and_mann_whitney_tests": pos_vs_neg_tests,
        "kruskal_wallis_tests": kruskal_tests,
        "sentiment_correlations": correlations_senti,
        "daily_bullish_share_vs_sp500": daily_sp_analysis
    }
    save_results(results)
    print("[03_finance] Finance analysis completed successfully.")

if __name__ == "__main__":
    run_finance_analysis()
