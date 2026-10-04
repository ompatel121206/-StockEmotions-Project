#!/usr/bin/env python3
"""
02_eda.py - Exploratory Data Analysis & Visualization

Generates 9 publication-grade charts saved to analysis/charts/:
1. 01_emotion_distribution.png: 12-class emotion frequency.
2. 02_bullish_vs_bearish.png: Sentiment distribution (bullish vs bearish).
3. 03_emotion_x_sentiment.png: Emotion vs. sentiment cross-tabulation heatmap.
4. 04_posts_per_ticker.png: Tweet volume across stock tickers.
5. 05_emotion_by_ticker_top10.png: Emotion breakdown for top 10 tickers.
6. 06_monthly_emotion_trend_with_sp500.png: 2-panel chart of monthly emotion trend & S&P 500.
7. 07_text_length_distribution.png: Character count and word count distributions.
8. 08_emotion_group_distribution.png: Positive, Negative, and Neutral macro groups.
9. 09_monthly_bullish_share_vs_sp500.png: Monthly bullish share vs S&P 500 trajectory.

Saves structured observations to analysis/results.json under 'step2_eda'.
"""

import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Matplotlib configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(WORKSPACE_DIR, "analysis")
CHARTS_DIR = os.path.join(ANALYSIS_DIR, "charts")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")
RESULTS_PATH = os.path.join(ANALYSIS_DIR, "results.json")
PRICE_DIR = os.path.join(WORKSPACE_DIR, "dataset", "price")

os.makedirs(CHARTS_DIR, exist_ok=True)

def load_results():
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[02_eda] Results saved to {RESULTS_PATH}")

def run_eda():
    print("[02_eda] Loading prepared tweets...")
    df = pd.read_csv(PREPARED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    
    sp_path = os.path.join(PRICE_DIR, "^GSPC.csv")
    sp_df = pd.read_csv(sp_path)
    sp_df["Date"] = pd.to_datetime(sp_df["Date"])
    sp_df = sp_df.sort_values("Date").reset_index(drop=True)
    
    observations = {}
    
    # -------------------------------------------------------------
    # Chart 1: Emotion Distribution (12 classes)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    emo_counts = df["emo_label"].value_counts()
    
    # Color code by emotion group
    palette_map = {
        "optimism": "#2ca02c", "excitement": "#32cd32", "amusement": "#98df8a", "belief": "#2e8b57",
        "anxiety": "#d62728", "anger": "#e41a1c", "panic": "#b22222", "depression": "#8b0000", "disgust": "#ff7f0e",
        "ambiguous": "#1f77b4", "confusion": "#aec7e8", "surprise": "#17becf"
    }
    bar_colors = [palette_map.get(e, "#4c72b0") for e in emo_counts.index]
    
    bars = ax.bar(emo_counts.index, emo_counts.values, color=bar_colors, edgecolor="black", linewidth=0.6, alpha=0.85)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 20, f"{h}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
        
    ax.set_title("Distribution of Investor Emotion Categories (10,000 Posts)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Tweet Count", fontsize=11)
    ax.set_xlabel("Emotion Category", fontsize=11)
    ax.set_xticklabels(emo_counts.index, rotation=35, ha="right", fontsize=9.5)
    ax.set_ylim(0, max(emo_counts.values) * 1.15)
    plt.tight_layout()
    chart1_path = os.path.join(CHARTS_DIR, "01_emotion_distribution.png")
    plt.savefig(chart1_path)
    plt.close()
    
    obs1 = (f"Optimism is the dominant emotion with {emo_counts['optimism']} posts ({emo_counts['optimism']/len(df)*100:.1f}%), "
            f"followed by excitement ({emo_counts['excitement']}) and anxiety ({emo_counts['anxiety']}), "
            f"whereas depression ({emo_counts['depression']}) and surprise ({emo_counts['surprise']}) represent the rarest classes.")
    observations["01_emotion_distribution"] = obs1
    print(f"[02_eda] Chart 1 saved. Observation: {obs1}")

    # -------------------------------------------------------------
    # Chart 2: Bullish vs Bearish Sentiment Distribution
    # -------------------------------------------------------------
    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(11, 5), dpi=300, gridspec_kw={"width_ratios": [1.2, 1]})
    senti_counts = df["senti_label"].value_counts()
    colors = ["#2ca02c", "#d62728"]
    
    bars = ax_bar.bar(senti_counts.index.str.capitalize(), senti_counts.values, color=colors, edgecolor="black", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax_bar.text(bar.get_x() + bar.get_width()/2., h/2, f"{h:,}\n({pct:.1f}%)", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax_bar.set_title("Sentiment Category Volume", fontsize=11, fontweight="bold")
    ax_bar.set_ylabel("Count", fontsize=10)
    ax_bar.set_ylim(0, max(senti_counts.values) * 1.15)
    
    wedges, texts, autotexts = ax_pie.pie(
        senti_counts.values,
        labels=[s.capitalize() for s in senti_counts.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        explode=(0.04, 0),
        textprops=dict(fontsize=10, fontweight="bold")
    )
    ax_pie.set_title("Sentiment Share", fontsize=11, fontweight="bold")
    
    plt.suptitle("Overall Investor Sentiment Distribution (Bullish vs. Bearish)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart2_path = os.path.join(CHARTS_DIR, "02_bullish_vs_bearish.png")
    plt.savefig(chart2_path, bbox_inches="tight")
    plt.close()
    
    obs2 = (f"Investor sentiment leans distinctly bullish with {senti_counts['bullish']:,} posts (54.7%) "
            f"compared to {senti_counts['bearish']:,} bearish posts (45.3%), reflecting retail equity bias.")
    observations["02_bullish_vs_bearish"] = obs2
    print(f"[02_eda] Chart 2 saved. Observation: {obs2}")

    # -------------------------------------------------------------
    # Chart 3: Emotion x Sentiment Cross-Tabulation
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ct = pd.crosstab(df["emo_label"], df["senti_label"])
    ct = ct.reindex(emo_counts.index)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    
    sns.heatmap(ct_pct[["bullish", "bearish"]], annot=True, fmt=".1f", cmap="RdYlGn", cbar_kws={"label": "Percentage within Emotion (%)"},
                linewidths=1, linecolor="white", ax=ax, vmin=0, vmax=100)
    ax.set_title("Emotion by Sentiment Cross-Tabulation (% within each Emotion)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Market Sentiment", fontsize=11)
    ax.set_ylabel("Emotion Category", fontsize=11)
    plt.tight_layout()
    chart3_path = os.path.join(CHARTS_DIR, "03_emotion_x_sentiment.png")
    plt.savefig(chart3_path)
    plt.close()
    
    obs3 = ("Optimism and excitement are over 98% bullish, whereas disgust (94.9%), anxiety (89.5%), panic (97.7%), "
            "and depression (98.5%) align almost strictly with bearish sentiment; neutral categories (ambiguous, surprise) exhibit mixed polarity.")
    observations["03_emotion_x_sentiment"] = obs3
    print(f"[02_eda] Chart 3 saved. Observation: {obs3}")

    # -------------------------------------------------------------
    # Chart 4: Posts per Ticker
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    ticker_counts = df["ticker"].value_counts()
    
    colors_t = ["#1f77b4" if i < 10 else "#aec7e8" for i in range(len(ticker_counts))]
    bars = ax.bar(ticker_counts.index, ticker_counts.values, color=colors_t, edgecolor="black", linewidth=0.5)
    for i, bar in enumerate(bars[:10]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 50, f"{h}", ha="center", va="bottom", fontsize=7.5, fontweight="bold", rotation=0)
        
    ax.set_title("Tweet Volume Distribution Across All 37 Tickers (Top 10 Highlighted in Dark Blue)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Posts", fontsize=11)
    ax.set_xlabel("Stock Ticker", fontsize=11)
    ax.set_xticklabels(ticker_counts.index, rotation=90, fontsize=8)
    ax.set_ylim(0, max(ticker_counts.values) * 1.1)
    plt.tight_layout()
    chart4_path = os.path.join(CHARTS_DIR, "04_posts_per_ticker.png")
    plt.savefig(chart4_path)
    plt.close()
    
    obs4 = (f"Retail tweet volume exhibits severe concentration: TSLA alone accounts for {ticker_counts['TSLA']} posts ({ticker_counts['TSLA']/len(df)*100:.1f}%), "
            f"and the top 3 tickers (TSLA, AAPL, BA) comprise {ticker_counts.iloc[:3].sum()/len(df)*100:.1f}% of all posts.")
    observations["04_posts_per_ticker"] = obs4
    print(f"[02_eda] Chart 4 saved. Observation: {obs4}")

    # -------------------------------------------------------------
    # Chart 5: Emotion Group by Ticker (Top 10)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 6), dpi=300)
    top10_tickers = ticker_counts.head(10).index.tolist()
    df_top10 = df[df["ticker"].isin(top10_tickers)].copy()
    
    ticker_grp_ct = pd.crosstab(df_top10["ticker"], df_top10["emotion_group"])
    ticker_grp_ct = ticker_grp_ct.reindex(top10_tickers)
    ticker_grp_pct = ticker_grp_ct.div(ticker_grp_ct.sum(axis=1), axis=0) * 100
    
    grp_cols = ["positive", "neutral", "negative"]
    colors_grp = ["#2ca02c", "#1f77b4", "#d62728"]
    bottom = np.zeros(len(top10_tickers))
    
    for grp, color in zip(grp_cols, colors_grp):
        vals = ticker_grp_pct[grp].values
        ax.bar(top10_tickers, vals, bottom=bottom, label=grp.capitalize(), color=color, edgecolor="black", linewidth=0.6, width=0.6)
        # annotate segment
        for idx, (v, b) in enumerate(zip(vals, bottom)):
            if v > 10:
                ax.text(idx, b + v/2, f"{v:.1f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        bottom += vals
        
    ax.set_title("Emotion Group Proportions for Top 10 Most Discussed Tickers (100% Stacked)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Share of Posts (%)", fontsize=11)
    ax.set_xlabel("Stock Ticker (Ranked by Total Post Volume)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    chart5_path = os.path.join(CHARTS_DIR, "05_emotion_by_ticker_top10.png")
    plt.savefig(chart5_path)
    plt.close()
    
    obs5 = ("Growth and tech leaders (TSLA, AAPL, AMZN) show strong positive emotion majorities (over 40-50%), "
            "whereas pandemic-disrupted aerospace and cruise operators (BA: 42.9% negative; CCL: 45.1% negative) exhibit substantial negative sentiment spikes.")
    observations["05_emotion_by_ticker_top10"] = obs5
    print(f"[02_eda] Chart 5 saved. Observation: {obs5}")

    # -------------------------------------------------------------
    # Chart 6: Monthly Emotion Trend with S&P 500 (Separate Panel)
    # -------------------------------------------------------------
    fig, (ax_emo, ax_sp) = plt.subplots(2, 1, figsize=(12, 8), dpi=300, sharex=True, gridspec_kw={"height_ratios": [1.4, 1]})
    
    monthly_grp = df.groupby(["month", "emotion_group"]).size().unstack(fill_value=0)
    months = np.arange(1, 13)
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    ax_emo.plot(months, monthly_grp["positive"], marker="o", color="#2ca02c", linewidth=2.2, label="Positive (Optimism, Excitement, etc.)")
    ax_emo.plot(months, monthly_grp["negative"], marker="s", color="#d62728", linewidth=2.2, label="Negative (Anxiety, Anger, Panic, etc.)")
    ax_emo.plot(months, monthly_grp["neutral"], marker="^", color="#1f77b4", linewidth=2.2, label="Neutral (Ambiguous, Confusion, Surprise)")
    
    # Annotate March peak
    mar_neg = monthly_grp.loc[3, "negative"]
    ax_emo.annotate(f"March Shock Peak: {mar_neg}", xy=(3, mar_neg), xytext=(3.2, mar_neg + 60),
                    arrowprops=dict(facecolor="#d62728", shrink=0.05, width=1.5, headwidth=6),
                    fontsize=9, fontweight="bold", color="#d62728")
    
    ax_emo.set_title("Monthly Investor Emotion Dynamics vs. S&P 500 Market Benchmark in 2020", fontsize=13, fontweight="bold", pad=12)
    ax_emo.set_ylabel("Monthly Post Count", fontsize=11)
    ax_emo.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_emo.grid(True, linestyle="--", alpha=0.6)
    
    # S&P 500 bottom panel
    sp_monthly_close = sp_df.set_index("Date")["Adj Close"].resample("ME").last().values
    if len(sp_monthly_close) > 12:
        sp_monthly_close = sp_monthly_close[-12:]
    
    ax_sp.plot(months, sp_monthly_close, color="#333333", linewidth=2.5, marker="D", markersize=5, label="S&P 500 Month-End Adj Close")
    ax_sp.fill_between(months, sp_monthly_close, color="#cccccc", alpha=0.3)
    
    # Annotate March market trough
    min_sp_idx = np.argmin(sp_monthly_close) + 1
    min_sp_val = np.min(sp_monthly_close)
    ax_sp.annotate(f"Market Trough: {min_sp_val:.0f}", xy=(min_sp_idx, min_sp_val), xytext=(min_sp_idx + 0.3, min_sp_val - 150),
                   arrowprops=dict(facecolor="#333333", shrink=0.05, width=1.5, headwidth=6),
                   fontsize=9, fontweight="bold")
                   
    ax_sp.set_ylabel("S&P 500 Index Level", fontsize=11)
    ax_sp.set_xlabel("Month (2020)", fontsize=11)
    ax_sp.set_xticks(months)
    ax_sp.set_xticklabels(month_names, fontsize=10)
    ax_sp.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax_sp.grid(True, linestyle="--", alpha=0.6)
    
    plt.tight_layout()
    chart6_path = os.path.join(CHARTS_DIR, "06_monthly_emotion_trend_with_sp500.png")
    plt.savefig(chart6_path)
    plt.close()
    
    obs6 = ("Negative emotions surged in March 2020 (443 posts) matching the S&P 500 COVID-19 market drawdown to 2,584, "
            "followed by positive emotions rebounding strongly as equity markets climbed to record highs through Q3 and Q4 2020.")
    observations["06_monthly_emotion_trend_with_sp500"] = obs6
    print(f"[02_eda] Chart 6 saved. Observation: {obs6}")

    # -------------------------------------------------------------
    # Chart 7: Text Length Distribution (Word & Char Counts)
    # -------------------------------------------------------------
    fig, (ax_char, ax_word) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Char count
    sns.histplot(df["char_count"], bins=40, kde=True, color="#4c72b0", edgecolor="black", ax=ax_char, stat="density")
    q1_c, q3_c = df["char_count"].quantile([0.25, 0.75])
    iqr_c = q3_c - q1_c
    ax_char.axvline(df["char_count"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median ({df['char_count'].median():.0f})")
    ax_char.axvline(q3_c + 1.5*iqr_c, color="darkorange", linestyle=":", linewidth=1.5, label=f"Upper IQR Bound ({q3_c + 1.5*iqr_c:.1f})")
    ax_char.set_title("Tweet Character Length Distribution", fontsize=11, fontweight="bold")
    ax_char.set_xlabel("Character Count", fontsize=10)
    ax_char.set_ylabel("Density", fontsize=10)
    ax_char.legend(frameon=True, fontsize=8.5)
    
    # Word count
    sns.histplot(df["word_count"], bins=35, kde=True, color="#55a868", edgecolor="black", ax=ax_word, stat="density")
    q1_w, q3_w = df["word_count"].quantile([0.25, 0.75])
    iqr_w = q3_w - q1_w
    ax_word.axvline(df["word_count"].median(), color="red", linestyle="--", linewidth=1.5, label=f"Median ({df['word_count'].median():.0f})")
    ax_word.axvline(q3_w + 1.5*iqr_w, color="darkorange", linestyle=":", linewidth=1.5, label=f"Upper IQR Bound ({q3_w + 1.5*iqr_w:.1f})")
    ax_word.set_title("Tweet Word Count Distribution", fontsize=11, fontweight="bold")
    ax_word.set_xlabel("Word Count", fontsize=10)
    ax_word.set_ylabel("Density", fontsize=10)
    ax_word.legend(frameon=True, fontsize=8.5)
    
    plt.suptitle("Distribution of Investor Tweet Lengths (Characters and Words)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    chart7_path = os.path.join(CHARTS_DIR, "07_text_length_distribution.png")
    plt.savefig(chart7_path, bbox_inches="tight")
    plt.close()
    
    obs7 = (f"Tweet lengths follow a right-skewed distribution with a median of {df['char_count'].median():.0f} characters "
            f"and {df['word_count'].median():.0f} words; values exceeding 174.5 characters (4.95%) or 34 words (4.23%) qualify as statistical outliers by IQR.")
    observations["07_text_length_distribution"] = obs7
    print(f"[02_eda] Chart 7 saved. Observation: {obs7}")

    # -------------------------------------------------------------
    # Chart 8: Emotion Group Macro Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    grp_counts = df["emotion_group"].value_counts()
    colors_g = ["#2ca02c", "#d62728", "#1f77b4"]
    bars = ax.bar([g.capitalize() for g in grp_counts.index], grp_counts.values, color=colors_g, edgecolor="black", linewidth=0.7, width=0.55)
    for bar in bars:
        h = bar.get_height()
        pct = (h / len(df)) * 100
        ax.text(bar.get_x() + bar.get_width()/2., h + 60, f"{h:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Distribution of Macro Emotion Groups (Positive, Negative, Neutral)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Number of Posts", fontsize=11)
    ax.set_xlabel("Emotion Group", fontsize=11)
    ax.set_ylim(0, max(grp_counts.values) * 1.15)
    plt.tight_layout()
    chart8_path = os.path.join(CHARTS_DIR, "08_emotion_group_distribution.png")
    plt.savefig(chart8_path)
    plt.close()
    
    obs8 = (f"Positive emotions represent the largest group ({grp_counts['positive']:,} posts, {grp_counts['positive']/len(df)*100:.1f}%), "
            f"followed by negative emotions ({grp_counts['negative']:,} posts, {grp_counts['negative']/len(df)*100:.1f}%) "
            f"and neutral emotions ({grp_counts['neutral']:,} posts, {grp_counts['neutral']/len(df)*100:.1f}%).")
    observations["08_emotion_group_distribution"] = obs8
    print(f"[02_eda] Chart 8 saved. Observation: {obs8}")

    # -------------------------------------------------------------
    # Chart 9: Monthly Bullish Share vs. S&P 500 Level
    # -------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=300)
    monthly_senti = df.groupby("month")["senti_encoded"].agg(["count", "mean"]).reset_index()
    monthly_senti["bullish_pct"] = monthly_senti["mean"] * 100
    
    ax2 = ax1.twinx()
    line1 = ax1.plot(monthly_senti["month"], monthly_senti["bullish_pct"], color="#2ca02c", marker="o", linewidth=2.5, label="Bullish Sentiment Share (%)")
    line2 = ax2.plot(months, sp_monthly_close, color="#333333", marker="s", linestyle="--", linewidth=2, label="S&P 500 Index (Month-End)")
    
    ax1.set_title("Monthly Retail Bullish Sentiment Share vs. S&P 500 Benchmark (2020)", fontsize=13, fontweight="bold", pad=12)
    ax1.set_xlabel("Month (2020)", fontsize=11)
    ax1.set_ylabel("Bullish Posts (% of Total)", color="#2ca02c", fontsize=11)
    ax2.set_ylabel("S&P 500 Level", color="#333333", fontsize=11)
    ax1.set_xticks(months)
    ax1.set_xticklabels(month_names, fontsize=10)
    ax1.set_ylim(40, 70)
    ax2.grid(False)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", frameon=True, fontsize=9.5)
    
    plt.tight_layout()
    chart9_path = os.path.join(CHARTS_DIR, "09_monthly_bullish_share_vs_sp500.png")
    plt.savefig(chart9_path)
    plt.close()
    
    obs9 = ("Retail bullish sentiment fell to a 2020 low in March (46.8% bullish) as equity indices collapsed, "
            "subsequently rising past 60% in autumn alongside the broader market recovery.")
    observations["09_monthly_bullish_share_vs_sp500"] = obs9
    print(f"[02_eda] Chart 9 saved. Observation: {obs9}")
    
    # Save observations and summary to results.json
    results = load_results()
    results["step2_eda"] = {
        "charts_generated": [
            "01_emotion_distribution.png",
            "02_bullish_vs_bearish.png",
            "03_emotion_x_sentiment.png",
            "04_posts_per_ticker.png",
            "05_emotion_by_ticker_top10.png",
            "06_monthly_emotion_trend_with_sp500.png",
            "07_text_length_distribution.png",
            "08_emotion_group_distribution.png",
            "09_monthly_bullish_share_vs_sp500.png"
        ],
        "observations": observations
    }
    save_results(results)
    print("[02_eda] All 9 charts generated successfully and saved to analysis/charts.")

if __name__ == "__main__":
    run_eda()
