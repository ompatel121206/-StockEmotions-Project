#!/usr/bin/env python3
"""
Experiment No: 14
Aim: Analyze a real-world dataset for descriptive analytics using Python.
Course Outcome: CO-5 (Data Science Basics, Python Programming)
Author: Om Patel

Objectives:
1. Select a real-world dataset from GitHub / public repository.
2. Identify and apply appropriate textual (numerical) and graphical descriptive analytics.
3. Implement built-in Python functions for central tendency, dispersion, and distribution shape.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Matplotlib headless configuration
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

# ==============================================================================
# Step 1 & 2: Load the Real-World Dataset
# ==============================================================================
print("=" * 75)
print("EXPERIMENT 14: DESCRIPTIVE ANALYTICS USING PYTHON")
print("=" * 75)

# Locate dataset path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, "analysis", "post_finance_metrics.csv")
output_dir = os.path.join(base_dir, "practical", "output_charts")
os.makedirs(output_dir, exist_ok=True)

data = pd.read_csv(data_path)
print(f"\n[Step 2] Dataset Loaded Successfully!")
print(f"Dataset Dimensions: {data.shape[0]:,} Rows × {data.shape[1]} Columns")
print("Selected Numerical Columns: ['word_count', 'char_count', 'same_day_return', 'intraday_range', 'fwd_vol_5d']")
print("Selected Categorical Columns: ['emo_label', 'senti_label', 'ticker']")

# ==============================================================================
# Step 3: Textual Descriptive Analytics Using Built-in Functions
# ==============================================================================
print("\n" + "=" * 75)
print("STEP 3: TEXTUAL DESCRIPTIVE ANALYTICS (CENTRAL TENDENCY & DISPERSION)")
print("=" * 75)

num_cols = ['word_count', 'char_count', 'same_day_return', 'intraday_range', 'fwd_vol_5d']

# 1. Central Tendency
print("\n--- 1. MEASURES OF CENTRAL TENDENCY ---")
mean_vals = data[num_cols].mean()
median_vals = data[num_cols].median()
mode_vals = data[num_cols].mode().iloc[0]

central_df = pd.DataFrame({
    "Mean": mean_vals.round(4),
    "Median": median_vals.round(4),
    "Mode": mode_vals.round(4)
})
print(central_df.to_string())

# 2. Dispersion
print("\n--- 2. MEASURES OF DISPERSION ---")
min_vals = data[num_cols].min()
max_vals = data[num_cols].max()
range_vals = max_vals - min_vals
var_vals = data[num_cols].var()
std_vals = data[num_cols].std()

dispersion_df = pd.DataFrame({
    "Minimum": min_vals.round(4),
    "Maximum": max_vals.round(4),
    "Range": range_vals.round(4),
    "Variance": var_vals.round(6),
    "Std Deviation": std_vals.round(4)
})
print(dispersion_df.to_string())

# 3. Shape of Distribution
print("\n--- 3. MEASURES OF DISTRIBUTION SHAPE ---")
skew_vals = data[num_cols].skew()
kurt_vals = data[num_cols].kurt()

shape_df = pd.DataFrame({
    "Skewness": skew_vals.round(4),
    "Kurtosis": kurt_vals.round(4),
    "Shape Interpretation": [
        "Right-skewed (leptokurtic)" if s > 0 and k > 0 else "Left-skewed"
        for s, k in zip(skew_vals, kurt_vals)
    ]
})
print(shape_df.to_string())

# Summary using describe()
print("\n--- 4. PANDAS BUILT-IN 5-NUMBER STATISTICAL SUMMARY ---")
print(data[num_cols].describe().round(4).to_string())

# Categorical Analysis
print("\n--- 5. CATEGORICAL ATTRIBUTE ANALYSIS (MODE & FREQUENCY) ---")
print("Emotion Label Frequency:")
print(data['emo_label'].value_counts().to_string())
print("\nSentiment Label Frequency:")
print(data['senti_label'].value_counts().to_string())

# ==============================================================================
# Step 4: Graphical Descriptive Analytics
# ==============================================================================
print("\n" + "=" * 75)
print("STEP 4: GRAPHICAL DESCRIPTIVE ANALYTICS")
print("=" * 75)

# Chart 1: Histogram (Data Distribution)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
data['word_count'].hist(bins=25, color="#2563EB", edgecolor="black", alpha=0.75, ax=ax)
ax.set_title("Histogram: Distribution of Tweet Word Counts", fontsize=12, fontweight="bold")
ax.set_xlabel("Word Count", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.axvline(data['word_count'].mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean ({data['word_count'].mean():.2f})")
ax.axvline(data['word_count'].median(), color="green", linestyle="-", linewidth=1.5, label=f"Median ({data['word_count'].median():.2f})")
ax.legend()
plt.tight_layout()
chart1_path = os.path.join(output_dir, "01_histogram.png")
plt.savefig(chart1_path)
plt.close()
print(f"1. Histogram saved to: {chart1_path}")

# Chart 2: Box Plot (Spread & Outliers)
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
data.boxplot(column='intraday_range', ax=ax, patch_artist=True,
             boxprops=dict(facecolor="#93C5FD", color="#1E3A8A"),
             medianprops=dict(color="red", linewidth=2),
             whiskerprops=dict(color="#1E3A8A"),
             capprops=dict(color="#1E3A8A"),
             flierprops=dict(marker='o', color='orange', alpha=0.5))
ax.set_title("Box Plot: Intraday Stock Price Range & Outlier Detection", fontsize=12, fontweight="bold")
ax.set_ylabel("Intraday Range ((High - Low) / Low)", fontsize=10)
plt.tight_layout()
chart2_path = os.path.join(output_dir, "02_boxplot.png")
plt.savefig(chart2_path)
plt.close()
print(f"2. Box Plot saved to: {chart2_path}")

# Chart 3: Bar Chart (Categorical Comparisons)
fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
data['emo_label'].value_counts().plot(kind='bar', color="#10B981", edgecolor="black", ax=ax, width=0.6)
ax.set_title("Bar Chart: Frequency of 12 Emotion Classes", fontsize=12, fontweight="bold")
ax.set_xlabel("Emotion Category", fontsize=10)
ax.set_ylabel("Post Count", fontsize=10)
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
plt.tight_layout()
chart3_path = os.path.join(output_dir, "03_barchart.png")
plt.savefig(chart3_path)
plt.close()
print(f"3. Bar Chart saved to: {chart3_path}")

# Chart 4: Line Chart (Trend Over Time)
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
data['date'] = pd.to_datetime(data['date'])
daily_avg = data.groupby('date')['same_day_return'].mean()
ax.plot(daily_avg.index, daily_avg.values * 100, color="#8B5CF6", linewidth=1.5, label="Daily Average Return (%)")
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title("Line Chart: Daily Average Stock Return (%) Trend in 2020", fontsize=12, fontweight="bold")
ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Average Return (%)", fontsize=10)
ax.legend()
plt.tight_layout()
chart4_path = os.path.join(output_dir, "04_linechart.png")
plt.savefig(chart4_path)
plt.close()
print(f"4. Line Chart saved to: {chart4_path}")

print("\n" + "=" * 75)
print("EXPERIMENT 14 EXECUTION COMPLETE! ALL TABLES & CHARTS GENERATED.")
print("=" * 75)
