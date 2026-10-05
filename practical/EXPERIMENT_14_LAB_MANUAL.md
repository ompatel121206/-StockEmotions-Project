# EXPERIMENT NO: 14
## Descriptive Analytics on a Real-World Dataset Using Python

**Student Name**: Om Patel  
**Branch**: Information Technology  
**Semester**: 5th Semester  
**College**: L.D. College of Engineering, Ahmedabad  
**Aim**: Analyze a real-world dataset for descriptive analytics using Python.  
**Competency and Practical Skills**: Data Science Basics, Statistical Computing, Python Programming (`pandas`, `numpy`, `matplotlib`)  
**Relevant Course Outcome (CO)**: **CO-5**  

---

### Objectives:
1. To select a real-world dataset from a publicly available repository (GitHub / UCI / Kaggle).
2. To identify and apply appropriate textual (numerical) and graphical descriptive analytics techniques.
3. To implement descriptive analytics using built-in Python functions for analyzing central tendency, dispersion, and data distribution.

---

### 1. Selection of Real-World Dataset
- **Dataset Name**: StockEmotions & Financial Market Performance Dataset (2020)
- **Source**: Public Repository on GitHub (`https://github.com/ompatel121206/-StockEmotions-Project`)
- **Dataset Dimensions**: 10,000 Records $\times$ 22 Attributes
- **Attributes Analyzed**:
  - `word_count` (Continuous/Discrete): Number of words in financial social media posts.
  - `char_count` (Continuous/Discrete): Number of characters per post.
  - `same_day_return` (Continuous): Same-day stock price return on trading session $T_0$.
  - `intraday_range` (Continuous): Intraday price volatility spread: $(\text{High} - \text{Low}) / \text{Low}$.
  - `fwd_vol_5d` (Continuous): 5-day forward return standard deviation.
  - `emo_label` (Categorical): 12 granular emotional categories (optimism, anxiety, excitement, etc.).
  - `senti_label` (Categorical): Binary market sentiment (bullish vs. bearish).

---

### 2. Theory & Formulations

#### 2.1 Measures of Central Tendency
- **Mean ($\mu$ or $\bar{x}$)**: Arithmetic average:
  $$\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$
- **Median**: The middle value of a sorted distribution (50th percentile); robust against extreme outliers.
- **Mode**: The value that appears with the highest frequency in the dataset.

#### 2.2 Measures of Dispersion
- **Minimum & Maximum**: The extreme bounds of the observed data.
- **Range**:
  $$\text{Range} = \text{Max} - \text{Min}$$
- **Variance ($s^2$)**: Average of squared deviations from the mean:
  $$s^2 = \frac{1}{N-1} \sum_{i=1}^N (x_i - \bar{x})^2$$
- **Standard Deviation ($s$)**: Square root of the sample variance ($s = \sqrt{s^2}$), measured in original data units.

#### 2.3 Measures of Distribution Shape
- **Skewness**: Measure of distributional symmetry:
  $$\text{Skewness} = \frac{N}{(N-1)(N-2)} \sum_{i=1}^N \left(\frac{x_i - \bar{x}}{s}\right)^3$$
  - $\text{Skewness} = 0$: Perfectly symmetric distribution.
  - $\text{Skewness} > 0$: Positively skewed (right tail is longer).
  - $\text{Skewness} < 0$: Negatively skewed (left tail is longer).
- **Kurtosis**: Measure of tail thickness and peak sharpness (Fisher's definition):
  - $\text{Kurtosis} > 0$: Leptokurtic (heavy-tailed with extreme outlier vulnerability).
  - $\text{Kurtosis} = 0$: Mesokurtic (normal distribution).

---

### 3. Implementation Steps in Python (VS Code)

```python
# Step 1: Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Load the Dataset
data = pd.read_csv("analysis/post_finance_metrics.csv")
num_cols = ['word_count', 'char_count', 'same_day_return', 'intraday_range', 'fwd_vol_5d']

# Step 3: Textual Descriptive Analytics Using Built-in Functions
# Central Tendency
print("Mean:\n", data[num_cols].mean())
print("Median:\n", data[num_cols].median())
print("Mode:\n", data[num_cols].mode().iloc[0])

# Dispersion
print("Min:\n", data[num_cols].min())
print("Max:\n", data[num_cols].max())
print("Variance:\n", data[num_cols].var())
print("Standard Deviation:\n", data[num_cols].std())

# Distribution Shape
print("Skewness:\n", data[num_cols].skew())
print("Kurtosis:\n", data[num_cols].kurt())

# Step 4: Graphical Descriptive Analytics
# 1. Histogram
data['word_count'].hist(bins=25, color='#2563EB', edgecolor='black')
plt.title("Histogram of Tweet Word Counts")
plt.show()

# 2. Box Plot
data.boxplot(column='intraday_range')
plt.title("Box Plot of Intraday Price Range")
plt.show()

# 3. Bar Chart
data['emo_label'].value_counts().plot(kind='bar', color='#10B981')
plt.title("Bar Chart of Emotion Classes")
plt.show()

# 4. Line Chart
data['date'] = pd.to_datetime(data['date'])
data.groupby('date')['same_day_return'].mean().plot()
plt.title("Daily Return Trend in 2020")
plt.show()
```

---

### 4. Textual Descriptive Analytics Results (Verified Output)

#### Table 1: Measures of Central Tendency
| Attribute | Mean (`data.mean()`) | Median (`data.median()`) | Mode (`data.mode()`) |
| :--- | :---: | :---: | :---: |
| **`word_count`** | 15.2609 | 13.0000 | 7.0000 |
| **`char_count`** | 79.0921 | 66.0000 | 47.0000 |
| **`same_day_return`** | 0.0019 (+0.19%) | 0.0016 (+0.16%) | -0.0560 (-5.60%) |
| **`intraday_range`** | 0.0652 (6.52%) | 0.0483 (4.83%) | 0.0483 (4.83%) |
| **`fwd_vol_5d`** | 0.0449 (4.49%) | 0.0358 (3.58%) | 0.0608 (6.08%) |

#### Table 2: Measures of Dispersion
| Attribute | Minimum | Maximum | Range | Variance | Std Deviation |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`word_count`** | 4.0000 | 56.0000 | 52.0000 | 69.9268 | 8.3622 |
| **`char_count`** | 19.0000 | 256.0000 | 237.0000 | 1989.6484 | 44.6055 |
| **`same_day_return`** | -0.3318 (-33.2%) | 0.3929 (+39.3%) | 0.7247 (72.5%) | 0.004165 | 0.0645 (6.45%) |
| **`intraday_range`** | 0.0069 (0.69%) | 0.5306 (53.1%) | 0.5238 (52.4%) | 0.002640 | 0.0514 (5.14%) |
| **`fwd_vol_5d`** | 0.0004 (0.04%) | 0.2093 (20.9%) | 0.2089 (20.9%) | 0.001108 | 0.0333 (3.33%) |

#### Table 3: Distribution Shape (Skewness & Kurtosis)
| Attribute | Skewness (`data.skew()`) | Kurtosis (`data.kurt()`) | Shape Interpretation |
| :--- | :---: | :---: | :--- |
| **`word_count`** | +1.4882 | +2.1228 | Positively right-skewed; leptokurtic |
| **`char_count`** | +1.4597 | +1.9746 | Positively right-skewed; leptokurtic |
| **`same_day_return`** | -0.0373 | +2.8055 | Near-symmetric; heavy-tailed fat tails |
| **`intraday_range`** | +2.4623 | +11.1369 | Highly right-skewed; extreme outlier peaks |
| **`fwd_vol_5d`** | +1.5451 | +2.4753 | Positively skewed; clustered volatility |

#### Table 4: Categorical Frequency Analysis
- **Emotion Classes (`emo_label`)**: Optimism (1,624), Excitement (1,386), Anxiety (1,366), Disgust (1,279), Belief (908), Ambiguous (871), Amusement (818), Confusion (609), Anger (386), Panic (304), Surprise (244), Depression (205).
- **Sentiment Polarity (`senti_label`)**: Bullish = 5,474 (54.74%) vs. Bearish = 4,526 (45.26%).

---

### 5. Graphical Descriptive Analytics & Inferences

1. **Histogram (`01_histogram.png`)**:
   - *Observation*: Tweet word count exhibits a distinct right-skew with peak concentration between 7 and 15 words. The mean (15.26) lies to the right of the median (13.00), visually confirming positive skewness.
2. **Box Plot (`02_boxplot.png`)**:
   - *Observation*: The box plot of intraday price range clearly highlights the interquartile range ($Q_1 = 3.20\%$, $Q_3 = 8.37\%$) and flags extreme outlier sessions (extending up to $53.06\%$ range) during the March 2020 COVID shock.
3. **Bar Chart (`03_barchart.png`)**:
   - *Observation*: Compares the discrete frequencies of all 12 emotion classes, clearly demonstrating that optimism ($16.2\%$) and excitement ($13.9\%$) dominate retail social media discussion.
4. **Line Chart (`04_linechart.png`)**:
   - *Observation*: Tracks the chronological time series of average daily returns across 2020, revealing the massive volatility swings in March 2020 followed by progressive stabilization.

---

### 6. Conclusion
In this practical, descriptive analytics was implemented on a real-world dataset of 10,000 observations using Python's `pandas`, `numpy`, and `matplotlib` libraries:
1. **Central tendency** techniques revealed that financial social media posts are typically short (median: 13 words, 66 characters), reflecting the fast-paced nature of retail financial discourse.
2. **Measures of dispersion** demonstrated substantial financial market variance ($s = 6.45\%$ daily return standard deviation, spanning a total range of $72.47\%$).
3. **Distribution shape analytics** confirmed that financial risk metrics like intraday range ($\text{Kurtosis} = 11.14$) deviate significantly from normal distributions, exhibiting heavy tails and extreme risk clusters.
4. **Graphical analytics** effectively visualized patterns, skewness, and outliers that tabular values alone cannot easily communicate.

---

### 7. Viva-Voce Questions & Answers

**Q1: What is the difference between Mean and Median, and when should Median be preferred?**  
> **Answer**: The mean is the arithmetic average of all values, while the median is the 50th percentile (middle value). The median should be preferred when data has strong skewness or extreme outliers (e.g., tweet length or stock volume), as the mean is heavily distorted by extreme values, whereas the median remains robust.

**Q2: What does a positive skewness value indicate?**  
> **Answer**: A positive skewness ($\text{Skewness} > 0$) indicates that the distribution's tail on the right side is longer or fatter than the left side, meaning the majority of observations are concentrated below the mean, with a few high-value outliers stretching the right tail.

**Q3: What does high Kurtosis signify in financial data?**  
> **Answer**: High kurtosis (leptokurtosis, $\text{Kurtosis} > 0$) signifies heavy tails and a sharp peak compared to a normal distribution. In finance, this indicates "fat tail" risk—meaning extreme price crashes or spikes occur much more frequently than predicted by a standard Gaussian model.

**Q4: Which Python function generates a 5-number summary of a DataFrame?**  
> **Answer**: The `DataFrame.describe()` function automatically outputs the count, mean, standard deviation, minimum, 25th percentile ($Q_1$), 50th percentile (median), 75th percentile ($Q_3$), and maximum.

**Q5: Why is a Box Plot particularly useful in descriptive analytics?**  
> **Answer**: A box plot graphically displays the median, the interquartile range ($\text{IQR} = Q_3 - Q_1$), the minimum and maximum within $1.5 \times \text{IQR}$, and explicitly plots individual data points beyond the whiskers as statistical outliers.
