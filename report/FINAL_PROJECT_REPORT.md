# INVESTOR EMOTIONS AND STOCK MARKET BEHAVIOUR
## An Empirical Behavioral Finance Investigation, Machine Learning Benchmark, and Interactive Intelligence Platform

**Course/Degree**: Capstone Research Project in Financial Technology & Data Science  
**Academic Year**: 2026  
**Author**: Om Patel  
**Live Production Application**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)  
**Repository**: [StockEmotions-Project](https://github.com/ompatel/StockEmotions-Project)  
**Methodological Parameters**: Random Seed = 42 | Sample Size $N = 10,000$ Posts | Evaluation: Test Split Reported Once  

---

## EXECUTIVE SUMMARY

This research project conducts an exhaustive, mathematically rigorous empirical study investigating the relationship between fine-grained retail investor emotions expressed on social media and stock market behavior across 37 major equities and the S&P 500 benchmark throughout the turbulent calendar year 2020. 

Guided by four foundational pillars—**Real-World Case Studies**, **Empirical Research**, **Model Development**, and **Software Engineering**—this project establishes a critical distinction in financial economics: **contemporaneous market association versus predictive forward causality**.

Key project findings and deliverables:
1. **Linguistic & Emotional Structure**: Across 10,000 audited posts, optimism (16.24%) and excitement (13.86%) dominate the positive spectrum, while anxiety (13.66%) and disgust (12.79%) represent the largest negative segments. Overall sentiment exhibits retail equity bias, leaning 54.74% bullish versus 45.26% bearish.
2. **Empirical Finance & Hypothesis Testing**: Retail sentiment demonstrates an overwhelming contemporaneous association with same-day stock price returns (Welch’s $t = 13.52, p = 3.59 \times 10^{-41}$; Mann-Whitney $U = 9,839,196, p = 1.29 \times 10^{-43}$; Kruskal-Wallis across 12 emotions $H = 244.41, p = 4.14 \times 10^{-46}$). On positive emotion days, mean same-day return is $+1.028\%$ (95% CI: $[+0.878\%, +1.178\%]$), compared to $-0.898\%$ on negative emotion days (95% CI: $[-1.149\%, -0.646\%]$).
3. **The Efficiency of Forward Prices**: In strict contrast to same-day co-movement, out-of-sample forward predictive tests show **zero statistically significant predictive return alpha** (Welch’s $t = -0.080, p = 0.936$; Kruskal-Wallis $H = 15.07, p = 0.179$). Average next-day returns following positive posts ($+0.233\%$) and negative posts ($+0.244\%$) are statistically indistinguishable.
4. **Machine Learning Model Development**: Using TF-IDF representation (10,000 features, $n$-grams 1–2), we benchmarked Majority, Multinomial Naive Bayes, Logistic Regression, and Linear SVM models tuned exclusively on validation data. For 12-class emotion classification, Linear SVM achieved the highest test performance (Accuracy: 37.70%, Macro-F1: 0.3298), dramatically beating the 16.30% majority baseline. For binary sentiment, Logistic Regression attained 77.10% test accuracy (Macro-F1: 0.7676).
5. **Directional Market Forecasting**: In a chronological out-of-sample experiment ($N = 1,992$ test observations), classifiers trained on emotion features failed to outperform the majority baseline (Logistic Regression: 51.00% vs. Baseline: 52.56%), empirically validating the Weak-Form Efficient Market Hypothesis (EMH).
6. **Production Application**: A full-stack, cloud-deployed Streamlit application featuring live inference, ticker intelligence, case studies, and research matrices is live at [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app).

---

## 1. INTRODUCTION & RESEARCH MOTIVATION

### 1.1 Background
The classical paradigm of financial economics, anchored by Eugene Fama’s Efficient Market Hypothesis (EMH), posits that financial asset prices fully reflect all available information and that price changes follow a random walk driven solely by unanticipated fundamental news. However, the modern market microstructure has been fundamentally altered by the democratization of trading platforms and the exponential growth of investor social media communities (e.g., Twitter/X, StockTwits, Reddit).

Retail market participants frequently trade based on emotional heuristics, speculative momentum, and cognitive biases rather than discounted cash flow valuations. The calendar year 2020 provides an unparalleled natural experiment to study these dynamics, encapsulating the catastrophic COVID-19 liquidity shock, unprecedented monetary stimulus, and a retail-driven speculative technology rally.

### 1.2 Core Research Objectives
This project addresses three primary research inquiries:
1. **Emotional Representation**: Can natural language processing (NLP) models accurately identify nuanced human emotional states (12 fine-grained classes) and sentiment polarities from short-form financial texts containing financial slang, ticker cashtags, and emojis?
2. **Econometric Association**: To what extent do investor emotional states correlate with contemporaneous stock returns, forward returns, forward volatility, and intraday trading range?
3. **Forecasting Feasibility**: Can social media emotional states generate statistically significant out-of-sample predictive alpha for next-day price direction, or do they merely act as a reactive mirror of current market reality?

---

## 2. DATA ARCHITECTURE & PREPROCESSING

### 2.1 Verified Dataset Provenance
The project utilizes two verified, uncorrupted primary data repositories covering the full 2020 annual window (`2020-01-01` to `2020-12-31`):
1. **Tweet Corpus (`dataset/tweet/`)**:
   - `train_stockemo.csv`: 8,000 annotated posts
   - `val_stockemo.csv`: 1,000 annotated posts
   - `test_stockemo.csv`: 1,000 annotated posts
   - Total volume: Exactly 10,000 posts across 37 equities.
   - Core fields: `id`, `date`, `ticker`, `emo_label`, `senti_label`, `original`, `processed`.
2. **Market Price Series (`dataset/price/`)**:
   - 41 CSV files containing daily Open, High, Low, Close, Adjusted Close, and Volume for each ticker plus the S&P 500 benchmark (`^GSPC.csv`).
   - Specific entity mapping: `FB` $\rightarrow$ `FB.csv`; `BRK.B` $\rightarrow$ `BRK-B.csv`.

### 2.2 Data Integrity & Quality Audit
As established in `analysis/01_prepare.py` and recorded in `analysis/results.json`:
- **Missing Values**: Zero missing cells across all 7 fields ($0/10,000$).
- **Duplicate Records**: Zero ID duplicates ($0$), zero original text duplicates ($0$), and zero processed text duplicates ($0$).
- **Text Length IQR Outlier Detection**:
  - Character count: Median = $66.0$, Mean = $79.25$, $Q_1 = 47.0$, $Q_3 = 98.0$, $\text{IQR} = 51.0$. Upper bound ($Q_3 + 1.5 \times \text{IQR}$) = $174.5$ characters. Outlier count: $495$ posts ($4.95\%$).
  - Word count: Median = $13.0$, Mean = $15.65$, $Q_1 = 9.0$, $Q_3 = 19.0$, $\text{IQR} = 10.0$. Upper bound = $34.0$ words. Outlier count: $423$ posts ($4.23\%$).
- **Market Return Volatility Outliers**:
  Across 9,375 ticker trading day observations in 2020, $185$ daily returns ($1.97\%$) exceeded $10\%$ in absolute magnitude ($|r| > 0.10$). The maximum single-day gain observed was Carnival ($CCL$) on November 9, 2020 ($+39.29\%$), while the maximum single-day decline was Carnival on April 1, 2020 ($-33.18\%$).

### 2.3 Feature Engineering & Group Encoding
Each post was enriched with engineered linguistic and macro-financial variables:
- **Macro Emotion Groups**:
  - **Positive** ($N = 4,736, 47.36\%$): `optimism` (1,624), `excitement` (1,386), `amusement` (818), `belief` (908).
  - **Negative** ($N = 3,540, 35.40\%$): `anxiety` (1,366), `anger` (386), `panic` (304), `depression` (205), `disgust` (1,279).
  - **Neutral** ($N = 1,724, 17.24\%$): `ambiguous` (871), `confusion` (609), `surprise` (244).
- **Emoji Detection**: Verified that $100.0\%$ ($10,000/10,000$) of posts contain expressive emojis (either unicode glyphs or tokenized descriptors).

---

## 3. EXPLORATORY DATA ANALYSIS & VISUAL FINDINGS

Nine publication-grade visualization figures were generated by `analysis/02_eda.py` and saved to `analysis/charts/`.

### Summary of Empirical Observations

| Chart File | Title | Primary Empirical Observation |
| :--- | :--- | :--- |
| `01_emotion_distribution.png` | 12-Class Emotion Distribution | Optimism is the dominant emotion ($1,624$ posts, $16.2\%$), followed by excitement ($1,386$) and anxiety ($1,366$), whereas depression ($205$) and surprise ($244$) represent the rarest classes. |
| `02_bullish_vs_bearish.png` | Sentiment Distribution | Investor sentiment leans distinctly bullish with $5,474$ posts ($54.7\%$) compared to $4,526$ bearish posts ($45.3\%$), reflecting retail equity bias. |
| `03_emotion_x_sentiment.png` | Emotion $\times$ Sentiment Heatmap | Optimism and excitement are over $98\%$ bullish, whereas disgust ($94.9\%$), anxiety ($89.5\%$), panic ($97.7\%$), and depression ($98.5\%$) align almost strictly with bearish sentiment; neutral categories exhibit mixed polarity. |
| `04_posts_per_ticker.png` | Tweet Volume per Ticker | Retail tweet volume exhibits severe concentration: TSLA alone accounts for $4,341$ posts ($43.4\%$), and the top 3 tickers (TSLA, AAPL, BA) comprise $69.8\%$ of all posts. |
| `05_emotion_by_ticker_top10.png` | Emotion Breakdown (Top 10) | Growth/tech leaders (TSLA, AAPL, AMZN) show strong positive emotion majorities ($>40\text{--}50\%$), whereas pandemic-disrupted cyclicals (BA: $42.9\%$ negative; CCL: $45.1\%$ negative) exhibit elevated negative sentiment. |
| `06_monthly_emotion_trend_with_sp500.png` | Monthly Emotion Trend vs. S&P 500 | Negative emotions surged in March 2020 ($443$ posts) matching the S&P 500 COVID-19 market drawdown to $2,584$, followed by positive emotions rebounding strongly as equity markets climbed to record highs through Q3/Q4. |
| `07_text_length_distribution.png` | Character & Word Lengths | Tweet lengths follow a right-skewed distribution with a median of $66$ characters and $13$ words; values exceeding $174.5$ characters ($4.95\%$) or $34$ words ($4.23\%$) qualify as statistical outliers by IQR. |
| `08_emotion_group_distribution.png` | Macro Emotion Groups | Positive emotions represent the largest group ($4,736$ posts, $47.4\%$), followed by negative emotions ($3,540$ posts, $35.4\%$) and neutral emotions ($1,724$ posts, $17.2\%$). |
| `09_monthly_bullish_share_vs_sp500.png` | Monthly Bullish Share vs. S&P 500 | Retail bullish sentiment fell to a 2020 low in March ($46.8\%$ bullish) as equity indices collapsed, subsequently rising past $60\%$ in autumn alongside the broader market recovery. |

---

## 4. FINANCIAL MAPPING & ECONOMETRIC ANALYSIS

### 4.1 Temporal Alignment Methodology
To prevent lookahead bias and handle non-trading calendar dates (weekends and federal market holidays), each tweet post was aligned to the immediate next valid trading session:
$$T_0 = \min \{ D \in \text{TradingDays}_{\text{ticker}} \mid D \ge D_{\text{post}} \}$$
Of the $10,000$ posts, all $10,000$ successfully mapped to a valid trading day $T_0$. Exactly $9,959$ posts possessed valid next-day price records $T_1$ (with 41 posts occurring on the final trading session of the year, December 31, 2020).

### 4.2 Financial Metric Definitions
1. **Same-Day Return ($R_{T_0}$)**:
   $$R_{T_0} = \frac{\text{Adj Close}_{T_0} - \text{Adj Close}_{T_{-1}}}{\text{Adj Close}_{T_{-1}}}$$
2. **Next-Day Return ($R_{T_1}$)**:
   $$R_{T_1} = \frac{\text{Adj Close}_{T_1} - \text{Adj Close}_{T_0}}{\text{Adj Close}_{T_0}}$$
3. **Excess Next-Day Return ($R^{\text{excess}}_{T_1}$)**:
   $$R^{\text{excess}}_{T_1} = R^{\text{ticker}}_{T_1} - R^{\text{SP500}}_{T_1}$$
4. **5-Day Forward Volatility ($\sigma^{\text{fwd}}_{5d}$)**:
   Sample standard deviation of daily returns across the forward window $[T_1, T_2, T_3, T_4, T_5]$:
   $$\sigma^{\text{fwd}}_{5d} = \sqrt{\frac{1}{4}\sum_{k=1}^5 (R_{T_k} - \bar{R}_{5d})^2}$$
5. **Intraday Trading Range**:
   $$\text{Range}_{T_0} = \frac{\text{High}_{T_0} - \text{Low}_{T_0}}{\text{Low}_{T_0}}$$

### 4.3 Group Confidence Intervals (95% CI)

The empirical distributions and 95% Student-$t$ confidence intervals across emotion groups are summarized below:

```
Financial Metric Comparison Across Macro Emotion Groups:
┌─────────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric                  │ Positive     │ Negative     │ Neutral      │
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Same-Day Return         │ +1.028%      │ -0.898%      │ +0.399%      │
│ 95% Confidence Interval │ [+0.88%, +1.18%]│ [-1.15%, -0.65%]│ [+0.11%, +0.69%]│
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Next-Day Return         │ +0.233%      │ +0.244%      │ +0.389%      │
│ 95% Confidence Interval │ [+0.11%, +0.36%]│ [+0.04%, +0.45%]│ [+0.14%, +0.63%]│
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Excess Next-Day Return  │ +0.243%      │ +0.347%      │ +0.346%      │
│ 95% Confidence Interval │ [+0.12%, +0.37%]│ [+0.15%, +0.54%]│ [+0.11%, +0.58%]│
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ 5-Day Forward Volatility│ 4.291%       │ 4.642%       │ 4.505%       │
│ 95% Confidence Interval │ [4.22%, 4.36%]│ [4.51%, 4.77%]│ [4.36%, 4.65%]│
├─────────────────────────┼──────────────┼──────────────┼──────────────┤
│ Intraday Price Range    │ 6.042%       │ 7.029%       │ 6.456%       │
│ 95% Confidence Interval │ [5.93%, 6.16%]│ [6.82%, 7.23%]│ [6.24%, 6.67%]│
└─────────────────────────┴──────────────┴──────────────┴──────────────┘
```

### 4.4 Hypothesis Testing: Welch’s t-Test & Mann-Whitney U Test
We test the null hypothesis $H_0: \mu_{\text{positive}} = \mu_{\text{negative}}$ against the two-sided alternative $H_1: \mu_{\text{positive}} \ne \mu_{\text{negative}}$:

1. **Same-Day Return ($R_{T_0}$)**:
   - Positive Mean: $+0.010277$, Negative Mean: $-0.008976$, Difference: $+0.019253$ ($+1.925\%$).
   - Welch’s $t = 13.5223$ ($p = 3.5873 \times 10^{-41}$).
   - Mann-Whitney $U = 9,839,196.0$ ($p = 1.2914 \times 10^{-43}$).
   - **Conclusion**: Null hypothesis overwhelmingly rejected ($p < 0.001$). Strong contemporaneous market co-movement.
2. **Next-Day Forward Return ($R_{T_1}$)**:
   - Positive Mean: $+0.002331$, Negative Mean: $+0.002439$, Difference: $-0.000108$ ($-0.011\%$).
   - Welch’s $t = -0.0799$ ($p = 0.93635$).
   - Mann-Whitney $U = 7,996,555.0$ ($p = 0.00294$).
   - **Conclusion**: Null hypothesis cannot be rejected for mean return difference ($p = 0.936$). There is **no forward return alpha**.
3. **Forward Volatility ($\sigma^{\text{fwd}}_{5d}$)**:
   - Positive Mean: $0.042908$, Negative Mean: $0.046421$, Difference: $-0.003513$ ($-0.351\%$).
   - Welch’s $t = -4.7359$ ($p = 2.2228 \times 10^{-6}$).
   - **Conclusion**: Negative emotions coincide with significantly higher subsequent 5-day return volatility.
4. **Intraday Price Range**:
   - Positive Mean: $0.060423$, Negative Mean: $0.070290$, Difference: $-0.009867$ ($-0.987\%$).
   - Welch’s $t = -8.5665$ ($p = 1.3030 \times 10^{-17}$).
   - **Conclusion**: Market turbulence and wide intraday swings trigger substantial retail anxiety and panic.

### 4.5 Non-Parametric ANOVA: Kruskal-Wallis Across 12 Emotions
To test whether medians differ across all 12 individual emotional states:
- **Same-Day Return**: $H = 244.4076, p = 4.1376 \times 10^{-46}$ (**Significant**)
- **Next-Day Return**: $H = 15.0729, p = 0.17918$ (**Not Significant**)
- **Excess Next-Day Return**: $H = 14.5511, p = 0.20398$ (**Not Significant**)
- **5-Day Forward Volatility**: $H = 197.7500, p = 2.1787 \times 10^{-36}$ (**Significant**)
- **Intraday Range**: $H = 285.7177, p = 8.8964 \times 10^{-55}$ (**Significant**)

### 4.6 Market-Level Sentiment: Daily Bullish Share vs. S&P 500
Aggregating tweet sentiment across 253 trading sessions in 2020:
- **Same-Day S&P 500 Return**: Pearson $r = +0.24008$ ($p = 0.000115$), Spearman $\rho = +0.21683$ ($p = 0.000514$). Statistically significant positive co-movement.
- **Next-Day S&P 500 Return**: Pearson $r = +0.03697$ ($p = 0.55907$), Spearman $\rho = +0.00722$ ($p = 0.90926$).
- **Predictive OLS Regression**:
  $$R^{\text{SP500}}_{t+1} = -0.00073 + 0.00267 \times S^{\text{bullish}}_t + \epsilon_t \quad (R^2 = 0.00137, p = 0.55907)$$
The slope $\beta$ is statistically indistinguishable from zero, and the coefficient of determination $R^2 \approx 0.14\%$ demonstrates negligible predictive utility.

---

## 5. REAL-WORLD CASE STUDIES BASED LEARNING

### Case Study 1: The March 2020 COVID-19 Liquidity Shock & Panic Peak
- **Historical Episode**: Between February 19 and March 23, 2020, the S&P 500 suffered a $34\%$ collapse, triggering four Level-1 market-wide circuit breakers.
- **Observed Retail Dynamics**: Negative emotions surged to an annual peak of $443$ posts in March ($+124\%$ above January levels), while retail bullish sentiment plummeted to its annual trough of $46.8\%$.
- **Behavioral Finance Takeaway**: Retail investors exhibited peak loss aversion and panic selling at the exact market bottom. The subsequent 50-day rally was the strongest in modern market history, demonstrating how retail emotional capitulation marks market turning points rather than forward trends.

### Case Study 2: The November 2020 "Vaccine Monday" Relief Rally
- **Historical Episode**: On Monday, November 9, 2020, Pfizer and BioNTech announced that their BNT162b2 mRNA vaccine demonstrated over $90\%$ efficacy, removing catastrophic downside tail risk for global commerce.
- **Observed Retail Dynamics**: Depressed cyclical assets experienced historic price spikes. Carnival Corporation ($CCL$) surged $+39.29\%$ in a single trading session (the maximum daily return in the entire 2020 dataset), while Boeing ($BA$) advanced $+13.71\%$.
- **Behavioral Finance Takeaway**: Prior to this date, $CCL$ and $BA$ had accumulated the highest negative emotion concentration in the dataset ($45.1\%$ and $42.9\%$ negative, respectively). Investor emotion pivoted instantaneously from `panic` and `anxiety` to `excitement` and `belief`, pricing in recovery expectations before actual travel revenue materialized.

### Case Study 3: The Retail Trading Explosion & Electric Vehicle Euphoria ($TSLA$)
- **Historical Episode**: Driven by commission-free trading apps (Robinhood) and stay-at-home lockdown conditions, retail equity volume doubled in 2020.
- **Observed Retail Dynamics**: Out of 37 stocks, Tesla ($TSLA$) accounted for $4,341$ of the $10,000$ posts ($43.41\%$), exhibiting a strong power-law distribution. $TSLA$ gained over $+743\%$ in 2020, supported by persistent retail optimism ($64.2\%$ bullish share).
- **Behavioral Finance Takeaway**: Retail sentiment created a self-reinforcing reflexivity loop. Retail call buying generated gamma squeezes on market makers, forcing institutional delta-hedging and driving momentum beyond traditional fundamental valuation metrics.

---

## 6. MACHINE LEARNING MODEL DEVELOPMENT & BENCHMARKS

All models were trained on the official training split ($N = 8,000$), tuned strictly on the validation split ($N = 1,000$), and evaluated once on the test split ($N = 1,000$). The text representation utilized TF-IDF with $10,000$ maximum features, sublinear term frequency scaling, and unigram/bigram tokenization.

### 6.1 Task 1: 12-Class Emotion Classification Performance (Test Split)

```
Test Split Performance Across 12 Emotion Classes:
┌───────────────────────┬────────────┬──────────┬─────────────┐
│ Model Architecture    │ Accuracy   │ Macro-F1 │ Weighted-F1 │
├───────────────────────┼────────────┼──────────┼─────────────┤
│ Majority Baseline     │ 16.30%     │ 0.0234   │ 0.0457      │
│ Multinomial NB (α=0.1)│ 31.50%     │ 0.2541   │ 0.2962      │
│ Logistic Reg (C=5.0)  │ 37.50%     │ 0.3150   │ 0.3602      │
│ Linear SVM (C=0.5)    │ 37.70%     │ 0.3298   │ 0.3657      │
└───────────────────────┴────────────┴──────────┴─────────────┘
```

**Analysis**: Linear SVM attained the highest classification power (37.70% accuracy, 0.3298 Macro-F1), more than doubling the majority baseline (16.30%). Fine-grained classification of 12 classes is inherently challenging due to subtle lexical overlap between adjacent emotional states (e.g., `optimism` vs. `excitement` vs. `belief`).

### 6.2 Task 2: Binary Sentiment Classification (Bullish vs. Bearish)

```
Test Split Performance on Binary Sentiment:
┌───────────────────────┬────────────┬──────────┬─────────────┐
│ Model Architecture    │ Accuracy   │ Macro-F1 │ Weighted-F1 │
├───────────────────────┼────────────┼──────────┼─────────────┤
│ Majority Baseline     │ 55.50%     │ 0.3569   │ 0.3962      │
│ Multinomial NB (α=0.5)│ 74.50%     │ 0.7412   │ 0.7447      │
│ Logistic Reg (C=1.0)  │ 77.10%     │ 0.7676   │ 0.7707      │
│ Linear SVM (C=0.1)    │ 77.00%     │ 0.7665   │ 0.7697      │
└───────────────────────┴────────────┴──────────┴─────────────┘
```

**Analysis**: Logistic Regression and Linear SVM achieve near-identical high performance ($\approx 77.1\%$ accuracy, $0.768$ Macro-F1), dramatically outperforming the majority baseline ($55.50\%$). Both models effectively learn financial lexicon indicators (e.g., "calls", "moon", "puts", "tanking", "short squeeze").

### 6.3 Task 3: Chronological Market Direction Prediction Out-of-Sample

To rigorously test whether social emotions predict next-day price direction (Up = 1 vs. Down = 0), we executed a strict chronological split on $9,959$ clean observations:
- **Training Set**: Earliest 80% ($N = 7,967$; dates `2020-01-01` to `2020-10-06`). Up proportion: $51.31\%$.
- **Test Set**: Final unseen 20% ($N = 1,992$; dates `2020-10-06` to `2020-12-30`). Up proportion: $52.56\%$.
- **Predictor Matrix**: One-hot emotion indicators, macro-emotion group, sentiment flag, and emoji presence.

```
Out-of-Sample Chronological Next-Day Direction Forecasting:
┌───────────────────────┬────────────┬──────────┬──────────────┐
│ Model Architecture    │ Test Acc   │ Macro-F1 │ Beats Base?  │
├───────────────────────┼────────────┼──────────┼──────────────┤
│ Majority Baseline     │ 52.56%     │ 0.3445   │ Benchmark    │
│ Bernoulli Naive Bayes │ 51.86%     │ 0.5170   │ False        │
│ Linear SVM (C=0.1)    │ 50.95%     │ 0.5088   │ False        │
│ Logistic Regression   │ 51.00%     │ 0.5089   │ False        │
└───────────────────────┴────────────┴──────────┴──────────────┘
```

**Critical Scientific Finding**:
The Majority Baseline achieves **52.56%** test accuracy. The machine learning models utilizing emotion features attain **50.95%–51.86%** test accuracy—**failing to beat the simple naive baseline**. This provides empirical confirmation that social media investor sentiment reflects contemporaneous market reactions rather than predictive information, consistent with the Weak-Form Efficient Market Hypothesis.

---

## 7. APPLICATION & SOFTWARE DEVELOPMENT

### 7.1 Architecture & Deployment
An interactive, responsive software application was designed in `application/app.py` and deployed live to **Streamlit Community Cloud**:
- **Live URL**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)
- **Framework**: Streamlit 1.65 + Plotly 7.1 + Scikit-Learn 1.9
- **Model Backend**: Serialized `joblib` artifacts located in `model/`:
  - `tfidf_vectorizer.joblib`
  - `emotion_model.joblib`
  - `sentiment_model.joblib`
  - `metadata.json`

### 7.2 Functional Modules
1. **Live Emotion & Sentiment Inference**: Accepts user-entered text or presets; performs real-time vectorization and outputs calibrated probability bars across all 12 emotion classes alongside bullish/bearish donut gauges.
2. **Ticker Intelligence Hub**: Allows selection of any of the 37 equities to view 2020 annual performance, total post count, bullish ratio, emotion distribution, and interactive daily price charts.
3. **Real-World Case Study Explorer**: Guided interactive walkthroughs of the March 2020 Crash, November Vaccine Day, and Tesla Retail Wave.
4. **Empirical Research Hub**: Displays interactive tables of Welch's $t$-tests, Mann-Whitney $U$ tests, Kruskal-Wallis tests, and 95% Confidence Intervals.
5. **Machine Learning Benchmarks**: Documents model evaluation matrices, confusion matrices, and chronological forecasting results.

---

## 8. CONCLUSIONS & ACADEMIC TAKEAWAYS

1. **Association vs. Causation**: Investor emotions on social media exhibit a massive, statistically undeniable **contemporaneous association** with market movements ($t = 13.52, p < 10^{-40}$), but possess **no forward predictive alpha** ($t = -0.080, p = 0.936$). Social media is an emotional barometer of what *has happened*, not an oracle of what *will happen*.
2. **Asymmetric Volatility Transmission**: While sentiment does not predict directional returns, negative emotions transmit significant forward volatility signals ($p < 10^{-5}$), confirming that retail panic clusters during periods of elevated structural market turbulence.
3. **Machine Learning Utility**: NLP classifiers achieve high precision in categorizing financial sentiment ($77.10\%$) and fine-grained emotions ($37.70\%$), providing valuable tools for real-time market sentiment monitoring and risk management.
4. **Market Efficiency Validation**: The failure of sentiment features to outperform the majority baseline out-of-sample ($51.00\%$ vs. $52.56\%$) reinforces Fama’s Weak-Form Efficient Market Hypothesis in modern, retail-participated equity markets.

---

## APPENDIX: PROJECT REPRODUCIBILITY GUIDE

```bash
# 1. Clone repository
git clone https://github.com/ompatel/StockEmotions-Project.git
cd StockEmotions-Project

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Reproduce empirical pipeline
python3 analysis/01_prepare.py    # Data preparation & IQR audit
python3 analysis/02_eda.py        # Generate 9 charts in analysis/charts/
python3 analysis/03_finance.py    # Statistical tests & CI calculations
python3 analysis/04_model.py      # Train & evaluate ML models
python3 model/train_and_export.py # Export model artifacts to model/

# 4. Launch interactive dashboard locally
streamlit run application/app.py
```
