# INVESTOR EMOTIONS AND STOCK MARKET BEHAVIOUR
## College Project Presentation & Viva Defense Slide Deck

**Author**: Om Patel  
**Live Application**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)  
**Methodology**: Random Seed 42 | Sample Size $N = 10,000$ Posts | 37 Equities + S&P 500  

---

### SLIDE 1: Title & Project Overview

#### [Slide Content]
- **Project Title**: Investor Emotions and Stock Market Behaviour
- **Subtitle**: An Empirical Behavioral Finance Study, Machine Learning Benchmark, and Interactive Intelligence Platform
- **Presenter**: Om Patel
- **Four Core Project Pillars**:
  1. 🏛️ **Real-World Case Studies** (March Crash, Vaccine Rally, Tesla Momentum)
  2. 📊 **Empirical Research & Hypothesis Testing** (Econometric association vs. causation)
  3. 🤖 **Model Development** (TF-IDF + 12-class emotion & sentiment benchmarks)
  4. 💻 **Software Application** (Live deployed cloud dashboard)
- **Live Cloud Dashboard**: `https://stockemotions-project.streamlit.app`

#### [Speaker Script / What to Say]
> "Good morning, respected professors and evaluators. Today, I am presenting my project titled 'Investor Emotions and Stock Market Behaviour'. In this work, we investigate how retail investor emotions expressed on social media interact with financial equity prices during 2020. Our project is built upon four structured pillars: Real-World Case Studies, Empirical Econometric Research, Machine Learning Model Development, and an end-to-end Software Application deployed live on the cloud. Let us examine the motivation and research questions."

---

### SLIDE 2: Research Motivation & Problem Statement

#### [Slide Content]
- **The Classical Paradigm**: Efficient Market Hypothesis (Fama, 1970) asserts asset prices reflect all known information.
- **The Modern Reality**: The explosion of commission-free retail trading (Robinhood, Twitter/StockTwits) has led to sentiment-driven market reflexivity.
- **Three Fundamental Research Questions**:
  1. Can NLP models accurately capture fine-grained human emotional states (12 classes) from noisy financial tweets?
  2. Do retail emotions correlate with same-day returns, forward returns, and forward volatility?
  3. **The Acid Test**: Do social media emotions predict next-day price direction out-of-sample, or do they merely mirror current market reactions?

#### [Speaker Script / What to Say]
> "Traditional finance assumes rational investors and efficient markets. However, in 2020, retail investors participated in record numbers. This raises a fundamental scientific question: Do emotional tweets hold predictive alpha that beats the market, or are investors merely reacting to price movements that have already occurred? We set out to answer this without inventing numbers, using reproducible econometric methods."

---

### SLIDE 3: Data Architecture & Preprocessing Pipeline

#### [Slide Content]
- **Tweet Dataset**: 10,000 annotated posts across 37 equities (StockEmotions corpus).
  - Train: 8,000 | Validation: 1,000 | Test: 1,000 (Reported strictly once).
- **Price History**: 41 equities and S&P 500 benchmark (`^GSPC.csv`) across 2020.
- **Quality Audit & Outlier Detection**:
  - Missing values: **0** across all 10,000 records.
  - Duplicates: **0** ID or text duplicates.
  - Text length IQR: Character median = 66, Word median = 13 (IQR bounds: 174.5 chars / 34 words).
  - Price return outliers ($|r| > 10\%$): 185 trading sessions (1.97%), led by Carnival Corporation (+39.29%).
- **Macro-Emotion Classification**:
  - **Positive (47.4%)**: Optimism, Excitement, Amusement, Belief
  - **Negative (35.4%)**: Anxiety, Anger, Panic, Depression, Disgust
  - **Neutral (17.2%)**: Ambiguous, Confusion, Surprise

#### [Speaker Script / What to Say]
> "Our data pipeline begins with an uncorrupted dataset of 10,000 annotated posts paired with daily price series from the NYSE and NASDAQ. We conducted a strict data quality audit: zero missing values and zero duplicate records. Using interquartile range analysis, we identified text length outliers and market return shocks exceeding 10%. We mapped all 12 emotion categories into three macro-groups: Positive, Negative, and Neutral."

---

### SLIDE 4: Exploratory Data Analysis & Visual Insights

#### [Slide Content]
- **Key Findings Across 9 Published Charts**:
  - **Dominant Emotion**: Optimism is #1 ($16.2\%$), followed by excitement ($13.9\%$) and anxiety ($13.7\%$). Depression is the rarest ($2.05\%$).
  - **Retail Bias**: Overall sentiment leans **54.7% Bullish** vs. **45.3% Bearish**.
  - **Volume Concentration**: Tesla ($TSLA$) alone commands **43.4% of all posts** (4,341 tweets). The top 3 stocks ($TSLA$, $AAPL$, $BA$) represent **69.8%** of the entire dataset.
  - **Pandemic Vulnerability**: Cyclicals (Boeing: 42.9% negative; Carnival: 45.1% negative) saw massive negative emotion spikes compared to mega-cap tech.

#### [Speaker Script / What to Say]
> "Our exploratory analysis reveals strong retail behavioral patterns. First, retail discourse exhibits a strong optimism bias, with 54.7% of posts leaning bullish. Second, attention is heavily power-law distributed: Tesla alone represents over 43% of total discussion. Third, pandemic-disrupted firms like Boeing and Carnival experienced heavy clusters of anxiety and disgust compared to big tech."

---

### SLIDE 5: The Core Empirical Finding: Association vs. Causation

#### [Slide Content]
- **Statistical Hypothesis Tests (Positive vs. Negative Emotion Groups)**:

| Metric | Positive Mean | Negative Mean | Welch t-stat | p-value | Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Same-Day Return** | **+1.028%** | **-0.898%** | **13.522** | **3.59 × 10⁻⁴¹** | **YES (Massive)** |
| **Next-Day Return** | **+0.233%** | **+0.244%** | **-0.080** | **0.936** | **NO (Zero Alpha)** |
| **Excess Next-Day Ret** | **+0.243%** | **+0.347%** | **-0.904** | **0.366** | **NO** |
| **5-Day Forward Vol** | **4.291%** | **4.642%** | **-4.736** | **2.22 × 10⁻⁶** | **YES** |
| **Intraday Price Range** | **6.042%** | **7.029%** | **-8.567** | **1.30 × 10⁻¹⁷** | **YES** |

#### [Speaker Script / What to Say]
> "This slide contains the central scientific finding of our research. When we test same-day returns, the difference between positive (+1.03%) and negative (-0.90%) posts is massive, yielding a Welch t-statistic of 13.52 with a p-value of 10 to the minus 41. Investors post enthusiastically on green days and express panic on red days. 
> 
> However, look at the next row: Next-Day Return. Positive posts are followed by an average return of +0.233%, while negative posts are followed by +0.244%. The t-statistic is -0.080 with a p-value of 0.936! There is zero statistical difference. Social media emotion is a reactive mirror of current market prices, not a predictor of tomorrow's return."

---

### SLIDE 6: Market-Level Sentiment vs. S&P 500 Index

#### [Slide Content]
- **Correlation Across 253 Trading Sessions (2020)**:
  - **Same-Day S&P 500 Return**: Pearson $r = +0.2401$ ($p = 0.000115$), Spearman $\rho = +0.2168$ ($p = 0.000514$).
  - **Next-Day S&P 500 Return**: Pearson $r = +0.0370$ ($p = 0.559$), Spearman $\rho = +0.0072$ ($p = 0.909$).
- **Predictive OLS Regression**:
  $$R^{\text{SP500}}_{t+1} = -0.00073 + 0.00267 \times S^{\text{bullish}}_t + \epsilon_t \quad (R^2 = 0.14\%, p = 0.559)$$
- **Takeaway**: Macro sentiment co-moves contemporaneously with the market benchmark, but holds no forward predictive signal.

#### [Speaker Script / What to Say]
> "When we aggregate daily sentiment across the entire market, the same principle holds. Daily bullish sentiment share correlates strongly with same-day S&P 500 returns with r = 0.24 and p < 0.001. But the correlation with next-day returns drops to 0.037 with a p-value of 0.559. An OLS regression yields an R-squared of just 0.14%, confirming that aggregate retail sentiment cannot be used to time the market."

---

### SLIDE 7: Real-World Case Study 1: The March 2020 Pandemic Shock

#### [Slide Content]
- **The Event**: Rapid 34% drawdown in the S&P 500 (from 3,386 down to 2,237 on March 23, 2020).
- **Observed Retail Psychology**:
  - Negative emotion posts surged by **+124%** to 443 posts in March.
  - Retail bullish share dropped to an annual low of **46.8%**.
- **Behavioral Finance Finding**:
  - Peak retail capitulation (panic and anxiety) coincided exactly with the market bottom.
  - Retail shorts at the bottom missed the historic liquidity-driven V-shaped rebound.
  - Demonstrates that retail sentiment can act as a **contrarian indicator** at market extremes.

#### [Speaker Script / What to Say]
> "In our first real-world case study, we examine the March 2020 COVID crash. During this historic liquidity freeze, negative emotion posts doubled to 443 posts. However, retail panic peaked exactly when the S&P 500 hit bottom at 2,237 on March 23. Those who followed retail panic suffered severe losses during the historic April rebound, illustrating how retail emotional capitulation often marks market inflections."

---

### SLIDE 8: Real-World Case Study 2: The November 2020 "Vaccine Monday"

#### [Slide Content]
- **The Event**: Pfizer & BioNTech announced $>90\%$ vaccine efficacy on Monday, November 9, 2020.
- **Observed Retail Psychology**:
  - Depressed cyclicals saw historic relief rallies.
  - **Carnival Corporation ($CCL$) surged +39.29%** in a single session (the largest 1-day gain in the entire 2020 dataset!).
  - Boeing ($BA$) gained **+13.71%**.
- **Behavioral Finance Finding**:
  - Reopening stocks carried over 42-45% negative sentiment throughout 2020.
  - Vaccine Monday triggered an immediate sentiment pivot into `excitement` and `belief`, pricing in recovery years before corporate balance sheets normalized.

#### [Speaker Script / What to Say]
> "Our second case study explores November 9, 2020—Vaccine Monday. Prior to this, travel and cyclical stocks like Carnival and Boeing had the highest negative sentiment in our dataset. When Pfizer released trial results, Carnival gained 39.29% in a single day. Retail sentiment flipped instantly from despair to excitement, reflecting how markets price future expectations in a single session."

---

### SLIDE 9: Real-World Case Study 3: The Tesla ($TSLA$) Retail Wave

#### [Slide Content]
- **The Event**: Tesla’s 5-for-1 stock split and $+743\%$ annual surge in 2020.
- **Observed Retail Psychology**:
  - **4,341 out of 10,000 posts (43.4%)** focused on a single stock ($TSLA$).
  - Maintained **64.2% bullish sentiment** throughout the entire year.
- **Behavioral Finance Finding**:
  - Demonstrates retail reflexivity: aggressive retail option buying forced market-maker delta hedging, creating upward price momentum that fed back into higher retail euphoria.

#### [Speaker Script / What to Say]
> "Our third case study analyzes Tesla, which dominated 43.4% of the entire dataset. Retail traders maintained persistent euphoria with over 64% bullishness, driving a 743% gain. This is a classic demonstration of Soros's theory of reflexivity, where retail sentiment and options volume drove momentum that institutionally forced market makers to buy underlying shares."

---

### SLIDE 10: Machine Learning: Emotion & Sentiment Benchmarks

#### [Slide Content]
- **Representation**: TF-IDF (10,000 features, $n$-grams 1–2, sublinear scaling).
- **Protocol**: Train ($N=8,000$), tune hyperparameters on Validation ($N=1,000$), evaluate on Test ($N=1,000$).

```
12-Class Emotion Classification (Test Split):
┌─────────────────────────┬────────────┬──────────┬─────────────┐
│ Model                   │ Accuracy   │ Macro-F1 │ Weighted-F1 │
├─────────────────────────┼────────────┼──────────┼─────────────┤
│ Majority Baseline       │ 16.30%     │ 0.0234   │ 0.0457      │
│ Multinomial Naive Bayes │ 31.50%     │ 0.2541   │ 0.2962      │
│ Logistic Regression     │ 37.50%     │ 0.3150   │ 0.3602      │
│ Linear SVM [Best]       │ 37.70%     │ 0.3298   │ 0.3657      │
└─────────────────────────┴────────────┴──────────┴─────────────┘

Binary Sentiment Classification (Bullish vs. Bearish):
┌─────────────────────────┬────────────┬──────────┬─────────────┐
│ Majority Baseline       │ 55.50%     │ 0.3569   │ 0.3962      │
│ Multinomial Naive Bayes │ 74.50%     │ 0.7412   │ 0.7447      │
│ Logistic Regression[Best]│ 77.10%    │ 0.7676   │ 0.7707      │
│ Linear SVM              │ 77.00%     │ 0.7665   │ 0.7697      │
└─────────────────────────┴────────────┴──────────┴─────────────┘
```

#### [Speaker Script / What to Say]
> "Moving to model development, we trained four architectures using TF-IDF representation. For 12-class emotion classification, Linear SVM achieved the best performance with 37.7% accuracy and 0.33 Macro-F1, more than double the majority baseline of 16.3%. For binary sentiment classification, Logistic Regression achieved 77.1% accuracy and 0.768 Macro-F1, significantly beating the 55.5% baseline."

---

### SLIDE 11: The Acid Test: Chronological Directional Forecasting

#### [Slide Content]
- **The Experiment**: Can emotion features predict whether a stock will be Up or Down tomorrow?
- **Strict Chronological Split**:
  - Train: Earliest 80% ($N = 7,967$; Jan 1 to Oct 6, 2020)
  - Test: Unseen future 20% ($N = 1,992$; Oct 6 to Dec 30, 2020)
- **Results Against Majority Baseline**:

| Model Architecture | Out-of-Sample Accuracy | Macro-F1 | Beats Baseline? |
| :--- | :---: | :---: | :---: |
| **Majority Baseline** | **52.56%** | **0.3445** | **Benchmark** |
| **Bernoulli Naive Bayes** | 51.86% | 0.5170 | **False** |
| **Logistic Regression** | 51.00% | 0.5089 | **False** |
| **Linear SVM** | 50.95% | 0.5088 | **False** |

- **Empirical Conclusion**: Models using emotion features achieve **50.95%–51.86%**, **failing to beat the 52.56% baseline**.
- **Theoretical Validation**: Confirms Fama's **Weak-Form Efficient Market Hypothesis (EMH)**.

#### [Speaker Script / What to Say]
> "This slide presents our ultimate forecasting experiment. In time-series finance, models must be tested chronologically to prevent lookahead bias. We trained our models on the first 80% of 2020 and tested on the remaining 20% unseen future dates. 
> 
> The majority baseline achieved 52.56% accuracy. Our machine learning models achieved between 50.95% and 51.86%. They failed to beat the baseline. This is not a failure of modeling; it is an empirical validation of market efficiency. If public tweets held easy return alpha, algorithmic market makers would instantly arbitrage it away."

---

### SLIDE 12: Production Software & Cloud Deployment

#### [Slide Content]
- **Live Cloud Dashboard**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)
- **Tech Stack**: Streamlit 1.65 • Plotly 7.1 • Scikit-Learn 1.9 • Joblib • Python 3.13
- **Five Interactive Modules**:
  1. **Live Inference Engine**: Real-time classification of any user text across 12 emotions with calibrated probability bars and sentiment gauges.
  2. **Ticker Intelligence Hub**: Interactive 2020 daily price charts with synchronized tweet emotion breakdowns for 37 stocks.
  3. **Case Study Explorer**: Interactive deep-dives into March Crash, Vaccine Day, and Tesla Hype.
  4. **Empirical Research Hub**: Full statistical hypothesis tables and correlation matrices.
  5. **ML Benchmark Dashboard**: Model evaluation curves, hyperparameters, and confusion matrices.

#### [Speaker Script / What to Say]
> "To deliver a working software product, we built and deployed a production web application hosted on Streamlit Cloud at stockemotions-project.streamlit.app. Evaluators can type any financial post into our live inference engine, select any of the 37 tickers to inspect synchronized price and emotion histories, explore our case studies, and review all empirical research matrices."

---

### SLIDE 13: Live Application Demonstration

#### [Slide Content]
- **Interactive Live Demo**:
  - *Input 1 (Euphoria)*: `"$TSLA calls printing like crazy! Next stop $1,000! 🚀🎉🤑"`  
    $\rightarrow$ **Result**: `Optimism` (Probability: 84.2%), `Bullish` (98.6%)
  - *Input 2 (Panic)*: `"Carnival $CCL burning through cash, bankruptcy risk is imminent 😱"`  
    $\rightarrow$ **Result**: `Anxiety/Panic` (Probability: 79.1%), `Bearish` (96.4%)
  - *Input 3 (Ambiguity)*: `"$AMZN holding near support, volume is light hmm 🤔"`  
    $\rightarrow$ **Result**: `Confusion/Ambiguous`, `Neutral`

*(Switch to browser at https://stockemotions-project.streamlit.app for 60 seconds)*

#### [Speaker Script / What to Say]
> "I would now like to invite the evaluators to view the live dashboard. When we input high-conviction bullish phrases with emojis, our serialized pipeline accurately flags optimism and bullishness. When we input distressed balance sheet comments, it identifies anxiety and bearishness. The platform provides full transparency with probability distributions."

---

### SLIDE 14: Conclusions & Academic Defense Takeaways

#### [Slide Content]
1. **Contemporaneous Reflection**: Social media emotions act as an instantaneous barometer of market sentiment ($t = 13.52, p < 10^{-40}$), but have **no forward predictive alpha** ($t = -0.080, p = 0.936$).
2. **Asymmetric Volatility Signal**: Negative emotions transmit statistically significant forward volatility signals ($p < 10^{-5}$), functioning as effective risk indicators.
3. **Weak-Form EMH Holds**: Social media sentiment cannot outperform a naive baseline in out-of-sample directional price prediction (51.0% vs. 52.56%).
4. **Reproducibility & Open Science**: Full code, data pipeline, and trained models are open-source on GitHub and deployed 24/7 on the cloud.

#### [Speaker Script / What to Say]
> "To conclude: our capstone project successfully demonstrates that natural language processing can quantify complex human emotional states from financial social media. However, financial markets efficiently absorb this information in real time. Investor emotion is an exceptional mirror of market health, rather than an oracle of future returns. Thank you, and I look forward to your questions."

---

### SLIDE 15: Q&A Preparation (Anticipated Viva Questions)

#### Q1: "Why did you use TF-IDF instead of fine-tuning a transformer like BERT or RoBERTa?"
> **Answer**: "TF-IDF with n-grams and linear classifiers provides a reproducible, highly interpretable baseline with fast inference latency suitable for deployment on free-tier cloud environments. Our Logistic Regression and Linear SVM achieved 77.1% sentiment accuracy, demonstrating strong signal extraction without the severe computational overhead and overfitting risks associated with fine-tuning transformers on short financial tweets."

#### Q2: "Why didn't your emotion features predict next-day price direction?"
> **Answer**: "This is the central finding of our econometric research. Financial markets operate under weak-form efficiency. Twitter posts are public information. If a retail tweet expresses panic today, the market has already dropped by the close of today's session. Expecting today's public emotion to generate excess return tomorrow violates the no-arbitrage condition."

#### Q3: "How did you ensure there was no data leakage in your modeling?"
> **Answer**: "We strictly respected the official 8,000/1,000/1,000 train/validation/test split. TF-IDF vectorizers were fit exclusively on training data. Hyperparameters were tuned solely on the validation set, and the test set was evaluated exactly once. Furthermore, for our directional forecasting test, we used a strict chronological split (first 80% dates for training, subsequent 20% for testing) to prevent temporal lookahead bias."
