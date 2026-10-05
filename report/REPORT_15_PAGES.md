# INVESTOR EMOTIONS AND STOCK MARKET BEHAVIOUR
## 15-Page Academic Project Report (Data Science & Behavioral Finance)

**Submitted by**: Om Patel  
**Branch**: Information Technology  
**Semester**: 5th Semester  
**College**: L.D. College of Engineering, Ahmedabad  
**Academic Year**: 2026–27  
**Live Application**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)  
**Repository**: [StockEmotions-Project](https://github.com/ompatel121206/-StockEmotions-Project)  

---

### PAGE 1: TITLE & PROJECT COVER
- **Dataset Records**: 10,000 Verified Tweets
- **Best Sentiment Accuracy**: 77.10% (Logistic Regression)
- **Welch t-stat**: 13.52 (p < 10⁻⁴⁰) for Same-Day Return Co-movement
- **S&P 500 Equities**: 37 Major Tickers + S&P 500 Benchmark (^GSPC)
- **Core Pillars**: Case Studies, Econometric Research, Model Development, Software Engineering

---

### PAGE 2: PROJECT OVERVIEW & WORKFLOW
- **Problem Statement**: Investigating whether retail investor emotions cause future price changes or contemporaneously reflect ongoing price movements.
- **Workflow**: Ingestion ➔ Data Audit ➔ Price Mapping ➔ Econometric Testing ➔ ML Benchmarking ➔ Streamlit Cloud.
- **Tech Stack**: Python 3.13, Scikit-Learn 1.9, Pandas 3.0, SciPy 1.18, Streamlit 1.65, Plotly 7.1.

---

### PAGE 3: DATASET SPECIFICATIONS
- **Data Sources**: StockEmotions 10,000 tweets corpus & Yahoo Finance 2020 daily price series.
- **Variables**: `id`, `date`, `ticker`, `emo_label` (12 classes), `senti_label` (bullish/bearish), `original`, `processed`, `same_day_return`, `next_day_return`, `fwd_vol_5d`, `intraday_range`.

---

### PAGE 4: DATASET EXPLORATION & AUDIT PREVIEW
- **Quality Audit**: 0 missing values, 0 duplicate IDs, 0 text duplicates.
- **Emoji Coverage**: 100.0% of posts contain expressive emojis.
- **Data Preview**: Clean schema with sentiment and emotion labels.

---

### PAGE 5: DATA PREPARATION & OUTLIER AUDIT
- **Text Length IQR Outliers**: Median character length = 66 (IQR upper = 174.5 chars, 4.95% outliers). Median word count = 13 (IQR upper = 34 words, 4.23% outliers).
- **Return Outliers (|r| > 10%)**: 185 sessions (1.97%), led by Carnival Corporation (+39.29% on Nov 9, 2020).
- **Macro Groups**: Positive (47.4%), Negative (35.4%), Neutral (17.2%).

---

### PAGE 6: EDA PART 1 — EMOTION & SENTIMENT DISTRIBUTIONS
- **Chart 1 (Emotion Distribution)**: Optimism is dominant (16.24%), followed by excitement (13.86%) and anxiety (13.66%). Depression is rarest (2.05%).
- **Chart 2 (Bullish vs. Bearish)**: Sentiment leans 54.74% Bullish vs. 45.26% Bearish.

---

### PAGE 7: EDA PART 2 — HEATMAP & TICKER VOLUME
- **Chart 3 (Emotion x Sentiment Heatmap)**: Optimism and excitement are >98% bullish; disgust (94.9%), anxiety (89.5%), panic (97.7%), depression (98.5%) align strictly with bearish sentiment.
- **Chart 4 (Ticker Concentration)**: TSLA alone accounts for 4,341 posts (43.4%). Top 3 stocks (TSLA, AAPL, BA) comprise 69.8%.

---

### PAGE 8: EDA PART 3 — TICKER BREAKDOWN & S&P 500 DYNAMICS
- **Chart 5 (Top 10 Tickers)**: Tech leaders show positive majorities; cyclical reopening stocks (BA, CCL) carry 42-45% negative sentiment.
- **Chart 6 (Monthly Emotion vs S&P 500)**: Negative emotions peaked in March 2020 (443 posts) matching the S&P 500 COVID-19 market drawdown to 2,584.

---

### PAGE 9: EDA PART 4 & PROPOSED DATA SCIENCE APPROACH
- **Chart 7 (Text Length Distribution)**: Unimodal right-skewed distribution.
- **Methodology**: 12-Class Emotion Classification, Binary Sentiment Classification, and Chronological Market Directional Forecasting.
- **Algorithms**: Majority Baseline, Multinomial Naive Bayes, Logistic Regression, Linear SVM.

---

### PAGE 10: ECONOMETRIC IMPLEMENTATION & RESULTS
- **Same-Day Return (T₀)**: Pos mean = +1.028%, Neg mean = -0.898%, Welch t = 13.522 (p = 3.59e-41). **Massive co-movement**.
- **Next-Day Return (T₁)**: Pos mean = +0.233%, Neg mean = +0.244%, Welch t = -0.080 (p = 0.936). **Zero return alpha**.
- **Kruskal-Wallis Across 12 Emotions**: Same-day H = 244.41 (p < 10⁻⁴⁵); Next-day H = 15.07 (p = 0.179, not significant).
- **Core Law**: Social sentiment is a reactive emotional mirror, not a predictive oracle.

---

### PAGE 11: REAL-WORLD CASE STUDIES
1. **March 2020 COVID Crash**: Negative emotions surged +124% to 443 posts. Retail panic capitulation marked the exact market bottom.
2. **November 2020 Vaccine Monday**: Pfizer announced >90% efficacy; Carnival ($CCL$) surged +39.29% in 1 day; sentiment flipped instantly to excitement.
3. **Tesla ($TSLA$) Retail Wave**: 4,341 posts (43.4%), +743% gain, 64.2% bullishness; retail call buying created reflexive gamma squeezes.

---

### PAGE 12: MACHINE LEARNING BENCHMARKS
- **12-Class Emotion**: Majority (16.30%), Naive Bayes (31.50%), Logistic Regression (37.50%), **Linear SVM (37.70%, Macro-F1: 0.3298)**.
- **Binary Sentiment**: Majority (55.50%), Naive Bayes (74.50%), **Logistic Regression (77.10%, Macro-F1: 0.7676)**, Linear SVM (77.00%).
- **Chronological Directional Forecasting**: Majority Baseline achieves 52.56%; ML models achieve 51.00%, **failing to beat the naive baseline**, validating Weak-Form EMH.

---

### PAGE 13: INTERACTIVE SOFTWARE APPLICATION
- **Live URL**: [https://stockemotions-project.streamlit.app](https://stockemotions-project.streamlit.app)
- **5 Modules**: Live NLP Inference, Ticker Intelligence Hub, Case Study Explorer, Empirical Research Matrix, ML Benchmark Dashboard.
- **Real-Time Examples**: Demonstrated live probability bars across all 12 classes and bullish/bearish gauges.

---

### PAGE 14: FINDINGS, STRENGTHS & LIMITATIONS
- **Findings**: Instantaneous market mirror, no forward return alpha, forward volatility signaling, power-law attention concentration.
- **Strengths**: Automated reproducible pipeline, serialized production models, 24/7 cloud accessibility, zero invented numbers.
- **Limitations**: Daily granularity vs tick data, retail vs institutional flow, social bot spam.

---

### PAGE 15: FUTURE WORK, SUMMARY & REFERENCES
- **Future Work**: FinBERT contextual embeddings, millisecond order book tick latency, options delta-hedging imbalances.
- **Summary**: Full integration of all four pillars (Case Studies, Research, Modeling, Software).
- **References**: Fama (1970), Bollen et al. (2011), Scikit-Learn (2011), Streamlit (2026).
