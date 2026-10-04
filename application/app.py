import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Investor Emotions & Stock Market Behaviour",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")
PRICE_DIR = os.path.join(BASE_DIR, "dataset", "price")
RESULTS_JSON = os.path.join(ANALYSIS_DIR, "results.json")
PREPARED_CSV = os.path.join(ANALYSIS_DIR, "prepared_tweets.csv")
POST_FIN_CSV = os.path.join(ANALYSIS_DIR, "post_finance_metrics.csv")

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .case-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 10px;
    }
    .badge-pos {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-neg {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load models and data
@st.cache_resource
def load_ml_assets():
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
    emo_model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
    senti_model = joblib.load(os.path.join(MODEL_DIR, "sentiment_model.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return tfidf, emo_model, senti_model, meta

@st.cache_data
def load_datasets():
    df_tweets = pd.read_csv(PREPARED_CSV)
    df_tweets["date"] = pd.to_datetime(df_tweets["date"])
    
    df_fin = pd.read_csv(POST_FIN_CSV)
    df_fin["date"] = pd.to_datetime(df_fin["date"])
    
    with open(RESULTS_JSON) as f:
        results = json.load(f)
        
    return df_tweets, df_fin, results

try:
    tfidf, emo_model, senti_model, meta = load_ml_assets()
    df_tweets, df_fin, results = load_datasets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    st.error(f"Error loading system assets: {e}")

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/bullish--v1.png", width=70)
st.sidebar.title("StockEmotions AI")
st.sidebar.markdown("**Project Hub**: Investor Emotions & Market Behaviour")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📌 Project Metadata")
st.sidebar.markdown("- **Scope**: 10,000 Verified Tweets (2020)")
st.sidebar.markdown("- **Tickers**: 37 S&P 500 / NASDAQ Equities")
st.sidebar.markdown("- **Emotions**: 12 Fine-Grained Classes")
st.sidebar.markdown("- **Benchmark**: S&P 500 Index (^GSPC)")
st.sidebar.markdown("- **Reproducibility**: Random Seed 42")
st.sidebar.markdown("---")
st.sidebar.info("💡 **Academic Note**: Realized statistical results strictly refute causal return forecasting, proving emotions reflect contemporaneous price reaction (market association).")

# Main Header
st.markdown("<div class='main-header'>Investor Emotions & Stock Market Behaviour</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Machine Learning Platform • Empirical Behavioral Finance • Real-World Case Studies</div>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Live Emotion & Sentiment Inference",
    "📈 Ticker Market Intelligence",
    "🏛️ Real-World Case Studies",
    "📊 Empirical Research & Association",
    "🤖 Machine Learning Benchmarks"
])

# ==============================================================================
# TAB 1: LIVE INFERENCE ENGINE
# ==============================================================================
with tab1:
    st.subheader("Interactive Natural Language Inference Engine")
    st.markdown("Test the trained **12-Class Emotion Classifier** and **Bullish/Bearish Sentiment Model** on any custom tweet, financial news headline, or forum post.")
    
    sample_presets = {
        "Custom Input": "",
        "Tesla Euphoria 🚀": "$TSLA calls printing like crazy! Next stop $1,000, shorts are totally ruined! 🚀🎉🤑",
        "COVID Crash Panic 😱": "$AAPL breaking down hard, circuit breakers hit again, market is in complete freefall! 😱📉",
        "Cruise Line Distress 🚢": "Carnival $CCL is burning through cash at alarming rates. Bankruptcy odds are skyrocketing.",
        "Market Ambiguity 🤔": "$AMZN holding near support, but volumes are low. Not sure if this bounce will hold or fail 🤔",
        "Home Depot Value Belief 🤞": "Added more $HD on today's dip. Solid balance sheet and strong housing trend. Stars are aligned."
    }
    
    selected_preset = st.selectbox("Choose a pre-configured sample or write your own:", list(sample_presets.keys()))
    default_text = sample_presets[selected_preset]
    
    user_text = st.text_area("Enter financial post text:", value=default_text, height=100, placeholder="Type or paste any investor tweet here (e.g. '$AAPL hitting new highs today! 🚀')...")
    
    col_btn1, col_btn2 = st.columns([1, 5])
    with col_btn1:
        run_predict = st.button("Classify Post", type="primary", use_container_width=True)
        
    if user_text.strip():
        # Feature transformation
        X_vec = tfidf.transform([user_text])
        
        # Predictions
        emo_id = emo_model.predict(X_vec)[0]
        emo_probs = emo_model.predict_proba(X_vec)[0]
        predicted_emo = meta["id_to_emotion"][str(emo_id)]
        
        senti_id = senti_model.predict(X_vec)[0]
        senti_probs = senti_model.predict_proba(X_vec)[0]
        predicted_senti = meta["id_to_sentiment"][str(senti_id)]
        
        # Determine emotion group
        emo_group = "Neutral"
        for grp, emos in meta["emotion_groups"].items():
            if predicted_emo in emos:
                emo_group = grp.capitalize()
                break
                
        st.markdown("---")
        st.markdown("### 🎯 Model Classification Results")
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            emo_emoji_map = {
                "optimism": "🌱", "excitement": "🔥", "amusement": "😄", "belief": "🙏",
                "anxiety": "😰", "anger": "😡", "panic": "🚨", "depression": "🌧️", "disgust": "🤢",
                "ambiguous": "🌀", "confusion": "❓", "surprise": "⚡"
            }
            emoji_icon = emo_emoji_map.get(predicted_emo, "💬")
            st.metric(label="Predicted Emotion (12-Class)", value=f"{emoji_icon} {predicted_emo.capitalize()}")
        with c2:
            senti_color = "🟢" if predicted_senti == "bullish" else "🔴"
            st.metric(label="Investor Sentiment", value=f"{senti_color} {predicted_senti.capitalize()}")
        with c3:
            st.metric(label="Macro Emotion Group", value=emo_group)
        with c4:
            senti_confidence = senti_probs[senti_id] * 100
            st.metric(label="Sentiment Confidence", value=f"{senti_confidence:.1f}%")
            
        # Visual distribution
        col_prob1, col_prob2 = st.columns([1.5, 1])
        with col_prob1:
            st.markdown("#### Emotion Probability Distribution (12 Classes)")
            prob_df = pd.DataFrame({
                "Emotion": [meta["id_to_emotion"][str(i)].capitalize() for i in range(len(emo_probs))],
                "Probability": emo_probs
            }).sort_values("Probability", ascending=True)
            
            fig_bar = px.bar(
                prob_df, x="Probability", y="Emotion", orientation="h",
                color="Probability", color_continuous_scale="Blues",
                labels={"Probability": "Model Confidence Score"},
                height=380
            )
            fig_bar.update_layout(showlegend=False, margin=dict(l=0, r=0, t=10, b=10))
            st.plotly_chart(fig_bar, use_container_width=True)
            
        with col_prob2:
            st.markdown("#### Bullish vs. Bearish Gauge")
            fig_donut = go.Figure(data=[go.Pie(
                labels=["Bearish", "Bullish"],
                values=[senti_probs[0], senti_probs[1]],
                hole=0.55,
                marker=dict(colors=["#EF4444", "#10B981"])
            )])
            fig_donut.update_layout(
                margin=dict(l=20, r=20, t=20, b=20),
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption(f"**Linguistic Analysis**: Word Count: {len(user_text.split())} | Character Count: {len(user_text)}")

# ==============================================================================
# TAB 2: TICKER MARKET INTELLIGENCE
# ==============================================================================
with tab2:
    st.subheader("Ticker-Level Sentiment & Financial Profiler")
    st.markdown("Explore social sentiment dynamics across all **37 equities** in the dataset mapped directly against 2020 price movements.")
    
    tickers_list = sorted(df_tweets["ticker"].unique().tolist())
    selected_ticker = st.selectbox("Select Equity Ticker:", tickers_list, index=tickers_list.index("TSLA") if "TSLA" in tickers_list else 0)
    
    ticker_tweets = df_fin[df_fin["ticker"] == selected_ticker].sort_values("date")
    
    # Load actual stock price series
    ticker_file = "FB.csv" if selected_ticker == "FB" else ("BRK-B.csv" if selected_ticker == "BRK.B" else f"{selected_ticker}.csv")
    price_path = os.path.join(PRICE_DIR, ticker_file)
    
    if os.path.exists(price_path):
        price_df = pd.read_csv(price_path)
        price_df["Date"] = pd.to_datetime(price_df["Date"])
        price_df = price_df.sort_values("Date").reset_index(drop=True)
        
        # Summary metrics
        annual_ret = (price_df["Adj Close"].iloc[-1] - price_df["Adj Close"].iloc[0]) / price_df["Adj Close"].iloc[0] * 100
        total_ticker_posts = len(ticker_tweets)
        bullish_share = (ticker_tweets["senti_encoded"].mean() * 100) if total_ticker_posts > 0 else 0
        top_emo = ticker_tweets["emo_label"].mode()[0] if total_ticker_posts > 0 else "N/A"
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("2020 Post Volume", f"{total_ticker_posts:,}")
        m2.metric("Bullish Share", f"{bullish_share:.1f}%")
        m3.metric("Dominant Emotion", top_emo.capitalize())
        m4.metric("2020 Total Return", f"{annual_ret:+.1f}%")
        m5.metric("Avg Intraday Range", f"{price_df['High'].sub(price_df['Low']).div(price_df['Low']).mean()*100:.2f}%")
        
        st.markdown("---")
        col_chart1, col_chart2 = st.columns([2, 1])
        
        with col_chart1:
            st.markdown(f"#### 2020 Daily Price Trajectory (${selected_ticker})")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=price_df["Date"], y=price_df["Adj Close"],
                mode="lines", name=f"{selected_ticker} Adj Close",
                line=dict(color="#2563EB", width=2.2)
            ))
            fig_price.update_layout(
                xaxis_title="Date", yaxis_title="Adjusted Close ($)",
                height=400, margin=dict(l=10, r=10, t=20, b=20),
                hovermode="x unified"
            )
            st.plotly_chart(fig_price, use_container_width=True)
            
        with col_chart2:
            st.markdown("#### Emotion Breakdown")
            if total_ticker_posts > 0:
                emo_dist = ticker_tweets["emo_label"].value_counts().reset_index()
                emo_dist.columns = ["Emotion", "Count"]
                fig_pie = px.pie(emo_dist, names="Emotion", values="Count", hole=0.45, color_discrete_sequence=px.colors.qualitative.Safe)
                fig_pie.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=400)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No posts recorded for this ticker.")
                
        # Recent posts sample
        st.markdown("#### Representative Post Samples")
        st.dataframe(
            ticker_tweets[["date", "emo_label", "senti_label", "original", "same_day_return", "next_day_return"]].head(5).rename(
                columns={
                    "date": "Date", "emo_label": "Emotion", "senti_label": "Sentiment",
                    "original": "Tweet Text", "same_day_return": "Same-Day Ret", "next_day_return": "Next-Day Ret"
                }
            ),
            use_container_width=True
        )

# ==============================================================================
# TAB 3: REAL-WORLD CASE STUDIES
# ==============================================================================
with tab3:
    st.subheader("Real-World Case Studies: Behavioral Finance in Crisis & Euphoria")
    st.markdown("Educational case studies demonstrating how investor sentiment mirrored major macro shocks during 2020.")
    
    case_choice = st.radio(
        "Select a Case Study to Investigate:",
        [
            "Case Study 1: The March 2020 COVID-19 Liquidity Shock & Panic Peak",
            "Case Study 2: The November 2020 'Vaccine Monday' Rotation & Relief Rally",
            "Case Study 3: The Retail Trading Explosion & Electric Vehicle Euphoria (TSLA)"
        ],
        horizontal=True
    )
    
    if "Case Study 1" in case_choice:
        st.markdown("<div class='case-title'>Case 1: The March 2020 COVID-19 Crash & Retail Panic</div>", unsafe_allow_html=True)
        st.markdown("""
        **Context**: In March 2020, the onset of the global COVID-19 pandemic caused the fastest 30% drawdown in stock market history.
        The S&P 500 dropped from an all-time high of **3,386** down to **2,237** on March 23, 2020, accompanied by multiple circuit breakers.
        """)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Negative Posts in March", "443 Posts", "+124% vs. Jan")
        c2.metric("March S&P 500 Return", "-16.36%", "Historic Drawdown")
        c3.metric("Bullish Share Low", "46.8%", "Annual Minimum")
        
        st.markdown("""
        #### Key Empirical Findings:
        1. **Contemporaneous Panic vs. Predictive Alpha**:
           - Investors flooded social platforms with `panic`, `anxiety`, and `disgust` on days the market plunged (Welch $t = 13.52, p < 10^{-40}$).
           - However, retail panic did **not** predict that the market would continue falling the next day. In fact, aggressive retail shorting at the bottom preceded the historic April recovery rally.
        2. **Volatility Clustering**:
           - Negative emotions coincided with 5-day forward return volatility of **4.64%**, compared to **4.29%** for positive emotions ($p = 2.22 \\times 10^{-6}$).
        """)
        
        # Display March emotion excerpt
        march_tweets = df_fin[(df_fin["date"] >= "2020-03-01") & (df_fin["date"] <= "2020-03-31") & (df_fin["emotion_group"] == "negative")]
        st.markdown("**Sample of March 2020 Panic Posts from Dataset:**")
        st.dataframe(march_tweets[["date", "ticker", "emo_label", "original"]].head(4), use_container_width=True)

    elif "Case Study 2" in case_choice:
        st.markdown("<div class='case-title'>Case 2: The November 2020 'Vaccine Monday' Rotation</div>", unsafe_allow_html=True)
        st.markdown("""
        **Context**: On Monday, November 9, 2020, Pfizer and BioNTech released interim Phase 3 clinical trial results showing their COVID-19 vaccine was over 90% effective. 
        This triggered a massive relief rally, particularly in heavily depressed cyclicals and reopening stocks (airlines, cruises, energy).
        """)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Carnival ($CCL$) Nov 9 Return", "+39.29%", "Highest in 2020")
        c2.metric("Boeing ($BA$) Nov 9 Return", "+13.71%", "Reopening Relief")
        c3.metric("Retail Excitement/Optimism", "+84% Spike", "Sentiment Pivot")
        
        st.markdown("""
        #### Key Empirical Findings:
        1. **Asymmetric Reopening Euphoria**:
           - Reopening stocks like Carnival ($CCL$) and Boeing ($BA$) had accumulated the highest proportion of negative emotion posts (over **42-45%** negative posts) throughout 2020.
           - On November 9, 2020, $CCL$ experienced a **+39.29%** single-day gain, the single largest 1-day price return in the entire dataset!
        2. **Sentiment Shift Preceded Fundamentals**:
           - Retail investor emotion flipped instantly to `excitement` and `belief`, reflecting future expectations long before corporate revenues actually recovered.
        """)

    else:
        st.markdown("<div class='case-title'>Case 3: The Retail Trading Explosion & EV Euphoria ($TSLA$)</div>", unsafe_allow_html=True)
        st.markdown("""
        **Context**: Throughout 2020, retail trading volume reached unprecedented records, driven by commission-free brokerage apps and social media discussion.
        Tesla ($TSLA$) became the epicenter of retail market culture, culminating in its 5-for-1 stock split and S&P 500 inclusion.
        """)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("TSLA Dataset Share", "4,341 Posts", "43.4% of Total Data")
        c2.metric("TSLA 2020 Total Gain", "+743%", "Historic Bull Run")
        c3.metric("TSLA Bullish Sentiment", "64.2%", "Persistent Euphoria")
        
        st.markdown("""
        #### Key Empirical Findings:
        1. **Extreme Retail Concentration**:
           - Out of 37 stocks, $TSLA$ alone accounted for **4,341 of the 10,000 tweets (43.4%)**, demonstrating that financial social media volume is heavily power-law distributed.
        2. **Perpetual Bullish Momentum**:
           - Despite frequent institutional skepticism and high short interest, retail posts for $TSLA$ maintained over 64% bullish sentiment, characterized by `optimism` and `excitement`.
        """)

# ==============================================================================
# TAB 4: EMPIRICAL RESEARCH & ASSOCIATION
# ==============================================================================
with tab4:
    st.subheader("Econometric Research: Association vs. Causation")
    st.markdown("""
    A foundational principle of empirical finance: **Contemporaneous correlation does not imply predictive causality.**
    Below are the hypothesis tests computed in `03_finance.py`.
    """)
    
    fin_stats = results["step3_finance"]
    tests = fin_stats["welch_and_mann_whitney_tests"]
    
    st.markdown("### 1. Statistical Hypothesis Tests: Positive vs. Negative Emotion Groups")
    
    test_rows = []
    for metric, data in tests.items():
        test_rows.append({
            "Financial Metric": metric.replace("_", " ").title(),
            "Positive Mean": f"{data['pos_mean']*100:.3f}%" if "return" in metric or "vol" in metric or "range" in metric else f"{data['pos_mean']:.4f}",
            "Negative Mean": f"{data['neg_mean']*100:.3f}%" if "return" in metric or "vol" in metric or "range" in metric else f"{data['neg_mean']:.4f}",
            "Difference": f"{data['difference (pos - neg)']*100:+.3f}%",
            "Welch t-stat": f"{data['welch_t_stat']:.3f}",
            "Welch p-value": f"{data['welch_p_val']:.4e}",
            "Statistically Significant (α=0.05)": "✅ Yes" if data["is_significant_05"] else "❌ No"
        })
    st.table(pd.DataFrame(test_rows))
    
    st.markdown("---")
    st.markdown("### 2. The Core Scientific Takeaway")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.success("""
        #### 🟢 Strong Contemporaneous Association (Same-Day)
        - **Welch $t$-statistic**: `13.52` ($p = 3.59 \\times 10^{-41}$)
        - **Kruskal-Wallis $H$**: `244.41` ($p = 4.14 \\times 10^{-46}$)
        - **Daily Bullish Share vs. S&P 500**: $r = +0.240$ ($p = 0.0001$)
        
        **Meaning**: When stocks go up today, retail investors write optimistic, excited posts. When stocks plunge, investors post anxious, panic-driven tweets. **Social media emotions act as an instantaneous mirror of the market.**
        """)
        
    with c_right:
        st.warning("""
        #### 🔴 Absence of Predictive Alpha (Next-Day)
        - **Welch $t$-statistic**: `-0.080` ($p = 0.936$)
        - **Kruskal-Wallis $H$**: `15.07` ($p = 0.179$)
        - **Daily Bullish Share vs. S&P 500 (T+1)**: $r = +0.037$ ($p = 0.559$)
        
        **Meaning**: Positive sentiment today does **not** generate positive returns tomorrow. Next-day average returns following positive emotions (+0.233%) and negative emotions (+0.244%) are statistically identical.
        """)

# ==============================================================================
# TAB 5: ML BENCHMARKS & EVALUATION
# ==============================================================================
with tab5:
    st.subheader("Machine Learning Performance & Model Benchmarks")
    st.markdown("Evaluated out-of-sample on the test set (`1,000 posts`) with validation-tuned hyperparameters.")
    
    model_stats = results["step4_model"]
    
    # Tables for Sentiment and Emotion
    c_m1, c_m2 = st.columns(2)
    
    with c_m1:
        st.markdown("#### 1. Sentiment Classification (Bullish vs. Bearish)")
        senti_perf = []
        for m_name, m_val in model_stats["sentiment_classification_bullish_bearish"]["models"].items():
            senti_perf.append({
                "Model": m_name.replace("_", " ").title(),
                "Accuracy": f"{m_val['accuracy']*100:.2f}%",
                "Macro-F1": f"{m_val['macro_f1']:.4f}",
                "Weighted-F1": f"{m_val['weighted_f1']:.4f}"
            })
        st.dataframe(pd.DataFrame(senti_perf), hide_index=True, use_container_width=True)
        
    with c_m2:
        st.markdown("#### 2. Emotion Classification (12 Classes)")
        emo_perf = []
        for m_name, m_val in model_stats["emotion_classification_12classes"]["models"].items():
            emo_perf.append({
                "Model": m_name.replace("_", " ").title(),
                "Accuracy": f"{m_val['accuracy']*100:.2f}%",
                "Macro-F1": f"{m_val['macro_f1']:.4f}",
                "Weighted-F1": f"{m_val['weighted_f1']:.4f}"
            })
        st.dataframe(pd.DataFrame(emo_perf), hide_index=True, use_container_width=True)
        
    st.markdown("---")
    st.markdown("#### 3. Chronological Market Direction Prediction Out-of-Sample (Up vs. Down)")
    st.markdown("Testing whether emotion features can beat the **Majority Baseline** in predicting next-day price direction on unseen future dates:")
    
    dir_stats = model_stats["next_day_direction_prediction"]
    dir_perf = []
    for m_name, m_val in dir_stats["models"].items():
        dir_perf.append({
            "Forecasting Model": m_name.replace("_", " ").title(),
            "Out-of-Sample Accuracy": f"{m_val['accuracy']*100:.2f}%",
            "Macro-F1": f"{m_val['macro_f1']:.4f}",
            "Beats Baseline?": "Benchmark" if "majority" in m_name else ("✅ Yes" if m_val['accuracy'] > 0.5256 else "❌ No")
        })
    st.table(pd.DataFrame(dir_perf))
    
    st.info(f"**Conclusion**: {dir_stats['conclusion']['findings']}")

# Footer
st.markdown("---")
st.caption("Investor Emotions & Stock Market Behaviour Capstone Project | Built with Streamlit, Scikit-Learn, Plotly & Pandas")
