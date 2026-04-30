import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import time
import re
from io import BytesIO

from src.preprocess import clean_tweet
from src.sentiment_model import analyze_sentiment
from src.emotion_model import detect_emotion
from src.crime_mapper import classify_crime
from src.visualization import (
    plot_sentiment_pie,
    plot_emotion_bar,
    plot_crime_heatmap,
    plot_timeline,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TweetLens · Human Behavior Sentinel",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
    border: 1px solid #4338ca;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    color: white;
}
.metric-card h2 { font-size: 2rem; margin: 0; color: #a5b4fc; }
.metric-card p  { margin: 0; font-size: 0.85rem; color: #c7d2fe; letter-spacing: 0.05em; text-transform: uppercase; }

.badge-pos  { background:#14532d; color:#86efac; border-radius:8px; padding:2px 10px; font-size:0.8rem; }
.badge-neg  { background:#450a0a; color:#fca5a5; border-radius:8px; padding:2px 10px; font-size:0.8rem; }
.badge-neu  { background:#1c1917; color:#d6d3d1; border-radius:8px; padding:2px 10px; font-size:0.8rem; }
.badge-crime{ background:#422006; color:#fdba74; border-radius:8px; padding:2px 10px; font-size:0.8rem; }

stApp { background-color: #0f0e17; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 TweetLens")
    st.markdown("*Human Behavior & Sentiment Intelligence*")
    st.divider()

    mode = st.radio("**Analysis Mode**", ["📂 Dataset Analysis", "✍️ Live Tweet Analyzer"])
    st.divider()

    if mode == "📂 Dataset Analysis":
        uploaded = st.file_uploader("Upload tweets CSV", type=["csv"])
        st.caption("Columns expected: `tweets`, optionally `time`, `username`, `location`")
        sample_n = st.slider("Sample size (rows)", 100, 5000, 1000, step=100)
    else:
        user_tweet = st.text_area("Paste a tweet:", height=120, placeholder="Type or paste tweet text here…")
        analyze_btn = st.button("🚀 Analyze Tweet", use_container_width=True)

    st.divider()
    st.markdown("**Built with:** VADER · TextBlob · Streamlit · Plotly")

# ── Helper: sentiment badge ───────────────────────────────────────────────────
def sentiment_badge(s):
    s = s.lower()
    if s == "positive": return f'<span class="badge-pos">● Positive</span>'
    if s == "negative": return f'<span class="badge-neg">● Negative</span>'
    return f'<span class="badge-neu">● Neutral</span>'

def crime_badge(c):
    if c != "none": return f'<span class="badge-crime">⚠ {c.title()}</span>'
    return "—"

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE TWEET ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════
if mode == "✍️ Live Tweet Analyzer":
    st.title("✍️ Live Tweet Analyzer")

    if "analyze_btn" in dir() and analyze_btn and user_tweet.strip():
        t0 = time.time()
        cleaned = clean_tweet(user_tweet)
        sentiment, scores = analyze_sentiment(cleaned)
        emotion = detect_emotion(cleaned)
        crime = classify_crime(cleaned)
        elapsed = round((time.time() - t0) * 1000)

        col1, col2, col3, col4 = st.columns(4)
        col1.markdown(f'<div class="metric-card"><h2>{sentiment.upper()}</h2><p>Sentiment</p></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card"><h2>{emotion.title()}</h2><p>Emotion</p></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card"><h2>{crime.upper() if crime!="none" else "CLEAN"}</h2><p>Crime Flag</p></div>', unsafe_allow_html=True)
        col4.markdown(f'<div class="metric-card"><h2>{elapsed}ms</h2><p>Inference Time</p></div>', unsafe_allow_html=True)

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("VADER Score Breakdown")
            fig = go.Figure(go.Bar(
                x=["Positive", "Neutral", "Negative"],
                y=[scores["pos"], scores["neu"], scores["neg"]],
                marker_color=["#22c55e", "#94a3b8", "#ef4444"],
            ))
            fig.update_layout(template="plotly_dark", height=300, margin=dict(t=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Cleaned Text")
            st.code(cleaned, language="text")
            st.caption(f"Compound score: **{scores['compound']:.4f}**")
    elif mode == "✍️ Live Tweet Analyzer":
        st.info("👈 Paste a tweet in the sidebar and click **Analyze Tweet**.")

# ═══════════════════════════════════════════════════════════════════════════════
#  DATASET ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.title("📂 Dataset Sentiment & Crime Dashboard")

    # Load data
    if "uploaded" in dir() and uploaded:
        df_raw = pd.read_csv(uploaded)
    else:
        try:
            df_raw = pd.read_csv("data/tweets.csv")
        except FileNotFoundError:
            st.warning("No dataset found. Upload a CSV in the sidebar or place `tweets.csv` in `data/`.")
            st.stop()

    # Normalise tweet column
    tweet_col = next((c for c in df_raw.columns if "tweet" in c.lower()), df_raw.columns[-1])
    df_raw = df_raw.rename(columns={tweet_col: "tweets"})
    df = df_raw.dropna(subset=["tweets"]).sample(min(sample_n, len(df_raw)), random_state=42).reset_index(drop=True)

    with st.spinner("🔄 Running NLP pipeline…"):
        t0 = time.time()
        df["cleaned"]   = df["tweets"].apply(clean_tweet)
        df["sentiment"], df["scores"] = zip(*df["cleaned"].apply(analyze_sentiment))
        df["emotion"]   = df["cleaned"].apply(detect_emotion)
        df["crime"]     = df["cleaned"].apply(classify_crime)
        df["compound"]  = df["scores"].apply(lambda x: x["compound"])
        elapsed = round(time.time() - t0, 2)

    st.success(f"✅ Processed **{len(df):,}** tweets in **{elapsed}s**")

    # ── KPI row ───────────────────────────────────────────────────────────────
    pos_pct = round((df["sentiment"] == "positive").mean() * 100, 1)
    neg_pct = round((df["sentiment"] == "negative").mean() * 100, 1)
    crime_pct = round((df["crime"] != "none").mean() * 100, 1)
    avg_compound = round(df["compound"].mean(), 3)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><h2>{pos_pct}%</h2><p>Positive</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><h2>{neg_pct}%</h2><p>Negative</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><h2>{crime_pct}%</h2><p>Crime Flagged</p></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="metric-card"><h2>{avg_compound}</h2><p>Avg Compound</p></div>', unsafe_allow_html=True)

    st.divider()

    # ── Row 1: Sentiment Pie + Emotion Bar ───────────────────────────────────
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("Sentiment Distribution")
        st.plotly_chart(plot_sentiment_pie(df), use_container_width=True)
    with r1c2:
        st.subheader("Emotion Distribution")
        st.plotly_chart(plot_emotion_bar(df), use_container_width=True)

    # ── Row 2: Crime Heatmap + Word Cloud ────────────────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("Crime Category Heatmap")
        st.plotly_chart(plot_crime_heatmap(df), use_container_width=True)

    with r2c2:
        st.subheader("Word Cloud")
        crime_filter = st.selectbox("Filter by crime", ["All"] + sorted(df["crime"].unique().tolist()))
        wc_df = df if crime_filter == "All" else df[df["crime"] == crime_filter]
        text = " ".join(wc_df["cleaned"].tolist())
        if text.strip():
            wc = WordCloud(width=800, height=400, background_color="#0f0e17",
                           colormap="cool", max_words=150).generate(text)
            fig_wc, ax = plt.subplots(figsize=(8, 4))
            fig_wc.patch.set_facecolor("#0f0e17")
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
        else:
            st.info("No text available for selected filter.")

    # ── Timeline ─────────────────────────────────────────────────────────────
    if "time" in df.columns:
        st.subheader("📅 Sentiment Over Time")
        st.plotly_chart(plot_timeline(df), use_container_width=True)

    # ── Data Table ────────────────────────────────────────────────────────────
    st.subheader("🔬 Tweet Explorer")
    sent_filter = st.multiselect("Filter sentiment", ["positive", "negative", "neutral"],
                                  default=["positive", "negative", "neutral"])
    disp = df[df["sentiment"].isin(sent_filter)][["tweets", "sentiment", "emotion", "crime", "compound"]].copy()
    disp["sentiment"] = disp["sentiment"].apply(lambda x: sentiment_badge(x))
    disp["crime"]     = disp["crime"].apply(crime_badge)
    st.write(disp.to_html(escape=False, index=False), unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    csv_out = df[["tweets", "cleaned", "sentiment", "emotion", "crime", "compound"]].to_csv(index=False)
    st.download_button("⬇️ Download Results CSV", csv_out, "results.csv", "text/csv", use_container_width=True)
