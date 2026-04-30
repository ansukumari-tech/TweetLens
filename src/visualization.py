"""
visualization.py – Plotly chart builders for the Streamlit dashboard
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

SENTIMENT_COLORS = {
    "positive": "#22c55e",
    "negative": "#ef4444",
    "neutral":  "#94a3b8",
}

EMOTION_COLORS = {
    "joy":      "#facc15",
    "anger":    "#ef4444",
    "sadness":  "#60a5fa",
    "fear":     "#a78bfa",
    "surprise": "#fb923c",
    "disgust":  "#4ade80",
    "neutral":  "#94a3b8",
}

CRIME_COLORS = {
    "phishing":       "#f97316",
    "scam":           "#eab308",
    "fraud":          "#ef4444",
    "hate_speech":    "#dc2626",
    "harassment":     "#9333ea",
    "misinformation": "#0ea5e9",
    "none":           "#374151",
}

_DARK = "plotly_dark"


def plot_sentiment_pie(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["sentiment", "count"]
    fig = px.pie(
        counts, names="sentiment", values="count",
        color="sentiment", color_discrete_map=SENTIMENT_COLORS,
        hole=0.45, template=_DARK,
    )
    fig.update_traces(textinfo="label+percent", textfont_size=13)
    fig.update_layout(showlegend=False, margin=dict(t=10, b=10))
    return fig


def plot_emotion_bar(df: pd.DataFrame) -> go.Figure:
    counts = df["emotion"].value_counts().reset_index()
    counts.columns = ["emotion", "count"]
    counts["color"] = counts["emotion"].map(EMOTION_COLORS).fillna("#94a3b8")
    fig = go.Figure(go.Bar(
        x=counts["emotion"], y=counts["count"],
        marker_color=counts["color"].tolist(),
        text=counts["count"], textposition="outside",
    ))
    fig.update_layout(template=_DARK, margin=dict(t=10), xaxis_title="", yaxis_title="Count")
    return fig


def plot_crime_heatmap(df: pd.DataFrame) -> go.Figure:
    crime_sent = df.groupby(["crime", "sentiment"]).size().reset_index(name="count")
    pivot = crime_sent.pivot(index="crime", columns="sentiment", values="count").fillna(0)
    fig = px.imshow(
        pivot, color_continuous_scale="Plasma",
        labels=dict(color="Count"), template=_DARK,
        aspect="auto",
    )
    fig.update_layout(margin=dict(t=10))
    return fig


def plot_timeline(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["time"] = pd.to_datetime(df2["time"], errors="coerce")
    df2 = df2.dropna(subset=["time"])
    df2["date"] = df2["time"].dt.date
    grp = df2.groupby(["date", "sentiment"]).size().reset_index(name="count")
    fig = px.line(
        grp, x="date", y="count", color="sentiment",
        color_discrete_map=SENTIMENT_COLORS, template=_DARK, markers=True,
    )
    fig.update_layout(margin=dict(t=10), xaxis_title="Date", yaxis_title="Tweet Count")
    return fig
