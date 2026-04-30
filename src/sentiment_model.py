"""
sentiment_model.py – VADER + TextBlob hybrid sentiment analysis
"""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

_vader = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str):
    """
    Returns (label, scores_dict).
    label: 'positive' | 'negative' | 'neutral'
    scores_dict: {'pos', 'neg', 'neu', 'compound', 'tb_polarity', 'tb_subjectivity'}
    """
    scores = _vader.polarity_scores(text)
    tb = TextBlob(text)
    scores["tb_polarity"] = tb.sentiment.polarity
    scores["tb_subjectivity"] = tb.sentiment.subjectivity

    # Hybrid decision: VADER compound weighted with TextBlob
    hybrid = 0.7 * scores["compound"] + 0.3 * scores["tb_polarity"]

    if hybrid >= 0.05:
        label = "positive"
    elif hybrid <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return label, scores
