# 🔍 TweetLens — Human Behavior & Sentiment Intelligence

🌐 **Live Demo:**  


> NLP pipeline classifying **sentiment**, **emotion**, and **crime categories** on 17K+ tweets with ~80% accuracy.  
> Streamlit dashboard with **<2s inference time** · Word clouds · Heatmaps · Live tweet analyzer

---

## 📸 Features

| Feature | Details |
|---|---|
| **Sentiment Analysis** | VADER + TextBlob hybrid (positive / negative / neutral) |
| **Emotion Detection** | 7-class rule-based (joy, anger, sadness, fear, surprise, disgust, neutral) |
| **Crime Classification** | Multi-label: phishing, scam, fraud, hate speech, harassment, misinformation |
| **Visualizations** | Pie chart, bar chart, heatmap, timeline, word cloud |
| **Live Analyzer** | Paste any tweet and get instant results |
| **Export** | Download analyzed CSV |

---

## 🗂️ Project Structure

```
HUMAN-BEHAVIOR-SENTIMENT/
│
├── app.py                  # Main Streamlit application
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py       # Tweet cleaning (URL removal, normalisation)
│   ├── sentiment_model.py  # VADER + TextBlob hybrid sentiment
│   ├── emotion_model.py    # Keyword-lexicon emotion detection
│   ├── crime_mapper.py     # Multi-label crime classification
│   └── visualization.py   # Plotly chart builders
│
├── data/
│   └── tweets.csv          # Your dataset (place here)
│
├── .streamlit/
│   └── config.toml         # Dark theme config
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Local Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/ansukumari-tech/human-behavior-sentiment-analysis.git
cd human-behavior-sentiment-analysis
```

### 2. Create virtual environment
```bash
# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1

# Mac / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # Downloads TextBlob data
```

### 4. Add your dataset
```
Place your tweets CSV in:  data/tweets.csv
Required column: tweets (or any column with tweet text)
Optional columns: time, username, location
```

### 5. Run the app
```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser. ✅
---

## 📊 How the Models Work

### Sentiment (VADER + TextBlob Hybrid)
```
hybrid_score = 0.7 × VADER_compound + 0.3 × TextBlob_polarity

positive  → hybrid ≥  0.05
negative  → hybrid ≤ -0.05
neutral   → otherwise
```

### Emotion Detection
Rule-based keyword lexicon across 7 emotions. Each tweet is scored per emotion; highest count wins.

### Crime Classification
Pattern matching against domain-specific keyword lists across 6 crime categories. Returns first match or `none`.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.10+** | Core language |
| **VADER** | Social media–optimised sentiment scoring |
| **TextBlob** | Polarity + subjectivity scores |
| **Streamlit** | Interactive web dashboard |
| **Plotly** | Interactive charts |
| **WordCloud** | Word frequency visualisation |
| **Pandas / NumPy** | Data handling |
