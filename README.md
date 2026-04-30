# 🧠 Human Behavior Sentiment Analysis using NLP

## 📌 Project Overview

The **Human Behavior Sentiment Analysis System** analyzes social media tweets to understand public sentiment and behavioral patterns related to crime discussions.

The system processes tweet data using a **Natural Language Processing (NLP) pipeline** including text preprocessing, crime category detection, sentiment analysis, emotion detection, and visualization.

An **interactive Streamlit dashboard** allows users to explore insights, visualize crime-related sentiment trends, and analyze new tweets in real time.

---
## 🚀 Features

- Tweet text preprocessing and cleaning
- Crime category detection (Phishing, Scam, Cyber Crime, Fraud, etc.)
- Sentiment analysis using **VADER Sentiment Analyzer**
- Emotion detection using **TextBlob**
- Interactive **Streamlit dashboard**
- Sentiment and emotion distribution visualizations
- Crime vs Sentiment analysis
- Tweet **Word Cloud visualization**
- Map visualization of tweet locations
- Real-time **Sentiment Predictor**
- Dataset download from the dashboard

---

## 🛠 Technologies Used

- **Python**
- Pandas
- NumPy
- NLTK
- TextBlob
- VADER Sentiment Analyzer
- Matplotlib
- Seaborn
- WordCloud
- Streamlit

---

## ⚙️ Project Workflow

```
tweets.csv
   ↓
collect_data.py
   ↓
cleaned_data.csv
   ↓
preprocess.py
   ↓
sentiment_model.py
   ↓
emotion_model.py
   ↓
visualization.py
   ↓
Streamlit Dashboard (app.py)
```

---

## 📊 Dataset

The dataset contains tweet information used for sentiment and behavior analysis.

Important columns include:

- **tweet** → Original tweet text  
- **clean_text** → Preprocessed tweet text  
- **crime_type** → Detected crime category  
- **city** → Simulated city location  
- **latitude / longitude** → Map coordinates  
- **sentiment** → Predicted sentiment label  
- **sentiment_score** → Sentiment intensity score  
- **emotion** → Detected emotional tone  

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 2️⃣ Run Data Processing

```
python collect_data.py
python preprocess.py
python crime_mapper.py
python sentiment_model.py
python emotion_model.py
```

### 3️⃣ Launch the Streamlit Dashboard

```
streamlit run app.py
```

The application will open in your browser.

---

## 📊 Dashboard Features

The Streamlit dashboard allows users to:

- View dataset statistics
- Analyze sentiment distribution
- Explore crime vs sentiment relationships
- Visualize crime distribution by city
- View tweet word cloud
- Explore tweet locations on a map
- Predict sentiment for new tweets

---

## 🔮 Future Improvements

- Integrate **deep learning models (LSTM / BERT)**
- Add **real-time tweet collection using Twitter API**
- Improve crime classification using ML models
- Deploy the application online

---

## 📄 License

This project is open-source and intended for **educational and research purposes**.