# Future YouTube Viral Predictor AI

An AI-powered web application that predicts whether a YouTube Short has the potential to go viral using Machine Learning and YouTube API data.

---

## Features

- Predict future virality of YouTube Shorts
- Paste any YouTube Shorts URL
- Real-time YouTube API integration
- AI-based viral probability score
- Engagement analysis
- Future reach estimation
- Interactive Streamlit UI
- Feature importance analytics

---

## Tech Stack

- Python
- Streamlit
- XGBoost
- Scikit-learn
- Pandas
- NumPy
- YouTube Data API v3

---

## Machine Learning

The model was trained using:

- Title Length
- Hashtag Count
- Tag Count
- Likes
- Comments
- Upload Hour
- Upload Month
- Engagement Rate

Algorithm Used:
- XGBoost Classifier

---

## Project Structure

```text
.
├── app.py
├── train_future_model.py
├── future_viral_predictor.pkl
├── requirements.txt
├── logo.png
├── feature_importance.png
└── README.md
