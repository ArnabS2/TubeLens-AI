import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================

df = pd.read_csv(
    "youtube_dataset_5k.csv"
)

print(df.head())

# =========================
# CLEAN DATA
# =========================

df = df.dropna()

# =========================
# DATETIME FEATURES
# =========================

df['publishedAt'] = pd.to_datetime(
    df['publishedAt']
)

df['upload_hour'] = (
    df['publishedAt'].dt.hour
)

df['upload_month'] = (
    df['publishedAt'].dt.month
)

# =========================
# ENGAGEMENT FEATURE
# =========================

df['engagement_rate'] = (

    (df['likes'] + df['comments'])

    / df['views']

)

# REMOVE INF VALUES
df = df.replace(
    [float("inf"), -float("inf")],
    0
)

# =========================
# VIRAL LABEL
# =========================

viral_threshold = df['views'].median()

df['viral'] = (
    df['views'] > viral_threshold
).astype(int)

print(
    df['viral'].value_counts()
)

# =========================
# FEATURES
# =========================

X = df[[

    'title_length',
    'hashtag_count',
    'tag_count',
    'likes',
    'comments',
    'upload_hour',
    'upload_month',
    'engagement_rate'

]]

# =========================
# TARGET
# =========================

y = df['viral']

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.2,
    random_state=42

)

# =========================
# MODEL
# =========================

model = XGBClassifier(

    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42

)

# =========================
# TRAIN MODEL
# =========================

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICTIONS
# =========================

predictions = model.predict(
    X_test
)

# =========================
# ACCURACY
# =========================

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    f"\nAccuracy: {accuracy}"
)

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "future_viral_predictor.pkl"
)

print(
    "\nModel Saved Successfully"
)