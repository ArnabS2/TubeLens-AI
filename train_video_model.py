import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# =========================
# LOAD DATASET
# =========================

print("=" * 50)
print("YouTube Videos Viral Predictor")
print("Training Script — videos_model.pkl")
print("=" * 50)

df = pd.read_csv(
    "youtube_videos_dataset.csv"
)

print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} cols")
print(df.head())

# =========================
# CLEAN DATA
# =========================

df = df.dropna()

print(f"\nAfter dropna: {df.shape[0]} rows")

# =========================
# DATETIME FEATURES
# =========================

df['publishedAt'] = pd.to_datetime(
    df['publishedAt'],
    utc=True,
    errors='coerce'
)

df = df.dropna(subset=['publishedAt'])

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
    / df['views'].clip(lower=1)
)

# remove inf / nan
df = df.replace(
    [float("inf"), -float("inf")],
    0
)

df = df.dropna()

# =========================
# VIRAL LABEL
# =========================

viral_threshold = df['views'].median()

print(f"\nViral threshold (median views): {viral_threshold:,.0f}")

df['viral'] = (
    df['views'] > viral_threshold
).astype(int)

print("\nClass distribution:")
print(df['viral'].value_counts())
print(f"Viral ratio: {df['viral'].mean():.2%}")

# =========================
# FEATURES
# — exact same 8 features
#   as shorts model
#   so app code stays clean
# =========================

FEATURES = [
    'title_length',
    'hashtag_count',
    'tag_count',
    'likes',
    'comments',
    'upload_hour',
    'upload_month',
    'engagement_rate',
]

X = df[FEATURES]
y = df['viral']

print(f"\nFeatures used: {FEATURES}")
print(f"X shape: {X.shape}")

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y        # balanced split
)

print(f"\nTrain size : {X_train.shape[0]}")
print(f"Test size  : {X_test.shape[0]}")

# =========================
# MODEL
# — same hyperparams as
#   shorts model
# =========================

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss',
    use_label_encoder=False
)

# =========================
# TRAIN
# =========================

print("\nTraining model...")

model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("Training complete!")

# =========================
# EVALUATE
# =========================

predictions  = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)

print("\n" + "=" * 50)
print(f"Accuracy : {accuracy:.4f} ({accuracy*100:.2f}%)")
print("=" * 50)

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    target_names=["Not Viral", "Viral"]
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, predictions)
print(cm)

# =========================
# FEATURE IMPORTANCE
# =========================

importances = model.feature_importances_
feat_imp = sorted(
    zip(FEATURES, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nFeature Importance:")
for feat, imp in feat_imp:
    bar = "█" * int(imp * 50)
    print(f"  {feat:<20} {imp:.4f}  {bar}")

# =========================
# SAVE MODEL
# =========================

joblib.dump(
    model,
    "videos_model.pkl"
)

print("\n" + "=" * 50)
print("Model saved: videos_model.pkl")
print("=" * 50)
print("\nNow update app.py to load both models:")
print("  shorts_model = joblib.load('future_viral_predictor.pkl')")
print("  videos_model = joblib.load('videos_model.pkl')")
