import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime

# Read dataset
df = pd.read_csv("file:///home/iamsy/Project/output/cleaned_dataset/part-00000-8586b7f3-deed-4f23-86a6-3960fe488fbf-c000.csv", on_bad_lines='warn')

# Drop duplicates based on video_id
df = df.drop_duplicates(subset="video_id")

# Convert date strings to datetime
df["published_at"] = pd.to_datetime(df["published_at"], errors='coerce')
df["fetch_date"] = pd.to_datetime(df["fetch_date"], errors='coerce')

# Current date for feature engineering
current_date = pd.to_datetime(datetime.now().date())

# Calculate time-based features
df["days_since_published"] = (current_date - df["published_at"]).dt.days

# Fill NaNs with 0 or a reasonable default
df.fillna({
    "likes": 0,
    "comments": 0,
    "views": 0,
    "days_since_published": 0,
    "title": ""
}, inplace=True)

# Feature engineering: ratios and datetime features
df["like_ratio"] = df["likes"] / (df["views"] + 1)
df["comment_ratio"] = df["comments"] / (df["views"] + 1)
df["title_length"] = df["title"].astype(str).apply(len)
df["day_of_week"] = df["published_at"].dt.dayofweek

# Handle NaN in hour_of_day
df["hour_of_day"] = df["published_at"].dt.hour.fillna(0).astype(int)

# Encode categorical columns
le_channel = LabelEncoder()
le_country = LabelEncoder()
df["channel_index"] = le_channel.fit_transform(df["channel"].astype(str))
df["country_index"] = le_country.fit_transform(df["country"].astype(str))

# Save mappings for reverse lookup
channel_map = dict(zip(df["channel_index"], df["channel"]))
country_map = dict(zip(df["country_index"], df["country"]))

# Features and target
feature_cols = [
    "channel_index", "country_index", "likes", "comments", "days_since_published",
    "like_ratio", "comment_ratio", "title_length", "day_of_week", "hour_of_day"
]
X = df[feature_cols]
y = df["views"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X_scaled, y, df, test_size=0.2, random_state=42
)

# Gradient Boosting Regressor for better accuracy
model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.02, max_depth=8, subsample=0.9, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save predictions with meta info for Tableau
output_df = df_test.copy()
output_df["prediction"] = y_pred
output_df["actual_views"] = y_test.values
output_df.to_csv("/home/iamsy/Project/output/predictions_for_tableau1.csv", index=False)
