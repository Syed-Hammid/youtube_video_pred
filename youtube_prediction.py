import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
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
df["days_since_fetch"] = (current_date - df["fetch_date"]).dt.days

# Fill NaNs with 0 or a reasonable default
df.fillna({
    "likes": 0,
    "comments": 0,
    "days_since_published": 0,
    "days_since_fetch": 0
}, inplace=True)

# Encode categorical columns
le_channel = LabelEncoder()
le_country = LabelEncoder()
df["channel_index"] = le_channel.fit_transform(df["channel"].astype(str))
df["country_index"] = le_country.fit_transform(df["country"].astype(str))

# Features and target
feature_cols = ["channel_index", "country_index", "likes", "comments", "days_since_published", "days_since_fetch"]
X = df[feature_cols]
y = df["views"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save predictions to CSV for Tableau
output_df = pd.DataFrame({
    "prediction": y_pred,
    "label": y_test.values
})
output_df.to_csv("/home/iamsy/Project/output/predictions_for_tableau.csv", index=False)
