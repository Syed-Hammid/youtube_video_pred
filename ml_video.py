import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# 1. Simulate a dataset (replace this with your own CSV if available)
# For real usage: df = pd.read_csv('your_cleaned_dataset.csv')
df = pd.read_csv("file:///home/iamsy/Project/output/cleaned_dataset/part-00000-8586b7f3-deed-4f23-86a6-3960fe488fbf-c000.csv", on_bad_lines='warn')

# 2. Drop missing target values
df.dropna(subset=["views"], inplace=True)

# 3. Split features and target
X = df.drop(columns=["views"])
y = df["views"]

# 4. Identify column types
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# 5. Define preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
])

# 6. Build pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42))
])

# 7. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 8. Train model
pipeline.fit(X_train, y_train)

# 9. Predict and evaluate
y_pred = pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Root Mean Squared Error:", rmse)
print("R² Score:", r2)

# 10. Add prediction to original data and save for Tableau
df["prediction"] = pipeline.predict(X)
df.to_csv("/home/iamsy/Project/youtube_predictions_for_tableau.csv", index=False)
