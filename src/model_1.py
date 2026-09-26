# src/model_1x.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('../data/sampregdata.csv')

# Use the single best predictor (x4)
X = df[['x4']]
y = df['y']

# Train Model 1
model = LinearRegression()
model.fit(X, y)

print(f"Model 1 (Single Variable) R-squared: {r2_score(y, model.predict(X)):.4f}")

# Save the model
joblib.dump(model, '../models/saved_model_1x.pkl')