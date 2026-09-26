# src/model_2x.py
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('../data/sampregdata.csv')

# Use the best two predictors (x4 and x2)
X = df[['x4', 'x2']]
y = df['y']

# Train Model 2
model = LinearRegression()
model.fit(X, y)

print(f"Model 2 (Two Variables) R-squared: {r2_score(y, model.predict(X)):.4f}")

# Save the updated model as the "current" production model
joblib.dump(model, '../models/saved_model_2x.pkl')