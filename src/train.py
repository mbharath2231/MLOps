import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('../data/sampregdata.csv')

X = df[['x4']]
y = df['y']

model = LinearRegression()
model.fit(X, y)

print(f"Model 1 R-squared: {r2_score(y, model.predict(X)):.4f}")

joblib.dump(model, '../models/current_model.pkl')