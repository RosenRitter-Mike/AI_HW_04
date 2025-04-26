import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Create simple data with 10 data points
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)  # reshape for sklearn
y = np.array([2, 4, 5, 4, 5, 7, 8, 9, 11, 12])

# Fit the model
model = LinearRegression()
model.fit(X, y)
print("Model fitted")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

new_data = np.array([5]).reshape(-1, 1)
new_predictions = model.predict(new_data)
print("Predictions on new data:", new_predictions)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])
model.fit(X, y)
joblib.dump(model, 'linear_model.joblib')

print("Model saved successfully!")