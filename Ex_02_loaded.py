import joblib
import numpy as np
# Load the model from disk
loaded_model = joblib.load('linear_model.joblib')
print("Model loaded successfully!")

new_X = int(input("Input your number? "))
new_X = np.array([[new_X]])
loaded_model = joblib.load('linear_model.joblib')
prediction = loaded_model.predict(new_X)
print("Predictions on new data:", prediction)
