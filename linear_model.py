import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate sample data for regression
X, y = make_regression(n_samples=100, n_features=1, noise=10)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict the target variable for new data
new_data = np.array([[1.5]])
prediction = model.predict(new_data)

print("Prediction:", prediction)
