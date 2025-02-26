# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Boston housing dataset
boston = load_boston()

# Create a DataFrame with the dataset
df = pd.DataFrame(boston.data, columns=boston.feature_names)

# Add the target (house prices) as the last column
df['Price'] = boston.target

# Display the first few rows of the dataset
print(df.head())

# Visualize the distribution of house prices
plt.figure(figsize=(8,6))
sns.histplot(df['Price'], kde=True)
plt.title("Distribution of House Prices")
plt.show()

# Split the data into training and test sets
X = df.drop('Price', axis=1)
y = df['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the Mean Squared Error and R-squared value
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Plot the true vs predicted prices
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel('True Prices')
plt.ylabel('Predicted Prices')
plt.title('True Prices vs Predicted Prices')
plt.show()
