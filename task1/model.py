import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\aslan\Documents\python\internship\task1\train.csv")

# Select features
df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'salary', 'age']]
df = df.dropna()

# Split data
X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Custom prediction
sample = np.array([[2000, 3, 2]])
print("Predicted Price:", model.predict(sample)[0])

# Line graph (sorted by age so the line doesn't zigzag)
df_sorted = df.sort_values('age')
plt.plot(df_sorted['age'], df_sorted['salary'], marker='o')

plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

# Actual vs Predicted scatter plot
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.show()