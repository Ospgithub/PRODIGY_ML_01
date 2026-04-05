import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\aslan\Documents\python\internship\train.csv")

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

# line graph
plt.plot(df['age'], df['salary'])

plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")

plt.show()