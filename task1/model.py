import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv(r"C:\Users\aslan\Documents\python\internship\task1\train.csv")

df = df[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'salary', 'age']]
df = df.dropna()

X = df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = df['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

sample = np.array([[2000, 3, 2]])
print("Predicted Price:", model.predict(sample)[0])

df_sorted = df.sort_values('age')
plt.plot(df_sorted['age'], df_sorted['salary'], marker='o')

plt.xlabel("Age")
plt.ylabel("Salary")
plt.title("Age vs Salary")
plt.show()

plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Salary")
plt.ylabel("Predicted Salary")
plt.title("Actual vs Predicted")
plt.show()