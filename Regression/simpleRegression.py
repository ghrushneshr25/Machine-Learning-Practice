from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# PREPROCESSING STEPS

dataSet = pd.read_csv("Salary_Data.csv")
x = dataSet.iloc[:, :-1].values
y = dataSet.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

"print(x_train, x_test, y_train, y_test)"

# PREPROCESSING ENDS HERE

# TRAINING THE SIMPLE LINEAR REGRESSION MODEL ON THE TRAINING SET

regressor = LinearRegression()
regressor.fit(x_train, y_train)

# PREDICTING THE TEST SET RESULT

y_pred = regressor.predict(x_test)

# VISUALIZING THE TRAINING SET RESULTS

plt.scatter(x_train, y_train, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Train Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# VISUALIZING THE TEST SET RESULTS

plt.scatter(x_test, y_test, color="red")
plt.plot(x_train, regressor.predict(x_train), color="blue")
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.grid(b=None, which='major', axis='both')
plt.show()
